"""SFT runner for Qwen3-8B (Unsloth + TRL).

This script is **GPU-only**. It is committed in skeleton form so that
once a GPU host is available, the user runs::

    uv run python -m exact_agent.train.run_sft_qwen \\
        --config configs/training/sft_qwen3_8b.yaml

…and gets a LoRA adapter dropped into ``outputs/sft_qwen3_8b/``.

The implementation deliberately depends on ``unsloth`` (and therefore on
CUDA), so we import lazily — the whole module is importable on a CPU box
for tests; the heavy work fires only inside :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_gpu_available() -> None:
    """Fail loudly on CPU-only hosts before we touch heavyweight imports."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        print(f"[fatal] torch is required but not installed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if not torch.cuda.is_available():
        print(
            "[fatal] CUDA GPU not detected. SFT requires a GPU; spin up a Runpod / "
            "Colab / Vast.ai instance and re-run.",
            file=sys.stderr,
        )
        raise SystemExit(3)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "training" / "sft_qwen3_8b.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + dataset existence; do not load model or train.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    print(f"[info] using config {args.config}")

    dataset_path = Path(cfg.get("dataset_path", ""))
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path
    if not dataset_path.exists():
        print(
            f"[fatal] training dataset missing: {dataset_path}\n"
            "  -> run: uv run python -m exact_agent.train.prepare_sft",
            file=sys.stderr,
        )
        return 1

    train_rows = _read_jsonl(dataset_path)
    print(f"[info] dataset rows: {len(train_rows)}")

    if args.dry_run:
        print("[ok] dry-run passed; config + dataset look usable.")
        return 0

    _ensure_gpu_available()
    return _run_unsloth_sft(cfg)


# ---------------------------------------------------------------------------
# Heavyweight: Unsloth model load + training. Imported lazily so the file is
# importable on CPU-only CI machines.
# ---------------------------------------------------------------------------


def _run_unsloth_sft(cfg: dict[str, Any]) -> int:
    """Run QLoRA SFT via Unsloth + TRL. Kept slim — see Unsloth's own docs.

    Imports are deliberately lazy: this function only fires inside ``main()``
    after the GPU check, so importing on a CPU box still works for tests.
    """
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model_id = cfg.get("base_model", "Qwen/Qwen3-8B")
    quant = cfg.get("quantization", {})
    lora = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})

    print(f"[info] loading {model_id} via Unsloth (4bit={quant.get('load_in_4bit', True)}) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=train_cfg.get("max_seq_length", 2048),
        load_in_4bit=quant.get("load_in_4bit", True),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        random_state=cfg.get("seed", 42),
    )

    train_path = REPO_ROOT / "data" / "processed" / "sft_chat_train.jsonl"
    val_path = REPO_ROOT / "data" / "processed" / "sft_chat_val.jsonl"
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )

    output_dir = REPO_ROOT / train_cfg.get("output_dir", "outputs/sft_qwen3_8b")
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 50),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        bf16=train_cfg.get("bf16", True),
        packing=train_cfg.get("packing", True),
        max_seq_length=train_cfg.get("max_seq_length", 2048),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=sft_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    print(f"[ok] adapter saved to {output_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
