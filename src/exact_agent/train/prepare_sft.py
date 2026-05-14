"""Convert ``sft_train_mixed_solver_clean.jsonl`` into a Qwen3 chat-template
SFT dataset, ready for Unsloth / TRL ``SFTTrainer``.

Input rows (from the cleaned dataset) look like::

    {
      "sample_id": "...",
      "task_type": "logic" | "physics",
      "prompt":   {"task": "...", "premises-NL": [...], "question": "..."},
      "completion": {"answer": "...", "explanation": "...", ...}
    }

We render each row into the Qwen3 chat schema:

    [
      {"role": "system",    "content": "You are an explainable QA agent..."},
      {"role": "user",      "content": <serialized prompt>},
      {"role": "assistant", "content": <serialized completion>},
    ]

The system prompt commits the model to (a) JSON-only output, (b) the
canonical schema (answer/explanation/cot/premises/confidence). At inference
time the pipeline parses the JSON; no schema drift.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "cleaned" / "sft_train_mixed_solver_clean.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "processed"

SYSTEM_PROMPT = (
    "You are EXACT 2026 — an explainable educational QA agent. "
    "For every query, return ONLY a single JSON object with these keys: "
    "answer (string), explanation (string), cot (list of strings, optional), "
    "premises (list of premise IDs like 'P1', optional), "
    "confidence (number in [0, 1], optional). "
    "Numerical answers MUST come from the deterministic solver trace — "
    "never invent numbers. For logic tasks, cite the exact premise IDs "
    "you used in the 'premises' field."
)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                yield json.loads(line)


def _format_user_message(prompt: dict) -> str:
    """Stringify the prompt so the LLM sees a stable, machine-readable shape.

    We keep the JSON form because the Phase-4 inference path will feed
    structured payloads through the same template.
    """
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def _format_assistant_message(completion: dict) -> str:
    """Assistant target: the JSON envelope the API also emits."""
    canonical: dict[str, object] = {
        "answer": completion.get("answer", ""),
        "explanation": completion.get("explanation", ""),
    }
    if completion.get("cot"):
        canonical["cot"] = completion["cot"]
    if completion.get("premises"):
        canonical["premises"] = completion["premises"]
    if "confidence" in completion:
        canonical["confidence"] = completion["confidence"]
    return json.dumps(canonical, ensure_ascii=False, indent=2)


def to_chat_record(row: dict) -> dict | None:
    """Render one cleaned-SFT row as a Qwen3 chat-format example."""
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, dict) or not isinstance(completion, dict):
        return None
    if not str(completion.get("answer") or "").strip():
        return None  # skip rows with empty gold answer
    return {
        "sample_id": row.get("sample_id", ""),
        "task_type": row.get("task_type", "unknown"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_user_message(prompt)},
            {"role": "assistant", "content": _format_assistant_message(completion)},
        ],
    }


def prepare(
    input_path: Path = DEFAULT_INPUT,
    out_dir: Path = DEFAULT_OUT,
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path, dict]:
    """Build the chat-format SFT dataset and split into train / val.

    Returns ``(train_path, val_path, stats)``.
    """
    rows: list[dict] = []
    skipped = 0
    for raw in _read_jsonl(input_path):
        record = to_chat_record(raw)
        if record is None:
            skipped += 1
            continue
        rows.append(record)

    rng = random.Random(seed)
    rng.shuffle(rows)
    split = max(1, round(len(rows) * val_ratio))
    val = rows[:split]
    train = rows[split:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "sft_chat_train.jsonl"
    val_path = out_dir / "sft_chat_val.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for row in val:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "input_rows": len(rows) + skipped,
        "skipped_empty_answer": skipped,
        "train": len(train),
        "val": len(val),
    }
    return train_path, val_path, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[fatal] input not found: {args.input}")
        return 1

    train_path, val_path, stats = prepare(
        args.input, args.out, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"[ok] {stats}")
    print(f"[ok] wrote {train_path}")
    print(f"[ok] wrote {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
