#!/usr/bin/env python3
"""P1 — capture SAE features + solver ground truth per sample (orchestrator).

Two halves with very different requirements:

* **Ground truth** (CPU): runs the symbolic solver to derive the concepts each
  sample uses. Needs the physics/logic extras (sympy/pint/z3) but NO GPU.
* **Activations** (GPU): loads the base model + Qwen-Scope SAE, hooks the
  residual stream, decodes the active TopK features, maps them to concepts.

Run ``--ground-truth-only`` on a CPU box to validate the data path + measure
ground-truth coverage before touching a GPU. The full run fills ``active_*``.

Deps for the GPU run:  pip install torch transformers huggingface_hub
                       (+ bitsandbytes only if --load-in-4bit)

Examples:
    # CPU dry-run (coverage check, no model)
    python scripts/interp/capture_activations.py --task physics \
      --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \
      --layer 18 --out outputs/interp/records_physics_l18.jsonl --ground-truth-only

    # Full run on Colab (dev preset = 2B, fits any GPU)
    python scripts/interp/capture_activations.py --task physics --preset qwen3.5-2b \
      --split .../physics_eval_sft_unseen.jsonl --layer 12 \
      --out outputs/interp/records_physics_l12.jsonl --labels outputs/interp/labels.json

    # Headline run (8B, needs ~24GB; add --load-in-4bit on a 16GB T4)
    python scripts/interp/capture_activations.py --task logic --preset qwen3-8b \
      --split .../logic_eval.jsonl --layer 18 --out outputs/interp/records_logic_l18.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from exact_agent.interp import ground_truth as gt  # noqa: E402
from exact_agent.interp.features import (  # noqa: E402
    concepts_from_features,
    load_label_cache,
    sae_encode,
)
from exact_agent.interp.records import ActivationRecord  # noqa: E402


def _load_split(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _ground_truth(task: str, sid: str, row: dict) -> gt.GroundTruthConcepts:
    if task == "physics":
        return gt.ground_truth_for_physics(sid, str(row.get("question", "")))
    return gt.ground_truth_for_logic(
        sid,
        str(row.get("question", "")),
        list(row.get("premises-NL") or row.get("premises_NL") or []),
        list(row.get("premises-FOL") or row.get("premises_FOL") or []),
    )


def _build_prompt(task: str, row: dict) -> str:
    """Plain-text input we capture activations over (Base model, v1).

    Physics: the question. Logic: premises then the question — so the
    residual stream represents the domain concepts the solver will use.
    """
    q = str(row.get("question", ""))
    if task == "logic":
        prem = list(row.get("premises-NL") or row.get("premises_NL") or [])
        if prem:
            return "\n".join(prem) + "\n" + q
    return q


def _build_cfg(args):  # type: ignore[no-untyped-def]
    from exact_agent.interp.sae_loader import PRESETS, SAEConfig  # noqa: PLC0415

    if args.preset:
        base = PRESETS[args.preset]
        return SAEConfig(
            base_model=args.model or base.base_model,
            sae_repo=args.sae_repo or base.sae_repo,
            top_k=args.top_k or base.top_k,
            layers=(args.layer,),
            device=args.device,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
        )
    return SAEConfig(
        base_model=args.model or SAEConfig.base_model,
        sae_repo=args.sae_repo or SAEConfig.sae_repo,
        top_k=args.top_k or 50,
        layers=(args.layer,),
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["physics", "logic"], required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=None, help="autointerp label cache JSON")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ground-truth-only", action="store_true",
                    help="skip the GPU activation capture (CPU validation run)")
    # model selection (charter §10 decision #1) — preset or explicit override
    ap.add_argument("--preset", choices=["qwen3-8b", "qwen3.5-2b", "qwen3-1.7b"], default=None)
    ap.add_argument("--model", default=None, help="override base model id")
    ap.add_argument("--sae-repo", default=None, help="override Qwen-Scope SAE repo id")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-features", type=int, default=None,
                    help="cap active features per sample (most-frequent first)")
    args = ap.parse_args()

    rows = _load_split(args.split, args.limit)
    labels = load_label_cache(args.labels) if args.labels else {}

    model = tok = sae = None
    if not args.ground_truth_only:
        # ---- GPU path -------------------------------------------------------
        import torch  # noqa: PLC0415

        from exact_agent.interp.hooks import capture_residual  # noqa: PLC0415
        from exact_agent.interp.sae_loader import (  # noqa: PLC0415
            load_model_and_tokenizer,
            load_sae,
        )
        cfg = _build_cfg(args)
        print(f"loading {cfg.base_model} + SAE {cfg.sae_repo} layer {args.layer} ...")
        model, tok = load_model_and_tokenizer(cfg)
        sae = load_sae(cfg, args.layer)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with args.out.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            sid = str(row.get("id", row.get("sample_id", i)))
            g = _ground_truth(args.task, sid, row)
            n_ok += int(g.solver_ok)

            active_ids: list[int] = []
            active_concepts: list[str] = []
            if model is not None:
                prompt = _build_prompt(args.task, row)
                enc = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad(), capture_residual(model, [args.layer]) as store:
                    model(**enc)
                residual = store[args.layer][0]  # [seq, d_model]
                active_ids = sae_encode(residual, sae, max_features=args.max_features)
                active_concepts = sorted(concepts_from_features(active_ids, labels))

            rec = ActivationRecord(
                sample_id=sid, task_type=args.task, question=str(row.get("question", "")),
                layer=args.layer, active_feature_ids=active_ids,
                active_concepts=active_concepts, gold_concepts=g.concepts,
                solver_ok=g.solver_ok, solver_meta={"gt_source": g.source},
            )
            out.write(rec.to_json() + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)} ...")

    print(f"wrote {len(rows)} records -> {args.out}  (solver_ok on {n_ok}/{len(rows)})")
    if args.ground_truth_only:
        print("ground-truth-only run: active_* fields are empty by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
