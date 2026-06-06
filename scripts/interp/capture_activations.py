#!/usr/bin/env python3
"""P1 — capture SAE features + solver ground truth per sample (orchestrator).

Two halves with very different requirements:

* **Ground truth** (CPU): runs the symbolic solver to derive the concepts each
  sample uses. Needs the physics/logic extras (sympy/pint/z3) but NO GPU.
* **Activations** (GPU): loads the base model + Qwen-Scope SAE, hooks the
  residual stream, and decodes top-k features. This is the only part that
  requires a GPU and is currently a clearly-isolated NotImplementedError
  (see exact_agent.interp.sae_loader / hooks / features.sae_encode).

Run ``--ground-truth-only`` on a CPU box to produce a records file with empty
``active_*`` fields (useful to validate the data path + ground-truth coverage
before touching a GPU). The full run fills in the SAE side.

Usage:
    python scripts/interp/capture_activations.py \
        --task physics \
        --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \
        --layer 18 --out outputs/interp/records_physics_layer18.jsonl \
        [--ground-truth-only] [--labels outputs/interp/feature_labels.json] [--limit N]
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
    args = ap.parse_args()

    rows = _load_split(args.split, args.limit)
    labels = load_label_cache(args.labels) if args.labels else {}

    sae_runtime = None
    if not args.ground_truth_only:
        # ---- GPU path (isolated) -------------------------------------------
        from exact_agent.interp.sae_loader import (  # noqa: PLC0415
            SAEConfig,
            load_model_and_tokenizer,
            load_sae,
        )
        cfg = SAEConfig(layers=(args.layer,))
        model, tok = load_model_and_tokenizer(cfg)  # raises until implemented
        sae = load_sae(cfg, args.layer)
        sae_runtime = (cfg, model, tok, sae)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with args.out.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            sid = str(row.get("id", row.get("sample_id", i)))
            g = _ground_truth(args.task, sid, row)
            n_ok += int(g.solver_ok)

            active_ids: list[int] = []
            active_concepts: list[str] = []
            if sae_runtime is not None:
                from exact_agent.interp.features import sae_encode  # noqa: PLC0415
                from exact_agent.interp.hooks import capture_residual  # noqa: PLC0415

                cfg, model, tok, sae = sae_runtime
                prompt = str(row.get("question", ""))
                with capture_residual(model, [args.layer]) as store:
                    _ = model  # TODO(P1): tokenize+forward(prompt) to fill store
                active_ids = sae_encode(store[args.layer], sae)
                active_concepts = sorted(concepts_from_features(active_ids, labels))

            rec = ActivationRecord(
                sample_id=sid, task_type=args.task, question=str(row.get("question", "")),
                layer=args.layer, active_feature_ids=active_ids,
                active_concepts=active_concepts, gold_concepts=g.concepts,
                solver_ok=g.solver_ok, solver_meta={"gt_source": g.source},
            )
            out.write(rec.to_json() + "\n")

    print(f"wrote {len(rows)} records -> {args.out}  (solver_ok on {n_ok}/{len(rows)})")
    if args.ground_truth_only:
        print("ground-truth-only run: active_* fields are empty by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
