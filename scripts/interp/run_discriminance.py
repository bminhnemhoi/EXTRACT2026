#!/usr/bin/env python3
"""RQ1-v0 (label-free) — do active SAE features discriminate the solver class?

Pure metric (no GPU, no autointerp labels). Reads the JSONL from
``capture_activations.py`` and reports, per solver gold_label (physics:
formula_id; logic: answer): same-class vs different-class feature overlap, an
AUC, and leave-one-out 1-NN accuracy vs the majority baseline.

Usage:
    python scripts/interp/run_discriminance.py \
        --records outputs/interp/records_physics_l12.jsonl \
        --out outputs/interp/discriminance_physics_l12
    python scripts/interp/run_discriminance.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from exact_agent.interp import discriminance as dc  # noqa: E402
from exact_agent.interp.records import ActivationRecord, read_records  # noqa: E402


def _samples(records: list[ActivationRecord]) -> list[tuple[set[int], str]]:
    out: list[tuple[set[int], str]] = []
    for rec in records:
        if not rec.gold_label or not rec.active_feature_ids:
            continue
        out.append((set(rec.active_feature_ids), rec.gold_label))
    return out


def _render_md(name: str, sep: dc.Separation, nn: dc.NNResult, n_classes: int) -> str:
    return (
        f"# RQ1-v0 (label-free) — {name}\n\n"
        f"- samples used: **{nn.n}**  ·  classes: **{n_classes}**  ·  pairs: {sep.n_pairs}\n"
        f"- within-class overlap: **{sep.within_mean:.3f}**  ·  "
        f"across-class: **{sep.across_mean:.3f}**  ·  gap: **{sep.gap:+.3f}**\n"
        f"- separation AUC: **{sep.auc:.3f}**  (0.5 = no signal)\n"
        f"- LOO 1-NN accuracy: **{nn.accuracy:.3f}**  vs majority **{nn.majority_baseline:.3f}**  "
        f"(lift **{nn.lift:+.3f}**)\n\n"
        "> RQ1-v0 success (charter §8): AUC well above 0.5 AND 1-NN lift > 0. "
        "This is the no-autointerp signal; the lexical alignment (run_alignment) "
        "adds the human-readable layer once labels exist.\n"
    )


def _selfcheck() -> int:
    # two clean classes with disjoint feature sets -> strong signal
    recs = []
    for i in range(6):
        recs.append(ActivationRecord(
            sample_id=f"a{i}", task_type="physics", question="q", layer=12,
            active_feature_ids=[1, 2, 3, i], active_concepts=[], gold_concepts=[],
            solver_ok=True, gold_label="capacitor_energy"))
    for i in range(6):
        recs.append(ActivationRecord(
            sample_id=f"b{i}", task_type="physics", question="q", layer=12,
            active_feature_ids=[7, 8, 9, 100 + i], active_concepts=[], gold_concepts=[],
            solver_ok=True, gold_label="ohms_law"))
    s = _samples(recs)
    sep, nn = dc.jaccard_separation(s), dc.loo_1nn_accuracy(s)
    print(_render_md("selfcheck", sep, nn, 2))
    assert sep.auc > 0.9 and nn.lift > 0.0
    print("selfcheck OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return _selfcheck()
    if not args.records or not args.out:
        ap.error("--records and --out are required (or use --selfcheck)")

    with args.records.open(encoding="utf-8") as f:
        samples = _samples(read_records(f))
    sep = dc.jaccard_separation(samples)
    nn = dc.loo_1nn_accuracy(samples)
    n_classes = len({lab for _, lab in samples})
    md = _render_md(args.records.stem, sep, nn, n_classes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"separation": sep.__dict__, "nn": nn.__dict__,
                    "n_classes": n_classes}, indent=2), encoding="utf-8")
    print(md)
    print(f"wrote {args.out.with_suffix('.md')} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
