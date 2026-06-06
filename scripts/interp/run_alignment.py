#!/usr/bin/env python3
"""P2 / RQ1 — score SAE↔solver concept alignment from captured records.

Runs on the PURE metric modules (no GPU/torch). Input is the JSONL produced by
``capture_activations.py``; output is a markdown + JSON report comparing the
observed alignment against a random baseline (charter §8).

Usage:
    python scripts/interp/run_alignment.py \
        --records outputs/interp/records_physics_layer18.jsonl \
        --out outputs/interp/alignment_physics_layer18 \
        [--probe-records outputs/interp/probe_physics_layer18.jsonl] \
        [--trials 200] [--seed 0]

A tiny synthetic self-check runs with ``--selfcheck`` (no input needed) so the
metric path can be exercised before any activations exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the script runnable without installing the package (mirrors run_eval.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from exact_agent.interp import alignment as al  # noqa: E402
from exact_agent.interp.records import ActivationRecord, read_records  # noqa: E402


def _samples_and_vocab(
    records: list[ActivationRecord],
) -> tuple[list[tuple[set[str], set[str], int]], list[str]]:
    samples: list[tuple[set[str], set[str], int]] = []
    vocab: set[str] = set()
    for rec in records:
        active = set(rec.active_concepts)
        gold = set(rec.gold_concepts)
        samples.append((active, gold, len(active)))
        vocab |= active | gold
    return samples, sorted(vocab)


def _render_md(name: str, cmp: al.Comparison) -> str:
    return (
        f"# Alignment report — {name}\n\n"
        f"- samples: **{cmp.n}**\n"
        f"- observed macro-F1: **{cmp.observed_macro_f1:.3f}**\n"
        f"- random-baseline macro-F1: **{cmp.random_macro_f1:.3f}**\n"
        f"- lift (observed − random): **{cmp.lift:+.3f}**\n"
        f"- win-rate (observed > random per sample): **{cmp.win_rate:.1%}**\n\n"
        "> RQ1 success criterion (charter §8): lift meaningfully > 0 and "
        "win-rate well above 50%. If lift ≈ 0, that is an honest negative "
        "result — pivot per charter §8.\n"
    )


def _selfcheck() -> int:
    """Exercise the metric on synthetic records (no model needed)."""
    recs = [
        ActivationRecord(
            sample_id="s1", task_type="physics", question="q", layer=18,
            active_feature_ids=[1, 2], active_concepts=["capacitor", "voltage"],
            gold_concepts=["capacitor", "voltage", "energy"], solver_ok=True,
        ),
        ActivationRecord(
            sample_id="s2", task_type="physics", question="q", layer=18,
            active_feature_ids=[9], active_concepts=["poetry", "novel"],
            gold_concepts=["resistance", "current"], solver_ok=True,
        ),
    ]
    samples, vocab = _samples_and_vocab(recs)
    cmp = al.compare(samples, vocab, n_trials=50, seed=0)
    print(_render_md("selfcheck", cmp))
    # s1 aligns (lift should be >= 0); s2 is pure noise. Just assert it runs.
    assert cmp.n == 2
    print("selfcheck OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, help="captured ActivationRecord JSONL")
    ap.add_argument("--probe-records", type=Path, default=None,
                    help="optional linear-probe baseline records (same format)")
    ap.add_argument("--out", type=Path, help="output stem (writes .md and .json)")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return _selfcheck()
    if not args.records or not args.out:
        ap.error("--records and --out are required (or use --selfcheck)")

    with args.records.open(encoding="utf-8") as f:
        records = read_records(f)
    samples, vocab = _samples_and_vocab(records)
    cmp = al.compare(samples, vocab, n_trials=args.trials, seed=args.seed)

    report: dict = {"sae": cmp.__dict__}
    if args.probe_records and args.probe_records.exists():
        with args.probe_records.open(encoding="utf-8") as f:
            psamples, pvocab = _samples_and_vocab(read_records(f))
        report["linear_probe"] = al.compare(
            psamples, pvocab, n_trials=args.trials, seed=args.seed
        ).__dict__

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(_render_md(args.records.stem, cmp), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(_render_md(args.records.stem, cmp))
    print(f"wrote {args.out.with_suffix('.md')} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
