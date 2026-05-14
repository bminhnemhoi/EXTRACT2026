"""Sample failure diagnostics from a per-sample eval report.

Usage::

    uv run python scripts/inspect_eval.py outputs/eval/debug/physics_report.json
    uv run python scripts/inspect_eval.py ... --prefix LD --limit 5
    uv run python scripts/inspect_eval.py ... --fail-reason missing_input
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--formula", default=None)
    parser.add_argument("--fail-reason", default=None)
    parser.add_argument("--correct", action="store_true", help="Show only successes")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print per-prefix fail-reason histogram instead of samples.",
    )
    args = parser.parse_args()

    with args.report.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples") or []

    selected = []
    for s in samples:
        if args.prefix and s.get("prefix") != args.prefix:
            continue
        if args.formula and s.get("formula_id") != args.formula:
            continue
        if args.fail_reason and (s.get("fail_reason") or "").split(":")[0] != args.fail_reason:
            continue
        if args.correct and not s.get("full_correct"):
            continue
        selected.append(s)

    if args.summary:
        per_prefix: dict[str, Counter] = {}
        for s in samples:
            p = s.get("prefix", "?")
            reason = (s.get("fail_reason") or "ok").split(":")[0]
            per_prefix.setdefault(p, Counter())[reason] += 1
        for prefix in sorted(per_prefix):
            print(f"\n[{prefix}]")
            for reason, count in per_prefix[prefix].most_common():
                print(f"  {reason:30s} {count}")
        return 0

    n_show = min(args.limit, len(selected))
    print(f"matched {len(selected)} / {len(samples)} samples; showing {n_show}\n")
    for s in selected[: args.limit]:
        print(f"id={s['sample_id']} prefix={s['prefix']} formula={s.get('formula_id')!r}")
        print(f"  fail   : {s.get('fail_reason')!r}")
        print(f"  Q      : {s['question'][:240]}")
        print(f"  expect : {s['expected_answer']} {s['expected_unit']}")
        conf = s["confidence"]
        print(f"  predict: {s['predicted_answer']} {s['predicted_unit']}  conf={conf:.2f}")
        warns = s.get("verifier_warnings") or []
        if warns:
            print(f"  warns  : {warns}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
