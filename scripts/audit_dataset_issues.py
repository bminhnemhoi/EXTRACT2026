"""Surface candidate dataset annotation issues for the Q22 bonus report.

Reads a physics eval JSON written with ``--include-samples`` and prints
rows where the solver computed a confident, formula-grounded answer that
disagrees with the gold by more than `--rel-tol` (default 50%). These
are the highest-priority candidates for a manual cross-check + Discord
``#dataset-issue-report`` submission.

Bonus criteria recap (QA Q22): "Teams that report verified dataset
issues (incorrect labels, ambiguous questions, formatting bugs) during
Phase 1 or Phase 2 receive a bonus on the final score."

Heuristic — a row is flagged if ALL of these hold:

* The solver completed (``solved=True``) — we trust our number.
* A formula was applied (``formula_id`` non-empty) — not a guess.
* The verifier raised no warnings (e.g. negative-where-positive, oom).
* ``|predicted - expected| / |expected| > rel_tol``.
* The unit matched (``unit_correct=True``) — pure value mismatch is
  more credible as a gold bug; unit mismatch is more likely OUR bug.

Usage::

    uv run python scripts/audit_dataset_issues.py \\
        --report outputs/eval/day22_v0515_physics_e6/physics_llm_report.json \\
        --rel-tol 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--rel-tol", type=float, default=0.5,
        help="Relative tolerance: only flag rows >this off (default 0.5 = 50 percent).",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/audit_candidates.md"))
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    samples = data.get("samples") or []
    if not samples:
        print(
            "[fatal] report has no samples — re-run run_eval.py with --include-samples",
            file=sys.stderr,
        )
        return 1

    flagged: list[dict] = []
    for s in samples:
        if not s.get("solved") or not s.get("formula_id"):
            continue
        if s.get("verifier_warnings"):
            continue
        if not s.get("unit_correct"):
            continue
        # The eval's `full_correct` is the authoritative scorer verdict —
        # unit-aware, round-aware. A row already marked correct is not a
        # dataset-issue candidate (the system+gold agree). The audit only
        # surfaces rows the scorer marks WRONG despite confident, unit-
        # consistent extraction — those are the cross-check candidates.
        if s.get("full_correct"):
            continue
        try:
            pred = float(s.get("predicted_value") or 0.0)
            exp = float(s.get("expected_value") or 0.0)
        except (TypeError, ValueError):
            continue
        if exp == 0.0:
            continue
        # The raw rel-diff here is unit-naïve (we compare the SI float
        # against the gold-unit's number). It only ranks candidates —
        # the real "are these close?" question already answered No by
        # the scorer above.
        rel = abs(pred - exp) / abs(exp)
        if rel > args.rel_tol:
            flagged.append({**s, "rel_diff": round(rel, 3)})

    flagged.sort(key=lambda r: -r["rel_diff"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# Dataset issue audit — {args.report}",
        "",
        f"Flagged **{len(flagged)} rows** with rel_diff > {args.rel_tol}, "
        "unit-correct, formula-applied, no verifier warnings.",
        "",
        "Each is a manual cross-check candidate for a Q22 Discord report.",
        "",
        "| id | prefix | formula | predicted | expected | rel_diff | question (truncated) |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in flagged[:40]:
        q = (r.get("question") or "").replace("\n", " ")[:100]
        lines.append(
            f"| {r['sample_id']} | {r.get('prefix','')} | {r.get('formula_id','')} | "
            f"{r.get('predicted_answer','')} {r.get('predicted_unit','')} | "
            f"{r.get('expected_answer','')} {r.get('expected_unit','')} | "
            f"{r['rel_diff']} | {q} |"
        )
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {args.out} ({len(flagged)} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
