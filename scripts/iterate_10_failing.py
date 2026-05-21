"""Day-25/26 iteration tool — agent error-driven refinement on 10 rows.

Runs only 10 specific failing rows through the current solver and prints
a compact diagnostic per row so we can identify root cause fast (where
the agent failed: classifier? extractor? formula? unit?).

Workflow:
  1. uv run python scripts/iterate_10_failing.py
  2. Read the table; pick the highest-impact fix
  3. Apply fix in src/ + add tests
  4. Re-run this script — measure delta
  5. Iterate until target accuracy on the 10

Target: >= 5/10 = 50% on these 10 specific rows.

The 10 rows are picked once and pinned by sample_id below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "data" / "official_v20260515" / "physics_safe.jsonl"

# Pinned 10 diverse failing rows from outputs/eval/f1f2_3b_unseen
# (picked Day-25 by hand; cover routing / extractor / formula gap / conceptual).
PINNED_IDS = (
    "LD025", "LD026",        # routing: 2-3 charge force → wrong formula
    "TD010", "TD013",        # formula gap: voltage-distance change / ε_r reverse
    "DT005", "DT006",        # routing: asked FORCE got FIELD formula
    "THCB070", "THCB083",    # multi-step + conceptual (likely uncrackable)
    "NL005", "NL007",        # routing + extractor
)


def main() -> int:
    # UTF-8 stdout for tables with Unicode glyphs (μ, ×, →).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Lazy imports so the script's startup latency stays low when iterating.
    from exact_agent.llm.vllm_client import VLLMClient  # noqa: PLC0415
    from exact_agent.physics.solver import PhysicsSolver  # noqa: PLC0415

    rows_by_id: dict[str, dict] = {}
    with SPLIT_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in PINNED_IDS:
                rows_by_id[row["id"]] = row

    client = VLLMClient()
    solver = PhysicsSolver(llm_client=client)

    correct = 0
    print(f"{'='*100}\nITERATION RUN — 10 pinned failing rows\n{'='*100}")
    for i, rid in enumerate(PINNED_IDS, 1):
        row = rows_by_id.get(rid)
        if row is None:
            print(f"{i}. [{rid}] NOT FOUND in split")
            continue
        question = row["question_clean"]
        gold_answer = row.get("answer_clean", "")
        gold_unit = row.get("unit_clean", "")
        print(f"\n{i}. [{rid}] prefix={row.get('prefix')}")
        print(f"   Q: {question[:140]}")
        print(f"   GOLD:  {gold_answer!r} {gold_unit!r}")
        try:
            result = solver.solve(question)
        except Exception as exc:
            print(f"   ERROR: {exc}")
            continue
        print(
            f"   PRED:  {result.answer_str!r} {result.answer_unit!r}    "
            f"formula={result.formula_id}    fail={result.fail_reason}"
        )
        # Top-3 trace lines (most informative).
        for trace_line in result.trace[:3]:
            print(f"   trace: {trace_line[:160]}")
        # Compare predicted vs gold using the SAME scorer the eval uses.
        from exact_agent.eval.metrics import quantity_match  # noqa: PLC0415
        ok = quantity_match(result.answer_str, result.answer_unit, gold_answer, gold_unit)
        if ok:
            correct += 1
            print("   ✓ MATCH")
        else:
            print("   ✗ MISS")

    print(f"\n{'='*100}\nRESULT: {correct}/10 = {correct*10}%\n{'='*100}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
