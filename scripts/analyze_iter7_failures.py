"""One-shot root-cause categoriser for Iter-7 physics + logic failures.

Reads outputs/eval/iter7_*/{*_report.json}, classifies every failing row
by what would need to change to fix it, and prints a compact report.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PHYS = ROOT / "outputs/eval/iter7_physics/physics_llm_report.json"
LOG = ROOT / "outputs/eval/iter7_logic_v3/logic_llm_report.json"

SYM_GLYPHS = ("√", "sqrt(", "log", "sin", "cos", "∞", "pi", "π")
QUAL_TOKENS = (
    "linear", "doubled", "halved", "increase", "decrease", "proportional",
    "the same", "the square", "approaches", "inductive characteristic",
    "capacitive characteristic", "depends on", "regardless", "neither",
    "either", "any", "all of the above", "none of the above",
)


def physics_root_cause(s: dict) -> str:
    g = str(s.get("expected_answer") or "")
    glow = g.lower()
    fr = s.get("fail_reason") or "numeric_mismatch"
    fr_root = fr.split(":")[0]

    # Out-of-scope: gold is symbolic expression
    if any(t in g for t in SYM_GLYPHS) or g.count("+") > 1 or g.count("/") > 2:
        return "OOS-symbolic_gold"

    # Out-of-scope: gold is qualitative
    if any(t in glow for t in QUAL_TOKENS) and not any(c.isdigit() for c in g[:4]):
        return "OOS-qualitative_gold"

    # Gold is enumeration ("A; B; C") — multi-target answer
    if ";" in g and len(g) < 30:
        return "OOS-multi_target_gold"

    if fr_root == "no_formula_matched":
        return "FORMULA-missing_from_registry"
    if fr_root == "missing_input":
        # Formula picked correctly but extractor (regex + LLM) failed
        return "EXTRACTOR-missing_input"
    if fr_root == "llm_recovery_failed":
        return "EXTRACTOR-llm_unit_mangle"
    if fr_root == "unit_conversion_failed":
        return "UNIT-pint_parse_error"

    # Numeric mismatch -- formula picked, all inputs found, value wrong
    # Heuristic: check if it's a multi-step problem (gold is significantly
    # different magnitude or sign from predicted).
    if s.get("predicted_value") is not None and s.get("expected_value") is not None:
        try:
            pv, ev = float(s["predicted_value"]), float(s["expected_value"])
            if ev != 0:
                ratio = abs(pv / ev)
                if 0.1 < ratio < 10:
                    return "VALUE-wrong_within_order"
                else:
                    return "VALUE-wrong_order_of_magnitude"
        except Exception:
            pass
    return "VALUE-other_mismatch"


def logic_root_cause(s: dict) -> str:
    p = s.get("predicted_answer") or ""
    g = s.get("expected_answer") or ""
    if not s.get("correct"):
        if s.get("abstained"):
            return f"ABSTAIN-should_be_{g}" if g != "Unknown" else "ABSTAIN-correct_unknown"
        # Hard wrong
        if p == "Yes" and g == "No":
            return "SURFACE-yes_when_no"
        if p == "No" and g == "Yes":
            return "SURFACE-no_when_yes"
        if g in ("Yes", "No") and p in ("A", "B", "C", "D"):
            return "TYPE-mc_when_yn"
        if g in ("A", "B", "C", "D") and p in ("A", "B", "C", "D"):
            return "MC-wrong_option"
        if g in ("Yes", "No") and p == "Unknown":
            return f"ABSTAIN-should_be_{g}"
        return f"OTHER-{p}_vs_{g}"
    return "correct"


def main() -> None:
    if not PHYS.exists():
        print("No physics eval at", PHYS)
        return
    if not LOG.exists():
        print("No logic eval at", LOG)
        return

    phys = json.load(PHYS.open(encoding="utf-8"))
    log = json.load(LOG.open(encoding="utf-8"))

    fail_p = [s for s in phys["samples"] if not s["full_correct"]]
    fail_l = [s for s in log["samples"] if not s["correct"]]

    print("=" * 72)
    print(f"PHYSICS ({phys['overall']['n']} rows, "
          f"{phys['overall']['n'] - len(fail_p)} correct, "
          f"{len(fail_p)} wrong = {100*len(fail_p)/phys['overall']['n']:.1f}%)")
    print("=" * 72)
    rc_p = Counter(physics_root_cause(s) for s in fail_p)
    for rc, n in rc_p.most_common():
        pct = 100 * n / phys["overall"]["n"]
        print(f"  {n:3d} rows ({pct:.1f}% of holdout)  {rc}")

    print()
    print("=" * 72)
    print(f"LOGIC ({log['overall']['n']} rows, "
          f"{log['overall']['n'] - len(fail_l)} correct, "
          f"{len(fail_l)} wrong = {100*len(fail_l)/log['overall']['n']:.1f}%)")
    print("=" * 72)
    rc_l = Counter(logic_root_cause(s) for s in fail_l)
    for rc, n in rc_l.most_common():
        pct = 100 * n / log["overall"]["n"]
        print(f"  {n:3d} rows ({pct:.1f}% of holdout)  {rc}")


if __name__ == "__main__":
    main()
