"""Bench premise_selector ranker (Jaccard vs TF-IDF) against the official
gold ``idx`` field — the 1-based premise indices the dataset says each
question actually uses (CHANGELOG_TYPE1 calls this the "P3 cited
premises gold").

Reports Recall@1 / @3 / @5 / @8 for each method on the official
2026-05-15 logic eval split. Run after Day-23 F3 to verify TF-IDF wins.

Usage::

    uv run python scripts/bench_premise_selector.py
"""

from __future__ import annotations

import json
from pathlib import Path

from exact_agent.logic.premise_selector import rank_premises

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGIC_EVAL = REPO_ROOT / "data" / "official_v20260515" / "eval_split" / "logic_eval.jsonl"


def _recall_at_k(retrieved: list[int], gold: set[int], k: int) -> float:
    if not gold:
        return float("nan")
    hit = len(set(retrieved[:k]) & gold)
    return hit / len(gold)


def main() -> int:
    if not LOGIC_EVAL.exists():
        print(f"[fatal] {LOGIC_EVAL} not found — run import_official first.")
        return 1
    raw = LOGIC_EVAL.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    print(f"[info] logic eval rows: {len(rows)}")

    bench: dict[str, dict[int, list[float]]] = {
        "jaccard": {1: [], 3: [], 5: [], 8: []},
        "tfidf": {1: [], 3: [], 5: [], 8: []},
    }
    scored = 0
    for row in rows:
        gold_idx_raw = row.get("used_premise_idx")
        if not isinstance(gold_idx_raw, list):
            continue
        gold = {int(i) for i in gold_idx_raw if isinstance(i, int) and i >= 1}
        if not gold:
            continue
        premises = list(row.get("premises_NL") or [])
        if not premises:
            continue
        question = str(row.get("question") or "")
        scored += 1
        for method in ("jaccard", "tfidf"):
            ranked = rank_premises(question, premises, method=method)
            retrieved_ids = [rp.index for rp in ranked]
            for k in (1, 3, 5, 8):
                bench[method][k].append(_recall_at_k(retrieved_ids, gold, k))

    print(f"[info] benched on {scored} rows with non-empty gold idx")
    print()
    print(f"{'method':<10} | {'R@1':>6} | {'R@3':>6} | {'R@5':>6} | {'R@8':>6}")
    print("-" * 50)
    for method in ("jaccard", "tfidf"):
        sc = bench[method]
        means = [
            sum(v) / len(v) if v else 0.0
            for v in (sc[1], sc[3], sc[5], sc[8])
        ]
        print(
            f"{method:<10} | {means[0]:>5.1%} | {means[1]:>5.1%} | "
            f"{means[2]:>5.1%} | {means[3]:>5.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
