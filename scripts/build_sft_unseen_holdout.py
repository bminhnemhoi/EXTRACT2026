"""Build a clean A/B holdout that the SFT adapter NEVER saw.

The 1765-row SFT training corpus (Day-13, ADR 0013) was derived from
the 2026-05-09 release, and 1189 of its rows (87%) also appear in the
new 2026-05-15 official release. Naïvely A/B-testing the SFT'd
Qwen2.5-7B against vanilla qwen2.5:3b on the existing
``data/official_v20260515/eval_split/physics_eval.jsonl`` is biased:
133 of those 135 rows ARE in SFT-train, so the adapter is judged on
questions it has memorised.

This script extracts the 163 official rows that the SFT trainer never
saw (= the 138 SFT-val rows + 25 rows that the SFT corpus pre-cleaned
away but are present in the official 2026-05-15 release). The result
is the only honest dataset for an SFT-vs-no-SFT comparison.

Output: ``data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl``
(N=163; full prefix coverage: LD 45 / CH 39 / TD 26 / NL 21 / DDT 16 /
THCB 9 / DT 6 / CHLT 1).

Usage::

    uv run python scripts/build_sft_unseen_holdout.py
    uv run python scripts/run_eval.py --task physics --with-llm \\
        --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \\
        --out outputs/eval/e7a_sft_unseen_<backbone>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = REPO_ROOT / "data" / "official_v20260515" / "physics_safe.jsonl"
SFT_TRAIN = REPO_ROOT / "data" / "processed" / "sft_chat_train.jsonl"
OUT = REPO_ROOT / "data" / "official_v20260515" / "eval_split" / "physics_eval_sft_unseen.jsonl"


def _ids(path: Path, key: str) -> set[str]:
    if not path.exists():
        print(f"[fatal] missing {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return {json.loads(line)[key] for line in f if line.strip()}


def main() -> int:
    sft_seen = _ids(SFT_TRAIN, "sample_id")
    print(f"[info] SFT train rows: {len(sft_seen)}")

    with OFFICIAL.open("r", encoding="utf-8") as f:
        official_rows = [json.loads(line) for line in f if line.strip()]
    print(f"[info] official 2026-05-15 physics rows: {len(official_rows)}")

    unseen = [r for r in official_rows if r["id"] not in sft_seen]
    print(f"[info] SFT-unseen rows (clean A/B holdout): {len(unseen)}")
    prefix_dist = Counter(r.get("prefix", "?") for r in unseen)
    print(f"[info] prefix distribution: {dict(prefix_dist.most_common())}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in unseen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
