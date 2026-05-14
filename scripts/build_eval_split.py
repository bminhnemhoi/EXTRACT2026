"""Carve a deterministic eval split from cleaned datasets.

Produces ``data/eval_split/{logic,physics}_eval.jsonl`` (the 10% holdout)
and ``data/processed/{logic,physics}_train.jsonl`` (the 90% remainder),
seeded for reproducibility.

Usage::

    uv run python scripts/build_eval_split.py
    uv run python scripts/build_eval_split.py --seed 7 --ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEANED = REPO_ROOT / "data" / "cleaned"
EVAL_DIR = REPO_ROOT / "data" / "eval_split"
PROCESSED = REPO_ROOT / "data" / "processed"

_DATASETS = {
    "physics": CLEANED / "physics_train_safe.jsonl",
    "logic": CLEANED / "logic_train_safe.jsonl",
}


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split(name: str, source: Path, seed: int, ratio: float) -> tuple[int, int]:
    rows = _load_jsonl(source)
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_eval = max(1, round(len(rows) * ratio))
    eval_idx = set(indices[:n_eval])

    eval_rows = [rows[i] for i in range(len(rows)) if i in eval_idx]
    train_rows = [rows[i] for i in range(len(rows)) if i not in eval_idx]

    _write_jsonl(EVAL_DIR / f"{name}_eval.jsonl", eval_rows)
    _write_jsonl(PROCESSED / f"{name}_train.jsonl", train_rows)
    return len(eval_rows), len(train_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument(
        "--only",
        choices=("physics", "logic", "all"),
        default="all",
        help="Restrict to one dataset (default: both).",
    )
    args = parser.parse_args()

    targets = list(_DATASETS) if args.only == "all" else [args.only]
    for name in targets:
        source = _DATASETS[name]
        if not source.exists():
            print(f"[skip] {name}: source not found at {source}")
            continue
        n_eval, n_train = split(name, source, args.seed, args.ratio)
        print(f"[ok] {name}: eval={n_eval}  train={n_train}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
