"""Carve a deterministic 10% holdout from the cleaned datasets for local eval.

Reads `data/cleaned/logic_train_safe.jsonl` and `data/cleaned/physics_train_safe.jsonl`,
writes `data/eval_split/{logic,physics}_eval.jsonl` and the corresponding train
remainders to `data/processed/`. Reproducible via the fixed seed.

Phase 1 / Day 3 will start consuming the eval split. The script is a stub today
so the import paths are stable; flesh it out when the eval harness lands.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.1)
    args = parser.parse_args()

    # TODO(Day 3): implement once eval/eval_logic.py and eval_physics.py exist.
    print(
        f"[stub] build_eval_split — will hold out {args.ratio * 100:.0f}% with seed={args.seed}. "
        "Wire up in Day 3."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
