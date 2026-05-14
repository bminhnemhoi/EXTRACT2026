"""Run the local evaluation harness. Stub until Day 3."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="eval", choices=["eval", "train"])
    args = parser.parse_args()
    print(f"[stub] run_eval split={args.split} — implementation lands in Day 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
