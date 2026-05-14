"""Run a local evaluation against the holdout split.

Usage::

    # default: physics eval split
    uv run python scripts/run_eval.py

    # explicit task and output dir
    uv run python scripts/run_eval.py --task physics --out outputs/eval/day3

    # dump per-sample diagnostics too (slower / larger file)
    uv run python scripts/run_eval.py --include-samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Force UTF-8 stdout so Markdown tables with Unicode arrows / glyphs render
# on Windows consoles defaulting to cp1252.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from exact_agent.eval.eval_logic import run_eval as run_logic_eval  # noqa: E402
from exact_agent.eval.eval_physics import run_eval as run_physics_eval  # noqa: E402
from exact_agent.eval.reports import render_markdown, write_outputs  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "outputs" / "eval"
EVAL_SPLITS = {
    "physics": REPO_ROOT / "data" / "eval_split" / "physics_eval.jsonl",
    "logic": REPO_ROOT / "data" / "eval_split" / "logic_eval.jsonl",
}
RUNNERS = {
    "physics": run_physics_eval,
    "logic": run_logic_eval,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("physics", "logic"), default="physics")
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="Override the eval split path (default: data/eval_split/<task>_eval.jsonl).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Embed per-sample evaluations in the JSON report.",
    )
    args = parser.parse_args()

    split_path: Path = args.split or EVAL_SPLITS[args.task]
    if not split_path.exists():
        print(f"[fatal] eval split not found at {split_path}", file=sys.stderr)
        print("  -> run: uv run python scripts/build_eval_split.py", file=sys.stderr)
        return 1

    print(f"[info] evaluating {args.task} on {split_path.name} ...")
    report = RUNNERS[args.task](split_path)
    md_path, json_path = write_outputs(
        report, args.out, stem=args.task, include_samples=args.include_samples
    )
    print(render_markdown(report))
    print(f"[ok] wrote {md_path}")
    print(f"[ok] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
