"""Run a local evaluation against the holdout split.

Usage::

    # default: rule-only physics eval split
    uv run python scripts/run_eval.py

    # explicit task and output dir
    uv run python scripts/run_eval.py --task physics --out outputs/eval/day3

    # dump per-sample diagnostics too (slower / larger file)
    uv run python scripts/run_eval.py --include-samples

    # exercise the LLM fallback paths (needs a running vLLM/Ollama endpoint
    # at configs/model.yaml::llm.vllm_base_url); --limit for a quick subset
    uv run python scripts/run_eval.py --task physics --with-llm --limit 15
"""

from __future__ import annotations

import argparse
import sys
import tempfile
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


def _truncate_split(split_path: Path, limit: int) -> Path:
    """Write the first ``limit`` rows to a temp JSONL and return its path.

    Keeps eval_physics/eval_logic untouched — they just read whatever path
    they're handed.
    """
    lines: list[str] = []
    with split_path.open("r", encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                lines.append(raw)
            if len(lines) >= limit:
                break
    tmp = Path(tempfile.gettempdir()) / f"exact_eval_subset_{split_path.stem}_{limit}.jsonl"
    tmp.write_text("".join(lines), encoding="utf-8")
    return tmp


def _build_runner_arg(task: str, with_llm: bool, ld_url: str | None = None):  # type: ignore[no-untyped-def]
    """Return the solver/pipeline to inject, or None for the rule-only path.

    ``ld_url`` (physics + ``--hybrid-ld-url``) wires a second LLM client
    for LD-domain extractions; the solver routes per formula. The
    ``default`` client still comes from ``configs/model.yaml`` /
    EXACT_LLM__BASE_URL (so non-LD formulas use whatever the env says).
    """
    if not with_llm:
        return None
    # Lazy imports: only touch the LLM stack when actually requested.
    from exact_agent.llm.vllm_client import LLMConfig, VLLMClient  # noqa: PLC0415

    default_client = VLLMClient()
    if task == "physics":
        from exact_agent.physics.solver import PhysicsSolver  # noqa: PLC0415

        ld_client = None
        if ld_url:
            # Build the LD client from the *raw* model.yaml file so we
            # get its model name verbatim (e.g. 'qwen2.5:3b-instruct'),
            # NOT whatever EXACT_LLM__MODEL the env override set for the
            # default client. Otherwise we'd be sending 'qwen2.5-7b' to
            # Ollama 3B and getting 404.
            import yaml as _yaml  # noqa: PLC0415

            from exact_agent.config import get_settings  # noqa: PLC0415
            with (get_settings().configs_dir / "model.yaml").open(encoding="utf-8") as f:
                file_data = (_yaml.safe_load(f) or {}).get("llm", {}) or {}
            ld_cfg = LLMConfig(
                base_url=ld_url.rstrip("/"),
                api_key=str(file_data.get("api_key", "local-no-auth")),
                model=str(file_data.get("backbone", "qwen2.5:3b-instruct")),
                max_tokens=int(file_data.get("max_tokens", 1024)),
                temperature=float(file_data.get("temperature", 0.2)),
                top_p=float(file_data.get("top_p", 0.9)),
                timeout_s=float(file_data.get("timeout_s", 120)),
                enable_lora=bool(file_data.get("enable_lora", False)),
                lora_adapter_path=str(file_data.get("lora_adapter_path", "")),
                mode=str(file_data.get("mode", "chat")).lower(),
                disable_thinking=bool(file_data.get("disable_thinking", False)),
            )
            ld_client = VLLMClient(ld_cfg)
            print(f"[info] hybrid LD client → {ld_url} (model={ld_cfg.model})")
        return PhysicsSolver(llm_client=default_client, llm_client_ld=ld_client)
    from exact_agent.logic.pipeline import LogicPipeline  # noqa: PLC0415

    return LogicPipeline(llm_client=default_client)


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
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Inject a VLLMClient so the LLM fallback paths fire (needs an endpoint).",
    )
    parser.add_argument(
        "--hybrid-ld-url",
        type=str,
        default=None,
        help=(
            "Physics only. Second LLM endpoint for LD-domain extractions "
            "(Coulomb/electric field). The default client comes from "
            "configs/model.yaml or EXACT_LLM__BASE_URL; this flag adds a "
            "non-LD/LD split so we can A/B a different backbone per domain. "
            "Q3 sequential: only one client is invoked per query."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N rows (0 = all). Use for a quick LLM sanity pass.",
    )
    args = parser.parse_args()

    split_path: Path = args.split or EVAL_SPLITS[args.task]
    if not split_path.exists():
        print(f"[fatal] eval split not found at {split_path}", file=sys.stderr)
        print("  -> run: uv run python scripts/build_eval_split.py", file=sys.stderr)
        return 1

    if args.limit > 0:
        split_path = _truncate_split(split_path, args.limit)
        print(f"[info] limited to first {args.limit} rows → {split_path.name}")

    runner_arg = _build_runner_arg(args.task, args.with_llm, ld_url=args.hybrid_ld_url)
    mode = "LLM-enabled" if args.with_llm else "rule-only"
    print(f"[info] evaluating {args.task} ({mode}) on {split_path.name} ...")

    if args.task == "physics":
        report = run_physics_eval(split_path, runner_arg)
    else:
        report = run_logic_eval(split_path, runner_arg)

    stem = f"{args.task}_llm" if args.with_llm else args.task
    md_path, json_path = write_outputs(
        report, args.out, stem=stem, include_samples=args.include_samples
    )
    print(render_markdown(report))
    print(f"[ok] wrote {md_path}")
    print(f"[ok] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
