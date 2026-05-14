"""Render an :class:`EvalReport` as Markdown / JSON."""

from __future__ import annotations

import json
from pathlib import Path

from exact_agent.eval.eval_physics import AggregateMetrics, EvalReport


def _row(label: str, agg: AggregateMetrics) -> str:
    d = agg.as_dict()
    return (
        f"| {label} | {d['n']} | "
        f"{d['solved_pct']:.1f}% | "
        f"{d['numeric_correct_pct']:.1f}% | "
        f"{d['unit_correct_pct']:.1f}% | "
        f"{d['full_correct_pct']:.1f}% | "
        f"{d['explanation_f1_mean']:.3f} | "
        f"{d['mean_elapsed_ms']:.1f} |"
    )


def render_markdown(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append(f"# Physics evaluation — `{Path(report.split_path).name}`")
    lines.append("")

    lines.append("| Slice | N | Solved | Numeric✓ | Unit✓ | Full✓ | Expl F1 | ms/sample |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(_row("**Overall**", report.overall))

    if report.by_prefix:
        for prefix, agg in sorted(report.by_prefix.items()):
            lines.append(_row(f"prefix `{prefix}`", agg))

    lines.append("")
    lines.append("## By formula chosen")
    lines.append("")
    lines.append("| Formula | N | Solved | Numeric✓ | Unit✓ | Full✓ | Expl F1 | ms/sample |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for formula_id, agg in sorted(report.by_formula.items()):
        lines.append(_row(f"`{formula_id}`", agg))

    if report.fail_reasons:
        lines.append("")
        lines.append("## Failure reasons (count)")
        for reason, count in report.fail_reasons.most_common():
            lines.append(f"- `{reason}` — {count}")

    return "\n".join(lines) + "\n"


def write_outputs(
    report: EvalReport,
    out_dir: Path,
    *,
    stem: str = "physics",
    include_samples: bool = False,
) -> tuple[Path, Path]:
    """Write both ``<stem>_report.md`` and ``<stem>_report.json``. Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{stem}_report.md"
    json_path = out_dir / f"{stem}_report.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(include_samples=include_samples), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return md_path, json_path
