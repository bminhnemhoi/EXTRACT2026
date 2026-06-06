"""Autointerp helpers — turn SAE features into natural-language labels.

Pure helpers (context selection, prompt building, output parsing) are stdlib
and unit-tested. The model forward + LLM call live in
``scripts/interp/build_labels.py`` (GPU). Pipeline (AutoInterp, arXiv 2410.13928):
for each target feature, collect the text windows where it activates most, ask
an LLM to summarize the shared pattern, cache the short label.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Context:
    """A high-activation window for a feature."""

    text: str
    activation: float
    sample_id: str


def select_top_contexts(
    hits: list[Context], top_n: int = 12, *, dedupe: bool = True
) -> list[Context]:
    """Keep the ``top_n`` highest-activation contexts (optionally de-duped)."""
    ordered = sorted(hits, key=lambda c: c.activation, reverse=True)
    if not dedupe:
        return ordered[:top_n]
    seen: set[str] = set()
    out: list[Context] = []
    for c in ordered:
        key = c.text.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= top_n:
            break
    return out


def build_label_prompt(contexts: list[Context]) -> str:
    """Prompt asking a labeler LLM to summarize the shared pattern.

    Asks for a SHORT noun-phrase label (so it tokenizes into clean concepts for
    lexical alignment) plus an explicit "uninterpretable" escape hatch — we do
    NOT want forced labels on polysemantic/dead features.
    """
    lines = [
        "You are interpreting one feature of a sparse autoencoder over a "
        "language model. Below are text snippets where this feature activates "
        "most strongly. Identify the SINGLE shared concept/pattern.",
        "",
        "Snippets:",
    ]
    for i, c in enumerate(contexts, 1):
        lines.append(f"{i}. {c.text.strip()}")
    lines += [
        "",
        "Answer with a SHORT noun phrase (2-6 words) naming the concept, e.g. "
        "'capacitor energy', 'electric field strength', 'graduation eligibility'. "
        "If there is no clear shared concept, answer exactly: UNINTERPRETABLE.",
        "Label:",
    ]
    return "\n".join(lines)


def parse_label(llm_output: str) -> str:
    """Extract the label; empty string for an uninterpretable / blank reply."""
    text = (llm_output or "").strip()
    # take the first non-empty line, strip a leading "Label:" if echoed
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if line.lower().startswith("label:"):
            line = line.split(":", 1)[1].strip()
        if not line:
            continue
        if line.strip().upper().startswith("UNINTERPRETABLE"):
            return ""
        return line.strip().strip('"').strip()
    return ""
