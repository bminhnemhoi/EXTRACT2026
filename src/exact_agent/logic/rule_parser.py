"""Parse NL premises into surface-level rules and grounded facts.

The educational dataset has three recurring premise shapes that cover the
majority of training samples we inspected:

* ``"If A (and B), then C."`` — a conditional rule with one or two conditions.
* ``"All X are Y."``           — universal subject/predicate fact.
* ``"<Entity> has completed <thing>."`` — a grounded fact about a named entity.

Anything else is kept as an opaque :class:`Fact` carrying the original text;
forward chaining can still surface it through Jaccard overlap.

Day-4 caveat: we deliberately stay at the surface level — full NL→FOL
translation needs a language model and lives on the Phase-4 roadmap. The
patterns are loaded from ``configs/logic_patterns.yaml`` so the rule set is
editable without touching code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from exact_agent.config import get_settings

# ---------------------------------------------------------------------------
# Data classes — kept lean so the forward chainer can consume them directly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A surface-level rule: ``conditions ⇒ conclusion``."""

    conditions: tuple[str, ...]
    conclusion: str
    raw: str
    premise_index: int
    """0-based index into the original premises list (the API converts to ``P{i+1}``)."""


@dataclass(frozen=True)
class Fact:
    """A grounded statement extracted directly from a premise."""

    text: str
    raw: str
    premise_index: int


@dataclass(frozen=True)
class ParsedPremise:
    rule: Rule | None
    fact: Fact | None


# ---------------------------------------------------------------------------
# Pattern loading from configs/logic_patterns.yaml
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CompiledPattern:
    pattern_id: str
    regex: re.Pattern[str]


@lru_cache(maxsize=1)
def _load_patterns() -> list[_CompiledPattern]:
    path: Path = get_settings().configs_dir / "logic_patterns.yaml"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    compiled: list[_CompiledPattern] = []
    for entry in data.get("rule_patterns", []) or []:
        try:
            compiled.append(
                _CompiledPattern(
                    pattern_id=str(entry["id"]),
                    regex=re.compile(str(entry["regex"]), flags=re.IGNORECASE),
                )
            )
        except (KeyError, re.error):
            continue
    return compiled


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip(" .")).strip()


def parse_premise(text: str, premise_index: int) -> ParsedPremise:
    """Best-effort parse of one NL premise.

    Order of pattern checks follows ``configs/logic_patterns.yaml`` (first
    match wins). Anything we can't recognize becomes a free-form fact —
    the chainer can still surface it via Jaccard overlap.
    """
    cleaned = _normalize(text)
    if not cleaned:
        return ParsedPremise(rule=None, fact=None)

    for pattern in _load_patterns():
        match = pattern.regex.match(cleaned)
        if not match:
            continue
        groups = match.groupdict()

        if pattern.pattern_id in {"if_and_then", "if_and_implicit"}:
            return ParsedPremise(
                rule=Rule(
                    conditions=(_normalize(groups["a"]), _normalize(groups["b"])),
                    conclusion=_normalize(groups["concl"]),
                    raw=cleaned,
                    premise_index=premise_index,
                ),
                fact=None,
            )

        if pattern.pattern_id in {
            "if_then_simple",
            "if_implicit",
            "students_who_x_are_y",
        }:
            cond_key = "cond" if "cond" in groups else "a"
            return ParsedPremise(
                rule=Rule(
                    conditions=(_normalize(groups[cond_key]),),
                    conclusion=_normalize(groups["concl"]),
                    raw=cleaned,
                    premise_index=premise_index,
                ),
                fact=None,
            )

        if pattern.pattern_id in {"all_x_are_y", "x_have_completed_y"}:
            return ParsedPremise(
                rule=None,
                fact=Fact(text=cleaned, raw=cleaned, premise_index=premise_index),
            )

    # Unrecognized — keep as a free-form fact.
    return ParsedPremise(
        rule=None,
        fact=Fact(text=cleaned, raw=cleaned, premise_index=premise_index),
    )


def parse_premises(premises: list[str]) -> tuple[list[Rule], list[Fact]]:
    """Parse every premise and split into ``(rules, facts)``.

    Premise indices in the returned objects are 0-based; downstream code
    is responsible for the ``P{i+1}`` label conversion.
    """
    rules: list[Rule] = []
    facts: list[Fact] = []
    for idx, raw in enumerate(premises):
        parsed = parse_premise(raw, premise_index=idx)
        if parsed.rule is not None:
            rules.append(parsed.rule)
        if parsed.fact is not None:
            facts.append(parsed.fact)
    return rules, facts
