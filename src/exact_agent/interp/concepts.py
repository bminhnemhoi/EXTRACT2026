"""Shared concept vocabulary (pure stdlib).

RQ1 compares two concept sets that come from different worlds — the symbolic
solver and the SAE autointerp labels — so they must live in **one comparable
space**. v1 uses a simple, defensible operationalization: reduce both sides to
a bag of lowercased English content-word tokens. (A curated concept ontology
is the planned v2 refinement — charter §8 / limitations.)
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-zA-Z]+")

# Small English stoplist — enough to drop function words from formula
# descriptions and autointerp labels without pulling in a dependency.
STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "be", "with", "by", "from", "as", "this", "that", "these", "those",
    "it", "its", "into", "via", "per", "between", "within", "across", "given",
    "use", "used", "using", "value", "values", "feature", "features", "token",
    "tokens", "text", "context", "contexts", "model", "activates", "activation",
    "related", "concept", "concepts", "about", "when", "where", "which", "such",
})

# Coarse topic expansion so the solver's TD/LD tag enters the lexical space.
_TOPIC_WORDS: dict[str, tuple[str, ...]] = {
    "TD": ("circuit", "current", "resistance", "voltage", "power"),
    "LD": ("electrostatic", "electric", "field", "charge", "force"),
}


def tokenize_concepts(text: str, *, min_len: int = 3) -> set[str]:
    """Lowercase content-word tokens of length >= ``min_len``, minus stopwords."""
    out: set[str] = set()
    for m in _TOKEN_RE.findall(text or ""):
        tok = m.lower()
        if len(tok) >= min_len and tok not in STOPWORDS:
            out.add(tok)
    return out


def split_identifier(name: str) -> set[str]:
    """Split a predicate/alias identifier into word tokens.

    ``eligible_for_graduation`` -> {eligible, graduation};
    ``gpaAbove3_5`` -> {gpa, above}. Stopwords/short tokens are dropped.
    """
    parts = re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])", name or "")
    out: set[str] = set()
    for p in parts:
        tok = re.sub(r"[^a-zA-Z]", "", p).lower()
        if len(tok) >= 3 and tok not in STOPWORDS:
            out.add(tok)
    return out


def topic_concepts(topic: str | None) -> set[str]:
    return set(_TOPIC_WORDS.get((topic or "").upper(), ()))
