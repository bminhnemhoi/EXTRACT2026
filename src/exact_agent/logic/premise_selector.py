"""Rank premises by relevance to the question.

We use token Jaccard overlap on lemma-like surface forms. Cheap, no model
download, fully deterministic. The selector is the primary feeder of the
``premises`` field in the API response (P3 reasoning-depth signal); the
answer verifier consumes the top-k ranking too.

A sentence-transformers backend can replace this without changing the
interface — we keep ``select`` as a function on text only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")

# Surface stopwords. Lemmatization is overkill at Day-4 baseline; we just
# drop common English glue words so signal tokens dominate Jaccard.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "having",
        "of",
        "in",
        "on",
        "at",
        "by",
        "to",
        "from",
        "for",
        "with",
        "and",
        "or",
        "not",
        "but",
        "if",
        "then",
        "else",
        "when",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "where",
        "why",
        "how",
        "according",
        "premises",
        "premise",
        "follow",
        "follows",
        "based",
        "above",
        "below",
        "regarding",
        "x",
        "y",
        "z",
        "it",
        "its",
        "they",
        "their",
        "them",
        "he",
        "she",
        "his",
        "her",
        "as",
        "so",
        "such",
        "than",
        "while",
        "also",
        "just",
        "can",
        "could",
        "may",
        "might",
        "must",
        "should",
        "will",
        "would",
    }
)


_SUFFIXES: tuple[str, ...] = ("ies", "ed", "ing", "es", "s")


def _stem(token: str) -> str:
    """Cheap suffix stripping. Maps 'receive', 'receives', 'receiving' all → 'receiv'.
    Not a real Porter stemmer — just enough to collapse singular/plural and tense
    morphology so Jaccard overlap works across paraphrases.
    """
    if len(token) < 5:
        return token
    # First absorb "es"/"ed"/"ing"/"ies"/"s" suffixes.
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            token = token[: -len(suf)]
            break
    # Then strip a trailing silent "e" so 'receive' and 'receives' collapse together.
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def _normalize(text: str) -> set[str]:
    """Lowercase, strip stopwords, light-stem, deduplicate."""
    tokens = (t.lower() for t in _TOKEN_RE.findall(text or ""))
    return {_stem(t) for t in tokens if t not in _STOPWORDS and len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class RankedPremise:
    """A single premise with its rank score and 1-based index id."""

    index: int
    """1-based premise number, used as the ``P1``/``P7`` reference."""

    text: str
    score: float

    @property
    def label(self) -> str:
        return f"P{self.index}"


def rank_premises(question: str, premises: list[str]) -> list[RankedPremise]:
    """Return all premises sorted by descending Jaccard to the question.

    Ties are broken by premise order (stable sort).
    """
    q_tokens = _normalize(question)
    scored: list[RankedPremise] = []
    for i, premise in enumerate(premises, start=1):
        p_tokens = _normalize(premise)
        scored.append(RankedPremise(index=i, text=premise, score=jaccard(q_tokens, p_tokens)))
    scored.sort(key=lambda rp: (-rp.score, rp.index))
    return scored


def select_top_k(question: str, premises: list[str], k: int = 8) -> list[RankedPremise]:
    """Convenience wrapper — return the top ``k`` ranked premises."""
    return rank_premises(question, premises)[: max(1, k)]
