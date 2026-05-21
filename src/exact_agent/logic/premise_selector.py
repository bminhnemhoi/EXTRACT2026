"""Rank premises by relevance to the question.

Day-23 (F3) upgrade: TF-IDF cosine over preprocessed (lowercase +
stopword-strip + light-stem) tokens, fit per query on the question +
premise set. This is uniformly stronger than the original Jaccard:

* **rare words count more** (e.g. "centroid" vs "the") — Jaccard
  treats both as 1 hit. TF-IDF down-weights ubiquitous tokens.
* still no model download, still deterministic.

Compared against the official dataset's ground-truth ``idx`` (the
1-based list of premises actually used per question — CHANGELOG_TYPE1
calls this out as the P3 cited-premises gold), TF-IDF gives a higher
Recall@k on the official 2026-05-15 logic split.

Legacy ``jaccard`` is kept as a public helper for code that wants the
older similarity, and the answer_verifier still imports ``_normalize``.
The Jaccard ranker is selectable via ``method="jaccard"`` on
:func:`rank_premises` to preserve the old behaviour for any caller
that depends on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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


def _preprocessed(text: str) -> str:
    """Stopword-strip + stem; return as space-joined tokens for TF-IDF."""
    return " ".join(sorted(_normalize(text)))


def _rank_tfidf(question: str, premises: list[str]) -> list[RankedPremise]:
    """F3 (Day-23): per-query TF-IDF cosine over preprocessed tokens.

    Sorts by cosine similarity descending; ties (incl. all-zero) preserve
    premise order (stable sort), matching the Jaccard implementation's
    contract so existing tests still pass.
    """
    docs = [_preprocessed(question)] + [_preprocessed(p) for p in premises]
    # Defensive: if every doc preprocesses to empty (question + all
    # premises were 100% stopwords), fall back to zero-score Jaccard
    # behaviour rather than letting sklearn raise.
    if not any(d.strip() for d in docs):
        return [
            RankedPremise(index=i + 1, text=p, score=0.0)
            for i, p in enumerate(premises)
        ]
    vec = TfidfVectorizer(
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=1,
        max_df=1.0,
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        matrix = vec.fit_transform(docs)
    except ValueError:
        # Empty vocabulary after vectoriser-internal filtering — same
        # graceful degrade as above.
        return [
            RankedPremise(index=i + 1, text=p, score=0.0)
            for i, p in enumerate(premises)
        ]
    sims = np.asarray((matrix[0] @ matrix[1:].T).todense()).ravel()
    scored = [
        RankedPremise(index=i + 1, text=p, score=float(s))
        for i, (p, s) in enumerate(zip(premises, sims, strict=False))
    ]
    scored.sort(key=lambda rp: (-rp.score, rp.index))
    return scored


def _rank_jaccard(question: str, premises: list[str]) -> list[RankedPremise]:
    """Legacy Jaccard ranker (Day-4). Kept for callers that need it."""
    q_tokens = _normalize(question)
    scored: list[RankedPremise] = []
    for i, premise in enumerate(premises, start=1):
        p_tokens = _normalize(premise)
        scored.append(RankedPremise(index=i, text=premise, score=jaccard(q_tokens, p_tokens)))
    scored.sort(key=lambda rp: (-rp.score, rp.index))
    return scored


def rank_premises(
    question: str, premises: list[str], *, method: str = "tfidf",
) -> list[RankedPremise]:
    """Return all premises sorted by descending similarity to the question.

    ``method="tfidf"`` (default, F3) — TF-IDF cosine over preprocessed
    tokens. ``method="jaccard"`` — the original Day-4 Jaccard ranker.
    Ties are broken by premise order (stable sort) under both methods.
    """
    if method == "jaccard":
        return _rank_jaccard(question, premises)
    return _rank_tfidf(question, premises)


def select_top_k(question: str, premises: list[str], k: int = 8) -> list[RankedPremise]:
    """Convenience wrapper — return the top ``k`` ranked premises."""
    return rank_premises(question, premises)[: max(1, k)]
