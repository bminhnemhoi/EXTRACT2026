"""TF-IDF retrieval over solved physics-training rows (E8).

Slide 28 of the organizer kickoff deck endorses RAG as a Type-2 strategy:
*"Build a knowledge base of physics formulas and solved examples,
retrieve similar problems as few-shot examples"*. This module owns the
"solved examples" half — for a given test question, it pulls the
``k`` most-similar (question, cot) pairs from the 90% training split
of the official 2026-05-15 release.

Backbone: ``sklearn.TfidfVectorizer`` over the *question* text. We
deliberately stay TF-IDF (not sentence-transformer embeddings) because

* the rest of the stack is already TF-IDF/Jaccard (Day-4 logic
  premise_selector — deliberate `embed` extra drop),
* zero new heavyweight dep on the API image (sklearn already required
  by the `physics` extra),
* the test queries are short, near-template-form physics statements
  where lexical overlap is a strong signal.

A future swap to embeddings is a one-class change without touching the
caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from exact_agent.config import get_settings


@dataclass(frozen=True)
class WorkedExample:
    """A solved row used as a few-shot demonstration."""

    sample_id: str
    question: str
    cot: str
    answer: str
    unit: str


class RagRetriever:
    """TF-IDF nearest-neighbour over solved training rows."""

    def __init__(self, examples: list[WorkedExample]) -> None:
        if not examples:
            raise ValueError("RagRetriever needs at least one example")
        self._examples = examples
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(
            [ex.question for ex in examples]
        )

    def top_k(self, query: str, k: int = 3) -> list[WorkedExample]:
        if not query.strip() or k <= 0:
            return []
        qv = self._vectorizer.transform([query])
        # Cosine similarity for L2-normalised TF-IDF == dot product.
        sims = (self._matrix @ qv.T).toarray().ravel()
        if not np.any(sims > 0):
            return []
        top_idx = np.argsort(-sims)[:k]
        return [self._examples[int(i)] for i in top_idx if sims[int(i)] > 0]


def _load_jsonl_examples(path: Path) -> list[WorkedExample]:
    """Parse a physics training JSONL into ``WorkedExample`` rows.

    Rows missing ``question`` or ``cot`` are silently skipped — a
    demonstration without a worked solution is no demonstration.
    """
    out: list[WorkedExample] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            question = (row.get("question_clean") or row.get("question_raw") or "").strip()
            cot = (row.get("explanation_clean") or row.get("cot_raw") or "").strip()
            if not question or not cot:
                continue
            out.append(WorkedExample(
                sample_id=str(row.get("id") or row.get("sample_id") or ""),
                question=question,
                cot=cot,
                answer=str(row.get("answer_clean") or row.get("answer_raw") or ""),
                unit=str(row.get("unit_clean") or row.get("unit_raw") or ""),
            ))
    return out


@lru_cache(maxsize=1)
def default_retriever() -> RagRetriever | None:
    """Build a retriever from the official-release physics training split.

    Returns ``None`` if the train split isn't present (e.g. in CI before
    the importer has been run) so the solver can quietly fall back to
    no-RAG extraction instead of crashing.
    """
    train_path = (
        get_settings().project_root
        / "data" / "official_v20260515" / "train" / "physics_train.jsonl"
    )
    examples = _load_jsonl_examples(train_path)
    if not examples:
        return None
    return RagRetriever(examples)
