"""Tests for the E8 TF-IDF RAG retriever."""

from __future__ import annotations

import pytest

from exact_agent.physics.rag_retriever import RagRetriever, WorkedExample

EXAMPLES = [
    WorkedExample(
        sample_id="TD001",
        question="Calculate the energy stored when C = 100 μF and U = 30 V.",
        cot="E = 0.5 × C × U² = 0.045 J",
        answer="0.045", unit="J",
    ),
    WorkedExample(
        sample_id="LD001",
        question="Find the Coulomb force between two charges q1 = 1e-6 C and q2 = 2e-6 C at distance 0.1 m.",
        cot="F = k × |q1·q2|/r² = 1.8 N",
        answer="1.8", unit="N",
    ),
    WorkedExample(
        sample_id="CH001",
        question="An RLC series circuit is in resonance with U = 110 V and R = 110 Ω. Power consumed?",
        cot="At resonance Z = R; P = U²/R = 110 W",
        answer="110", unit="W",
    ),
    WorkedExample(
        sample_id="TD002",
        question="A parallel-plate capacitor has area 30 cm² and gap 1 mm. Find capacitance.",
        cot="C = ε0·A/d",
        answer="2.6e-11", unit="F",
    ),
]


class TestRagRetriever:
    def test_retrieves_capacitor_for_capacitor_query(self) -> None:
        # The retriever returns the best-matching example FIRST.
        # `top_k` may return <=k entries (only those with positive
        # similarity — better to drop than to pad with off-topic noise).
        r = RagRetriever(EXAMPLES)
        out = r.top_k("Calculate energy when C = 50 μF and U = 20 V.", k=3)
        assert out and out[0].sample_id == "TD001"

    def test_retrieves_coulomb_for_charge_query(self) -> None:
        r = RagRetriever(EXAMPLES)
        out = r.top_k("Two charges q1 and q2 separated by r — find force.", k=1)
        assert out and out[0].sample_id == "LD001"

    def test_empty_query_returns_empty(self) -> None:
        r = RagRetriever(EXAMPLES)
        assert r.top_k("", k=3) == []

    def test_k_zero_returns_empty(self) -> None:
        r = RagRetriever(EXAMPLES)
        assert r.top_k("anything", k=0) == []

    def test_no_overlap_returns_empty_not_random(self) -> None:
        # A query that shares NO meaningful tokens with the corpus
        # (after stop-word removal) yields zero — better than
        # silently returning random demos.
        r = RagRetriever(EXAMPLES)
        out = r.top_k("xyzzy qwerty", k=3)
        assert out == []

    def test_empty_corpus_raises(self) -> None:
        with pytest.raises(ValueError):
            RagRetriever([])

    def test_top_k_respects_k(self) -> None:
        r = RagRetriever(EXAMPLES)
        out = r.top_k("capacitor", k=2)
        assert len(out) <= 2
