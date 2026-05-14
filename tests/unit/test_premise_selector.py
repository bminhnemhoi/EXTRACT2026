"""Tests for the Jaccard-based premise selector."""

from __future__ import annotations

from exact_agent.logic.premise_selector import jaccard, rank_premises, select_top_k


class TestJaccard:
    def test_zero_for_empty(self) -> None:
        assert jaccard(set(), {"x"}) == 0.0

    def test_perfect_overlap(self) -> None:
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_partial(self) -> None:
        assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


class TestRankPremises:
    def test_orders_by_overlap(self) -> None:
        question = "Is the curriculum well-structured?"
        premises = [
            "The faculty prioritizes pedagogical training.",
            "If a curriculum has exercises, it enhances engagement.",
            "All Python projects are well-structured.",
        ]
        ranked = rank_premises(question, premises)
        # P3 mentions 'well-structured' and 'curriculum' → highest overlap.
        assert ranked[0].index == 3
        assert ranked[0].score > 0
        assert all(rp.score >= 0.0 for rp in ranked)

    def test_stable_tiebreak_on_index(self) -> None:
        question = "What about logic?"
        premises = ["unrelated one", "unrelated two", "unrelated three"]
        ranked = rank_premises(question, premises)
        # All zero-score → preserve original order.
        assert [rp.index for rp in ranked] == [1, 2, 3]

    def test_label_format(self) -> None:
        ranked = rank_premises("q", ["p"])
        assert ranked[0].label == "P1"


class TestSelectTopK:
    def test_returns_at_most_k(self) -> None:
        ranked = select_top_k("q", ["a", "b", "c", "d"], k=2)
        assert len(ranked) == 2

    def test_k_larger_than_population_returns_all(self) -> None:
        ranked = select_top_k("q", ["only one"], k=8)
        assert len(ranked) == 1
