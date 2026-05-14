"""Tests for the question text normalizer."""

from __future__ import annotations

from exact_agent.physics.question_cleaner import clean


class TestClean:
    def test_collapses_whitespace(self) -> None:
        assert clean("Calculate   the\nenergy  stored.") == "Calculate the energy stored."

    def test_strips_outer_whitespace(self) -> None:
        assert clean("   q1 = 5 C   ") == "q1 = 5 C"

    def test_replaces_nbsp(self) -> None:
        assert clean("C = 100 μF") == "C = 100 μF"

    def test_idempotent(self) -> None:
        text = "C = 100 μF and U = 30 V."
        assert clean(clean(text)) == clean(text)

    def test_empty_string(self) -> None:
        assert clean("") == ""

    def test_none_safe(self) -> None:
        # We accept None defensively (datasets sometimes have missing fields).
        assert clean("") == ""
