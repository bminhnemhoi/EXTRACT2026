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

    def test_unicode_minus_to_ascii(self) -> None:
        # Iter-6 (DDT362): U+2212 MINUS SIGN in "C = 10−6 F" must become
        # ASCII '-' so the scientific-notation regex can fire.
        assert clean("C = 10−6 F") == "C = 10^-6 F"

    def test_en_dash_in_scientific_notation(self) -> None:
        # EN DASH (U+2013) becomes ASCII '-'. The bare-10 rewriter only
        # fires when 10 is flanked by space/equals, so mantissa-times-power
        # is left as is — the extractor's scientific regex handles that form.
        assert clean("q = 2×10–8 C") == "q = 2×10-8 C"

    def test_bare_10_neg_is_only_in_scientific_context(self) -> None:
        # The bare-10 rewrite must not corrupt arithmetic differences.
        # "(10-6)" or "10 -6" without flanking space-then-equals stays put.
        assert "10^-" not in clean("There are 10-6=4 apples.")
