"""Tests for the evaluation metric primitives."""

from __future__ import annotations

import pytest

from exact_agent.eval.metrics import (
    label_match,
    numeric_match,
    parse_number,
    token_overlap_f1,
    unit_match,
)


class TestParseNumber:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0.045", 0.045),
            ("45", 45.0),
            (" 10 ", 10.0),
            ("1e-4", 1e-4),
            ("6×10^-8", 6e-8),
            ("6 × 10 ^ -8", 6e-8),
            (0.045, 0.045),
            ("", None),
            ("hello", None),
            (None, None),
        ],
    )
    def test_parses(self, raw, expected) -> None:
        assert (
            parse_number(raw) == pytest.approx(expected)
            if expected is not None
            else (parse_number(raw) is None)
        )


class TestNumericMatch:
    def test_within_tolerance(self) -> None:
        assert numeric_match(0.0451, "0.045", rel_tol=0.01)

    def test_outside_tolerance(self) -> None:
        assert not numeric_match(0.05, "0.045", rel_tol=0.001)

    def test_string_vs_string(self) -> None:
        assert numeric_match("0.045", "0.045")

    def test_handles_units_in_string(self) -> None:
        # parse_number picks the leading number; the unit suffix is ignored.
        assert numeric_match("0.045 J", "0.045")


class TestUnitMatch:
    @pytest.mark.parametrize(
        ("p", "e"),
        [
            ("V", "volt"),
            ("joule", "J"),
            ("Ω", "ohm"),
            ("μF", "microfarad"),
        ],
    )
    def test_dimensional_match(self, p: str, e: str) -> None:
        assert unit_match(p, e)

    def test_mismatch(self) -> None:
        assert not unit_match("V", "F")

    def test_empty_both(self) -> None:
        assert unit_match("", "")


class TestLabelMatch:
    def test_case_insensitive(self) -> None:
        assert label_match("Yes", "yes")

    def test_strips_whitespace(self) -> None:
        assert label_match(" B ", "B")

    def test_difference(self) -> None:
        assert not label_match("A", "B")


class TestTokenOverlap:
    def test_identical(self) -> None:
        r = token_overlap_f1("alpha beta gamma", "alpha beta gamma")
        assert r.f1 == 1.0

    def test_disjoint(self) -> None:
        r = token_overlap_f1("alpha beta", "gamma delta")
        assert r.f1 == 0.0

    def test_partial(self) -> None:
        r = token_overlap_f1("alpha beta gamma", "alpha beta delta")
        assert 0.0 < r.f1 < 1.0
