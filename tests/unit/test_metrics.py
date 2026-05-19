"""Tests for the evaluation metric primitives."""

from __future__ import annotations

import pytest

from exact_agent.eval.metrics import (
    label_match,
    numeric_match,
    parse_number,
    quantity_match,
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


class TestQuantityMatch:
    def test_si_value_matches_prefixed_gold(self) -> None:
        # The exact Day-8 defect: solver emits SI coulombs, gold is in nC.
        # TD039: 5.987e-10 C  ==  0.6 nC
        assert quantity_match(5.987e-10, "coulomb", "0.6", "nC")

    def test_td060_and_td181_real_cases(self) -> None:
        assert quantity_match(7.27e-10, "coulomb", "0.73", "nC")
        assert quantity_match(1.454e-9, "coulomb", "1.46", "nC")

    def test_genuinely_wrong_still_fails(self) -> None:
        # TD397: solver 8927 C vs gold 26.55 nC — must stay False.
        assert not quantity_match(8927.464, "coulomb", "26.55", "nC")

    def test_same_unit_passthrough(self) -> None:
        assert quantity_match(0.045, "joule", "0.045", "J")

    def test_microfarad_gold(self) -> None:
        # 1e-4 F  ==  100 µF
        assert quantity_match(1e-4, "farad", "100", "μF")

    def test_missing_units_falls_back_to_raw(self) -> None:
        # No units → legacy numeric_match behavior.
        assert quantity_match(0.045, "", "0.045", "")
        assert not quantity_match(0.05, "", "0.045", "", rel_tol=0.001)

    def test_inconvertible_units_falls_back_to_raw(self) -> None:
        # volt vs farad are not convertible → raw compare of the magnitudes.
        assert quantity_match(3.0, "volt", "3.0", "farad")
        assert not quantity_match(3.0, "volt", "9.0", "farad")

    def test_none_inputs_return_false(self) -> None:
        assert not quantity_match(None, "coulomb", "0.6", "nC")
        assert not quantity_match(5e-10, "coulomb", None, "nC")


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

    @pytest.mark.parametrize(
        ("p", "e"),
        [
            ("dimensionless", "-"),
            ("dimensionless", ""),
            ("-", ""),
            ("dimensionless", "ratio"),
            ("none", "-"),
        ],
    )
    def test_dimensionless_tokens_all_match(self, p: str, e: str) -> None:
        # k-factor / power-factor answers: solver says "dimensionless",
        # gold writes "-" — must not fail on the cosmetic difference.
        assert unit_match(p, e)

    def test_dimensionless_does_not_match_real_unit(self) -> None:
        assert not unit_match("dimensionless", "V")
        assert not unit_match("-", "ohm")


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
