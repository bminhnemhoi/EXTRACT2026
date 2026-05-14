"""Unit-conversion tests."""

from __future__ import annotations

import math

import pytest

from exact_agent.physics.unit_converter import (
    UnitConversionError,
    convert,
    normalize_unit_string,
    parse_quantity,
)


class TestNormalize:
    def test_replaces_greek_mu(self) -> None:
        assert normalize_unit_string("μF") == "uF"

    def test_replaces_micro_sign(self) -> None:
        # MICRO SIGN U+00B5 vs GREEK SMALL LETTER MU U+03BC — both → 'u'.
        assert normalize_unit_string("µF") == "uF"

    def test_replaces_ohm(self) -> None:
        assert normalize_unit_string("Ω") == "ohm"

    def test_replaces_middot_and_superscript_inv(self) -> None:
        assert normalize_unit_string("V·m⁻¹") == "V*m**-1"


class TestConvert:
    @pytest.mark.parametrize(
        ("value", "src", "tgt", "expected"),
        [
            (100, "μF", "F", 1e-4),
            (100, "uF", "F", 1e-4),
            (3, "mC", "C", 3e-3),
            (5, "cm", "m", 0.05),
            (30, "V", "V", 30.0),
            (1, "kΩ", "ohm", 1000.0),
            (45, "mJ", "J", 0.045),
        ],
    )
    def test_canonical_si_conversion(
        self, value: float, src: str, tgt: str, expected: float
    ) -> None:
        result = convert(value, src, tgt)
        assert math.isclose(result.value_si, expected, rel_tol=1e-9)
        assert result.original_unit == src
        assert result.target_unit == tgt

    def test_trace_includes_both_values(self) -> None:
        result = convert(100, "μF", "F")
        assert "100" in result.trace
        assert "F" in result.trace

    def test_unparseable_unit_raises(self) -> None:
        with pytest.raises(UnitConversionError):
            convert(1, "florgs", "F")

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(UnitConversionError):
            convert(1, "V", "F")  # voltage ≠ capacitance


class TestParseQuantity:
    def test_parses_simple(self) -> None:
        value, unit = parse_quantity("30 V")
        assert math.isclose(value, 30.0)
        assert "volt" in unit

    def test_parses_with_micro(self) -> None:
        value, unit = parse_quantity("100 μF")
        assert math.isclose(value, 100.0)
        assert "farad" in unit  # pint expands μF -> microfarad

    def test_raises_on_pure_text(self) -> None:
        with pytest.raises(UnitConversionError):
            parse_quantity("hello world")
