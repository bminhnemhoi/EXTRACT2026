"""Tests for the physics plausibility verifier."""

from __future__ import annotations

import math

from exact_agent.physics.verifier import verify


class TestPlausibility:
    def test_plausible_capacitor_energy(self) -> None:
        result = verify(0.045, "joule")
        assert result.ok
        assert result.warnings == ()

    def test_implausible_capacitance(self) -> None:
        # 1e30 F is absurd for a circuit element.
        result = verify(1e30, "farad")
        assert not result.ok
        assert len(result.warnings) == 1
        assert "outside the plausible range" in result.warnings[0]

    def test_zero_is_allowed_for_field(self) -> None:
        # E = 0 V/m at the midpoint between equal charges is legitimate.
        result = verify(0.0, "volt/meter")
        assert result.ok

    def test_unknown_unit_passes_through(self) -> None:
        result = verify(1234.5, "moose")
        assert result.ok  # no bounds defined → no warning

    def test_nan_flagged(self) -> None:
        result = verify(math.nan, "joule")
        assert not result.ok
        assert "non-finite" in result.warnings[0]

    def test_inf_flagged(self) -> None:
        result = verify(math.inf, "joule")
        assert not result.ok
        assert "non-finite" in result.warnings[0]

    def test_none_value(self) -> None:
        result = verify(None, "joule")
        assert not result.ok
