"""Tests for the YAML-backed formula registry."""

from __future__ import annotations

import math

import pytest

from exact_agent.physics.formula_library import default_library, load_library


@pytest.fixture(scope="module")
def lib():
    return load_library()


class TestRegistryShape:
    def test_default_library_loads(self) -> None:
        lib = default_library()
        assert len(lib) >= 12  # we declared 13 formulas in the YAML
        assert "capacitor_energy" in lib

    def test_by_topic_groups(self, lib) -> None:
        td = lib.by_topic("TD")
        assert {f.id for f in td} >= {"capacitor_energy", "capacitance_from_charge"}

    def test_required_symbols_match_yaml(self, lib) -> None:
        f = lib["coulomb_force"]
        assert set(f.required_symbols()) == {"q1", "q2", "r"}

    def test_output_unit_populated(self, lib) -> None:
        assert lib["capacitor_energy"].output_unit == "joule"


class TestCompute:
    def test_capacitor_energy_textbook(self, lib) -> None:
        # E = 0.5 * 1e-4 F * (30 V)^2 = 0.045 J
        result = lib["capacitor_energy"].compute({"C": 1e-4, "U": 30.0})
        assert math.isclose(result, 0.045, rel_tol=1e-9)

    def test_ohm_law(self, lib) -> None:
        # V = 2 A * 5 ohm = 10 V
        result = lib["ohm_law_voltage"].compute({"I": 2.0, "R": 5.0})
        assert math.isclose(result, 10.0, rel_tol=1e-9)

    def test_coulomb_force(self, lib) -> None:
        # F = k * |6e-8 * 6e-8| / 0.05^2
        #   = 8.988e9 * 3.6e-15 / 0.0025 ≈ 0.01294 N
        result = lib["coulomb_force"].compute({"q1": 6e-8, "q2": 6e-8, "r": 0.05})
        assert math.isclose(result, 0.012942, rel_tol=1e-3)

    def test_parallel_resistance(self, lib) -> None:
        # 1/R = 1/2 + 1/3  ->  R = 6/5 = 1.2
        result = lib["parallel_resistance_two"].compute({"R1": 2.0, "R2": 3.0})
        assert math.isclose(result, 1.2, rel_tol=1e-9)

    def test_resonance_frequency(self, lib) -> None:
        # f0 = 1 / (2π sqrt(LC)) with L=1H, C=1F → 1/(2π) ≈ 0.1591549...
        result = lib["resonance_frequency"].compute({"L": 1.0, "C": 1.0})
        assert math.isclose(result, 1 / (2 * math.pi), rel_tol=1e-9)

    def test_missing_input_raises(self, lib) -> None:
        with pytest.raises(KeyError):
            lib["capacitor_energy"].compute({"C": 1e-4})  # no U
