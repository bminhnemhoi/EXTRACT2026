"""End-to-end tests for the physics solver chain."""

from __future__ import annotations

import math

import pytest

from exact_agent.physics.solver import PhysicsSolver


@pytest.fixture(scope="module")
def solver():
    return PhysicsSolver()


class TestCanonicalProblems:
    def test_capacitor_energy(self, solver: PhysicsSolver) -> None:
        question = "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
        result = solver.solve(question)
        assert result.success, result.fail_reason
        assert math.isclose(result.answer_value, 0.045, rel_tol=1e-9)
        assert result.answer_unit == "joule"
        assert result.formula_id == "capacitor_energy"
        assert result.confidence > 0.8
        # Trace should mention the substitution.
        assert any("Apply" in line for line in result.trace)

    def test_ohm_law(self, solver: PhysicsSolver) -> None:
        question = "Apply Ohm's law: with R = 5 ohm and I = 2 A, what is the voltage?"
        result = solver.solve(question)
        assert result.success, result.fail_reason
        assert math.isclose(result.answer_value, 10.0, rel_tol=1e-9)
        assert result.answer_unit == "volt"

    def test_resultant_two_forces(self, solver: PhysicsSolver) -> None:
        # LD015: law of cosines for vector addition of two given forces.
        question = (
            "Two electric forces with magnitudes of F1 = 5 N and F2 = 12 N "
            "act at an angle of theta = 60 degree to each other. "
            "Calculate the resultant force."
        )
        result = solver.solve(question)
        assert result.success, result.fail_reason
        assert result.formula_id == "resultant_two_forces"
        # sqrt(25 + 144 + 2*5*12*cos60°) = sqrt(229) ≈ 15.13 N
        assert math.isclose(result.answer_value, 15.13, rel_tol=1e-3)
        assert result.answer_unit == "newton"

    def test_parallel_resistance(self, solver: PhysicsSolver) -> None:
        question = "Two resistors R1 = 4 ohm and R2 = 6 ohm are connected in parallel."
        result = solver.solve(question)
        assert result.success, result.fail_reason
        assert math.isclose(result.answer_value, 2.4, rel_tol=1e-9)

    def test_capacitance_for_resonance(self, solver: PhysicsSolver) -> None:
        # CH062: C = 1/(L·(2πf)²); L=0.2 H, f=100 Hz → ~12.67 µF
        q = (
            "What capacitance must the capacitor have for an LC circuit "
            "with L = 0.2 H to resonate at f = 100 Hz?"
        )
        r = solver.solve(q)
        assert r.success, r.fail_reason
        assert r.formula_id == "capacitance_for_resonance"
        assert math.isclose(r.answer_value, 1.2665e-5, rel_tol=1e-3)

    def test_resonance_frequency_factor(self, solver: PhysicsSolver) -> None:
        # CH187: k = sqrt(X_C/X_L); 25, 225 → 3
        q = (
            "In a series RLC circuit, X_L = 25 ohm and X_C = 225 ohm. "
            "By what factor of ω0 should the angular frequency be set?"
        )
        r = solver.solve(q)
        assert r.success, r.fail_reason
        assert r.formula_id == "resonance_frequency_factor"
        assert math.isclose(r.answer_value, 3.0, rel_tol=1e-6)

    def test_resistance_at_resonance(self, solver: PhysicsSolver) -> None:
        q = (
            "In a resonant RLC circuit, the measured impedance is Z = 40 ohm. "
            "Find the pure resistance R."
        )
        r = solver.solve(q)
        assert r.success, r.fail_reason
        assert r.formula_id == "resistance_at_resonance"
        assert math.isclose(r.answer_value, 40.0, rel_tol=1e-9)

    def test_power_at_resonance(self, solver: PhysicsSolver) -> None:
        q = (
            "A resonant RLC circuit has a resistance R = 25 ohm and an "
            "applied voltage U = 100 V. Find the power."
        )
        r = solver.solve(q)
        assert r.success, r.fail_reason
        assert r.formula_id == "power_at_resonance"
        assert math.isclose(r.answer_value, 400.0, rel_tol=1e-9)

    def test_relative_error_percent(self, solver: PhysicsSolver) -> None:
        q = "A voltmeter with delta = 0.2 reads value = 5.6. What is the relative error percentage?"
        r = solver.solve(q)
        assert r.success, r.fail_reason
        assert r.formula_id == "relative_error_percent"
        assert math.isclose(r.answer_value, 3.5714, rel_tol=1e-3)

    def test_current_from_inductor_energy(self, solver: PhysicsSolver) -> None:
        q = "A coil with L = 0.1 H stores W = 0.2 J of magnetic energy. What current I flows?"
        r = solver.solve(q)
        assert r.success, r.fail_reason
        assert r.formula_id == "current_from_inductor_energy"
        assert math.isclose(r.answer_value, 2.0, rel_tol=1e-6)


class TestFailureModes:
    def test_no_formula_matches(self, solver: PhysicsSolver) -> None:
        result = solver.solve("Explain Newton's first law of motion.")
        assert not result.success
        assert result.fail_reason == "no_formula_matched"
        assert result.answer_value is None

    def test_missing_required_input(self, solver: PhysicsSolver) -> None:
        # "energy stored" → capacitor_energy, but no C provided.
        result = solver.solve("Energy stored in a capacitor with U = 30 V.")
        assert not result.success
        assert result.fail_reason and result.fail_reason.startswith("missing_input")


class TestUnitConversionInside:
    def test_converts_micro_farad_in_trace(self, solver: PhysicsSolver) -> None:
        question = "Energy stored when C = 100 μF and U = 30 V."
        result = solver.solve(question)
        assert any("0.0001" in line and "farad" in line.lower() for line in result.trace), (
            f"trace did not show μF→F conversion: {result.trace}"
        )
