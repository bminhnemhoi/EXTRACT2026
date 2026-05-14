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

    def test_parallel_resistance(self, solver: PhysicsSolver) -> None:
        question = "Two resistors R1 = 4 ohm and R2 = 6 ohm are connected in parallel."
        result = solver.solve(question)
        assert result.success, result.fail_reason
        assert math.isclose(result.answer_value, 2.4, rel_tol=1e-9)


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
