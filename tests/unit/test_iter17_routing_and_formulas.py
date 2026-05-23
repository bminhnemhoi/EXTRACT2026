"""Iter-17 (Day-29) — routing + numeric tests for the audit-driven
fixes that close additional clusters after Iter-16e (Physics 53.4%).

Scope:
* 17a inductor_magnetic_energy formula + routing (NL040)
* 17c voltage_from_energy_capacitance routing broadening (NL103)
* 17d midpoint trigger broadening for the "field at the midpoint of
  the line segment" phrasing (LD067 routing test; numeric path is
  blocked by a separate dielectric-correction gap and is not asserted)
* 17f electric_field_two_sources_at_angle formula + routing
  (LD362 perpendicular, LD367 60°)

THCB cluster (17e) is intentionally not exercised — the formula was
added to YAML but its routing rule is commented out pending a
solver-side input-unit-passthrough mechanism (gold units g/V/°C cannot
be satisfied by a dimensionless formula).
"""

from __future__ import annotations

import math

import pytest

from exact_agent.physics.formula_library import default_library
from exact_agent.physics.quantity_extractor import extract_quantities
from exact_agent.physics.question_cleaner import clean
from exact_agent.physics.topic_classifier import classify


def _route(question: str) -> str | None:
    q = clean(question)
    res = classify(q, extract_quantities(q), default_library())
    return res.formula_id if res is not None else None


# ---------------------------------------------------------------------------
# 17a — inductor_magnetic_energy
# ---------------------------------------------------------------------------


NL040_QUESTION = (
    "An inductor has L = 0.08 H, and a current of 1.5 A. "
    "Calculate the magnetic field energy (mJ)."
)


def test_17a_routes_nl040_to_inductor_magnetic_energy() -> None:
    assert _route(NL040_QUESTION) == "inductor_magnetic_energy"


def test_17a_numeric_within_one_percent() -> None:
    f = default_library()["inductor_magnetic_energy"]
    assert math.isclose(f.compute({"L": 0.08, "I": 1.5}), 0.090, rel_tol=0.01)


def test_17a_inverse_problem_still_routes_to_current_formula() -> None:
    """NL007-style: question gives W and L, asks for current. The new
    17a rule has stricter triggers ("calculate the magnetic field
    energy") so the existing current_from_inductor_energy rule still
    wins on inverse questions."""
    inverse_q = (
        "A coil stores magnetic energy W = 0.045 J when carrying current "
        "I = 1.5 A. Calculate the current at the moment when the magnetic "
        "field energy is 0.02 J."
    )
    routed = _route(inverse_q)
    assert routed in {"current_from_inductor_energy", "inductance_from_inductor_energy"}


# ---------------------------------------------------------------------------
# 17c — voltage_from_energy_capacitance broader trigger
# ---------------------------------------------------------------------------


NL103_QUESTION = (
    "A capacitor has an electric field energy of 1.8 mJ when its "
    "capacitance is 30 μF. Calculate the voltage across its plates "
    "(unit: V, round the result to two decimal places)."
)


def test_17c_routes_nl103_to_voltage_from_energy_capacitance() -> None:
    assert _route(NL103_QUESTION) == "voltage_from_energy_capacitance"


def test_17c_numeric_within_one_percent() -> None:
    f = default_library()["voltage_from_energy_capacitance"]
    # W=1.8e-3 J, C=30e-6 F -> U = sqrt(2W/C) = sqrt(0.12) ≈ 10.95 V
    pred = f.compute({"W": 1.8e-3, "C": 30e-6})
    assert math.isclose(pred, 10.95, rel_tol=0.01)


# ---------------------------------------------------------------------------
# 17d — midpoint trigger broadening (routing only; numeric path needs
# a separate dielectric-correction formula not in this batch)
# ---------------------------------------------------------------------------


LD067_QUESTION = (
    "Determine the electric field vector produced by a system of two "
    "point charges, q1 = 2 × 10^-7 C and q2 = -4 × 10^-7 C, at the "
    "midpoint of the line segment connecting the two charges. The two "
    "charges are 10 cm apart in air."
)


def test_17d_routes_ld067_to_midpoint_field_formula() -> None:
    """Trigger broadening for the 'electric field ... at the midpoint
    of the line segment' phrasing. Without dielectric, this maps to
    electric_field_two_opposite_charges_midpoint (LD067 in dielectric
    medium is deferred — requires ε_r-aware formula)."""
    assert _route(LD067_QUESTION) == "electric_field_two_opposite_charges_midpoint"


# ---------------------------------------------------------------------------
# 17f — electric_field_two_sources_at_angle
# ---------------------------------------------------------------------------


LD362_QUESTION = (
    "Two electric charges, q1 = 1.68 × 10^-6 C and q2 = 4.59 × 10^-6 C, "
    "are each located 2.80 cm from point M. The electric fields they "
    "produce at M are perpendicular to each other. Calculate the "
    "magnitude of the resultant electric field at M."
)

LD367_QUESTION = (
    "Two electric charges, q1 = 4.10 × 10^-6 C and q2 = 3.68 × 10^-6 C, "
    "are located 3.47 cm from point M. The electric fields they "
    "produce at M form an angle of 60° with each other. Calculate the "
    "magnitude of the total electric field at M."
)


@pytest.mark.parametrize("question", [LD362_QUESTION, LD367_QUESTION])
def test_17f_routes_to_at_angle_formula(question: str) -> None:
    assert _route(question) == "electric_field_two_sources_at_angle"


@pytest.mark.parametrize(
    ("q1", "q2", "r", "theta", "gold"),
    [
        (1.68e-6, 4.59e-6, 0.0280, 90, 5.611e7),  # LD362 perpendicular
        (4.10e-6, 3.68e-6, 0.0347, 60, 5.039e7),  # LD367 60°
    ],
)
def test_17f_numeric_within_one_percent(
    q1: float, q2: float, r: float, theta: float, gold: float
) -> None:
    f = default_library()["electric_field_two_sources_at_angle"]
    pred = f.compute({"q1": q1, "q2": q2, "r": r, "theta": theta})
    assert math.isclose(pred, gold, rel_tol=0.01)


def test_17f_force_asked_does_not_steal_field_formula() -> None:
    """F2 guard regression: a force-asking question with the same
    'angle of 60°' phrasing must NOT route to the field formula."""
    force_q = (
        "Two forces F1 = 3 N and F2 = 5 N act on a body and form an "
        "angle of 60° with each other. Calculate the magnitude of the "
        "resultant force."
    )
    routed = _route(force_q)
    assert routed != "electric_field_two_sources_at_angle"
