"""Iter-18 (Day-29) — tests for the 6-fix audit batch after Iter-17.

Coverage:
* 18a parse_number: "X . 10^Y" dot-multiplier + "10^{N}" Latex braces
* 18b energy_loss_percent output_unit == "percent" (NL092 fix)
* 18c LD067 routing + numeric (dielectric-aware midpoint field)
* 18d LD081 + LD387 routing + numeric (same-sign perp-bisector)
* 18e LD243 routing + numeric (right-angle 3-identical FORCE)
* 18f CH360 routing (ohm_law_current at resonance)
* Same-sign guard isolation (opposite-sign LD053-style stays on its formula)
"""

from __future__ import annotations

import math

import pytest

from exact_agent.eval.metrics import parse_number, quantity_match
from exact_agent.physics.formula_library import default_library
from exact_agent.physics.quantity_extractor import extract_quantities
from exact_agent.physics.question_cleaner import clean
from exact_agent.physics.topic_classifier import (
    _has_same_sign_two_charges,
    classify,
)


def _route(question: str) -> str | None:
    q = clean(question)
    res = classify(q, extract_quantities(q), default_library())
    return res.formula_id if res is not None else None


# ---------------------------------------------------------------------------
# 18a — parse_number dataset-notation extensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3 . 10^4", 3e4),                  # Vietnamese dot-multiplier (DT046)
        ("1.22 . 10^{-3}", 1.22e-3),         # dot-multiplier + Latex braces (DT051)
        ("5.6 . 10^{-12}", 5.6e-12),
        ("2 . 10^3", 2e3),
        # Regressions: existing forms must still parse correctly
        ("3.14", 3.14),                      # plain decimal — must NOT be touched
        ("6×10^-8", 6e-8),
        ("3.89*10^-3", 3.89e-3),
        ("33.6 × 10^5", 33.6e5),
        ("8.48 × 10⁶", 8.48e6),              # Iter-16e superscript regression
    ],
)
def test_18a_parser_handles_dot_multiplier_and_braces(raw: str, expected: float) -> None:
    assert parse_number(raw) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 18b — energy_loss_percent output unit
# ---------------------------------------------------------------------------


def test_18b_energy_loss_percent_unit_is_percent() -> None:
    """The formula expression returns a percentage value (75 for a 75%
    loss, not 0.75). Output unit must be "percent" so quantity_match
    against a "%" gold doesn't trigger the 100× pint conversion that
    false-failed NL092."""
    f = default_library()["energy_loss_percent"]
    assert f.output_unit == "percent"
    assert quantity_match("75", "percent", "75", "%") is True


# ---------------------------------------------------------------------------
# 18c — LD067 dielectric-aware midpoint field
# ---------------------------------------------------------------------------


LD067_QUESTION = (
    "Determine the electric field vector produced by a system of two "
    "point charges, q1 = 2 × 10^-7 C and q2 = -4 × 10^-7 C, at the "
    "midpoint of the line segment connecting the two charges. The two "
    "charges are separated by 10 cm and are located in alcohol, which "
    "has a dielectric constant ε = 2.2."
)


def test_18c_routes_ld067_to_dielectric_midpoint_formula() -> None:
    assert _route(LD067_QUESTION) == "electric_field_two_opposite_charges_midpoint_dielectric"


def test_18c_numeric_within_one_percent() -> None:
    f = default_library()["electric_field_two_opposite_charges_midpoint_dielectric"]
    pred = f.compute({"q1": 2e-7, "q2": -4e-7, "d": 0.10, "eps_r": 2.2})
    assert math.isclose(pred, 9.8e5, rel_tol=0.01)


# ---------------------------------------------------------------------------
# 18d — same-sign perpendicular bisector field
# ---------------------------------------------------------------------------


LD081_QUESTION = (
    "Point charges qA and qB, both equal to 3 × 10^-7 C, are placed "
    "respectively at two points A and B in the air. The distance AB "
    "is 12 cm. M is a point located on the perpendicular bisector of "
    "AB, 8 cm away from the line segment AB. What is the magnitude of "
    "the electric field at M?"
)

LD387_QUESTION = (
    "Two electric charges, q1 = 2.96 × 10^-6 C and q2 = 3.84 × 10^-6 C, "
    "are placed at two points 5.9 cm apart. Calculate the electric "
    "field strength at a point on the perpendicular bisector, "
    "equidistant from both charges at 2.95 cm."
)


@pytest.mark.parametrize("question", [LD081_QUESTION, LD387_QUESTION])
def test_18d_routes_same_sign_perp_bisector(question: str) -> None:
    assert _route(question) == "electric_field_perp_bisector_two_same_sign"


@pytest.mark.parametrize(
    ("q1", "q2", "d", "l", "gold"),
    [
        (3e-7, 3e-7, 0.12, 0.08, 4.32e5),       # LD081
        (2.96e-6, 3.84e-6, 0.059, 0.0, 9.09e6),  # LD387 (l=0, midpoint case)
    ],
)
def test_18d_numeric_within_one_percent(
    q1: float, q2: float, d: float, l: float, gold: float
) -> None:
    f = default_library()["electric_field_perp_bisector_two_same_sign"]
    pred = f.compute({"q1": q1, "q2": q2, "d": d, "l": l})
    assert math.isclose(pred, gold, rel_tol=0.01)


def test_18d_opposite_sign_still_uses_original_formula() -> None:
    """Regression: LD053-style q1 = -q2 perp-bisector field stays on
    the original opposite-sign formula, NOT the new same-sign one."""
    opp_q = (
        "Two point charges q1 = -q2 = 5e-6 C at A and B, AB = 10 cm. "
        "Find the electric field strength at point M on the "
        "perpendicular bisector of AB, 6 cm from the midpoint."
    )
    routed = _route(opp_q)
    assert routed != "electric_field_perp_bisector_two_same_sign"


# ---------------------------------------------------------------------------
# 18e — right-angle 3-identical FORCE formula
# ---------------------------------------------------------------------------


LD243_QUESTION = (
    "Three identical charges q = -3 × 10^-8 C are placed at the three "
    "vertices of an isosceles right triangle with legs of 15 cm. "
    "Calculate the net force acting on the charge at the right-angle "
    "vertex."
)


def test_18e_routes_ld243_to_force_right_angle_formula() -> None:
    assert _route(LD243_QUESTION) == "coulomb_force_right_angle_two_identical"


def test_18e_numeric_within_one_percent() -> None:
    f = default_library()["coulomb_force_right_angle_two_identical"]
    pred = f.compute({"q": 3e-8, "r": 0.15})
    assert math.isclose(pred, 5.091e-4, rel_tol=0.01)


def test_18e_field_asked_keeps_iter16d_field_formula() -> None:
    """Regression: a field-asked right-angle-vertex question (LD335
    shape) must keep routing to the Iter-16d field formula, NOT the
    new 18e force formula."""
    field_q = (
        "Three identical charges, q = 1.2 × 10^-6 C, are placed at the "
        "three vertices of an isosceles right triangle with legs of "
        "5.2 cm. Calculate the net electric field strength at the "
        "right-angle vertex."
    )
    assert _route(field_q) == "electric_field_right_angle_two_identical"


# ---------------------------------------------------------------------------
# 18f — CH360 ohm_law_current at resonance
# ---------------------------------------------------------------------------


CH360_QUESTION = "At resonance with U = 100 V, R = 25 Ω, what is I?"


def test_18f_routes_ch360_to_ohm_law_current() -> None:
    assert _route(CH360_QUESTION) == "ohm_law_current"


def test_18f_power_resonance_question_still_routes_to_power() -> None:
    """Regression: a power-asking resonance question (CH041 shape)
    must still route to power_at_resonance — the dim guard relies on
    the question explicitly asking for power."""
    power_q = (
        "An RLC series circuit in resonance has an RMS voltage across "
        "the circuit of 100 V, and a pure resistance R = 50 Ω. "
        "Calculate the power consumed."
    )
    routed = _route(power_q)
    assert routed != "ohm_law_current"


# ---------------------------------------------------------------------------
# Same-sign guard sanity (used by 18d)
# ---------------------------------------------------------------------------


def test_same_sign_guard_detects_qa_qb_both_equal() -> None:
    assert _has_same_sign_two_charges(
        "point charges qa and qb, both equal to 3 × 10^-7 c"
    ) is True


def test_same_sign_guard_rejects_opposite_sign() -> None:
    assert _has_same_sign_two_charges("q1 = -q2 = 5e-6 c") is False


def test_same_sign_guard_detects_two_positive_q1_q2() -> None:
    assert _has_same_sign_two_charges(
        "q1 = 2.96 × 10^-6 c and q2 = 3.84 × 10^-6 c"
    ) is True
