"""Iter-16 (Day-29) — routing + numeric + guard tests for the 3 new
electric-field vector formulas that close the
``electric_field_point_charge`` numeric_mismatch cluster.

Coverage:
* 16a routing — LD053 (isoceles, q1=-q2) and LD090 (equilateral, q1=-q2)
  hit :data:`electric_field_two_opposite_sources_isoceles_apex`.
* 16b routing — DT051, LD319, LD392 (same-sign equilateral, field at
  the remaining/queried vertex) hit
  :data:`electric_field_equilateral_three_identical`.
* 16d routing — LD335 (isoceles right triangle, 3 identical charges,
  field at the right-angle vertex) hits
  :data:`electric_field_right_angle_two_identical`.
* Numeric correctness vs gold for each row (<1% rel error).
* Force-vs-field guard (F2 inverse from Iter-4): an EQUILATERAL question
  asking for *force* must NOT steal the new field formulas; it must
  stay on :data:`coulomb_force_equilateral_three_identical`.
* Opposite-sign guard isolation: a same-sign isoceles question (LD052
  shape) does NOT route to 16a, falling through to point-charge until
  the future general-isoceles formula is added.
* Routing priority: the new rules sit ABOVE
  :data:`electric_field_point_charge`, so q + r symbol overlap alone
  cannot steal the wrong winner.
"""

from __future__ import annotations

import math

import pytest

from exact_agent.physics.formula_library import default_library
from exact_agent.physics.quantity_extractor import extract_quantities
from exact_agent.physics.question_cleaner import clean
from exact_agent.physics.topic_classifier import (
    _has_opposite_sign_two_charges,
    classify,
)


def _route(question: str) -> str | None:
    q = clean(question)
    quants = extract_quantities(q)
    res = classify(q, quants, default_library())
    return res.formula_id if res is not None else None


# ---------------------------------------------------------------------------
# Iter-16a — opposite-sign isoceles / equilateral apex
# ---------------------------------------------------------------------------


LD053_QUESTION = (
    "At two points A and B, separated by 10 cm in the air, two charges "
    "q1 = -q2 = 6 × 10^-6 C are placed. Determine the electric field "
    "strength caused by these two point charges at point C, knowing "
    "that AC = BC = 12 cm."
)

LD090_QUESTION = (
    "Two electric charges, q1 = 4.1 × 10^-10 C and q2 = -4.1 × 10^-10 C, "
    "are placed at points A and B in air, with AB = 2 cm. Determine the "
    "magnitude of the electric field E (in V/m) at point N, given that "
    "triangle NAB is an equilateral triangle."
)


def test_16a_routes_ld053_chained_equality_isoceles_opposite() -> None:
    assert _route(LD053_QUESTION) == "electric_field_two_opposite_sources_isoceles_apex"


def test_16a_routes_ld090_two_statement_opposite_equilateral() -> None:
    assert _route(LD090_QUESTION) == "electric_field_two_opposite_sources_isoceles_apex"


def test_16a_numeric_ld053_within_one_percent() -> None:
    f = default_library()["electric_field_two_opposite_sources_isoceles_apex"]
    pred = f.compute({"q": 6e-6, "d": 0.10, "r": 0.12})
    assert math.isclose(pred, 3.125e6, rel_tol=0.01)


def test_16a_numeric_ld090_within_three_percent() -> None:
    # Gold "9 × 10^3 V/m" is rounded to 1 sig fig; true value 9.21e3.
    # We accept up to 3% to reflect the gold's stated precision.
    f = default_library()["electric_field_two_opposite_sources_isoceles_apex"]
    pred = f.compute({"q": 4.1e-10, "d": 0.02, "r": 0.02})
    assert math.isclose(pred, 9.0e3, rel_tol=0.03)


# ---------------------------------------------------------------------------
# Iter-16b — same-sign equilateral, field at remaining vertex
# ---------------------------------------------------------------------------


DT051_QUESTION = (
    "Two point charges, q1 = q2 = 5.10-16 C, are placed at vertices B "
    "and C of an equilateral triangle ABC with a side length of 8 cm, "
    "in air. What is the magnitude of the electric field at vertex A?"
)

LD319_QUESTION = (
    "Three charges q1 = q2 = q3 = 2.6 × 10^-6 C are placed at the three "
    "vertices of an equilateral triangle with a side length of 10.3 cm. "
    "Calculate the net electric field strength at the position of q3. "
    "Give your answer rounded two decimal places."
)

LD392_QUESTION = (
    "Three charges, q1 = q2 = 4.71 × 10^-6 C and q3 = 3.38 × 10^-6 C, "
    "are placed at the three vertices of an equilateral triangle with a "
    "side length of 9.3 cm. Calculate the net electric field strength "
    "acting on q3. Give your answer rounded to two decimal places."
)


@pytest.mark.parametrize(
    "question",
    [DT051_QUESTION, LD319_QUESTION, LD392_QUESTION],
)
def test_16b_routes_same_sign_equilateral_field(question: str) -> None:
    assert _route(question) == "electric_field_equilateral_three_identical"


@pytest.mark.parametrize(
    ("q", "a", "gold"),
    [
        (5e-16, 0.08, 1.22e-3),    # DT051
        (2.6e-6, 0.103, 3.82e6),   # LD319
        (4.71e-6, 0.093, 8.48e6),  # LD392
    ],
)
def test_16b_numeric_within_one_percent(q: float, a: float, gold: float) -> None:
    f = default_library()["electric_field_equilateral_three_identical"]
    assert math.isclose(f.compute({"q": q, "a": a}), gold, rel_tol=0.01)


# ---------------------------------------------------------------------------
# Iter-16d — three identical charges, isoceles right triangle, field at
# the right-angle vertex
# ---------------------------------------------------------------------------


LD335_QUESTION = (
    "Three identical charges, q = 1.2 × 10^-6 C, are placed at the three "
    "vertices of an isosceles right triangle with legs of 5.2 cm. "
    "Calculate the net electric field strength at the right-angle vertex. "
    "Give your answer rounded two decimal places."
)


def test_16d_routes_ld335_isoceles_right_triangle_field() -> None:
    assert _route(LD335_QUESTION) == "electric_field_right_angle_two_identical"


def test_16d_numeric_ld335_within_one_percent() -> None:
    f = default_library()["electric_field_right_angle_two_identical"]
    pred = f.compute({"q": 1.2e-6, "r": 0.052})
    assert math.isclose(pred, 5.65e6, rel_tol=0.01)


# ---------------------------------------------------------------------------
# Guards — make sure the new rules don't steal questions they shouldn't.
# ---------------------------------------------------------------------------


def test_opposite_sign_detector_chained_equality() -> None:
    assert _has_opposite_sign_two_charges("q1 = -q2 = 6 × 10^-6 c") is True


def test_opposite_sign_detector_two_statement() -> None:
    text = "q1 = 4.1 × 10^-10 c and q2 = -4.1 × 10^-10 c"
    assert _has_opposite_sign_two_charges(text) is True


def test_opposite_sign_detector_rejects_same_sign() -> None:
    assert _has_opposite_sign_two_charges("q1 = q2 = 16 × 10^-8 c") is False


def test_same_sign_isoceles_does_not_route_to_16a() -> None:
    """LD052 (same-sign, AC = BC) must not steal 16a; it stays on
    electric_field_point_charge until a future 16c general-isoceles
    formula is shipped."""
    ld052 = (
        "Two point charges, q1 = q2 = 16 × 10^-8 C, are placed at points "
        "A and B, which are 10 cm apart in air. Determine the electric "
        "field strength produced by these two point charges at point C, "
        "given that AC = BC = 8 cm."
    )
    routed = _route(ld052)
    assert routed != "electric_field_two_opposite_sources_isoceles_apex"


def test_force_asked_equilateral_keeps_coulomb_formula() -> None:
    """F2 guard regression: an equilateral question asking for FORCE
    must not be stolen by Iter-16b. Newton-output formula must win."""
    force_q = (
        "Three identical charges q1 = q2 = q3 = 1.0e-6 C are placed at "
        "the vertices of an equilateral triangle with side 0.1 m. "
        "Calculate the net force on one charge."
    )
    assert _route(force_q) == "coulomb_force_equilateral_three_identical"


def test_force_asked_isoceles_keeps_coulomb_formula() -> None:
    """F2 guard regression for 16a: a force-asking isoceles question
    with opposite-sign sources stays on the force formula."""
    force_iso_q = (
        "Two opposite charges q1 = -q2 = 2.0e-6 C at A and B (AB = 10 cm), "
        "test charge q3 = 1.0e-9 C placed at C with AC = BC = 12 cm. "
        "Calculate the net force on q3."
    )
    routed = _route(force_iso_q)
    assert routed != "electric_field_two_opposite_sources_isoceles_apex"
    assert routed in {
        "coulomb_force_two_opposite_sources_isoceles_apex",
        "coulomb_force",
    }
