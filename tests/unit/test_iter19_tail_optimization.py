"""Iter-19 (Day-29) — 7-fix tail-optimization batch tests.

Coverage:
* 19a CH143/144/145 — voltage_capacitor_from_urc_resonance formula
  (RLC at resonance: U_C = sqrt(U_RC^2 - U^2)).
* 19b LD052 — same-sign general isoceles field formula.
* 19c THCB094/114/133 — 3 unit-specific mean formulas (mass/V/°C).
* 19d CH160 — Imax @ resonance routes to ohm_law_current.
* 19e CH247/249 — LCω²=1 power factor = 1 (constant formula).
* 19f NL334 — "what is its inductance" routes to inductance_from_inductor_energy.
* 19g √-literal cleaner — "100√2" → "141.421" before extractor.

Routing/numeric per cluster + regression checks that the new rules
don't steal pre-existing routes (Iter-15..18 wins still pass).
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
# 19a — CH RLC at resonance U_C = sqrt(U_RC^2 - U^2)
# ---------------------------------------------------------------------------


CH143_Q = (
    "Consider a series RLC AC circuit, powered by an RMS voltage of 220 V "
    "at a frequency of 50 Hz. It is known that the RMS voltage across the "
    "R-C combination and across the C-L combination are both 300 V, and "
    "the circuit is currently in resonance. What is the RMS voltage across "
    "the capacitor C?"
)
CH144_Q = (
    "A series RLC circuit at resonance is connected to an AC supply with "
    "an RMS voltage of 90 V. The RMS voltage measured across the R-C "
    "section and across the C-L section are both 110 V. Calculate the "
    "RMS voltage across the capacitor C."
)
CH145_Q = (
    "An RLC series circuit is connected to an AC power source with an RMS "
    "voltage U = 150 V. At resonance, the RMS voltages across the R-C "
    "combination and the C-L combination are equal, and both are 180 V. "
    "Calculate the RMS voltage across the capacitor C."
)


@pytest.mark.parametrize("question", [CH143_Q, CH144_Q, CH145_Q])
def test_19a_routes_ch_rlc_to_urc_resonance_formula(question: str) -> None:
    assert _route(question) == "voltage_capacitor_from_urc_resonance"


@pytest.mark.parametrize(
    ("U", "U_RC", "gold"),
    [
        (220, 300, 203.96),
        (90, 110, 63.25),
        (150, 180, 99.50),
    ],
)
def test_19a_numeric_within_one_percent(U: float, U_RC: float, gold: float) -> None:
    f = default_library()["voltage_capacitor_from_urc_resonance"]
    assert math.isclose(f.compute({"U": U, "U_RC": U_RC}), gold, rel_tol=0.01)


# ---------------------------------------------------------------------------
# 19b — LD052 same-sign isoceles general
# ---------------------------------------------------------------------------


LD052_Q = (
    "Two point charges, q1 = q2 = 16 × 10^-8 C, are placed at points A "
    "and B, which are 10 cm apart in air. Determine the electric field "
    "strength produced by these two point charges at point C, given that "
    "AC = BC = 8 cm."
)


def test_19b_routes_ld052_to_same_sign_isoceles() -> None:
    assert _route(LD052_Q) == "electric_field_isoceles_two_same_sign_apex"


def test_19b_numeric_within_one_percent() -> None:
    f = default_library()["electric_field_isoceles_two_same_sign_apex"]
    pred = f.compute({"q": 16e-8, "d": 0.10, "r": 0.08})
    assert math.isclose(pred, 3.51e5, rel_tol=0.01)


def test_19b_does_not_steal_equilateral_case() -> None:
    """Iter-16b equilateral route must still win when AC = BC = AB."""
    equilateral_q = (
        "Two point charges q1 = q2 = 5e-16 C are placed at vertices B "
        "and C of an equilateral triangle ABC with side 8 cm. What is "
        "the electric field at vertex A?"
    )
    assert _route(equilateral_q) == "electric_field_equilateral_three_identical"


# ---------------------------------------------------------------------------
# 19c — THCB per-unit mean formulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected_formula"),
    [
        (
            "Three mass measurements were taken: 100.2 g; 100.0 g; 100.4 g. "
            "Calculate the average mass and the average absolute error.",
            "mean_three_measurements_mass",
        ),
        (
            "Three voltage readings: 9.9V; 10.1V; 10.0V. Calculate the "
            "average voltage and the average absolute error.",
            "mean_three_measurements_voltage",
        ),
        (
            "Three temperature measurements: 20.1°C; 20.0°C; 19.9°C. "
            "Calculate the average temperature and the average absolute "
            "error.",
            "mean_three_measurements_temperature",
        ),
    ],
)
def test_19c_routes_thcb_per_unit(question: str, expected_formula: str) -> None:
    assert _route(question) == expected_formula


def test_19c_numeric_mean() -> None:
    f = default_library()["mean_three_measurements_mass"]
    assert math.isclose(f.compute({"x1": 100.2, "x2": 100.0, "x3": 100.4}), 100.2)


# ---------------------------------------------------------------------------
# 19d — CH160 Imax @ resonance
# ---------------------------------------------------------------------------


def test_19d_routes_ch160_max_current_at_resonance() -> None:
    q = (
        "An effective voltage U = 120 V is applied to a series RLC circuit "
        "with R = 80 Ω, operating at resonance. Calculate the maximum "
        "effective current Imax."
    )
    assert _route(q) == "ohm_law_current"


# ---------------------------------------------------------------------------
# 19e — CH247/249 power factor @ LC resonance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "For circuit AB, LCω2 = 1, and uAM and uMB are 90° out of phase. "
        "Given R1 = 70 Ω and R2 = 40 Ω. Calculate the power factor of the "
        "entire circuit.",
        "The circuit AB satisfies the condition LCω2 = 1, and the voltages "
        "u_AM and u_MB are 90° out of phase. Given R1 = 60 Ω and R2 = 15 Ω. "
        "Calculate the power factor of the entire circuit.",
    ],
)
def test_19e_routes_lc_resonance_power_factor_to_constant(question: str) -> None:
    assert _route(question) == "power_factor_at_resonance_constant"


def test_19e_numeric_returns_one() -> None:
    f = default_library()["power_factor_at_resonance_constant"]
    assert f.compute({"R1": 70, "R2": 40}) == 1.0


def test_19e_does_not_steal_non_resonance_power_factor() -> None:
    """A normal power-factor question (without LCω² = 1) must still
    route to power_factor_from_r_z."""
    normal_q = (
        "An RLC series circuit has R = 50 Ω and total impedance Z = 100 Ω. "
        "Calculate the power factor cosφ."
    )
    routed = _route(normal_q)
    assert routed != "power_factor_at_resonance_constant"


# ---------------------------------------------------------------------------
# 19f — NL334 "what is its inductance"
# ---------------------------------------------------------------------------


def test_19f_routes_nl334_to_inductance_formula() -> None:
    q = (
        "An inductor has a magnetic energy of 0.2 J when the current is "
        "2 A. What is its inductance (H)?"
    )
    assert _route(q) == "inductance_from_inductor_energy"


# ---------------------------------------------------------------------------
# 19g — √-literal cleaner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "needles"),
    [
        ("100√2 V", ["141.421"]),
        ("voltage 200√3 V", ["346.41"]),
        ("U = 50√2 V", ["70.71"]),
        ("E = √3 N/C", ["1.73205"]),
        ("100/√3 A", ["57.735"]),
    ],
)
def test_19g_sqrt_literal_cleaner(raw: str, needles: list[str]) -> None:
    out = clean(raw)
    for needle in needles:
        assert needle in out, f"{needle!r} not in {out!r}"


def test_19g_does_not_break_plain_text() -> None:
    """Cleaner must be idempotent on text without √."""
    plain = "A capacitor has C = 10 μF and U = 100 V."
    assert clean(plain) == "A capacitor has C = 10 μF and U = 100 V."
