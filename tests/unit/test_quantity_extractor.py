"""Tests for the regex-based variable extractor."""

from __future__ import annotations

import math

import pytest

from exact_agent.physics.quantity_extractor import extract_quantities, find_by_name


class TestSimpleAssignments:
    def test_capacitor_energy_question(self) -> None:
        text = "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
        quantities = extract_quantities(text)
        names = {q.name for q in quantities}
        assert {"C", "U"} <= names

        c = find_by_name(quantities, "C")
        assert c is not None
        assert math.isclose(c.value, 100.0)
        assert "F" in c.unit

        u = find_by_name(quantities, "U")
        assert u is not None
        assert math.isclose(u.value, 30.0)
        assert u.unit == "V"

    def test_handles_unicode_times_and_caret_exponent(self) -> None:
        text = "Two charges q1 = 6 × 10^-8 C and q2 = -6 × 10^-8 C are placed in air."
        quantities = extract_quantities(text)
        names = [q.name for q in quantities]
        assert "q1" in names
        assert "q2" in names
        q1 = find_by_name(quantities, "q1")
        assert q1 is not None
        assert math.isclose(q1.value, 6e-8, rel_tol=1e-9)
        q2 = find_by_name(quantities, "q2")
        assert q2 is not None
        assert math.isclose(q2.value, -6e-8, rel_tol=1e-9)

    def test_handles_unicode_superscript_exponent(self) -> None:
        text = "Charge q1 = 6 × 10⁻⁸ C is moved."
        quantities = extract_quantities(text)
        q1 = find_by_name(quantities, "q1")
        assert q1 is not None
        assert math.isclose(q1.value, 6e-8, rel_tol=1e-9)

    def test_scientific_e_notation(self) -> None:
        text = "q1 = 6e-8 C"
        q = extract_quantities(text)[0]
        assert math.isclose(q.value, 6e-8)
        assert q.unit == "C"

    def test_milli_prefix_passed_through(self) -> None:
        text = "Q = 3 mC at terminal A."
        q = extract_quantities(text)[0]
        assert q.name == "Q"
        assert q.value == 3.0
        assert q.unit == "mC"

    def test_resistance_in_ohms_word(self) -> None:
        text = "A 5 ohm resistor is connected. R = 5 ohm, I = 2 A."
        quantities = extract_quantities(text)
        r = find_by_name(quantities, "R")
        i = find_by_name(quantities, "I")
        assert r is not None and r.value == 5.0
        assert i is not None and i.value == 2.0


class TestBlocklist:
    def test_drops_stopword_as_unit(self) -> None:
        text = "If x = 5 and y = 3, what is x+y?"
        quantities = extract_quantities(text)
        x = find_by_name(quantities, "x")
        assert x is not None
        # "and" was the greedy match for the unit; we expect it stripped.
        assert x.unit == ""


class TestNoMatch:
    @pytest.mark.parametrize(
        "text",
        [
            "How many moles of gas are needed?",
            "Describe the relationship between voltage and current.",
        ],
    )
    def test_returns_empty(self, text: str) -> None:
        assert extract_quantities(text) == []
