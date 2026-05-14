"""Tests for the rule-based topic classifier."""

from __future__ import annotations

from exact_agent.physics.formula_library import default_library
from exact_agent.physics.quantity_extractor import ExtractedQuantity, extract_quantities
from exact_agent.physics.topic_classifier import classify


class TestKeywordRouting:
    def test_capacitor_energy_keyword(self) -> None:
        text = "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
        result = classify(text, extract_quantities(text), default_library())
        assert result is not None
        assert result.formula_id == "capacitor_energy"
        assert result.confidence >= 0.8

    def test_coulomb_keyword(self) -> None:
        text = (
            "Use Coulomb's law to find the force between q1 = 6e-8 C and q2 = -6e-8 C "
            "separated by r = 0.05 m."
        )
        result = classify(text, extract_quantities(text), default_library())
        assert result is not None
        assert result.formula_id == "coulomb_force"

    def test_parallel_resistance_keyword(self) -> None:
        text = "Two resistors R1 = 4 ohm and R2 = 6 ohm in parallel. What is R?"
        result = classify(text, extract_quantities(text), default_library())
        assert result is not None
        assert result.formula_id == "parallel_resistance_two"


class TestSymbolFallback:
    def test_no_keyword_but_unique_symbols_picks_formula(self) -> None:
        text = "Given L = 2 H and C = 0.5 F, compute the natural frequency."
        # No keyword; only {L, C} symbol set matches resonance_frequency.
        quantities = extract_quantities(text)
        result = classify(text, quantities, default_library())
        assert result is not None
        assert result.formula_id == "resonance_frequency"

    def test_returns_none_when_undecidable(self) -> None:
        # Empty quantities + no keyword → no match.
        result = classify("Describe Newton's first law.", [], default_library())
        assert result is None


class TestEdgeCases:
    def test_partial_symbol_set_rejected(self) -> None:
        # Only U present; capacitor_energy needs both C and U.
        quantities = [
            ExtractedQuantity(name="U", value=30.0, unit="V", raw_value="30", raw_unit="V")
        ]
        result = classify("Some question with U = 30 V.", quantities, default_library())
        # Partial-only matches must NOT route — would silently solve with junk C.
        assert result is None
