"""End-to-end physics solver.

Pipeline:

    question text
        │
        ▼
    quantity_extractor  ──▶  list[ExtractedQuantity]
        │
        ▼
    topic_classifier  ──▶  ClassificationResult (formula id)
        │
        ▼
    formula_library  ──▶  Formula (with required SI units)
        │
        ▼
    unit_converter   ──▶  values_si: dict[str, float]
        │
        ▼
    formula.compute  ──▶  numeric answer (SI)
        │
        ▼
    SolverResult     ──▶  trace + final number/unit + confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exact_agent.physics.formula_library import (
    Formula,
    FormulaLibrary,
    default_library,
)
from exact_agent.physics.quantity_extractor import (
    ExtractedQuantity,
    extract_quantities,
    find_by_name,
)
from exact_agent.physics.question_cleaner import clean as clean_question
from exact_agent.physics.topic_classifier import classify
from exact_agent.physics.unit_converter import (
    UnitConversionError,
    convert,
)
from exact_agent.physics.verifier import verify as verify_result


@dataclass(frozen=True)
class SolverResult:
    success: bool
    answer_value: float | None
    answer_unit: str
    formula_id: str | None
    formula_description: str
    trace: list[str] = field(default_factory=list)
    extracted: list[ExtractedQuantity] = field(default_factory=list)
    fail_reason: str | None = None
    confidence: float = 0.0
    verifier_warnings: tuple[str, ...] = ()

    @property
    def answer_str(self) -> str:
        """Compact numeric representation, mirroring competition expectations."""
        if self.answer_value is None:
            return ""
        v = self.answer_value
        if v == 0:
            return "0"
        abs_v = abs(v)
        if abs_v >= 1e4 or abs_v < 1e-3:
            return f"{v:.4g}"
        return f"{v:.6f}".rstrip("0").rstrip(".") or "0"


class PhysicsSolver:
    """Orchestrate classifier → extractor → unit conversion → SymPy compute."""

    def __init__(self, library: FormulaLibrary | None = None) -> None:
        self._library = library or default_library()

    def solve(self, question: str) -> SolverResult:
        trace: list[str] = []
        question = clean_question(question)
        extracted = extract_quantities(question)
        if extracted:
            quantities_text = ", ".join(
                f"{q.name} = {q.raw_value} {q.raw_unit}".rstrip() for q in extracted
            )
            trace.append(f"Extracted: {quantities_text}")
        else:
            trace.append("No numeric quantities found in the question.")

        classification = classify(question, extracted, self._library)
        if classification is None:
            return SolverResult(
                success=False,
                answer_value=None,
                answer_unit="",
                formula_id=None,
                formula_description="",
                trace=trace,
                extracted=extracted,
                fail_reason="no_formula_matched",
            )

        formula = self._library[classification.formula_id]
        trace.append(
            f"Select formula '{formula.id}': {formula.description} "
            f"(reason: {classification.reason})"
        )

        try:
            values_si = self._convert_inputs(formula, extracted, trace)
        except UnitConversionError as exc:
            return SolverResult(
                success=False,
                answer_value=None,
                answer_unit=formula.output_unit,
                formula_id=formula.id,
                formula_description=formula.description,
                trace=trace,
                extracted=extracted,
                fail_reason=f"unit_conversion_failed: {exc}",
            )
        except KeyError as exc:
            return SolverResult(
                success=False,
                answer_value=None,
                answer_unit=formula.output_unit,
                formula_id=formula.id,
                formula_description=formula.description,
                trace=trace,
                extracted=extracted,
                fail_reason=f"missing_input: {exc}",
            )

        try:
            numeric = formula.compute(values_si)
        except (ValueError, ZeroDivisionError) as exc:
            return SolverResult(
                success=False,
                answer_value=None,
                answer_unit=formula.output_unit,
                formula_id=formula.id,
                formula_description=formula.description,
                trace=trace,
                extracted=extracted,
                fail_reason=f"compute_failed: {exc}",
            )

        trace.append(self._format_substitution(formula, values_si, numeric))

        provisional = SolverResult(
            success=True,
            answer_value=numeric,
            answer_unit=formula.output_unit,
            formula_id=formula.id,
            formula_description=formula.description,
            trace=trace,
            extracted=extracted,
            confidence=min(0.95, 0.6 + 0.35 * classification.confidence),
        )

        verification = verify_result(provisional.answer_value, provisional.answer_unit)
        if verification.warnings:
            trace.extend(f"Verifier: {w}" for w in verification.warnings)
        return SolverResult(
            success=provisional.success,
            answer_value=provisional.answer_value,
            answer_unit=provisional.answer_unit,
            formula_id=provisional.formula_id,
            formula_description=provisional.formula_description,
            trace=trace,
            extracted=provisional.extracted,
            confidence=(
                provisional.confidence if verification.ok else min(0.4, provisional.confidence)
            ),
            verifier_warnings=verification.warnings,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_alias(
        formula: Formula,
        extracted: list[ExtractedQuantity],
        symbol: str,
    ) -> ExtractedQuantity | None:
        """Match by exact name first, then by formula alias names."""
        direct = find_by_name(extracted, symbol)
        if direct is not None:
            return direct
        spec = formula.inputs[symbol]
        for alias in spec.aliases:
            hit = find_by_name(extracted, alias)
            if hit is not None:
                return hit
        return None

    def _convert_inputs(
        self,
        formula: Formula,
        extracted: list[ExtractedQuantity],
        trace: list[str],
    ) -> dict[str, float]:
        values_si: dict[str, float] = {}
        for symbol, spec in formula.inputs.items():
            quantity = self._resolve_alias(formula, extracted, symbol)
            if quantity is None:
                raise KeyError(symbol)
            if not quantity.unit:
                # No unit text — assume the value is already in the SI target.
                values_si[symbol] = quantity.value
                trace.append(
                    f"Assume {symbol} = {quantity.value} {spec.unit} (no unit string in question)"
                )
                continue
            conversion = convert(quantity.value, quantity.unit, spec.unit)
            values_si[symbol] = conversion.value_si
            trace.append(conversion.trace)
        return values_si

    @staticmethod
    def _format_substitution(formula: Formula, values_si: dict[str, float], result: float) -> str:
        subs = ", ".join(f"{k} = {v}" for k, v in values_si.items())
        return (
            f"Apply {formula.expression_text} with {subs}; result = {result} {formula.output_unit}"
        )
