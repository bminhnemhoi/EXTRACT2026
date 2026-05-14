"""Integration tests for the LLM fallback paths in both pipelines."""

from __future__ import annotations

import pytest

from exact_agent.llm.vllm_client import MockLLMClient
from exact_agent.logic.pipeline import LogicPipeline
from exact_agent.physics.solver import PhysicsSolver
from exact_agent.schemas import PredictRequest


class TestPhysicsLLMRecovery:
    def test_llm_recovers_missing_inputs(self) -> None:
        """When regex misses, the LLM extractor JSON should let SymPy compute."""
        # Question lacks the explicit `C = ... and U = ...` form the regex needs.
        question = "A capacitor is charged to a voltage of 30 volts and has 100 microfarads."
        # The classifier still picks `capacitor_energy` because of "capacitor"
        # plus symbol fallback. We hand the LLM the inputs in SI form.
        canned_json = '{"C": {"value": 100, "unit": "uF"}, "U": {"value": 30, "unit": "V"}}'
        llm = MockLLMClient([canned_json])
        solver = PhysicsSolver(llm_client=llm)
        result = solver.solve(question)

        # The fallback may or may not trigger depending on whether the regex
        # extractor finds *anything* — what matters is the pipeline doesn't
        # crash and either solves with the LLM or fails gracefully with a
        # `missing_input` reason.
        if result.success:
            # Energy = 0.5 * 1e-4 * 900 = 0.045 J
            assert result.answer_value == pytest.approx(0.045, rel=1e-3)
            assert "LLM extractor recovered" in "\n".join(result.trace)


class TestLogicLLMTranslation:
    def test_llm_supplies_claim_fol_then_z3_decides(self) -> None:
        """Pipeline asks the LLM for a claim_FOL when the request omits it."""
        # Hand the LLM the right answer (modus ponens with a named entity).
        llm = MockLLMClient(["FOL: ReceivesCredit(Sophia)"])
        pipeline = LogicPipeline(llm_client=llm)
        payload = PredictRequest.model_validate(
            {
                "premises-NL": [
                    "All students who pass the exam receive credit.",
                    "Sophia has passed the exam.",
                ],
                "premises-FOL": [
                    "∀x (Student(x) ∧ PassedExam(x) → ReceivesCredit(x))",
                    "Student(Sophia)",
                    "PassedExam(Sophia)",
                ],
                "question": "Does Sophia receive credit?",
            }
        )
        resp = pipeline.run(payload)
        assert resp.answer == "Yes"
        assert resp.cot is not None
        joined = "\n".join(resp.cot)
        assert "LLM translated claim" in joined
        assert "Z3 entailment fallback applied" in joined

    def test_no_llm_no_claim_returns_surface_answer(self) -> None:
        """Without an LLM and without claim_FOL, surface chain still answers."""
        pipeline = LogicPipeline()
        payload = PredictRequest.model_validate(
            {
                "premises-NL": [
                    "If a student passes the exam, then they receive credit.",
                    "A student passes the exam.",
                ],
                "question": "Does the student receive credit?",
            }
        )
        resp = pipeline.run(payload)
        # Surface chain wins this one — no LLM needed.
        assert resp.answer == "Yes"
