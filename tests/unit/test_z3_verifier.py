"""Tests for the Z3 entailment backend."""

from __future__ import annotations

from exact_agent.logic.z3_verifier import verify_with_z3


class TestZ3Verifier:
    def test_modus_ponens_entails(self) -> None:
        premises = [
            "∀x (Student(x) ∧ PassedExam(x) → ReceivesCredit(x))",
            "Student(Alice)",
            "PassedExam(Alice)",
        ]
        result = verify_with_z3(premises, "ReceivesCredit(Alice)")
        assert result.verdict == "Yes"
        # All three premises participated in the assertion stack.
        assert set(result.supports) == {1, 2, 3}

    def test_contradicted_claim_yields_no(self) -> None:
        premises = [
            "∀x (Student(x) ∧ PassedExam(x) → ReceivesCredit(x))",
            "Student(Alice)",
            "PassedExam(Alice)",
            "¬ReceivesCredit(Bob)",
        ]
        result = verify_with_z3(premises, "ReceivesCredit(Bob)")
        # Bob: explicit ¬ReceivesCredit → claim is False.
        assert result.verdict == "No"

    def test_unrelated_claim_unknown(self) -> None:
        premises = ["Student(Alice)"]
        result = verify_with_z3(premises, "PassedExam(Alice)")
        assert result.verdict == "Unknown"

    def test_chain_of_two_rules(self) -> None:
        premises = [
            "∀x (Student(x) → Eligible(x))",
            "∀x (Eligible(x) → CanRegister(x))",
            "Student(Alice)",
        ]
        result = verify_with_z3(premises, "CanRegister(Alice)")
        assert result.verdict == "Yes"

    def test_unparseable_premises_skipped(self) -> None:
        premises = [
            "∀x (Student(x) → Eligible(x))",
            "this is not FOL at all",
            "Student(Alice)",
        ]
        result = verify_with_z3(premises, "Eligible(Alice)")
        assert result.verdict == "Yes"
        assert result.skipped_premises == 1

    def test_unparseable_claim_returns_unknown(self) -> None:
        result = verify_with_z3(["Student(Alice)"], "this is not FOL")
        assert result.verdict == "Unknown"
        assert "did not parse" in result.rationale


class TestMultiVarAndComparison:
    def test_multivar_transitivity(self) -> None:
        premises = [
            "∀a (ForAll(b, ForAll(c, (higher(a, b) ∧ higher(b, c)) → higher(a, c))))",
            "higher(PhD, MSc)",
            "higher(MSc, BA)",
        ]
        assert verify_with_z3(premises, "higher(PhD, BA)").verdict == "Yes"

    def test_comparison_threshold_entails(self) -> None:
        premises = [
            "membership_duration(Alex) = 8",
            "∀x ((membership_duration(x) ≥ 6) → eligible_trainer(x))",
        ]
        assert verify_with_z3(premises, "eligible_trainer(Alex)").verdict == "Yes"

    def test_below_threshold_not_entailed(self) -> None:
        premises = [
            "membership_duration(Bob) = 4",
            "∀x ((membership_duration(x) ≥ 6) → eligible_trainer(x))",
        ]
        # No rule forces eligibility at dur=4 → must not over-conclude Yes.
        assert verify_with_z3(premises, "eligible_trainer(Bob)").verdict != "Yes"

    def test_single_var_still_works(self) -> None:
        # Regression: the pre-existing single-var path is unaffected.
        premises = [
            "∀x (Student(x) ∧ PassedExam(x) → ReceivesCredit(x))",
            "Student(Alice)",
            "PassedExam(Alice)",
        ]
        assert verify_with_z3(premises, "ReceivesCredit(Alice)").verdict == "Yes"
