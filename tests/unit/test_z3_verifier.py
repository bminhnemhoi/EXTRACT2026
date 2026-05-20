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


class TestMixedArity:
    def test_same_name_different_arity_does_not_crash(self) -> None:
        # Regression: an upstream NL->FOL translator may emit the same
        # predicate name with two arities across premises (e.g. Has/1
        # ground predicate vs Has/2 binary relation). Pre-fix this raised
        # ``z3.z3types.Z3Exception: index out of bounds`` and killed the
        # whole eval run. Post-fix: P/1 and P/2 are distinct funcdecls,
        # the run completes, and the verdict is sound (Unknown here, no
        # entailment to be had between unrelated arities).
        premises = [
            "Has(Alice)",                    # arity 1
            "Has(Alice, BA)",                # arity 2
            "∀x (Has(x) → Eligible(x))",     # uses Has/1
        ]
        result = verify_with_z3(premises, "Eligible(Alice)")
        assert result.verdict == "Yes"      # via Has/1 path

    def test_mixed_arity_does_not_corrupt_unrelated_path(self) -> None:
        premises = [
            "Has(Alice, BA)",                # arity 2 only
            "∀x (∀y (Has(x, y) → Holds(x)))",
        ]
        assert verify_with_z3(premises, "Holds(Alice)").verdict == "Yes"
