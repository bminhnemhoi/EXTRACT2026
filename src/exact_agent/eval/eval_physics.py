"""Run the physics solver against the eval split and report metrics.

This is the ground-truth measurement we use to make Phase-1 trade-offs.
The harness is deliberately self-contained: no LLM, no network, no
randomness — same input -> same numbers across reruns.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from exact_agent.eval.metrics import (
    parse_number,
    quantity_match,
    token_overlap_f1,
    unit_match,
)
from exact_agent.physics.question_cleaner import clean
from exact_agent.physics.solver import PhysicsSolver, SolverResult


@dataclass
class SampleEval:
    sample_id: str
    prefix: str
    question: str
    expected_answer: str
    expected_unit: str
    expected_value: float | None

    predicted_answer: str
    predicted_unit: str
    predicted_value: float | None
    formula_id: str | None
    confidence: float
    elapsed_ms: float

    solved: bool
    numeric_correct: bool
    unit_correct: bool
    full_correct: bool
    explanation_f1: float
    fail_reason: str | None
    verifier_warnings: tuple[str, ...]


@dataclass
class AggregateMetrics:
    n: int = 0
    n_solved: int = 0
    n_numeric_correct: int = 0
    n_unit_correct: int = 0
    n_full_correct: int = 0
    explanation_f1_sum: float = 0.0
    total_elapsed_ms: float = 0.0

    def add(self, sample: SampleEval) -> None:
        self.n += 1
        self.n_solved += int(sample.solved)
        self.n_numeric_correct += int(sample.numeric_correct)
        self.n_unit_correct += int(sample.unit_correct)
        self.n_full_correct += int(sample.full_correct)
        self.explanation_f1_sum += sample.explanation_f1
        self.total_elapsed_ms += sample.elapsed_ms

    def as_dict(self) -> dict[str, float | int]:
        n = max(self.n, 1)
        return {
            "n": self.n,
            "solved": self.n_solved,
            "solved_pct": round(100 * self.n_solved / n, 1),
            "numeric_correct": self.n_numeric_correct,
            "numeric_correct_pct": round(100 * self.n_numeric_correct / n, 1),
            "unit_correct": self.n_unit_correct,
            "unit_correct_pct": round(100 * self.n_unit_correct / n, 1),
            "full_correct": self.n_full_correct,
            "full_correct_pct": round(100 * self.n_full_correct / n, 1),
            "explanation_f1_mean": round(self.explanation_f1_sum / n, 3),
            "mean_elapsed_ms": round(self.total_elapsed_ms / n, 2),
        }


@dataclass
class EvalReport:
    split_path: str
    overall: AggregateMetrics
    by_prefix: dict[str, AggregateMetrics]
    by_formula: dict[str, AggregateMetrics]
    fail_reasons: Counter
    samples: list[SampleEval] = field(default_factory=list)

    def to_dict(self, *, include_samples: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "split_path": self.split_path,
            "overall": self.overall.as_dict(),
            "by_prefix": {k: v.as_dict() for k, v in sorted(self.by_prefix.items())},
            "by_formula": {k: v.as_dict() for k, v in sorted(self.by_formula.items())},
            "fail_reasons": dict(self.fail_reasons.most_common()),
        }
        if include_samples:
            payload["samples"] = [asdict(s) for s in self.samples]
        return payload


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                yield json.loads(line)


def _score_sample(row: dict, solver: PhysicsSolver) -> SampleEval:
    question = clean(row.get("question_clean") or row.get("question_raw") or "")
    expected_answer = str(row.get("answer_clean") or row.get("answer_raw") or "").strip()
    expected_unit = str(row.get("unit_clean") or row.get("unit_raw") or "").strip()
    expected_explanation = str(row.get("explanation_clean") or row.get("cot_raw") or "")

    start = time.perf_counter()
    result: SolverResult = solver.solve(question)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    # The solver already invokes the verifier; we just surface its warnings.
    warnings = result.verifier_warnings

    predicted_unit = result.answer_unit
    predicted_value = result.answer_value
    predicted_answer = result.answer_str

    numeric_correct = quantity_match(
        predicted_value,
        predicted_unit,
        expected_answer,
        expected_unit,
        rel_tol=0.01,
    )
    unit_correct = unit_match(predicted_unit, expected_unit)
    full_correct = numeric_correct and unit_correct

    explanation = "\n".join(result.trace)
    explanation_f1 = token_overlap_f1(explanation, expected_explanation).f1

    return SampleEval(
        sample_id=str(row.get("id") or row.get("sample_id") or ""),
        prefix=str(row.get("prefix") or ""),
        question=question,
        expected_answer=expected_answer,
        expected_unit=expected_unit,
        expected_value=parse_number(expected_answer),
        predicted_answer=predicted_answer,
        predicted_unit=predicted_unit,
        predicted_value=predicted_value,
        formula_id=result.formula_id,
        confidence=result.confidence,
        elapsed_ms=round(elapsed_ms, 2),
        solved=result.success,
        numeric_correct=bool(numeric_correct),
        unit_correct=bool(unit_correct),
        full_correct=bool(full_correct),
        explanation_f1=explanation_f1,
        fail_reason=result.fail_reason,
        verifier_warnings=warnings,
    )


def run_eval(split_path: Path, solver: PhysicsSolver | None = None) -> EvalReport:
    """Run the solver on every row in ``split_path`` and aggregate."""
    solver = solver or PhysicsSolver()
    overall = AggregateMetrics()
    by_prefix: dict[str, AggregateMetrics] = defaultdict(AggregateMetrics)
    by_formula: dict[str, AggregateMetrics] = defaultdict(AggregateMetrics)
    fail_reasons: Counter = Counter()
    samples: list[SampleEval] = []

    for row in _read_jsonl(split_path):
        sample = _score_sample(row, solver)
        samples.append(sample)
        overall.add(sample)
        if sample.prefix:
            by_prefix[sample.prefix].add(sample)
        if sample.formula_id:
            by_formula[sample.formula_id].add(sample)
        if sample.fail_reason:
            fail_reasons[sample.fail_reason.split(":")[0]] += 1

    return EvalReport(
        split_path=str(split_path),
        overall=overall,
        by_prefix=dict(by_prefix),
        by_formula=dict(by_formula),
        fail_reasons=fail_reasons,
        samples=samples,
    )
