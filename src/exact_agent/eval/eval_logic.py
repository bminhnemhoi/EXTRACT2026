"""Run the logic pipeline against the eval split and report metrics.

Mirrors :mod:`exact_agent.eval.eval_physics` so the CLI stays uniform.
The dataset rows come from ``logic_train_safe.jsonl`` via the holdout
script; each row carries ``premises_NL``, ``question``, ``answer_clean``
(label), and ``question_type`` (mc / yes_no_unknown / true_false / open).
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from exact_agent.eval.metrics import label_match, token_overlap_f1
from exact_agent.logic.pipeline import LogicPipeline
from exact_agent.schemas import PredictRequest


@dataclass
class LogicSampleEval:
    sample_id: str
    question_type: str
    question: str
    expected_answer: str
    predicted_answer: str
    premises_used: tuple[str, ...]
    confidence: float
    elapsed_ms: float
    correct: bool
    abstained: bool
    explanation_f1: float


@dataclass
class LogicAggregate:
    n: int = 0
    n_correct: int = 0
    n_abstained: int = 0
    explanation_f1_sum: float = 0.0
    total_elapsed_ms: float = 0.0

    def add(self, sample: LogicSampleEval) -> None:
        self.n += 1
        self.n_correct += int(sample.correct)
        self.n_abstained += int(sample.abstained)
        self.explanation_f1_sum += sample.explanation_f1
        self.total_elapsed_ms += sample.elapsed_ms

    def as_dict(self) -> dict[str, float | int]:
        n = max(self.n, 1)
        return {
            "n": self.n,
            "correct": self.n_correct,
            "correct_pct": round(100 * self.n_correct / n, 1),
            "abstained": self.n_abstained,
            "abstained_pct": round(100 * self.n_abstained / n, 1),
            "explanation_f1_mean": round(self.explanation_f1_sum / n, 3),
            "mean_elapsed_ms": round(self.total_elapsed_ms / n, 2),
        }


@dataclass
class LogicEvalReport:
    split_path: str
    overall: LogicAggregate
    by_question_type: dict[str, LogicAggregate]
    fail_reasons: Counter
    samples: list[LogicSampleEval] = field(default_factory=list)

    def to_dict(self, *, include_samples: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "split_path": self.split_path,
            "overall": self.overall.as_dict(),
            "by_question_type": {k: v.as_dict() for k, v in sorted(self.by_question_type.items())},
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


def _build_payload(row: dict) -> PredictRequest:
    return PredictRequest.model_validate(
        {
            "premises-NL": list(row.get("premises_NL") or []),
            "question": str(row.get("question") or ""),
            "task_type": "logic",
        }
    )


def _categorize_failure(predicted: str, expected: str) -> str:
    if not predicted:
        return "abstained"
    if predicted.lower() in {"unknown", "uncertain"} and expected.lower() not in {
        "unknown",
        "uncertain",
    }:
        return "answered_unknown"
    return "wrong_label"


def _score_sample(row: dict, pipeline: LogicPipeline) -> LogicSampleEval:
    payload = _build_payload(row)
    expected = str(row.get("answer_clean") or row.get("answer_raw") or "").strip()
    expected_explanation = str(row.get("explanation") or "")

    start = time.perf_counter()
    response = pipeline.run(payload)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    predicted = response.answer or ""
    correct = label_match(predicted, expected)
    abstained = predicted == "" or predicted.lower() == "unknown"

    explanation_f1 = token_overlap_f1(response.explanation, expected_explanation).f1

    return LogicSampleEval(
        sample_id=str(row.get("sample_id") or row.get("id") or ""),
        question_type=str(row.get("question_type") or "unknown"),
        question=str(row.get("question") or ""),
        expected_answer=expected,
        predicted_answer=predicted,
        premises_used=tuple(response.premises or ()),
        confidence=float(response.confidence or 0.0),
        elapsed_ms=round(elapsed_ms, 2),
        correct=bool(correct),
        abstained=bool(abstained),
        explanation_f1=explanation_f1,
    )


def run_eval(split_path: Path, pipeline: LogicPipeline | None = None) -> LogicEvalReport:
    """Run the logic pipeline on every row in ``split_path`` and aggregate."""
    pipeline = pipeline or LogicPipeline()
    overall = LogicAggregate()
    by_qtype: dict[str, LogicAggregate] = defaultdict(LogicAggregate)
    fail_reasons: Counter = Counter()
    samples: list[LogicSampleEval] = []

    for row in _read_jsonl(split_path):
        sample = _score_sample(row, pipeline)
        samples.append(sample)
        overall.add(sample)
        by_qtype[sample.question_type].add(sample)
        if not sample.correct:
            fail_reasons[_categorize_failure(sample.predicted_answer, sample.expected_answer)] += 1

    return LogicEvalReport(
        split_path=str(split_path),
        overall=overall,
        by_question_type=dict(by_qtype),
        fail_reasons=fail_reasons,
        samples=samples,
    )
