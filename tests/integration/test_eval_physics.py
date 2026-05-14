"""Integration test for the physics eval harness.

We don't drive the full 133-row dataset here (that runs in `make eval`);
instead we feed two hand-crafted rows that exercise the success and
failure branches of :func:`run_eval`.
"""

from __future__ import annotations

import json
from pathlib import Path

from exact_agent.eval.eval_physics import run_eval


def _fixture_jsonl(tmp_path: Path) -> Path:
    rows = [
        {
            "id": "TD-test",
            "prefix": "TD",
            "task_type": "physics",
            "question_clean": (
                "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
            ),
            "answer_clean": "0.045",
            "unit_clean": "J",
            "explanation_clean": "Apply E = 0.5 C U^2.",
        },
        {
            "id": "GIBBERISH-test",
            "prefix": "QA",
            "task_type": "physics",
            "question_clean": "What is the meaning of life?",
            "answer_clean": "42",
            "unit_clean": "",
            "explanation_clean": "",
        },
    ]
    path = tmp_path / "fixture.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class TestRunEval:
    def test_aggregate_counts(self, tmp_path: Path) -> None:
        path = _fixture_jsonl(tmp_path)
        report = run_eval(path)
        assert report.overall.n == 2
        assert report.overall.n_solved == 1  # only the capacitor row solves
        assert report.overall.n_numeric_correct == 1
        assert report.overall.n_full_correct == 1

    def test_fail_reason_recorded(self, tmp_path: Path) -> None:
        path = _fixture_jsonl(tmp_path)
        report = run_eval(path)
        assert "no_formula_matched" in report.fail_reasons

    def test_to_dict_shape(self, tmp_path: Path) -> None:
        path = _fixture_jsonl(tmp_path)
        report = run_eval(path)
        d = report.to_dict()
        assert "overall" in d
        assert d["overall"]["full_correct"] == 1
        assert "by_prefix" in d
        assert "TD" in d["by_prefix"]
