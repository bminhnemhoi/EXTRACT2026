"""Record schemas for the interpretability harness (pure stdlib — no heavy deps).

These dataclasses are the on-disk contract between P1 (capture) and P2
(alignment). They use only the standard library so the analysis side runs
on any Python without torch/transformers installed.

A "concept" is a short string token (e.g. ``"formula:capacitor_energy"``,
``"quantity:C"``, ``"topic:TD"``, ``"premise:3"``, ``"pred:eligible"``).
Both the solver ground truth and the SAE feature labels are reduced to sets
of such tokens so RQ1 alignment is a set-overlap computation (see
:mod:`.alignment`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TextIO

TaskType = Literal["logic", "physics"]


@dataclass(frozen=True)
class GroundTruthConcepts:
    """Concepts the *symbolic solver* confirms were used for one sample."""

    sample_id: str
    task_type: TaskType
    concepts: list[str]
    solver_ok: bool
    source: str = ""  # e.g. "formula_id+inputs" / "z3_supports+predicates"

    def concept_set(self) -> set[str]:
        return set(self.concepts)


@dataclass(frozen=True)
class ActivationRecord:
    """One captured forward pass: SAE-active features + solver ground truth.

    ``active_feature_ids`` are the top-k SAE latent indices that fired
    (aggregated across the relevant tokens — see capture script). They are
    mapped to ``active_concepts`` via the autointerp label cache so RQ1 can
    compare against ``gold_concepts`` without re-touching the model.
    """

    sample_id: str
    task_type: TaskType
    question: str
    layer: int
    active_feature_ids: list[int]
    active_concepts: list[str]
    gold_concepts: list[str]
    solver_ok: bool
    solver_meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> ActivationRecord:
        data = json.loads(line)
        return cls(
            sample_id=str(data["sample_id"]),
            task_type=data["task_type"],
            question=str(data.get("question", "")),
            layer=int(data["layer"]),
            active_feature_ids=[int(x) for x in data.get("active_feature_ids", [])],
            active_concepts=[str(x) for x in data.get("active_concepts", [])],
            gold_concepts=[str(x) for x in data.get("gold_concepts", [])],
            solver_ok=bool(data.get("solver_ok", False)),
            solver_meta=dict(data.get("solver_meta", {})),
        )


def write_records(records: list[ActivationRecord], fp: TextIO) -> int:
    """Write records as JSONL. Returns the number written."""
    n = 0
    for rec in records:
        fp.write(rec.to_json())
        fp.write("\n")
        n += 1
    return n


def read_records(fp: TextIO) -> list[ActivationRecord]:
    """Read JSONL records, skipping blank lines."""
    out: list[ActivationRecord] = []
    for line in fp:
        line = line.strip()
        if not line:
            continue
        out.append(ActivationRecord.from_json(line))
    return out
