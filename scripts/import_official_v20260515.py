"""Import the official EXACT 2026 dataset (release 2026-05-15).

Replaces ``cleaned_exact_dataset_package`` (which we processed Day 1 from
the 2026-05-09 snapshot) with the organizer's authoritative cleaned drop.
The official 2026-05-15 release fixes 37 FOL syntax bugs, 8 UNSAT
records, 103 mislabelled MCQs, the QA-prefix annotation pipe leak
(401 unusable physics rows dropped) and several physics rounding/Vietnamese
unit bugs. Per CHANGELOG_TYPE{1,2}.md.

Outputs (under ``data/official_v20260515/``):

* ``physics_safe.jsonl``  — 1,352 rows  (1,755 raw - 401 QA-prefix - 2 empty cot)
* ``logic_safe.jsonl``    —   808 rows  (1 row per question, flattened from 411 records)
* ``eval_split/{physics,logic}_eval.jsonl``   — 10% holdout (seed 42)
* ``train/{physics,logic}_train.jsonl``       — 90% remainder

We also (E11) drop ``LD348, LD350`` from physics: they are in our prior
cleaned set but the organizer dropped them from the official cleanup,
so they are not legitimate eval rows.

Usage::

    uv run python scripts/import_official_v20260515.py \\
        --source "D:/Exact2026/tailieu_moi/Datasets/unpacked_0515"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "official_v20260515"

# E11: rows present in our prior cleaned set that the organizer's official
# 2026-05-15 cleanup *excludes*. We mirror their exclusions for parity.
DROP_PHYSICS_IDS: frozenset[str] = frozenset({"LD348", "LD350"})

_ALPHA_PREFIX_RE = re.compile(r"^([A-Za-z]+)")


def _alpha_prefix(row_id: str) -> str:
    m = _ALPHA_PREFIX_RE.match(row_id or "")
    return m.group(1) if m else ""


def _classify_logic_qtype(answer: object) -> str:
    """Map answer string to the eval module's question_type bucket.

    The eval CLI groups by this field, so the bucket names must match
    those already in use (``mc``, ``yes_no_unknown``, ``open``).
    """
    if not isinstance(answer, str):
        return "open"
    a = answer.strip()
    if a in {"Yes", "No", "Unknown"}:
        return "yes_no_unknown"
    if a in {"A", "B", "C", "D"}:
        return "mc"
    return "open"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def import_physics(source_csv: Path) -> list[dict]:
    rows: list[dict] = []
    n_dropped = 0
    with source_csv.open("r", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rid = (raw.get("id") or "").strip()
            if rid in DROP_PHYSICS_IDS:
                n_dropped += 1
                continue
            question = (raw.get("question") or "").strip()
            cot = (raw.get("cot") or "").strip()
            answer = (raw.get("answer") or "").strip()
            unit = (raw.get("unit") or "").strip()
            rows.append({
                "id": rid,
                "prefix": _alpha_prefix(rid),
                "task_type": "physics",
                "question_raw": question,
                "question_clean": question,
                "cot_raw": cot,
                "answer_raw": answer,
                "answer_clean": answer,
                "unit_raw": unit,
                "unit_clean": unit,
                "explanation_clean": cot,
                "source": "official_v20260515",
            })
    print(f"[physics] imported {len(rows)} rows (dropped {n_dropped} per E11)")
    return rows


def import_logic(source_json: Path) -> list[dict]:
    """Flatten 411 records into one row per question.

    Carries the official ``idx`` field per question as ``used_premise_idx``
    — this is the ground-truth premise reference the P3 rubric rewards
    citing (CHANGELOG_TYPE1: ``idx`` is 1-based within ``[1, n]``).
    """
    with source_json.open("r", encoding="utf-8") as f:
        records = json.load(f)
    out: list[dict] = []
    for rec_idx, record in enumerate(records):
        questions = record.get("questions") or []
        answers = record.get("answers") or []
        # Official `idx` is parallel to questions: idx[i] = list of premise
        # indices used for questions[i]. May be missing for some questions
        # (CHANGELOG dropped invalid indices) — fall back to None.
        raw_idx = record.get("idx")
        per_q_idx: list[object] = (
            raw_idx if isinstance(raw_idx, list) and len(raw_idx) == len(questions)
            else [None] * len(questions)
        )
        premises_NL = list(record.get("premises-NL") or [])
        premises_FOL = list(record.get("premises-FOL") or [])
        explanation = record.get("explanation") or ""
        for q_idx, (question, answer) in enumerate(zip(questions, answers, strict=False)):
            out.append({
                "sample_id": f"{rec_idx}_q{q_idx}",
                "record_id": rec_idx,
                "question_index": q_idx,
                "task_type": "logic",
                "question_type": _classify_logic_qtype(answer),
                "premises_NL": premises_NL,
                "premises_FOL_raw": premises_FOL,
                "premises_FOL_clean": premises_FOL,
                "question": str(question).strip(),
                "choices": None,
                "answer_raw": str(answer).strip(),
                "answer_clean": str(answer).strip(),
                "explanation": str(explanation),
                "used_premise_idx": per_q_idx[q_idx],
                "source": "official_v20260515",
            })
    print(f"[logic] flattened {len(records)} records -> {len(out)} questions")
    return out


def split_and_write(name: str, rows: list[dict], seed: int, ratio: float) -> tuple[int, int]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_eval = max(1, round(len(rows) * ratio))
    eval_set = set(idx[:n_eval])
    eval_rows = [rows[i] for i in range(len(rows)) if i in eval_set]
    train_rows = [rows[i] for i in range(len(rows)) if i not in eval_set]
    _write_jsonl(OUT_DIR / f"{name}_safe.jsonl", rows)
    _write_jsonl(OUT_DIR / "eval_split" / f"{name}_eval.jsonl", eval_rows)
    _write_jsonl(OUT_DIR / "train" / f"{name}_train.jsonl", train_rows)
    return len(eval_rows), len(train_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("D:/Exact2026/tailieu_moi/Datasets/unpacked_0515"),
        help="Unpacked 2026-05-15 release directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.1)
    args = parser.parse_args()

    src: Path = args.source
    physics_csv = src / "Physics_Problems_Text_Only" / "Physics_Problems_Text_Only.csv"
    logic_dir = src / "Logic_Based_Educational_Queries_Text_Only"
    logic_json = logic_dir / "Logic_Based_Educational_Queries.json"
    for required in (physics_csv, logic_json):
        if not required.exists():
            print(f"[fatal] not found: {required}", file=sys.stderr)
            return 1

    phys_rows = import_physics(physics_csv)
    logic_rows = import_logic(logic_json)

    pe, pt = split_and_write("physics", phys_rows, args.seed, args.ratio)
    le, lt = split_and_write("logic", logic_rows, args.seed, args.ratio)
    print(f"[physics] eval={pe} train={pt}  total={pe + pt}")
    print(f"[logic]   eval={le} train={lt}  total={le + lt}")
    print(f"[ok] wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
