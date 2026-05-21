"""Day-26 (MAX PUSH plan): convert the official 2026-05-15 training split
into the Qwen2.5 chat-format SFT dataset for Colab retrain.

Source:
  data/official_v20260515/train/physics_train.jsonl   (1,217 rows)
  data/official_v20260515/train/logic_train.jsonl     (727 rows)

Output:
  data/processed/sft_chat_train_v20260515.jsonl
  data/processed/sft_chat_val_v20260515.jsonl
  (10% val split, seed 42 — identical split logic to the Day-13 prep)

Why a separate script (not just rerunning ``src/exact_agent/train/
prepare_sft.py``): the Day-13 prep consumed
``sft_train_mixed_solver_clean.jsonl`` from the 2026-05-09 release.
That intermediate file does not exist for 2026-05-15; the official
release ships per-task JSONLs and we built the importer
``scripts/import_official_v20260515.py`` to produce the splits we use
everywhere else. This script bridges those splits straight into the
chat shape SFT trainers expect, mirroring the system prompt and
assistant envelope of the Day-13 dataset bit-for-bit so the SFTTrainer
hyperparameters and loss curves stay comparable.

Usage (local):

    uv run python scripts/prepare_sft_data_v20260515.py

Then on Colab: upload the two output files renamed to
``sft_chat_train.jsonl`` / ``sft_chat_val.jsonl`` (the
``sft_qwen_colab.ipynb`` notebook expects those names).
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSICS_TRAIN = REPO_ROOT / "data" / "official_v20260515" / "train" / "physics_train.jsonl"
LOGIC_TRAIN = REPO_ROOT / "data" / "official_v20260515" / "train" / "logic_train.jsonl"
OUT_DIR = REPO_ROOT / "data" / "processed"

# Bit-for-bit identical to src/exact_agent/train/prepare_sft.py — the Day-13
# SFT was trained with this prompt, so reusing it keeps the apples-to-apples
# A/B comparison clean.
SYSTEM_PROMPT = (
    "You are EXACT 2026 — an explainable educational QA agent. "
    "For every query, return ONLY a single JSON object with these keys: "
    "answer (string), explanation (string), cot (list of strings, optional), "
    "premises (list of premise IDs like 'P1', optional), "
    "confidence (number in [0, 1], optional). "
    "Numerical answers MUST come from the deterministic solver trace — "
    "never invent numbers. For logic tasks, cite the exact premise IDs "
    "you used in the 'premises' field."
)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                yield json.loads(line)


def _physics_to_chat(row: dict) -> dict | None:
    """Render one physics train row as a chat-format record."""
    question = (row.get("question_clean") or row.get("question_raw") or "").strip()
    answer = (row.get("answer_clean") or row.get("answer_raw") or "").strip()
    unit = (row.get("unit_clean") or row.get("unit_raw") or "").strip()
    cot = (row.get("explanation_clean") or row.get("cot_raw") or "").strip()
    if not question or not answer:
        return None
    # Combine answer + unit into a single string (matches the Day-13
    # assistant envelope format like "34.64 Ω").
    full_answer = f"{answer} {unit}".strip() if unit else answer
    prompt = {"task": "physics_qa", "question": question}
    completion = {"answer": full_answer, "explanation": cot}
    return {
        "sample_id": row.get("id") or row.get("sample_id") or "",
        "task_type": "physics",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, indent=2)},
            {"role": "assistant", "content": json.dumps(completion, ensure_ascii=False, indent=2)},
        ],
    }


def _logic_to_chat(row: dict) -> dict | None:
    """Render one logic train row as a chat-format record."""
    question = (row.get("question") or "").strip()
    answer = (row.get("answer_clean") or row.get("answer_raw") or "").strip()
    if not question or not answer:
        return None
    premises_nl = list(row.get("premises_NL") or [])
    explanation = (row.get("explanation") or "").strip()
    # Carry the gold premise indices into the assistant envelope as
    # "P1"/"P3"/... labels — the very evidence the P3 rubric rewards
    # the inference pipeline for citing.
    used = row.get("used_premise_idx")
    premise_labels: list[str] = []
    if isinstance(used, list):
        for i in used:
            if isinstance(i, int) and 1 <= i <= len(premises_nl):
                premise_labels.append(f"P{i}")

    prompt = {
        "task": "logic_qa",
        "premises-NL": premises_nl,
        "question": question,
    }
    completion: dict[str, object] = {"answer": answer, "explanation": explanation}
    if premise_labels:
        completion["premises"] = premise_labels
    completion["confidence"] = 0.9

    return {
        "sample_id": row.get("sample_id") or row.get("id") or "",
        "task_type": "logic",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, indent=2)},
            {"role": "assistant", "content": json.dumps(completion, ensure_ascii=False, indent=2)},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for path in (PHYSICS_TRAIN, LOGIC_TRAIN):
        if not path.exists():
            print(f"[fatal] missing {path} — run scripts/import_official_v20260515.py first")
            return 1

    rows: list[dict] = []
    skipped = {"physics": 0, "logic": 0}
    for row in _read_jsonl(PHYSICS_TRAIN):
        record = _physics_to_chat(row)
        (rows.append(record) if record else skipped.__setitem__(
            "physics", skipped["physics"] + 1
        ))
    for row in _read_jsonl(LOGIC_TRAIN):
        record = _logic_to_chat(row)
        (rows.append(record) if record else skipped.__setitem__(
            "logic", skipped["logic"] + 1
        ))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, round(len(rows) * args.val_ratio))
    val = rows[:n_val]
    train = rows[n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "sft_chat_train_v20260515.jsonl"
    val_path = OUT_DIR / "sft_chat_val_v20260515.jsonl"
    for path, batch in ((train_path, train), (val_path, val)):
        with path.open("w", encoding="utf-8") as f:
            for record in batch:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_phys = sum(1 for r in rows if r["task_type"] == "physics")
    n_logic = sum(1 for r in rows if r["task_type"] == "logic")
    print(
        f"[ok] {len(rows)} total ({n_phys} physics + {n_logic} logic); "
        f"skipped {skipped}; train={len(train)} val={len(val)}"
    )
    print(f"[ok] {train_path}")
    print(f"[ok] {val_path}")
    print()
    print("Colab upload step: rename to sft_chat_train.jsonl / sft_chat_val.jsonl")
    print("when uploading (the sft_qwen_colab.ipynb cell expects those names).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
