---
geometry: margin=0.7in
fontsize: 10pt
colorlinks: true
---

**EXACT 2026 -- Data Disclosure Document**
Team Veriscope * IEEE IJCNN 2026 EXACT Challenge

Per QA Q11 / Q23: every external data source touching our pipeline
(training, fine-tuning, retrieval, evaluation) is declared below.

## 1. Datasets used

| Source | Release / version | Where used | Volume |
|---|---|---|---|
| **EXACT2026 official dataset** | `2026-05-15` (drop-in replacement for `2026-05-09`; see CHANGELOG_TYPE{1,2}.md) | Training (90% split), internal self-eval (10% holdout, seed 42) | 1,352 physics + 808 logic questions (411 records) |

No other datasets are used. Specifically, **no** external academic corpora,
**no** web-crawled data, **no** Vietnamese physics textbooks, **no** prior
EXACT 2025 data.

## 2. Synthetic data from closed-source models

**None.** No GPT, Claude, Gemini, or other closed-source model was used
to generate any training, fine-tuning, retrieval, or evaluation data at
any stage of this submission.

## 3. Crawled / scraped data

**None.**

## 4. Knowledge base / retrieval corpora

The deployed system retrieves only from the official EXACT2026 release
above. No external knowledge base is built or queried at inference.

If a retrieval-augmented variant is shipped (currently planned but not
yet enabled): the retrieval corpus consists of the *training-split* rows
(90% of the 2026-05-15 release) used as few-shot exemplars. No data
outside section 1 is added.

## 5. Models used at inference

| Role | Model | License | Params | Hosting |
|---|---|---|---|---|
| Utility extractor / NL->FOL translator / explanation phrasing | **Qwen2.5-3B-Instruct** (`qwen2.5:3b-instruct`, Q4_K_M) | Qwen License (permissive research use; per QA Q4) | 3.1B (nominal 3B-class, <= 8B) | Self-hosted on rented GPU via Ollama (OpenAI-compatible `/v1/models` and `/v1/chat/completions`; verified by team before deploy) |

**Only one LLM is loaded at inference at any moment** (per QA Q3 rule of
thumb). If a sequential second model is added (e.g. a specialised
NL->FOL translator for Type 1), it will be declared here in the final
version of this document.

No closed-source LLM is called at inference. No third-party hosted
inference API (Together AI, Fireworks, Groq, Replicate, HF Inference,
etc.) is used.

## 6. Fine-tuning / training disclosure

| Artefact | Base | Data | Recipe | Used at inference? |
|---|---|---|---|---|
| `models/exact_qwen25_7b_lora` (kept as ablation) | `unsloth/qwen2.5-7b-instruct` | `sft_train_mixed_solver_clean.jsonl` (1,765/196 train/val), derived from EXACT2026 official release 2026-05-09 (no augmentation, no synthetic data) | QLoRA, rank=16, alpha=32, 2 epochs, Unsloth on free Colab T4 | **No** -- omitted from deployed endpoint for latency (ADR 0013); retained as a documented, reproducible ablation only |

If the production deploy is upgraded to use the SFT'd adapter as a
sequential specialised extractor (option E7a in our improvement plan),
this row will be moved to section 5 and re-declared.

## 7. Tools (per QA Q7 -- do not count toward the parameter limit)

- **SymPy** -- symbolic algebra / numerical solver for physics formulas
- **pint** -- SI unit normalisation and conversion
- **Z3** -- first-order-logic entailment / theorem proving for Type 1
- **scikit-learn** TF-IDF / Jaccard similarity (premise selector)
- **FastAPI / uvicorn** -- HTTP serving

All tool invocations are reflected in the `cot`/`explanation` fields of
the API response so that evaluators can verify the reasoning trace
(per QA Q7 visibility requirement).

## 8. Contact

ngobinhminh2322006@gmail.com (Team Veriscope contact)
