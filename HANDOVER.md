# HANDOVER — `exact-veriscope-agent` (EXACT 2026)

> **Status at handover:** May 28, 2026. ~32 days to challenge deadline. Pipeline functional end-to-end. Physics deterministic = **~63% Full✓**, Logic = **~24% Correct**. See [docs/solution_description.md](docs/solution_description.md) for the official 1-page solution write-up.

---

## 1. What this project is

A **solver-first neuro-symbolic QA agent** for the IEEE IJCNN 2026 **EXACT 2026 Challenge** (Explainable Educational QA). Two task types:

- **Logic** — first-order-logic reasoning over university regulations (464 records, 913 Qs). Forms: MC, Yes/No/Unknown, open.
- **Physics** — numerical word problems on circuits + electrostatics (5,520 problems).

**Core invariant:** the LLM never decides the final answer. SymPy/pint solves physics; forward-chaining + Z3 decides logic. The LLM (Qwen2.5-3B-Instruct via Ollama) only (a) extracts quantities from prose, (b) translates NL → FOL for Z3, (c) phrases explanations from solver traces.

Submission target: `POST /predict` returning `{answer, explanation, cot, premises, fol, confidence}`.

---

## 2. Repo map (where to look first)

| Path | Purpose |
| --- | --- |
| [README.md](README.md) | Quickstart, public-facing overview |
| [docs/solution_description.md](docs/solution_description.md) | **1-page submission** — read this first |
| [docs/architecture.md](docs/architecture.md) | Module-level data flow |
| [docs/decisions/](docs/decisions/) | **41 ADRs** — every iteration's rationale + measurements |
| [docs/data_disclosure.md](docs/data_disclosure.md) | Dataset compliance statement |
| [docs/notes/idea.md](docs/notes/idea.md) | Original research vision (Qwen-Scope / GRPO) — partly unimplemented |
| [docs/notes/debai.md](docs/notes/debai.md) | Official challenge brief (VN) |
| [src/exact_agent/](src/exact_agent/) | All production code (src-layout) |
| [configs/](configs/) | YAML configs + Jinja2 prompts |
| [data/official_v20260515/](data/official_v20260515/) | **Frozen dataset** used for all reported numbers |
| [data/cleaned/](data/cleaned/) | Cleaned SFT subset |
| [outputs/eval/final_logic/](outputs/eval/final_logic/) | Latest logic eval report |
| [outputs/eval/final_physics/](outputs/eval/final_physics/) | Latest physics eval report |
| [archive/](archive/) | LoRA weights zip + cleaned-dataset zip (Git LFS) |
| [external/tailieu_moi/](external/tailieu_moi/) | Raw competition materials (slides, QA, datasets 05-09 & 05-15) |
| [models/exact_qwen25_7b_lora/](models/exact_qwen25_7b_lora/) | QLoRA adapter (NOT deployed — see §6) |

---

## 3. Architecture in 30 seconds

```
POST /predict
   → api/routes.py → agent/orchestrator.py
                          │
            ┌─────────────┴─────────────┐
            ▼ task=logic                ▼ task=physics
       logic/pipeline.py           physics/pipeline.py
       premise_selector            question_cleaner
       rule_parser                 topic_classifier
       forward_chainer             quantity_extractor (6-pass regex + LLM fallback)
       z3_verifier                 unit_converter (pint)
       fol_parser                  formula_library (47-entry YAML)
       llm_translator (NL→FOL)     solver (SymPy)
       answer_verifier             verifier
       explanation                 explanation
                          ▼
              agent/output_formatter.py → PredictResponse
                          │
                          ▼
              agent/self_correction.py (≤2 rounds, solver-grounded)
```

Full details: [docs/architecture.md](docs/architecture.md).

---

## 4. How to run

### Local dev
```bash
uv sync                                # Python 3.11+, installs all deps
uv run pytest -q                       # 271 tests, ~30s
uv run uvicorn exact_agent.api.app:app --reload --port 8000
```

Smoke test:
```bash
uv run python scripts/smoke_api.py
```

### Local eval (reproduces all reported numbers)
```bash
# Physics on the SFT-unseen holdout (163 rows, the headline number)
uv run python scripts/run_eval.py --task physics --with-llm \
  --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \
  --out outputs/eval/my_physics_run

# Logic on the eval split
uv run python scripts/run_eval.py --task logic --with-llm \
  --split data/official_v20260515/eval_split/logic_eval.jsonl \
  --out outputs/eval/my_logic_run
```

**Requires** a running Ollama with `qwen2.5:3b` pulled:
```bash
ollama serve &
ollama pull qwen2.5:3b
```

### Docker (submission topology)
```bash
docker compose -f docker/docker-compose.yml up -d
# GPU Ollama sidecar (qwen2.5:3b, OLLAMA_KEEP_ALIVE=-1) + CPU FastAPI
curl http://localhost:8000/healthz
```

---

## 5. Current results (frozen on 163-row SFT-unseen holdout, dataset v2026-05-15)

| Variant | Physics Full✓ | Logic Correct | Source |
| --- | --: | --: | --- |
| Day-1 rule-only baseline | 0.8% | 35.9% | ADR-0001..3 |
| Day-22 (F1+F2 audit, FOL loop) | 27.6% | 23.5% | ADR-0019..21 |
| Day-25 Iter-7 | 46.0% | 24.7% | ADR-0029 |
| Day-28 Iter-18 | 60.7% | ~24% | ADR-0040 |
| **Day-29 Iter-19 (current HEAD)** | **~63%** | **~24%** | ADR-0041 |

- Physics formula registry: **47 formulas** in `configs/physics_formulas.yaml`.
- Logic Z3 backend: multi-variable universals + comparison literals, NL→FOL reject-and-retry loop.
- Scoring: pint dimension match + 1% rel tolerance OR round-aware at gold precision. **Exact** when extraction succeeds.

Latest reports:
- [outputs/eval/final_physics/physics_llm_report.md](outputs/eval/final_physics/physics_llm_report.md)
- [outputs/eval/final_logic/logic_llm_report.md](outputs/eval/final_logic/logic_llm_report.md)

---

## 6. What is intentionally NOT deployed

| Thing | Why excluded | Where it lives |
| --- | --- | --- |
| QLoRA SFT-7B | Single = 24.5% (-6.6pp regression vs 3B-deterministic-only). Hybrid 30.1% — still below deterministic 46%. | [models/exact_qwen25_7b_lora/](models/exact_qwen25_7b_lora/), [archive/exact_qwen25_7b_lora.zip](archive/exact_qwen25_7b_lora.zip), ADR-0013 |
| GRPO (TRL) | Not yet attempted. Stub: [src/exact_agent/train/](src/exact_agent/train/). | — |
| Qwen-Scope SAE | Listed in research vision ([docs/notes/idea.md](docs/notes/idea.md)) but no implementation. | — |
| Qwen3-8B backbone | Day-23 A/B with Qwen3-8B did not lift, chose Qwen2.5-3B for latency. | ADR-0022 |

If you reopen any of these, document the lift on the **same 163-row SFT-unseen holdout** to stay comparable.

---

## 7. Where the biggest wins still are (ranked by ROI for 32-day push)

### 7.1 Logic translator (HIGHEST leverage)
Logic at 24% Correct / 40% abstain is **translator-bound** on 3B. Options:
- Few-shot RAG over `data/official_v20260515/train/logic_train.jsonl` for premise→FOL templates.
- Small SFT (LoRA) targeted **only** at NL→FOL translation (not answer generation).
- Improve `src/exact_agent/logic/fol_parser.py` vocabulary expansion (see ADR-0014, ADR-0017).

Target: 24% → 35% Correct ≈ same overall lift as physics 60→70%.

### 7.2 Physics residual clusters
Failure breakdown ([outputs/eval/final_physics/physics_llm_report.md](outputs/eval/final_physics/physics_llm_report.md)):
- `no_formula_matched` — 33 rows. Audit + add via the proven Iter-3..19 pattern (ADRs 0030-0041).
- `llm_recovery_failed` — 20 rows. Strengthen [src/exact_agent/physics/llm_extractor.py](src/exact_agent/physics/llm_extractor.py) prompts / N-self-consistency.
- `missing_input` — 17 rows. Extend role-aware regex in [src/exact_agent/physics/quantity_extractor.py](src/exact_agent/physics/quantity_extractor.py).

NL prefix is the worst (Full✓=9.5%) — prose problems with implicit setup. Likely needs a dedicated extractor pass.

### 7.3 Documentation sync (REQUIRED before final submission)
- [README.md](README.md) and [docs/notes/idea.md](docs/notes/idea.md) still mention **Qwen3-8B + Qwen-Scope + GRPO** as the headline architecture. The actual submission uses **Qwen2.5-3B + deterministic solver**. Reconcile before the deadline.

---

## 8. Iteration workflow (proven — keep using it)

Every iteration follows this loop (see ADR-0029 onward):
1. Run eval on the 163-row holdout → identify wrong rows.
2. Cluster failures by pattern (formula type, extraction failure, routing miss).
3. Pick a cluster of ≥3 rows for a single iter batch.
4. Write fix (new formula in YAML / new routing guard / extractor regex).
5. Add **dedicated tests** in [tests/](tests/) — one per row fixed.
6. Re-run eval, confirm lift, write ADR `docs/decisions/00NN-dayXX-iterYY-name.md`.
7. Commit in 3-4 small commits: `feat(iter-NN-fmt)`, `feat(iter-NN-route)`, `fix(iter-NN-text)`, `test(iter-NN)`.

**Never** edit production code without an ADR if it changes a measured number.

---

## 9. Important gotchas

- **`models/` and `data/raw/` are gitignored** (see [.gitignore](.gitignore)). The LoRA adapter shipped here is via [archive/exact_qwen25_7b_lora.zip](archive/exact_qwen25_7b_lora.zip) over Git LFS. Unzip into `models/exact_qwen25_7b_lora/` if you want to experiment with SFT.
- **`uv.lock` and `poetry.lock` are gitignored** intentionally — pin via `pyproject.toml`. If you need reproducible builds, commit a lockfile.
- **Score field convention:** physics uses `pint` dimensional match + 1% rel-tol OR round-aware. Don't compare raw floats — use `src/exact_agent/eval/metrics.py`.
- **Holdout discipline:** the headline 163-row `physics_eval_sft_unseen.jsonl` was built by [scripts/build_sft_unseen_holdout.py](scripts/build_sft_unseen_holdout.py) after fixing a 99% leakage in the earlier holdout. Don't replace it without re-running the leakage check.
- **Cleanup files on Windows:** `Remove-Item` does NOT use the Recycle Bin. Be careful around `archive/` and `models/`.
- **CI:** ruff + mypy + pytest. PRs must keep all 271 tests green.

---

## 10. Contact & continuity

- All architectural decisions are in ADR form in [docs/decisions/](docs/decisions/) — read them in order before changing the pipeline.
- The Makefile entry points (`make eval`, `make test`, `make serve`) are the canonical run targets.
- For the official defense round (live unseen-query demo), see [docs/defense_demo_script.md](docs/defense_demo_script.md).

Good luck. The hardest part of this project was the dataset cleaning + the audit-driven iteration discipline. Keep both, and the numbers will keep moving.
