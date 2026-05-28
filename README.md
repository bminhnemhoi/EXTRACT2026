# exact-veriscope-agent

> **Solver-first neuro-symbolic QA agent for the IEEE IJCNN 2026 EXACT Challenge — Explainable Educational QA.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-271%20passing-brightgreen.svg)](tests/)
[![ADRs](https://img.shields.io/badge/ADRs-41-informational.svg)](docs/decisions/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **deterministic, verifiable** question-answering pipeline that answers two classes of educational queries — first-order-logic reasoning over university regulations and numerical physics problems on circuits / electrostatics — and always returns a solver-grounded explanation.

**Core invariant.** The language model never decides the final answer. SymPy + `pint` compute physics; forward-chaining + Z3 prove logic. The LLM (Qwen2.5-3B-Instruct, served locally via Ollama) only (a) extracts quantities from prose, (b) translates natural-language premises to FOL, and (c) phrases the explanation from the solver trace.

---

## Table of contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [API contract](#api-contract)
- [Reproducing the headline numbers](#reproducing-the-headline-numbers)
- [Project layout](#project-layout)
- [Development](#development)
- [Deployment](#deployment)
- [Documentation index](#documentation-index)
- [Compliance & data](#compliance--data)
- [License](#license)

---

## Highlights

| Aspect | Status |
| --- | --- |
| **Physics** (163-row SFT-unseen holdout, dataset v2026-05-15) | **~63 % Full✓** (value + unit, strict) |
| **Logic** (`logic_eval.jsonl`) | **~24 % Correct**, principled `Unknown` abstention |
| **LLM** | Qwen2.5-3B-Instruct via Ollama (OpenAI-compat), open-source, ≤ 8 B params |
| **Solvers** | SymPy / `pint` (physics), forward-chaining + Z3 (logic) |
| **Formula registry** | 47 physics formulas in [configs/physics_formulas.yaml](configs/physics_formulas.yaml) |
| **Tests** | 271 unit + integration + e2e, CI: ruff + mypy + pytest |
| **Decisions** | 41 ADRs in [docs/decisions/](docs/decisions/) — every iteration is measured |
| **Submission** | `POST /predict` returning `{answer, explanation, cot, premises, fol, confidence}` |

> The LLM contribution is intentionally narrow: numbers come from the solver, never from a generation. This makes every answer **inspectable** and every explanation **faithful by construction** — directly serving the EXACT 2026 criteria for accuracy (P1), explanation quality (P2), and reasoning depth (P3).

---

## Architecture

```
POST /predict
   │
   ▼
api/routes.py ──▶ agent/orchestrator.py
                       │
            ┌──────────┴──────────┐
            ▼ task = logic        ▼ task = physics
       logic/pipeline.py     physics/pipeline.py
       │                     │
       │ premise_selector    │ question_cleaner
       │ rule_parser         │ topic_classifier
       │ forward_chainer     │ quantity_extractor   ── LLM fallback
       │ z3_verifier         │ unit_converter (pint)
       │ fol_parser          │ formula_library (47 YAML formulas)
       │ llm_translator      │ solver (SymPy)
       │ answer_verifier     │ verifier
       │ explanation         │ explanation
       ▼                     ▼
            agent/output_formatter.py
                       │
                       ▼
            agent/self_correction.py   (≤ 2 rounds, solver-grounded)
                       │
                       ▼
                PredictResponse (Pydantic)
```

See [docs/architecture.md](docs/architecture.md) for module-level responsibilities and invariants.

---

## Quickstart

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com/) with the `qwen2.5:3b` model pulled (only if you want the LLM-assisted path)

```bash
# 1. Install deps
uv sync --all-extras

# 2. Pull the LLM (only if you want LLM extractor / FOL translator)
ollama serve &
ollama pull qwen2.5:3b

# 3. Run the test suite
uv run pytest -q

# 4. Start the API
uv run uvicorn exact_agent.api.app:app --reload --port 8000
```

Smoke test:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "Calculate the energy stored when C = 100 μF and U = 30 V."}'
```

Or use the bundled script:

```bash
uv run python scripts/smoke_api.py
```

Common workflows are exposed via the [Makefile](Makefile) — run `make help` to list them.

---

## API contract

The single endpoint is `POST /predict`. It accepts either kind of query and returns a uniform schema:

**Physics request**
```json
{ "question": "Calculate the energy stored when C = 100 μF and U = 30 V." }
```

**Logic request**
```json
{
  "premises-NL": [
    "If a student completes required courses, they are eligible for graduation.",
    "John completed required courses."
  ],
  "question": "Is John eligible for graduation?"
}
```

**Response** (Pydantic-validated — malformed JSON is impossible)
```json
{
  "answer": "45",
  "explanation": "Apply E = ½ C U² with C = 1e-4 F, U = 30 V → E = 45 J.",
  "cot": ["Step 1: ...", "Step 2: ..."],
  "premises": ["P1", "P3"],
  "fol": "Eligible(John)",
  "confidence": 0.92
}
```

If the LLM is unreachable, the response degrades to the deterministic answer (never a 500).

---

## Reproducing the headline numbers

All reported numbers use dataset **v2026-05-15** and the **163-row SFT-unseen holdout** built by [scripts/build_sft_unseen_holdout.py](scripts/build_sft_unseen_holdout.py) (closes a 99 % leakage in the prior split — see ADR-0029).

```bash
# Physics (headline metric)
uv run python scripts/run_eval.py --task physics --with-llm \
  --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \
  --out outputs/eval/repro_physics

# Logic
uv run python scripts/run_eval.py --task logic --with-llm \
  --split data/official_v20260515/eval_split/logic_eval.jsonl \
  --out outputs/eval/repro_logic
```

Trajectory (Physics Full✓ on the same holdout):

| Milestone | Physics Full✓ | Logic Correct | Reference |
| --- | --: | --: | --- |
| Day-1 rule-only baseline | 0.8 % | 35.9 % | ADR-0001..3 |
| Day-22 (F1+F2 audit, FOL loop) | 27.6 % | 23.5 % | ADR-0019..21 |
| Day-25 Iter-7 | 46.0 % | 24.7 % | ADR-0029 |
| Day-28 Iter-18 | 60.7 % | ~24 % | ADR-0040 |
| **Day-29 Iter-19 (current `HEAD`)** | **~63 %** | **~24 %** | ADR-0041 |

Latest formatted reports:

- [outputs/eval/final_physics/physics_llm_report.md](outputs/eval/final_physics/physics_llm_report.md)
- [outputs/eval/final_logic/logic_llm_report.md](outputs/eval/final_logic/logic_llm_report.md)

Scoring is **strict**: pint dimensional match + 1 % relative tolerance OR round-aware match at the gold's stated precision. The deterministic solver is exact where extraction succeeds; the residual error is bounded by LLM-extractor recall on hard multi-step prose.

---

## Project layout

```
exact-veriscope-agent/
├── src/exact_agent/             # All production code (src-layout)
│   ├── api/                     # FastAPI app, routes, middleware
│   ├── agent/                   # orchestrator, self_correction, output_formatter
│   ├── router.py                # logic vs physics dispatch
│   ├── logic/                   # premise selector, rule parser, Z3, FOL translator
│   ├── physics/                 # extractors, unit converter, SymPy solver, 47 formulas
│   ├── llm/                     # Ollama / OpenAI-compat client, prompt templates
│   ├── eval/                    # Metric harness + report generators
│   └── train/                   # QLoRA SFT scripts (not deployed — see ADR-0013)
├── configs/                     # YAML configs + Jinja2 prompts (split from code)
├── data/
│   ├── official_v20260515/      # Frozen competition dataset
│   ├── cleaned/                 # Cleaned SFT subset
│   └── eval_split/              # Held-out evaluation splits
├── docs/
│   ├── architecture.md          # Module-level architecture
│   ├── solution_description.md  # 1-page submission write-up
│   ├── data_disclosure.md       # Dataset compliance statement
│   ├── decisions/               # 41 ADRs — read in order
│   └── notes/                   # Research vision (idea.md) + challenge brief
├── tests/                       # 271 tests across unit / integration / e2e
├── scripts/                     # Eval drivers, holdout builders, audits
├── docker/                      # Compose files + API Dockerfile
├── archive/                     # LoRA adapter zip + cleaned dataset zip (Git LFS)
├── external/                    # Raw competition materials (slides, source datasets)
├── HANDOVER.md                  # ⭐ Continuation guide for new maintainers
└── Makefile                     # Common workflows — run `make help`
```

---

## Development

### Common commands

```bash
make install-all        # Install all extras
make test               # Run the full test suite
make test-unit          # Unit tests only
make lint               # ruff check
make typecheck          # mypy
make format             # Auto-format
make api                # Run dev server with reload
make smoke              # Hit the API with sample payloads
make eval               # Run the local eval harness
```

### Iteration workflow

Every functional change that moves a measured number follows this loop (see ADRs 0029 onward):

1. Run the eval harness on the 163-row holdout, cluster wrong rows by pattern.
2. Pick a cluster of ≥ 3 rows for a single iter batch.
3. Implement the fix (new formula in YAML / new routing guard / extractor regex).
4. Add a dedicated test per row fixed in [tests/](tests/).
5. Re-run eval, confirm the lift, write an ADR under `docs/decisions/`.
6. Commit in small, semantic units: `feat(iter-NN-fmt)`, `feat(iter-NN-route)`, `fix(iter-NN-text)`, `test(iter-NN)`.

**No measured change ships without an ADR.**

### Code style

- `ruff` for linting and formatting, `mypy` for typing, `pytest` for tests.
- src-layout (`src/exact_agent/`) with all configuration in [configs/](configs/) — no hard-coded numbers in the solver.
- All public types live in `src/exact_agent/*/types.py` or `schemas.py`.

---

## Deployment

The submission topology is a CPU FastAPI container plus a GPU Ollama sidecar with the LLM pinned in memory:

```bash
docker compose -f docker/docker-compose.yml up -d
curl http://localhost:8000/healthz
```

- `OLLAMA_KEEP_ALIVE=-1` keeps `qwen2.5:3b` resident.
- A cron pre-warm script ([scripts/prewarm_cron.sh](scripts/prewarm_cron.sh)) keeps tail latency stable.
- Liveness and readiness gates are wired through `/healthz`.

See [docs/deploy.md](docs/deploy.md) for the full deployment guide and [docs/defense_demo_script.md](docs/defense_demo_script.md) for the live-demo runbook used in the EXACT 2026 final round.

---

## Documentation index

| Document | Read it when… |
| --- | --- |
| [HANDOVER.md](HANDOVER.md) | **You are new to the project** — start here |
| [docs/solution_description.md](docs/solution_description.md) | You want the 1-page submission write-up |
| [docs/architecture.md](docs/architecture.md) | You want the module-level data flow |
| [docs/decisions/](docs/decisions/) | You want to understand why a choice was made |
| [docs/data_disclosure.md](docs/data_disclosure.md) | You need the data-compliance statement |
| [docs/deploy.md](docs/deploy.md) | You are deploying the API |
| [docs/defense_demo_script.md](docs/defense_demo_script.md) | You are preparing for the live final round |
| [docs/notes/idea.md](docs/notes/idea.md) | You want the original research vision (Qwen-Scope / GRPO) |
| [docs/notes/debai.md](docs/notes/debai.md) | You want the official challenge brief (VN) |

---

## Compliance & data

- Only the official **EXACT 2026 v2026-05-15** dataset is used for training, evaluation, and tuning. Full disclosure: [docs/data_disclosure.md](docs/data_disclosure.md).
- No closed-source LLM is invoked anywhere in the pipeline (no GPT, Claude, or Gemini).
- The Qwen2.5-3B-Instruct weights are open-source and served locally via Ollama; the model is well within the ≤ 8 B parameter limit of the challenge.
- Raw competition materials are mirrored in [external/tailieu_moi/](external/tailieu_moi/) for transparency.
- Large binaries (LoRA adapter, dataset archives) are tracked via **Git LFS** under [archive/](archive/).

---

## License

[MIT](LICENSE).

---

## Citation

If you build on this work, please cite the EXACT 2026 challenge and reference this repository:

```bibtex
@software{exact_veriscope_agent_2026,
  title  = {exact-veriscope-agent: a solver-first neuro-symbolic QA agent for EXACT 2026},
  year   = {2026},
  url    = {https://github.com/bminhnemhoi/EXTRACT2026}
}
```
