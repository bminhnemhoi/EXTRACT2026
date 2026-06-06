# CLAUDE.md — EXTRACT2026 Codebase Guide

Guide for Claude (and humans) working on this repo. Grounded in the **real** source tree — every path below was verified by reading the code, not inferred from the README.

---

## What this is

**Solver-first neuro-symbolic QA agent** for the IEEE IJCNN 2026 **EXACT** challenge (Explainable Educational QA). Two task types — **Logic** (FOL over university regulations) and **Physics** (circuits + electrostatics numerics).

**Core invariant — never break it:** the LLM **never decides the answer**. SymPy+pint compute physics; forward-chaining+Z3 decide logic. The LLM (Qwen2.5-3B via Ollama) only (a) extracts quantities, (b) translates NL→FOL, (c) phrases explanations from solver traces.

**Submission:** `POST /predict` → `{answer, explanation, cot, premises, fol, confidence}`.

**Current scores:** Physics ~63–67% Full✓ (README ~63% vs ADR-0041 66.9% — see discrepancy note below), Logic ~24–28% Correct (24.7% ADR-0035 → 28.4% ADR-0036, latest stable). Holdout: 163-row SFT-unseen physics / 81-row logic (v2026-05-15).

---

## ⚠️ Known-stale facts (verify, don't trust the README)

| Claim in docs | Reality in code | Action |
| --- | --- | --- |
| README / architecture.md / **solution_description.md**: "47 formulas" | `configs/physics_formulas.yaml` has **70** | trust the file |
| README badge: "271 tests" | **295** `def test_` funcs / 31 files (pytest collects ~362 w/ parametrize; ADR-0041 cites 362) | trust `pytest` |
| README/HANDOVER: Day-29 ~63% physics | ADR-0041 says **66.9%** | run eval to confirm |
| README/HANDOVER: logic ~24% | latest stable **28.4%** (ADR-0036) | run eval to confirm |
| `outputs/eval/` reports exist (HANDOVER links them) | **directory absent** — numbers come from docs/ADRs, not a saved run | run eval to regenerate |
| `docker/docker-compose.yml` comment: "28.6% physics / 37.5% logic" | those are **Day-8/14** scores; current 66.9% / ~28% | stale comment, ignore |
| `.env.example`: `Qwen/Qwen3-8B` @ vLLM:8001 | deploy overrides to Ollama `qwen2.5:3b-instruct` | use Ollama path |
| `make train-grpo` → `run_grpo.py` | module **does not exist** (only `prepare_sft.py`, `run_sft_qwen.py`) | don't run it |
| idea.md: Qwen3-8B + Qwen-Scope SAE + GRPO | **none deployed** — pure solver + 3B | docs are aspirational |

When a number matters, **run eval** rather than quoting a doc. The failure-cluster counts below (33/20/17) come from HANDOVER §7.2 and are **unverified until eval is re-run** post-Iter-19.

---

## Real source tree (`src/exact_agent/`)

```
exact_agent/
├── config.py          Settings + YAML load + env override (EXACT_ prefix, __ nesting)
├── router.py          route_task(payload) → "logic" if premises-NL present else "physics"
├── schemas.py         PredictRequest, PredictResponse, TaskType, HealthResponse (Pydantic)
├── logging_setup.py   loguru config
│
├── agent/
│   ├── orchestrator.py     Orchestrator.predict() — THE entry hub: route → pipeline → self-correct
│   ├── confidence.py       score_response() → ConfidenceAssessment (revision threshold)
│   ├── output_formatter.py format_response() → PredictResponse
│   └── self_correction.py  SelfCorrector.maybe_revise() — ≤2 rounds, solver-grounded
│
├── api/
│   ├── app.py         create_app(), global `app`, run() — uvicorn target: exact_agent.api.app:app
│   ├── routes.py      GET /healthz, POST /predict
│   ├── exceptions.py  PipelineError + handlers
│   └── middleware.py  request id + timing + CORS
│
├── logic/
│   ├── pipeline.py          LogicPipeline.run() — selector→parser→chainer→verifier (Z3 fallback)
│   ├── premise_selector.py  rank_premises(), select_top_k() — TF-IDF cosine (Jaccard fallback)
│   ├── rule_parser.py       parse_premises() — "If..then..", "All X are Y" → Rule/Fact
│   ├── forward_chainer.py   forward_chain() → ChainResult (Jaccard surface matching)
│   ├── llm_translator.py    translate_question_to_fol[_with_verify]() — NL→FOL + reject/retry
│   ├── answer_verifier.py   verify_yes_no/verify_multiple_choice() → VerifierResult
│   ├── z3_verifier.py       verify_with_z3() → Yes/No/Unknown (entailment fallback)
│   ├── fol_parser.py        parse_fol() — Unicode/ASCII ∀∃∧∨¬→ parsing
│   ├── explanation.py       render_explanation()
│   └── types.py             Atom, Comparison, Rule, Existential (dataclasses)
│
├── physics/
│   ├── pipeline.py          PhysicsPipeline.run() — facade
│   ├── solver.py            PhysicsSolver.solve() → SolverResult (classify→extract→convert→compute)
│   ├── topic_classifier.py  classify() → ClassificationResult (keyword + symbol heuristic)
│   ├── quantity_extractor.py extract_quantities() → ExtractedQuantity (regex)
│   ├── llm_extractor.py     extract_with_llm[_self_consistent]() — N-sample majority vote
│   ├── formula_library.py   Formula, FormulaLibrary, default_library() — loads the 70 YAML formulas
│   ├── unit_converter.py    convert(), normalize_unit_string() — pint → SI
│   ├── verifier.py          verify() → VerificationResult (magnitude sanity)
│   ├── question_cleaner.py  clean() — whitespace/unicode-minus/imperative-prefix
│   ├── rag_retriever.py     RagRetriever — TF-IDF few-shot demos (E8)
│   └── explanation.py       render_explanation()
│
├── llm/
│   ├── vllm_client.py       VLLMClient, LLMClient (protocol), LLMConfig, parse_json_completion()
│   ├── prompt_templates.py  render() — Jinja2 from configs/prompts/
│   └── retry.py             llm_retry decorator (tenacity, 3 attempts, exp backoff)
│
├── eval/
│   ├── eval_physics.py      run_eval_physics() → EvalReport, SampleEval
│   ├── eval_logic.py        run_eval_logic() → LogicEvalReport, LogicSampleEval
│   ├── metrics.py           label_match(), quantity_match(), unit_match(), parse_number()
│   └── reports.py           render_as_markdown(), render_as_json()
│
└── train/                   NOT DEPLOYED (ablation only — ADR-0013/0022)
    ├── prepare_sft.py       build Qwen chat-template SFT dataset
    └── run_sft_qwen.py      Unsloth + TRL QLoRA trainer (GPU only)
```

---

## Configs, data, tests

**`configs/`**
- `physics_formulas.yaml` — **70 formulas**. Schema per entry:
  ```yaml
  capacitor_energy:
    topic: TD                      # TD=circuits, LD=electrostatics
    description: "Energy stored in a capacitor: E = 0.5·C·U²"
    inputs:
      C: { unit: "farad", aliases: ["capacitance"] }
      U: { unit: "volt",  aliases: ["voltage", "V"] }
    output: { symbol: E, unit: "joule" }
    expression: "0.5 * C * U**2"
  ```
- `app.yaml`, `model.yaml`, `logic_patterns.yaml`, `training/`
- `prompts/` — **3** Jinja2 templates: `physics_extract.j2`, `nl_to_fol.j2`, `self_correction.j2`

**`data/`**
```
cleaned/              logic_train_safe(644), physics_train_safe(1329), sft_train_mixed(1961)
eval_split/           physics_eval(133), logic_eval(64)              ← make eval default
official_v20260515/   logic_safe(808), physics_safe(1352)
  ├ train/            logic_train(727), physics_train(1217)
  └ eval_split/       physics_eval(135), logic_eval(81),
                      physics_eval_sft_unseen.jsonl (163) ← HEADLINE holdout
```
The **163-row SFT-unseen holdout** is the canonical benchmark. Built by `scripts/build_sft_unseen_holdout.py` after fixing a 99% leakage (ADR-0022). Don't replace it.

**`tests/`** — 31 `test_*.py` files, **295 `def test_` functions** (pytest collects ~362 cases because of `@pytest.mark.parametrize`): `unit/` (25 files), `integration/` (5), `e2e/` (1). Iteration tests are named `test_iterNN_*`.

**`scripts/`** (12) — key ones: `run_eval.py` (eval harness), `smoke_api.py` (hit API), `build_sft_unseen_holdout.py` (holdout), `iterate_10_failing.py` (fast 10-row triage), `inspect_eval.py` (failure diagnostics), `audit_dataset_issues.py`.

---

## First-time setup (this machine is NOT yet ready — checked 2026-05-31)

None of the toolchain is installed (`uv`, `ollama`, `docker` all missing) and the system Python is **3.14**, which is too new — heavy deps (`z3-solver`, `pint`, `sentence-transformers`) may have no 3.14 wheels. Use **Python 3.11** (matches `.github/workflows/ci.yml` and `Dockerfile.api`).

```bash
# 1. uv (every command below needs it)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Python 3.11 + deps + hooks
uv python install 3.11
uv sync --all-extras          # or: uv sync --extra physics --extra logic --extra dev  (no-LLM path)
pre-commit install

# 3. Verify
uv run pytest -q              # expect all green (~362 collected)

# 4. LLM path — only needed for --with-llm eval / API / self-correction
brew install ollama && ollama serve &
ollama pull qwen2.5:3b        # ~2 GB
```

Rule-only eval runs without Ollama but collapses physics to ~0.8% (the extractor is LLM-backed, ADR-0016) — use the LLM path for realistic numbers. To experiment with the (not-deployed) SFT adapter: `unzip archive/exact_qwen25_7b_lora.zip -d models/exact_qwen25_7b_lora/`.

## How to run (verified Makefile targets)

```bash
make install        # uv sync --extra dev
make install-all    # uv sync --all-extras
make test           # pytest -x -v   (unit / integration / e2e via test-unit/-int/-e2e)
make lint           # ruff check src tests scripts
make typecheck      # mypy src
make format         # ruff format + ruff check --fix
make api            # uvicorn exact_agent.api.app:app --reload --host 0.0.0.0 --port 8000
make smoke          # python scripts/smoke_api.py
make eval           # python scripts/run_eval.py --split eval
make help           # list all targets
```

**LLM path (optional but needed for --with-llm eval & self-correction):**
```bash
ollama serve & ollama pull qwen2.5:3b
```

**Headline eval (the numbers everyone quotes):**
```bash
uv run python scripts/run_eval.py --task physics --with-llm \
  --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \
  --out outputs/eval/repro_physics
uv run python scripts/run_eval.py --task logic --with-llm \
  --split data/official_v20260515/eval_split/logic_eval.jsonl \
  --out outputs/eval/repro_logic
```
`run_eval.py` writes `{stem}_report.md` + `{stem}_report.json` to `--out`.

**Docker (submission topology):**
```bash
docker compose -f docker/docker-compose.yml up -d --build   # Ollama qwen2.5:3b + CPU FastAPI
curl http://localhost:8000/healthz
```
Docker sets `EXACT_SELF_CORRECTION__ENABLED=true` and points the client at `http://ollama:11434/v1`.

---

## Scoring (eval/metrics.py)

- **Physics Full✓** = `numeric_correct AND unit_correct`.
  - `quantity_match()`: pint dimensional convert + 1% rel tolerance OR round-aware at gold's stated precision.
  - `unit_match()`: dimensional equivalence (V ≡ volt).
- **Logic Correct** = `label_match()` (case-insensitive exact). `abstained` when pred is empty / "unknown" / "uncertain". Abstaining on under-determined premises is **intended**, not a bug.

Never compare raw floats — always go through `eval/metrics.py`.

---

## The iteration loop (how every measured change ships)

1. Run eval → cluster wrong rows by pattern.
2. Pick a cluster of **≥3 rows** for one iter batch.
3. Fix: new formula in `physics_formulas.yaml` / routing guard / extractor regex / prompt.
4. Add a **dedicated test per row fixed** in `tests/unit/test_iterNN_*.py`.
5. Re-run eval, confirm the lift, write `docs/decisions/00NN-dayXX-iterYY-name.md`.
6. Commit small semantic units: `feat(iter-NN-fmt)`, `feat(iter-NN-route)`, `fix(iter-NN-text)`, `test(iter-NN)`.

**No measured change ships without an ADR.** This discipline is what moved physics 27.6%→66.9%.

---

## Where the wins are (ROI-ranked)

> ⏱️ **Code-change window is nearly closed.** Main phase ended 5/30; the only remaining API-resubmit window is the **6/3–6/4 refinement window** ("last chance to improve"), then finals (live unseen queries) 6/15. Verified timeline: `docs/notes/competition_brief_verified.md`. Prioritise fixes that land by 6/4.

_(Row counts below are from HANDOVER §7.2 — re-run eval to confirm they still hold after Iter-19.)_

1. **Logic translator (highest)** — ~28% Correct / ~37% abstain is NL→FOL-bound on 3B. Few-shot RAG over `logic_train.jsonl` (reuse `physics/rag_retriever.py` pattern) or a tiny NL→FOL-only LoRA. Target →35%. Touch: `logic/llm_translator.py`, `configs/prompts/nl_to_fol.j2`, `logic/fol_parser.py`.
2. **Physics `no_formula_matched` (~33 rows)** — audit + add 6–8 formulas via the proven iter pattern. Touch: `configs/physics_formulas.yaml`.
3. **Physics `llm_recovery_failed` (~20 rows)** — strengthen `physics/llm_extractor.py` (more self-consistency samples, better prompt).
4. **Doc sync (required before submission)** — reconcile the stale-facts table above; README/HANDOVER/solution_description.md still cite 47 formulas / 271 tests / ~63%.

---

## Gotchas

- `models/` & `data/raw/` are gitignored. LoRA adapter is in `archive/*.zip` (Git LFS).
- `uv.lock`/`poetry.lock` gitignored on purpose — pin via `pyproject.toml`.
- All LLM calls are **local** (Ollama). No GPT/Claude/Gemini anywhere — competition rule (≤8B open model).
- Physics modules deliberately use Unicode glyphs (×, −, μ, Ω, ⁻²); ruff RUF001-003 are ignored there.
- CI = ruff + mypy + pytest. Keep all tests green.
- `outputs/eval/` is not committed — generate it locally with `run_eval.py`.

---

## Working with Claude on this repo

- **Bug fix / small change** → I read the file, fix, run the relevant tests.
- **New formula / extractor** → give me the failing cluster; I add to YAML/src, write a per-row test, run eval, draft the ADR stub.
- **Analysis question** → I read ADRs + code + (freshly generated) eval reports and summarize.
- **Before quoting a number** → I run eval rather than trusting a possibly-stale doc.

Read [ANALYSIS.md](ANALYSIS.md) for the full progress breakdown, [HANDOVER.md](HANDOVER.md) for onboarding, and `docs/decisions/` (read in order) for the why behind every choice.
