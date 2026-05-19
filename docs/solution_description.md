---
geometry: margin=0.7in
fontsize: 10pt
colorlinks: true
---

**EXACT 2026 -- Veriscope: a solver-first neuro-symbolic QA agent**
Team Veriscope * IEEE IJCNN 2026 EXACT Challenge * Submission: `POST /predict`

## Approach

The answer is **always computed by a deterministic symbolic solver**, never
by a language model. The LLM (open-source, **Qwen2.5-3B-Instruct**,
Apache-2.0, <= 8B params) is confined to three utility roles: (1) extract
physical quantities from prose when regex misses, (2) translate a
natural-language claim into first-order logic, (3) phrase the final
explanation. Every numeric result comes from SymPy/pint; every logical
verdict from forward-chaining or a Z3 proof. This makes the system
*verifiable*: the explanation is generated from the solver's own trace, so it
cannot diverge from the answer -- directly serving the rubric's P2
(explanation) and P3 (reasoning depth).

## Pipeline

- **Router**: presence of `premises-NL` -> logic, else physics.
- **Physics**: question-cleaner -> topic classifier (ID-prefix + keyword +
  symbol fallback) -> quantity extractor (regex; LLM fallback) -> unit
  normalisation (pint) -> formula registry (24 formulas, YAML->SymPy, incl.
  Coulomb vector superposition) -> solve -> unit/sanity verifier -> trace
  explanation.
- **Logic**: Jaccard premise selector -> rule parser -> forward chainer ->
  **Z3** entailment (multi-variable universals, comparison literals) when
  surface reasoning is inconclusive -> per-option entailment for MC; emits
  `premises:["P1",...]`, the FOL, and a step CoT.
- **Self-correction** (deployed, ON): the solver re-checks the LLM draft;
  feedback is *solver-grounded* (never invents numbers/premises), <= 2 rounds.
- Output schema: `{answer, explanation, cot, premises, fol, confidence}`;
  malformed JSON is impossible (Pydantic) and the LLM path degrades to the
  deterministic answer if the model is unreachable (never a 500).

## Training (ablation, not on the critical path)

We QLoRA-SFT'd Qwen2.5-7B purely to stabilise output *format* (ADR
0006/0013). Measured impact was marginal (physics +1.5 pp, logic flat) at
~5x latency: **the system is solver-bound, not LLM-bound.** The deployed
endpoint therefore omits the adapter and runs the small 3B extractor for
latency; the adapter is retained as a documented, reproducible ablation.

## Data & compliance

- **Only** `cleaned_exact_dataset_package` is used (no external data/models).
- LLM is open-source <= 8B; **no GPT/Claude/Gemini** anywhere in the pipeline.
- Eval = strict 10% holdout (seed 42); a frozen unit-/round-aware scorer
  (ADR 0009/0011/0012). 228 automated tests; ruff+mypy CI.

## Internal results (10% holdout, frozen scorer)

| Task | Rule-only | Deployed system | Lift |
|---|---|---|---|
| Physics (exact value+unit) | 0.8% | **28.6%** | ~36x |
| Logic (exact label) | 35.9% | **37.5%** | +1.6 pp |

Physics correctness is reported under a deliberately strict exact
value-and-unit match; the dominant residual is LLM extraction recall on
hard multi-step problems, not solver error (the solver is exact where
extraction succeeds). Logic is translator-bound: the symbolic core (Z3) is
sound; gains track NL->FOL quality.

## Deployment

`docker compose up` -> GPU Ollama sidecar + CPU FastAPI solver; always-on
with `/healthz`, model pinned resident, cron pre-warm, snapshot backup.
Reproduce: `uv run pytest -q`; `uv run python scripts/run_eval.py --task
{physics,logic} --with-llm`.
