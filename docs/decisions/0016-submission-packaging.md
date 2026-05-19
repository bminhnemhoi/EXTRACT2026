# 0016 — Submission packaging (deployable endpoint + 1-page PDF)

Date: 2026-05-20
Status: Accepted

## Context

ADR 0013/0015 closed the accuracy work ("Next: submission packaging — the
mandatory, highest-value remaining work"). The competition deliverable is an
always-on `POST /predict` endpoint + a 1-page solution PDF, due 2026-05-30.
Auditing the repo for deploy-readiness surfaced two blockers, not just a
missing compose file.

## Blockers found & fixed

1. **`Dockerfile.api` shipped a non-functional image.** It ran
   `uv sync --no-dev` — base deps only. The deterministic solver needs the
   `physics` (pint/sympy) and `logic` (z3/lark) extras; without them the
   image cannot answer anything. Also missing `COPY README.md` (hatchling
   `readme=` makes the package build fail without it). Fixed: install both
   extras (deliberately *not* the `llm` extra — inference is the remote
   sidecar, keeps the image ~300 MB), copy README, add curl + HEALTHCHECK
   start-period.
2. **LLM endpoint was not env-overridable.** `LLMConfig.from_yaml()` read
   `configs/model.yaml` only, so a container could not be pointed at an
   `ollama` service without editing a committed file. Added `EXACT_LLM__*`
   overrides (`BASE_URL`, `MODEL`, `API_KEY`, `LORA_ADAPTER_PATH`,
   `ENABLE_LORA`) — same `EXACT_…__…` convention `exact_agent.config`
   already uses for app.yaml. **Absent env ⇒ file value verbatim** (local
   dev and CI unchanged); +3 tests (228 → 231).

## Decisions

- **Topology** (`docker/docker-compose.yml`): GPU `ollama` sidecar +
  one-shot `model-pull` + CPU `api`. Named volume persists the model
  (fast always-on recovery); `OLLAMA_KEEP_ALIVE=-1` pins it resident (no
  ~5-min idle eviction → no mid-evaluation cold start); `restart:
  unless-stopped`; api gated on `model-pull` completing.
- **The deployed endpoint runs WITH the LLM, not rule-only.** Rule-only
  physics is 0.8% vs **28.6%** with the 3B extractor — shipping rule-only
  would throw away ~36× of the system's measured capability. The switch
  that builds the LLM client in the Orchestrator is
  `EXACT_SELF_CORRECTION__ENABLED=true`, set in the compose env.
- **SFT adapter omitted from the deployed endpoint** (ADR 0013: marginal
  +1.5pp, ~5× latency). Retained as a reproducible ablation; the endpoint
  runs the small qwen2.5:3b extractor for latency.
- **Solution PDF** = `docs/solution_description.md` → `pandoc
  --pdf-engine=pdflatex` → 1 page (verified). ASCII-sanitised (pdflatex
  rejects raw `≤ → ∀ ×`). Leads with the solver-first / LLM-never-computes
  story and states data/model compliance explicitly.

## Honest caveats

1. **Toggle coupling.** `self_correction.enabled` gates *both* the LLM
   fallback (extractor/translator) *and* the self-correction loop in
   `Orchestrator._resolve_llm` (the ADR-0008 design, kept so CI stays
   deterministic). Enabling it for the LLM extractor also enables the
   loop. This is acceptable for submission — the loop is solver-grounded
   (verified Day-8 live 5/5, never invents numbers) and improves P2/P3 —
   but it conflates two concerns. Flagged for a future
   `llm.enabled`-vs-`self_correction.enabled` decouple; not changing core
   orchestration at submission time (risk > benefit).
2. **Container runtime not exercised here.** The dev box has no docker
   daemon. The Python change, the test suite, and the PDF build are
   actually run; the compose/Dockerfile are authored + statically
   reviewed and validated end-to-end on the GPU host via
   `scripts/start_runpod.sh` (which health-gates before declaring up) —
   the same posture under which `Dockerfile.api` originally shipped Day-1.

## Artifacts

`docker/docker-compose.yml`, fixed `docker/Dockerfile.api`,
`scripts/start_runpod.sh`, `scripts/prewarm_cron.sh`, `docs/deploy.md`,
`docs/solution_description.md` (+ built `.pdf`), env override in
`src/exact_agent/llm/vllm_client.py` (+ `tests/unit/test_llm_client.py`).

## Reproduction

```powershell
uv run pytest -q                              # 231 tests
pandoc docs/solution_description.md -o docs/solution_description.pdf --pdf-engine=pdflatex
# GPU host:
bash scripts/start_runpod.sh                  # build → pull → health-gate
bash scripts/prewarm_cron.sh                  # full-path warm round-trip
```
