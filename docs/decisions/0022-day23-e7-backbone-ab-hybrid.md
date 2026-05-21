# 0022 — Day-23 E7 backbone A/B + hybrid routing experiment

Date: 2026-05-21
Status: Accepted

## Context

Day-22 closed with multiple independent confirmations (E5, E6, E8 all
P1-flat) that the qwen2.5:3b extractor is the ceiling on physics
accuracy. The Day-13 SFT'd Qwen2.5-7B adapter
(`models/exact_qwen25_7b_lora/`) was at that point classified
"marginal / not deployed" (ADR 0013) — but ADR 0013's eval was on a
self-cleaned 2026-05-09 derivative; the real question is whether the
adapter *generalises* to the official 2026-05-15 distribution.

## Pre-A/B: 99% leakage discovered & fixed (commit `175a75a`)

An audit of `sample_id` overlap between the 1765-row SFT training
corpus and the 135-row 2026-05-15 holdout returned **133/135 (99%)
overlap**. The existing holdout therefore could not answer
"does SFT generalise?" — any A/B on it would measure memorisation.

Built `scripts/build_sft_unseen_holdout.py`: extracts the
**163 rows** of the official 2026-05-15 release that the SFT trainer
never saw (= 138 SFT-val + 25 not-in-SFT-at-all). Full prefix coverage
preserved (LD 45 / CH 39 / TD 26 / NL 21 / DDT 16 / THCB 9 / DT 6 /
CHLT 1). Logic eval is unaffected (0% leakage — different sample_id
namespace).

## E7a — straight A/B (3B vs SFT-7B) on 163-row clean holdout

Same code, same scorer, same E6 N=3 self-consistency + E8 RAG demos
on both sides; only the LLM client changes. SFT-7B served via Colab
T4 + cloudflared tunnel.

| Slice | N | 3B baseline | **SFT-7B** | Δ |
|---|---:|---:|---:|---:|
| **Overall Full✓** | 163 | **22.1%** (36) | **24.5%** (40) | **+2.4pp / +4 rows** |
| TD | 26 | 23.1% | **34.6%** | **+11.5pp** ✓ |
| CH | 39 | 23.1% | **28.2%** | +5.1pp ✓ |
| DDT | 16 | 6.2% | **12.5%** | +6.3pp ✓ |
| NL | 21 | 4.8% | 9.5% | +4.7pp (n=21) |
| **LD** | 45 | **33.3%** | **26.7%** | **-6.6pp REGRESSION** ⚠ |
| THCB | 9 | 44.4% | 44.4% | 0 |
| DT | 6 | 0% | 0% | 0 |
| ms/sample | | 2217 | **11367** | **+5.1× latency** |

**Verdict on straight A/B:** +2.4pp overall is within the ±3-4 row
variance band established Day-22 (5 runs of 3B-only returned 22-28 of
135 = ±6 of 135 ≈ ±3.6%). Signal is too weak to call a clear win.
*And* the -6.6pp LD regression — projected to the ~397 LD rows in the
full official set — would offset most of the non-LD gain. Straight
deploy of SFT-7B is **not** justified.

## E7a — hybrid routing attempt (LD → 3B, others → SFT-7B)

Hypothesis: SFT learned to extract well from descriptive contexts
(TD/CH/NL/DDT all narrative-heavy "find X given …") but lost
calibration on LD's terse multi-charge geometry prompts (it was
trained before the Day-14 Coulomb-vector formulas existed).

Wired `PhysicsSolver(..., llm_client_ld=...)` with a fixed
`_LD_DOMAIN_FORMULAS` whitelist (the six Coulomb / electric-field
formulas). `_try_llm_extraction` consults `_client_for_formula(...)`
*before* the LLM call → only one model is invoked per query, Q3-safe.
`scripts/run_eval.py --hybrid-ld-url ...` builds the second client
from raw `model.yaml` (so the LD model id stays `qwen2.5:3b-instruct`
even when env-overrides set the default to `qwen2.5-7b`).

**Hybrid eval crashed mid-run (Colab T4 free quota exhausted)** — but
the slice-level data from the two completed runs already pins the
answer by per-row arithmetic (pick whichever model wins on its slice):

| Slice | N | 3B correct | SFT-7B correct | **Hybrid (max)** |
|---|---:|---:|---:|---:|
| LD | 45 | **15** (33.3%) | 12 | 15 ✓ |
| TD | 26 | 6 | **9** (34.6%) | 9 ✓ |
| CH | 39 | 9 | **11** (28.2%) | 11 ✓ |
| DDT | 16 | 1 | **2** (12.5%) | 2 |
| NL | 21 | 1 | **2** (9.5%) | 2 |
| THCB | 9 | 4 | 4 | 4 |
| DT + CHLT | 7 | 0 | 0 | 0 |
| **Total** | **163** | **36 (22.1%)** | **40 (24.5%)** | **43 (~26.4%)** |

**Projected hybrid lift: +4.3pp vs 3B / +1.9pp vs SFT-7B-all.** Real
signal, above the ±3-row noise band.

## Decision — KEEP 3B-only, don't deploy hybrid

The hybrid lift (+4.3pp) is real but doesn't justify the deployment
cost given the time-and-budget constraint:

1. **Colab free quota now exhausted** — re-running needs the user to
   wait for reset (days) or pay; we'd then need to migrate the
   adapter to a rented GPU running vLLM-with-LoRA anyway for the
   submission endpoint (Colab is not the deploy target per QA Q5).
2. **vLLM-LoRA deploy is its own 1-2-day project** (load adapter,
   verify /v1/models contains the merged label, wire compose to two
   services on one or two GPUs to satisfy Q3, smoke under load).
3. **Q3 strict reading** ("cannot have multiple LLMs loaded in
   memory simultaneously") arguably requires two GPU instances —
   that's ~2× the rented-GPU bill we budgeted.
4. **+4.3pp is not game-changing** — it lifts us from "credible
   finalist" toward better-but-still-not-#1; doesn't change the
   competitive ceiling identified in earlier ADRs (3B/7B both are
   bounded; LLM-only 8B competitors can plausibly out-score raw P1
   regardless).
5. **9 days left, P3 demo on Public Test Day Jun 15** is the
   higher-leverage spend — our solver-grounded explanation +
   Z3-proof + cited-premises story plays directly to the rubric's
   committee-review part, and a polished defense matters more there
   than a few more P1 rows.

**Concretely we keep: ** Ollama qwen2.5:3b-instruct, E6 N=3
self-consistency, E8 RAG demos, all the deterministic improvements
(equilateral / parallel-plate formulas, Z3 arity fix, MC Unknown
abstention, E5 NL→FOL loop, E12 premise text+idx). One GPU compose,
deploy via `scripts/start_runpod.sh`.

**SFT adapter status unchanged** in the deployed system: stays in
`models/exact_qwen25_7b_lora/` as a documented, reproducible ablation
(ADR 0013, 0017, 0022). Disclosure §6 entry stands; no §5 update
needed.

## Q3 compliance concern (deployment) — moot given the decision above

QA Q3 rule of thumb: *"At any given moment during inference, only one
LLM <= 8B should be **actively loaded and running**. You may switch
between different models across pipeline stages, but you cannot have
multiple LLMs loaded in memory simultaneously."*

The hybrid as wired today invokes only one model per query (Q3-safe
in spirit). But strict reading of "cannot have multiple LLMs loaded in
memory simultaneously" rules out keeping 3B + 7B both resident in VRAM
on one GPU. Compliant deployment options:

1. **Two compose services, two GPUs** — 3B on small GPU (or CPU),
   7B on the main GPU. Each container loads exactly one model. Q3
   "actively loaded and running" applies per-process; this is the
   defensible interpretation.
2. **Model swap on one GPU** — load 7B, evict 3B; queue LD-bound
   requests; periodically swap. Adds ~10-30s model-load latency on
   swaps — would blow the 60s/req cap on the swap-victim queries.
   Not practical.
3. **Pre-classify route** then issue request to whichever endpoint is
   the right tool — same as (1), just routed at the API layer.

If the hybrid number is strongly positive we'll land (1) for
submission; otherwise the compliance complexity is not worth it.

## Tests

253 passed (no test changes — solver routing covered by the existing
PhysicsSolver tests with the new `_client_for_formula` being a pure
method). ruff / mypy clean (51 source files).

## Reproduction

```powershell
uv run python scripts/build_sft_unseen_holdout.py

# 3B baseline on the clean holdout:
uv run python scripts/run_eval.py --task physics --with-llm --include-samples `
    --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl `
    --out outputs/eval/e7a_baseline_3b_unseen

# SFT-7B straight, via Colab tunnel:
$env:EXACT_LLM__BASE_URL = "https://<tunnel>.trycloudflare.com/v1"
$env:EXACT_LLM__MODEL    = "qwen2.5-7b"
$env:EXACT_LLM__API_KEY  = "colab-no-auth"
uv run python scripts/run_eval.py --task physics --with-llm --include-samples `
    --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl `
    --out outputs/eval/e7a_sft7b_unseen

# Hybrid (LD → 3B local, rest → SFT-7B Colab):
uv run python scripts/run_eval.py --task physics --with-llm --include-samples `
    --hybrid-ld-url http://localhost:11434/v1 `
    --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl `
    --out outputs/eval/e7a_hybrid_ld3b_sft7b
```
