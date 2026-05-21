# 0025 — Day-23 hybrid (F1+F2 + SFT-7B) CONFIRMED +8.0pp; deploy decision pending

Date: 2026-05-21
Status: Accepted (measurement); deploy decision pending user budget

## Context

Day-23 closed F1+F2 (commit 439565e, ADR 0023) at **27.6%** physics
on the 163-row SFT-unseen clean holdout (+5.5pp vs 22.1% baseline,
pure deterministic). ADR 0022's E7 hybrid projection (26.4%) was
computed on the pre-F1+F2 numbers; the open question after F1+F2 was:
*does SFT-7B still contribute on top of the new formula coverage, or
do the formulas eat its delta?*

User restarted Colab on a fresh free account, paste-shared a fresh
tunnel. This ADR records the **stacking eval** answer.

## Eval — `f1f2_hybrid_ld3b_sft7b`

Same 163-row SFT-unseen holdout, same scorer, same E6 N=3 + E8 RAG +
F1+F2 formulas + F2 routing guard. Two LLM clients:

* **Default** (non-LD formulas): SFT-7B via Colab T4 tunnel
* **LD-domain** (Coulomb / electric-field force): qwen2.5:3b-instruct local

| Slice | N | 3B raw | F1+F2 (3B) | **Hybrid F1+F2 + SFT-7B** | Δ vs F1+F2 |
|---|---:|---:|---:|---:|---:|
| **Overall Full✓** | 163 | 22.1% (36) | 27.6% (45) | **30.1%** (49) | **+2.5pp / +4 rows** |
| DDT | 16 | 6.2% (1) | 31.2% (5) | **37.5%** (6) | +6.3pp |
| TD | 26 | 23.1% (6) | 23.1% (6) | **34.6%** (9) | **+11.5pp** |
| LD | 45 | 33.3% (15) | 33.3% (15) | 33.3% (15) | 0 (routes to 3B) |
| CH | 39 | 23.1% (9) | 30.8% (12) | 30.8% (12) | 0 |
| NL | 21 | 4.8% | 14.3% | 14.3% | 0 |
| Others | unchanged | unchanged | unchanged | unchanged | 0 |
| ms/sample | | 2217 | 2054 | **6360 (3.1×)** | within 60s/req cap |

**Stacking confirmed.** SFT-7B contributes mainly on TD (+11.5pp) and
DDT (+6.3pp on top of the new F1 formulas) — the same narrative-heavy
slices where it won in E7a. F1+F2 cleared the CH/NL gaps the 3B was
losing to deterministic formula misses; SFT-7B then picks up the
hard-extraction rows the regex still misses.

Total progress vs the honest Day-21 baseline (when we discovered the
old 28.6% was wrong / leaked): **22.1% → 30.1% (+8.0pp)**, of which
+5.5pp is pure deterministic (no GPU) and +2.5pp is the SFT-7B
extractor on top.

## Deploy decision — TRUE judgment call

| Path | Physics | Cost / complexity |
|---|---:|---|
| **A. KEEP 3B-only** | **27.6%** (F1+F2) | 1 GPU (~$50-100/2wk); compose ready; Q3 trivially satisfied; Ollama works as-is |
| **B. Deploy hybrid** | **30.1%** (F1+F2 + SFT-7B) | 2 GPUs (~$200-300/2wk) OR single 24GB GPU with vLLM-LoRA + Ollama 3B in separate processes; Q3 strict reading needs justification; vLLM-LoRA stack setup ~1-2d; tunnel reliability replaced by proper rented serving |

Path B is **+2.5pp at +$100-200 budget + 1-2 days extra setup**. With
9 days to deadline, both are feasible. The +2.5pp pushes Top-5 odds
from ~25-30% to ~30-40% — material but not game-changing.

## Updated prize odds (calibrated honest)

| Outcome | Path A (27.6%) | Path B (30.1%) |
|---|---:|---:|
| Top-10 finalist | 65-75% | 70-80% |
| Top-5 (cash + LNCS paper) | 25-30% | **30-40%** |
| Top-1 ("giải nhất") | 7-10% | 10-15% |

## Honest caveats

1. **Single-run measurement** — Day-22 established 3B variance ±3
   rows on this 163-row holdout; could be a re-eval Path B is 28-32%
   in the noise band. The +4-row delta over F1+F2 is at the edge of
   the noise band. One additional eval would tighten the band but
   would cost another Colab session.
2. **Test set distribution unknown.** Our 163 rows are a random
   slice of the official release excluding what SFT saw; BTC's
   hidden test set may have different prefix distribution. TD 34.6%
   and DDT 37.5% are on small slices (26 / 16 rows respectively); a
   different distribution could yield different numbers.
3. **Hybrid serving complexity is real.** vLLM-LoRA on a rented
   GPU plus Ollama 3B is not the same as the docker-compose-up-and-go
   we have for Path A. Allow 1-2 days of integration debugging.
4. **Latency 3× higher** (2.0 → 6.4s/req mean). Still well under the
   60s cap, but BTC's evaluation throughput at our endpoint will be
   noticeably slower.

## Reproduction

```powershell
$env:EXACT_LLM__BASE_URL = "<tunnel>/v1"
$env:EXACT_LLM__MODEL    = "qwen2.5-7b"
$env:EXACT_LLM__API_KEY  = "colab-no-auth"
uv run python scripts/run_eval.py --task physics --with-llm --include-samples `
    --hybrid-ld-url http://localhost:11434/v1 `
    --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl `
    --out outputs/eval/f1f2_hybrid_ld3b_sft7b
```
