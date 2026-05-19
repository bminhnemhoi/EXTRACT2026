# 0013 — Day-13 SFT impact: marginal & mixed (system is solver-bound)

Date: 2026-05-20
Status: Accepted

## Context

Phase-4 SFT finally ran: QLoRA Qwen2.5-7B-Instruct on free Colab T4
(1765 train rows from `prepare_sft.py`), adapter served back through a
cloudflared tunnel, measured with the **frozen Day-9/11/12 scorer**
against the same 10% holdout used since ADR 0003.

Qualitative gate first (16 held-out rows, `inspect_sft_outputs.py`):
valid JSON 13/16 (the 3 misses were max-token CoT truncation, not
malformed generation), **0 prose rambling, 0 premise-ref
hallucination**, cot/premises keys correct. SFT did exactly its
ADR-0006 job — clean envelopes — and the model's own arithmetic stays
unreliable (case CH169: model wrote `P=U²/R=112²/75≈168 W`, gold
167.25 — right method, sloppy mental math; our solver computes it
exactly). Format gate: passed.

## Measured impact (quantitative, same holdout & scorer)

| | baseline (qwen2.5:3b) | SFT'd Qwen2.5-7B |
|---|---:|---:|
| **Physics Full✓** | 24.8% | **26.3%** (+1.5pp) |
| physics `missing_input` | 26 | **13** (halved) |
| physics solved | 56.4% | 58.6% |
| **Logic correct** | 37.5% (24/64) | **35.9%** (23/64) |
| ms / sample (via tunnel) | ~640 | ~3200 (~5×) |

Logic sample audit vs baseline: 2 regress, 1 gain → **net −1 row,
inside the ±1–2-row LLM-nondeterminism band** documented in ADR 0011.
Logic is therefore **flat**, not a real regression — and not the
improvement we hoped NL→FOL SFT would bring.

## Findings

1. **The system is solver-bound, not LLM-bound.** The SFT'd 7B is a
   structurally better extractor — `missing_input` halved (26→13) —
   yet end-to-end physics moved only +1.5pp (~2 rows). Most recovered
   extractions still hit **formula-coverage / solver-modelling gaps**
   (LD/DT vector composition, DDT conceptual, multi-step AC). More LLM
   quality cannot fix a missing formula.
2. **Envelope-SFT specialized the model away from the bare NL→FOL
   utility task.** The model was trained to emit the answer/explanation
   JSON envelope; asked instead to translate a question to one FOL
   line (`nl_to_fol.j2`), the SFT'd 7B is no better (slightly worse)
   than the untrained 3B. One model, several micro-tasks — optimizing
   one can dent another.
3. **SFT cost is real**: ~5× latency (7B vs 3B; tunnel adds more) plus
   GPU-serving fragility (4 notebook-debug cycles to get training to
   run at all).

## Decision

**The deterministic solver + frozen scorer (Days 1–12) is the product;
SFT is an optional, marginal add-on — not the critical path.**

- Committed default stays the small local LLM
  (`configs/model.yaml` reverted to `qwen2.5:3b-instruct` /
  `localhost:11434`; the tunnel URL was never committed).
- The SFT adapter (`models/exact_qwen25_7b_lora/`, gitignored) is a
  kept artifact. Deploy it for the submission endpoint **only** if a
  dedicated GPU is available and the +~1.5pp physics is judged worth
  the latency; otherwise submit the simpler, faster, robust
  deterministic pipeline.
- Do **not** spend the remaining runway on another SFT cycle. The
  highest-value remaining work is solver-side (formula coverage) and
  submission packaging — both GPU-free.

This is precisely the call the baseline-first strategy (ADR 0001) was
designed to enable: we are deciding SFT's role from measured data, not
from a bet made before any solver existed.

## Consequences for the last 2 days

1. Submission writeup leads with the solver-first / neuro-symbolic
   architecture (reproducible, P3-strong, no external data) — the
   defensible core. SFT mentioned as an ablation: format-stabilizing,
   accuracy-neutral, omitted from the deployed endpoint for latency.
2. Next work: package the submission (endpoint Dockerfile + 1-page
   PDF), optionally close a few more `no_formula_matched` if time.
3. Reference checkpoints: `outputs/eval/day13_sft/` (SFT) vs
   `outputs/eval/day12_llm/` (baseline) — both gitignored.

## Reproduction

```powershell
# serve adapter on Colab (notebooks/serve_sft_colab.ipynb) -> tunnel URL
# point configs/model.yaml at it (do not commit), then:
uv run python scripts/inspect_sft_outputs.py --url <tunnel>/v1 --n 16
uv run python scripts/run_eval.py --task physics --with-llm --out outputs/eval/day13_sft
uv run python scripts/run_eval.py --task logic   --with-llm --out outputs/eval/day13_sft
# revert configs/model.yaml afterwards
```
