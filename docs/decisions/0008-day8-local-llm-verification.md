# 0008 — Day-8 local LLM verification (Ollama)

Date: 2026-05-15
Status: Accepted

## Context

Phases 1–5 shipped with `MockLLMClient` only — the LLM fallback paths and
self-correction loop had never run against a real model. Before spending
rented-GPU budget on SFT, we verified the whole stack end-to-end locally
on the user's machine (RTX 4050 6 GB, 16 GB RAM, Ollama 0.24).

## Decisions

### 1. Local model = Qwen2.5-3B-Instruct, NOT Qwen3

`qwen3:4b` via Ollama **forces a reasoning channel**: the answer lands in
`message.reasoning` and `message.content` stays empty until thinking
finishes. None of `/no_think` (space or newline), `think:false`, or
`chat_template_kwargs:{enable_thinking:false}` suppressed it (all four
verified Day 8). For a utility extractor — where the *solver* does the
reasoning and we want a terse JSON envelope — this is fatal.

`qwen2.5:3b-instruct` returns clean direct JSON, no reasoning channel,
fast (~0.5–1.5 s/call on the 6 GB GPU). It is open-source and ≤8B, so it
satisfies the competition rule. The production SFT target on a rented
GPU is re-opened: **evaluate Qwen2.5-7B-Instruct alongside Qwen3-8B**;
the forced-thinking behavior makes Qwen2.5-Instruct the safer backbone
for the extract/translate role even at 8B.

### 2. `mode: chat` + `disable_thinking` config knobs

`LLMConfig` gained `mode` (chat|completion) and `disable_thinking`.
`VLLMClient.complete()` routes through `/v1/chat/completions` when
`mode==chat` (instruct models need the chat template) and prepends
`/no_think` when `disable_thinking` (no-op for Qwen2.5, kept for any
future Qwen3 path). `_extract_chat_text` falls back to `reasoning` when
`content` is empty so a thinking model still yields *something* the JSON
parser can salvage. Retry is applied once at the public boundary;
`_do_chat`/`_do_completion` are the undecorated impls.

### 3. Self-correction default stays OFF

Verified working (live API smoke 5/5 with it ON; penguins/birds went
`Unknown` → `No`). Committed default is `enabled: false` to keep CI
deterministic and the API LLM-free unless an endpoint is intended.
Production flips it on. The Orchestrator auto-builds a `VLLMClient`
from `configs/model.yaml` when the toggle is on.

## Measured numbers (local, qwen2.5:3b-instruct via Ollama)

Same 10% holdout, seed 42, same harness as ADR 0003/0004.

### Physics (133 rows)

| | Baseline (rule) | + LLM (Day 8) |
|---|---:|---:|
| Solved (formula ran) | 3.0% | **39.1%** |
| Numeric✓ (reported) | 0.8% | 3.8% |
| Unit✓ | 46.6% | 46.6% |
| `missing_input` failures | 83 | 26 |

The LLM extractor works: `missing_input` collapsed 83 → 26, and
`coulomb_force`/`charge_from_capacitance`/`electric_field` reach
85–100 % "solved". The deterministic solver still does every computation.

### Logic (64 rows)

| | Baseline (rule) | + LLM (Day 8) |
|---|---:|---:|
| Correct | 35.9% | **37.5%** |
| `yes_no_unknown` | 30.0% | 33.3% |

Modest. Bottleneck is exactly what ADR 0005 predicted: only ~47 % of
FOL parses, and a 3B model translating NL→FOL is weak. The SFT'd model
+ a multi-variable FOL parser are the real logic unlocks.

## Known measurement defect — the eval scorer is unit-naive

`eval/metrics.py::numeric_match` compares the solver's **SI float**
against the gold's **prefixed string** without unit normalization.
Evidence from `outputs/eval/day8_llm/physics_llm_report.json`:

| id | predicted | gold | physically correct? | scored |
|---|---|---|---|---|
| TD039 | 5.987e-10 C | 0.6 nC | ✅ (0.599 nC) | ❌ False |
| TD060 | 7.27e-10 C | 0.73 nC | ✅ (0.727 nC) | ❌ False |
| TD181 | 1.454e-9 C | 1.46 nC | ✅ (1.454 nC) | ❌ False |

`charge_from_capacitance` shows 100 % solved / 0 % numeric purely
because of this. **True physics correctness is materially above the
reported 3.8 %.** Fixing `numeric_match` to pint-normalize both sides
to SI before `math.isclose` is the highest-leverage Day-9 task — it
could lift the reported number to ~15–25 % without touching the solver.
(Deliberately out of scope today: ADR 0003 froze the harness as the
measurement source of truth; changing it is its own decision/commit.)

## Consequences

1. **Day 9 first task**: unit-aware `numeric_match`; re-run eval; the
   real Day-8 uplift becomes visible.
2. LD/DT remain capped by single-pair formulas (vector composition is a
   Phase-2 item, ADR 0003) — not an LLM problem.
3. Rented-GPU SFT proceeds with the plan's `run_sft_qwen.py`; backbone
   choice revisited (Qwen2.5-7B-Instruct vs Qwen3-8B) given the
   forced-thinking finding.
4. The local path is now a permanent dev affordance:
   `uv run python scripts/run_eval.py --task <t> --with-llm [--limit N]`.

## Reproduction

```powershell
ollama pull qwen2.5:3b-instruct
uv run python scripts/run_eval.py --task physics --with-llm --limit 15   # sanity
uv run python scripts/run_eval.py --task physics --with-llm              # full
uv run python scripts/run_eval.py --task logic   --with-llm
# live self-correction: set app.yaml self_correction.enabled=true, then
uv run uvicorn exact_agent.api.app:app --port 8000 ; uv run python scripts/smoke_api.py
```
