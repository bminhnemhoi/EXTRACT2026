# 0006 — Day-6 LLM fallback paths and SFT prep

Date: 2026-05-15
Status: Accepted

## Context

Day-3 / Day-4 measured the rule-only ceilings: physics 0.8% full-correct,
logic 35.9% correct. Both ceilings have the same root cause — the
deterministic extractor / chainer can't read NL paraphrases that don't
match its narrow grammar:

* Physics: ``"the voltage across its plates is 60 V"`` instead of
  ``U = 60 V``.
* Logic: question references ``Sophia`` but premises talk about
  ``a student``; surface Jaccard can't bridge the entity boundary.

Day-5 landed Z3, but it idles without a question→FOL translator.

## Decision

Land the **LLM fallback plumbing** on Day 6 — every part of the path
that does *not* require a GPU. Then ship a one-command SFT runner so
the Phase-4 training step is a single ``uv run`` away once a GPU host
is available.

What ships:

| Module | Role |
| --- | --- |
| `llm/vllm_client.py` | OpenAI-compatible HTTP client + `MockLLMClient` for tests |
| `llm/prompt_templates.py` | Jinja2 loader rooted at `configs/prompts/` |
| `llm/retry.py` | Tenacity policy: 3 attempts, exp backoff, retry on 5xx + network noise |
| `physics/llm_extractor.py` | JSON-keyed variable extraction against a `Formula` schema |
| `logic/llm_translator.py` | NL question → FOL claim, vocab-primed by `premises_FOL` |
| `train/prepare_sft.py` | Cleaned SFT JSONL → Qwen3 chat format (1765/196 split) |
| `train/run_sft_qwen.py` | Unsloth + TRL SFT runner; `--dry-run` works on CPU |
| `configs/training/sft_qwen3_8b.yaml` | Pointed at `data/processed/sft_chat_train.jsonl` |

Pipeline wiring:

* `PhysicsSolver(library, llm_client)` — when `_convert_inputs` raises
  `KeyError`, a single LLM extraction is attempted. The LLM returns
  `{symbol → {value, unit}}`; SymPy still does the math.
* `LogicPipeline(top_k, llm_client)` — when surface returns Unknown
  AND `premises_FOL` is set, the LLM is asked to produce `claim_FOL`
  if the request didn't supply one. Z3 then decides.

## What we explicitly did NOT do today

* Run training. ``run_sft_qwen.py --dry-run`` is the only thing that
  fires on CPU — the GPU path lazily imports `unsloth` / `trl`.
* Run the LLM in the API process. The pipeline accepts an optional
  client; default behavior (no client) is identical to Day-5.
* Re-measure the eval. The mock-driven integration test proves the
  path works end-to-end (logic Unknown → Yes via Z3 after LLM
  translation); a real measurement waits for a vLLM server with the
  SFT'd adapter loaded.

## Operational note

The pipeline is now safe to run with **any** OpenAI-compatible endpoint:

```python
from exact_agent.llm.vllm_client import VLLMClient
from exact_agent.physics.solver import PhysicsSolver

solver = PhysicsSolver(llm_client=VLLMClient())  # uses configs/model.yaml
```

Point ``configs/model.yaml::llm.vllm_base_url`` at any vLLM instance
(local Qwen3, Runpod, llama.cpp server) and the fallbacks light up
without touching code.

## Reproduction

```powershell
# Build the chat-format SFT split (CPU-only):
uv run python -m exact_agent.train.prepare_sft

# Validate the training command without GPU:
uv run python -m exact_agent.train.run_sft_qwen --dry-run

# (On a CUDA box) Actually train:
uv run python -m exact_agent.train.run_sft_qwen \
    --config configs/training/sft_qwen3_8b.yaml
```

## Consequences for Day 7+

1. **Self-correction loop** (Phase 5) becomes the next layer: solver
   verdict + LLM revision + solver re-verification, max 2 rounds.
2. **GRPO** stays optional — Day-7+ work, gated on time and the SFT
   adapter performing well enough to bother fine-tuning further.
3. The **vLLM endpoint URL** is the single deployment knob; the API
   container can stay slim (no GPU deps in `Dockerfile.api`).
