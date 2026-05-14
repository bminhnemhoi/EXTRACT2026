# 0002 — Qwen3-8B as the LLM backbone

Date: 2026-05-15
Status: Accepted

## Context

Rules cap us at open-source ≤8B parameters. Candidates considered:

- **Qwen3-8B** — strongest reasoning + multilingual support, agentic profile.
- **Qwen2.5-7B-Instruct** — mature, well-supported by Unsloth.
- **Llama-3.1-8B-Instruct** — good general baseline.
- **Qwen3-4B** — fallback for ≤16 GB VRAM hosts.

## Decision

Primary: `Qwen/Qwen3-8B`. Fallback: `Qwen/Qwen3-4B` if VRAM is tight or
inference latency exceeds 20 s/request.

## Rationale

- Qwen3 family ships strong step-by-step reasoning out of the box, which
  matters for the P2 explanation score even before SFT.
- The Qwen-Scope SAE checkpoints (interpretability tooling we may use in
  Phase 7) target Qwen3 specifically.
- Unsloth provides ready notebooks for Qwen3 QLoRA → fast iteration.

## Consequences

- `configs/model.yaml` hard-codes the backbone; only the model id field
  changes if we swap.
- Inference target host: a single 24 GB GPU (RTX 3090/4090 / A10 / L4)
  serving QLoRA-loaded weights via vLLM with `--enable-lora`.
