"""Solver-grounded interpretability harness (RESEARCH ONLY — NOT DEPLOYED).

This package implements P1–P2 of the VeriScope research charter
(``docs/notes/research_plan_solver_grounded_sae.md``):

* **P1 — capture:** run the LLM with forward hooks on the residual stream,
  decode Qwen-Scope SAE features, and record them alongside the solver's
  ground-truth derivation for each sample.
* **P2 — alignment (RQ1):** measure whether the SAE features active during
  extraction/translation align with the concepts the *symbolic solver*
  confirms were used — against a random baseline and a linear-probe baseline.

Design invariants (see charter §5):

1. The **solver is the ground truth / judge** — never the SAE, never the LLM.
2. This harness is an **offline, read-only** path. It does NOT touch the
   deployed solver-first pipeline and is never imported by ``api`` / ``agent``.
3. The deployed LLM client (:class:`exact_agent.llm.vllm_client.VLLMClient`)
   is HTTP-only and gives no activations; activation capture therefore loads
   the model locally via ``transformers`` (see :mod:`.sae_loader`). Heavy
   imports (torch/transformers/safetensors) live *inside* functions so the
   pure metric modules (:mod:`.alignment`, :mod:`.records`) import with the
   standard library alone and run without a GPU.

Module map:

* :mod:`.records`      — pure dataclasses + JSONL (no heavy deps).
* :mod:`.alignment`    — RQ1 metric: precision/recall/F1 + random baseline (pure).
* :mod:`.ground_truth` — solver result → ground-truth concept set.
* :mod:`.features`     — SAE top-k decode + autointerp label cache.
* :mod:`.sae_loader`   — load base model + Qwen-Scope SAE checkpoint (GPU).
* :mod:`.hooks`        — residual-stream forward hooks (GPU).
* :mod:`.steering`     — h' = h + α·d intervention + α-sweep (GPU; RQ2/P3).
"""

from __future__ import annotations

__all__ = ["__doc__"]
