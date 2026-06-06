"""Activation steering h' = h + α·d (GPU path — SKELETON; RQ2 / charter P3).

Used to *suppress* off-topic feature directions (e.g. code-switching, literary
style) during extraction/translation, to test whether task-focusing reduces a
measurable error class (RQ2). The steering vector ``d`` is a Qwen-Scope SAE
**decoder** column for the target feature.

GUARDRAILS baked into the design (charter §5 anti-patterns):
* α-sweep with a coherence check — steering degrades output monotonically at
  high α (Rogue Scalpel 2509.22067); never ship an α that hurts coherence.
* The solver remains the judge — steering only changes what the LLM extracts;
  it must never let the LLM decide the answer.
* Discover good (feature, α) OFFLINE, then deploy a STATIC vector (charter §6).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SteerSpec:
    layer: int
    feature_id: int
    alpha: float  # negative suppresses the feature, positive amplifies


def steering_vector(sae, feature_id: int):  # type: ignore[no-untyped-def]
    """TODO(P3): return the unit decoder direction d for ``feature_id``.

        d = sae.decoder.weight[:, feature_id]   # [d_model]
        return d / d.norm()
    """
    raise NotImplementedError("GPU path — read SAE decoder column (charter §9 P3).")


def make_steering_hook(specs: list[SteerSpec], sae_by_layer):  # type: ignore[no-untyped-def]
    """TODO(P3): forward hook applying h' = h + Σ α·d at the spec'd layers.

    Apply during generation (inference-time, no weight update). Aggregate
    multiple specs additively. Pair with an α-sweep driver that re-runs eval
    per α and records P1/P2 + a coherence proxy (e.g. repetition rate,
    perplexity, or judge score) so the sweep can pick the safe band.
    """
    raise NotImplementedError("GPU path — build the steering hook (charter §9 P3).")
