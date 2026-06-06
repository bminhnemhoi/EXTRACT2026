"""Activation steering h' = h + α·d (GPU path — IMPLEMENTED; RQ2 / charter P3).

Suppress (α<0) or amplify (α>0) Qwen-Scope SAE feature directions during a
forward pass to test task-focusing (RQ2). The direction is the SAE decoder
column for the target feature (:meth:`LoadedSAE.decoder_direction`).

GUARDRAILS (charter §5): solver stays the judge; α-sweep with a coherence
check (steering degrades at high α — Rogue Scalpel 2509.22067); discover
(feature, α) OFFLINE then deploy a STATIC vector.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class SteerSpec:
    layer: int
    feature_id: int
    alpha: float  # negative suppresses, positive amplifies


def steering_vector(sae, feature_id: int):  # type: ignore[no-untyped-def]
    """Unit decoder direction d for ``feature_id``."""
    return sae.decoder_direction(feature_id)


@contextmanager
def steer(model, specs, sae_by_layer):  # type: ignore[no-untyped-def]
    """Apply h' = h + Σ α·d at the spec'd layers for the duration of the block.

    ``sae_by_layer``: ``{layer: LoadedSAE}``. Use during generation; pair with
    an α-sweep driver that re-runs eval per α and logs P1/P2 + a coherence
    proxy so the sweep can pick the safe band.
    """
    by_layer: dict[int, list[SteerSpec]] = {}
    for s in specs:
        by_layer.setdefault(s.layer, []).append(s)

    handles = []

    def _mk(layer: int):
        sae = sae_by_layer[layer]
        vecs = [
            (s.alpha, steering_vector(sae, s.feature_id).to(sae.device))
            for s in by_layer[layer]
        ]

        def _hook(_module, _inp, out):  # type: ignore[no-untyped-def]
            hidden = out[0] if isinstance(out, tuple) else out
            for alpha, d in vecs:
                hidden = hidden + alpha * d.to(hidden.dtype)
            if isinstance(out, tuple):
                return (hidden, *out[1:])
            return hidden
        return _hook

    for layer in by_layer:
        handles.append(model.model.layers[layer].register_forward_hook(_mk(layer)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
