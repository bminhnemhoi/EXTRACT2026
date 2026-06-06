"""Residual-stream forward hooks (GPU path — SKELETON).

Captures the residual-stream activation at chosen layers during a forward pass,
so :func:`exact_agent.interp.features.sae_encode` can turn it into active SAE
latents. Heavy (torch) imports are deferred.

Hook point must match the SAE's training site: Qwen-Scope SAEs are trained on
the **residual stream** (``SAE-Res-...``), i.e. the input/output of each decoder
block, NOT the MLP/attention sublayer output. Register on the block module
(``model.model.layers[i]``) and read its output hidden state.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def capture_residual(model, layers):  # type: ignore[no-untyped-def]
    """TODO(P1): context manager yielding a dict ``{layer: Tensor[seq, d_model]}``.

    Sketch:

        store: dict[int, "Tensor"] = {}
        handles = []
        def mk_hook(i):
            def hook(_module, _inp, out):
                # out may be a tuple; the hidden state is out[0]
                hs = out[0] if isinstance(out, tuple) else out
                store[i] = hs.detach()[0]  # batch=1 -> [seq, d_model]
            return hook
        for i in layers:
            handles.append(model.model.layers[i].register_forward_hook(mk_hook(i)))
        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    Run the model once inside the ``with`` block, then read ``store``.
    """
    raise NotImplementedError(
        "GPU path — register residual-stream hooks (charter §9 P1)."
    )
    yield {}  # pragma: no cover  (keeps this a generator for @contextmanager)


def token_span_for_extraction(tokenizer, prompt: str, question: str):  # type: ignore[no-untyped-def]
    """TODO(P1): return the token index range over which to aggregate features.

    For physics we care about the activations while the model reads the
    *question* (where quantities/units live); for logic, while it reads the
    premises + question. Restricting aggregation to that span makes RQ1 align
    features with the actual extraction/translation work, not the boilerplate
    prompt scaffold. Default fallback: use the whole prompt span.
    """
    raise NotImplementedError("GPU path — compute the question token span (P1).")
