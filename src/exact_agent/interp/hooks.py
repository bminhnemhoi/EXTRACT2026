"""Residual-stream forward hooks (GPU path — IMPLEMENTED).

Captures the residual-stream output of chosen decoder blocks during a forward
pass. Matches the Qwen-Scope SAE training site (``SAE-Res-...`` = residual
stream), i.e. the output of ``model.model.layers[i]``, NOT a sublayer. torch is
imported lazily inside the function so this module loads on CPU boxes.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def capture_residual(model, layers):  # type: ignore[no-untyped-def]
    """Context manager yielding ``{layer: Tensor[batch, seq, d_model]}``.

    Run the model once inside the ``with`` block, then read the dict::

        with capture_residual(model, [18]) as store:
            model(**enc)
        residual = store[18][0]   # batch 0 -> [seq, d_model]
    """
    store: dict[int, object] = {}
    handles = []

    def _mk(i: int):
        def _hook(_module, _inp, out):  # type: ignore[no-untyped-def]
            hidden = out[0] if isinstance(out, tuple) else out
            store[i] = hidden.detach()
        return _hook

    for i in layers:
        handles.append(model.model.layers[i].register_forward_hook(_mk(i)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def question_token_span(tokenizer, prompt: str):  # type: ignore[no-untyped-def]
    """Token span to aggregate features over. v1 = all prompt tokens.

    Refinement (charter §8): restrict to the question/premise tokens (drop any
    fixed scaffold) so RQ1 aligns features with the actual extraction work.
    Returns ``(start, end)`` half-open indices.
    """
    n = len(tokenizer(prompt)["input_ids"])
    return 0, n
