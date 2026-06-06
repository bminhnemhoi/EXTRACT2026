"""SAE feature handling: top-k selection, autointerp label cache, concept map.

The pure parts (top-k from a Python list, label-cache JSON I/O, mapping feature
ids -> lexical concepts) are stdlib-only and unit-tested. The tensor encode
(:func:`sae_encode`) is GPU-gated and lives behind a lazy torch import.
"""

from __future__ import annotations

import json
from pathlib import Path

from exact_agent.interp.concepts import tokenize_concepts


def top_k_indices(values: list[float], k: int) -> list[int]:
    """Indices of the ``k`` largest values (descending). Pure; no numpy."""
    if k <= 0 or not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return order[: min(k, len(values))]


def load_label_cache(path: str | Path) -> dict[int, str]:
    """Load an autointerp label cache: ``{feature_id: "natural language"}``.

    Accepts JSON with string or int keys. Missing file -> empty cache.
    """
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in raw.items()}


def save_label_cache(labels: dict[int, str], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def concepts_from_features(
    feature_ids: list[int],
    labels: dict[int, str],
) -> set[str]:
    """Map active SAE feature ids -> lexical concept tokens via their labels.

    Features with no cached label contribute nothing (they stay diagnostic-only
    in the record, but cannot align). This is honest: an uninterpreted feature
    is not evidence of any concept.
    """
    out: set[str] = set()
    for fid in feature_ids:
        label = labels.get(int(fid))
        if label:
            out |= tokenize_concepts(label)
    return out


def sae_encode(residual, sae, *, max_features: int | None = None):  # type: ignore[no-untyped-def]
    """Active SAE feature ids for a residual span (GPU path).

    Thin delegate to :meth:`exact_agent.interp.sae_loader.LoadedSAE.active_feature_ids`.

        residual: Tensor [seq, d_model]
        returns:  list[int]  (active feature ids, frequency-ordered)

    Map the ids to lexical concepts with :func:`concepts_from_features`.
    """
    return sae.active_feature_ids(residual, max_features=max_features)
