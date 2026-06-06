"""Load the base model + a Qwen-Scope SAE checkpoint (GPU path — SKELETON).

Heavy imports (torch / transformers / safetensors) are deliberately deferred
into the functions so this module imports without a GPU. Fill in the TODOs on
a GPU box (charter §9 P0/P1).

VERIFIED FACTS the implementation must respect (web-verified 2026-06):
* Qwen-Scope SAEs target the **BASE** model (e.g. ``Qwen/Qwen3-8B-Base``); the
  only instruct-trained SAE is for Qwen3.5-27B (>8B, ineligible). So for an
  exact analysis run the agent on Instruct but capture activations on the
  matching **Base** model — or accept base-SAE-on-instruct as approximate
  (charter §10 decision 2).
* Checkpoint ``Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50``: residual-stream TopK
  SAE, d_sae = 65,536 (W64K), TopK = 50, 16x expansion, one SAE per layer
  0–35, d_model = 4096. License = custom "qwen" (NOT Apache-2.0) — confirm use.
* Eligible <=8B backbones with a Qwen-Scope SAE: Qwen3-1.7B, Qwen3.5-2B,
  Qwen3-8B (8.2B total / 6.95B non-embedding — borderline; see charter §7/§10).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SAEConfig:
    """Which backbone + SAE to analyze. Defaults = the research-primary pick."""

    base_model: str = "Qwen/Qwen3-8B-Base"
    sae_repo: str = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
    d_model: int = 4096
    d_sae: int = 65_536
    top_k: int = 50
    # Layers to hook. Mid-stack layers tend to be most semantically rich;
    # start with a few and expand. Full range for this SAE is 0..35.
    layers: tuple[int, ...] = (12, 18, 24)
    device: str = "cuda"
    dtype: str = "bfloat16"


def load_model_and_tokenizer(cfg: SAEConfig):  # type: ignore[no-untyped-def]
    """TODO(P0): load the base model + tokenizer via transformers.

        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=..., device_map=cfg.device,
            output_hidden_states=False,  # we hook the residual stream instead
        )
    """
    raise NotImplementedError(
        "GPU path — load transformers model on a GPU box (charter §9 P0)."
    )


def load_sae(cfg: SAEConfig, layer: int):  # type: ignore[no-untyped-def]
    """TODO(P0): load one per-layer Qwen-Scope SAE checkpoint.

    The Qwen-Scope repo ships one ``layer{N}.sae.pt`` per layer. Load with the
    Qwen-Scope reference loader (or sae_lens if a compatible config exists) and
    return a module exposing ``encode(residual) -> latents`` and a ``decoder``
    weight matrix (for steering directions, :mod:`.steering`). Verify
    d_sae/top_k match :class:`SAEConfig` after loading.
    """
    raise NotImplementedError(
        "GPU path — load the per-layer Qwen-Scope SAE (charter §9 P0). "
        f"repo={cfg.sae_repo!r} layer={layer}."
    )
