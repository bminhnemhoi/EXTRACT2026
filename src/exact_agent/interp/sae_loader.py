"""Load the base model + a Qwen-Scope SAE checkpoint (GPU path — IMPLEMENTED).

Heavy imports (torch / transformers / huggingface_hub) are deferred into the
functions so this module imports on a CPU box without them installed; the pure
metric modules never touch this file.

VERIFIED checkpoint format (Qwen-Scope model card, 2026-06):
* The repo ships ``layer{N}.sae.pt`` — a raw ``torch`` state dict per layer
  (NO custom class, NO sae_lens). Keys/shapes:
    W_enc (d_sae, d_model)   b_enc (d_sae,)
    W_dec (d_model, d_sae)   b_dec (d_model,)
  For Qwen3-8B-Base: d_model=4096, d_sae=65536 (W64K), one file per layer 0–35.
* Encode (TopK): ``pre = residual @ W_enc.T + b_enc`` then take TopK=50 along
  the feature dim. (We read the active *indices*; RQ1 needs the active set, not
  exact magnitudes.)
* Hook point: residual-stream output of ``model.model.layers[N]`` -> [B,T,d_model].
* Deps: ``pip install torch transformers huggingface_hub`` (+ ``bitsandbytes``
  only if load_in_4bit). Shapes are INFERRED from the checkpoint, so switching
  backbone (8B↔2B↔1.7B) is just changing base_model + sae_repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SAEConfig:
    """Which backbone + SAE to analyze.

    Default = the research-headline pick (Qwen3-8B-Base, richest SAE). For fast
    iteration / small Colab GPUs use the 2B dev config (see PRESETS below).
    """

    base_model: str = "Qwen/Qwen3-8B-Base"
    sae_repo: str = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
    top_k: int = 50
    layers: tuple[int, ...] = (12, 18, 24)
    device: str = "cuda"
    dtype: str = "bfloat16"
    load_in_4bit: bool = False  # last resort on <=16GB GPUs; perturbs the SAE


# Convenient presets (resolve charter §10 decision #1). Switching is one line.
PRESETS: dict[str, SAEConfig] = {
    # Headline: best capability + richest SAE. bf16 ≈ 16–18GB → L4 24GB / A100.
    "qwen3-8b": SAEConfig(
        base_model="Qwen/Qwen3-8B-Base",
        sae_repo="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
    ),
    # Dev / eligibility-safe (<=8B, newest family). bf16 ≈ 5GB → fits any Colab.
    "qwen3.5-2b": SAEConfig(
        base_model="Qwen/Qwen3.5-2B-Base",
        sae_repo="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50",
        layers=(8, 12, 16),
    ),
    # Smallest safe fallback.
    "qwen3-1.7b": SAEConfig(
        base_model="Qwen/Qwen3-1.7B-Base",
        sae_repo="Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
        layers=(8, 12, 16),
    ),
}


@dataclass
class LoadedSAE:
    """A loaded per-layer Qwen-Scope SAE (tensors live on ``device``)."""

    W_enc: Any  # (d_sae, d_model)
    b_enc: Any  # (d_sae,)
    W_dec: Any  # (d_model, d_sae)
    b_dec: Any  # (d_model,)
    top_k: int
    device: str

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[1])

    def topk_indices(self, residual):  # type: ignore[no-untyped-def]
        """``residual`` [..., d_model] -> LongTensor [..., top_k] of active ids.

        Follows the model card verbatim (no b_dec centering). If reconstruction
        looks off on the real weights, try ``(residual - b_dec) @ W_enc.T``.
        """
        x = residual.to(self.W_enc.dtype)
        pre = x @ self.W_enc.T + self.b_enc
        return pre.topk(self.top_k, dim=-1).indices

    def active_feature_ids(
        self, residual, *, max_features: int | None = None, min_count: int = 1
    ) -> list[int]:
        """Aggregate active features across a token span [seq, d_model].

        Returns feature ids that land in the per-token TopK for at least
        ``min_count`` tokens, ordered by frequency (desc), capped to
        ``max_features`` if given.
        """
        import torch  # noqa: PLC0415

        idx = self.topk_indices(residual).reshape(-1)
        counts = torch.bincount(idx, minlength=self.d_sae)
        active = (counts >= min_count).nonzero(as_tuple=False).flatten()
        order = counts[active].argsort(descending=True)
        active = active[order]
        if max_features is not None:
            active = active[:max_features]
        return active.detach().cpu().tolist()

    def decoder_direction(self, feature_id: int):  # type: ignore[no-untyped-def]
        """Unit steering direction d for ``feature_id`` (column of W_dec)."""
        d = self.W_dec[:, int(feature_id)]
        return d / d.norm()


def load_model_and_tokenizer(cfg: SAEConfig):  # type: ignore[no-untyped-def]
    """Load the base model + tokenizer via transformers (eval mode)."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    kwargs: dict[str, Any] = {"device_map": cfg.device}
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        kwargs["torch_dtype"] = getattr(torch, cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    model.eval()
    return model, tok


def load_sae(cfg: SAEConfig, layer: int) -> LoadedSAE:
    """Download + load one per-layer Qwen-Scope SAE checkpoint."""
    import torch  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    path = hf_hub_download(cfg.sae_repo, filename=f"layer{layer}.sae.pt")
    sd = torch.load(path, map_location=cfg.device, weights_only=True)
    # Cast weights to float32 for a numerically stable encode against a bf16
    # residual stream (we upcast the residual to match at encode time).
    return LoadedSAE(
        W_enc=sd["W_enc"].float().to(cfg.device),
        b_enc=sd["b_enc"].float().to(cfg.device),
        W_dec=sd["W_dec"].float().to(cfg.device),
        b_dec=sd["b_dec"].float().to(cfg.device),
        top_k=cfg.top_k,
        device=cfg.device,
    )
