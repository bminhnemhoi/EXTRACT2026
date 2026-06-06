#!/usr/bin/env python3
"""Build an autointerp label cache for the SAE features that fire (GPU).

For RQ1's lexical alignment (run_alignment.py), each active feature needs a
short natural-language label. This script: (1) reads a captured records file to
find the features that actually fired on the eval set (a tractable few hundred,
not all 65k); (2) re-runs the model over the corpus to find each feature's
top-activating samples; (3) asks a LOCAL open LLM to name the shared concept;
(4) writes ``labels.json`` (merged with any existing cache).

The labeler is a LOCAL open instruct model (default Qwen2.5-7B-Instruct) — no
external API, reproducible, rule-friendly (labels are analysis annotations, not
part of the submitted system). Override with --labeler-model.

Deps:  pip install torch transformers accelerate huggingface_hub

Usage:
    python scripts/interp/build_labels.py --preset qwen3.5-2b --task physics \
      --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \
      --records outputs/interp/records_physics_l12.jsonl --layer 12 \
      --out outputs/interp/labels.json --max-features 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from exact_agent.interp.autointerp import (  # noqa: E402
    Context,
    build_label_prompt,
    parse_label,
    select_top_contexts,
)
from exact_agent.interp.features import load_label_cache, save_label_cache  # noqa: E402
from exact_agent.interp.records import read_records  # noqa: E402


def _load_split(path: Path, limit: int | None) -> list[dict]:
    import json  # noqa: PLC0415

    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _target_features(records_path: Path, max_features: int | None) -> list[int]:
    """Union of fired features (frequency-ordered), capped to max_features."""
    counts: dict[int, int] = {}
    with records_path.open(encoding="utf-8") as f:
        for rec in read_records(f):
            for fid in rec.active_feature_ids:
                counts[fid] = counts.get(fid, 0) + 1
    ordered = sorted(counts, key=lambda k: counts[k], reverse=True)
    return ordered[:max_features] if max_features else ordered


def _build_prompt(task: str, row: dict) -> str:
    q = str(row.get("question", ""))
    if task == "logic":
        prem = list(row.get("premises-NL") or row.get("premises_NL") or [])
        if prem:
            return "\n".join(prem) + "\n" + q
    return q


def _make_labeler(model_id: str, device: str):  # type: ignore[no-untyped-def]
    """Return a callable(prompt)->str backed by a local instruct model."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    def _label(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=24, do_sample=False)
        return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    return _label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["physics", "logic"], required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True, help="capture output (which features fired)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True, help="labels.json (merged if exists)")
    ap.add_argument("--preset", choices=["qwen3-8b", "qwen3.5-2b", "qwen3-1.7b"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--sae-repo", default=None)
    ap.add_argument("--labeler-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-features", type=int, default=400)
    ap.add_argument("--top-contexts", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch  # noqa: PLC0415

    from exact_agent.interp.hooks import capture_residual  # noqa: PLC0415
    from exact_agent.interp.sae_loader import (  # noqa: PLC0415
        PRESETS,
        SAEConfig,
        load_model_and_tokenizer,
        load_sae,
    )

    base = PRESETS[args.preset] if args.preset else SAEConfig()
    cfg = SAEConfig(
        base_model=args.model or base.base_model,
        sae_repo=args.sae_repo or base.sae_repo,
        layers=(args.layer,), device=args.device,
    )
    targets = _target_features(args.records, args.max_features)
    print(f"labeling {len(targets)} features (layer {args.layer}) with {args.labeler_model}")

    model, tok = load_model_and_tokenizer(cfg)
    sae = load_sae(cfg, args.layer)
    rows = _load_split(args.split, args.limit)

    # Per-feature high-activation contexts (sample-level v1; token-window = TODO).
    hits: dict[int, list[Context]] = {t: [] for t in targets}
    for row in rows:
        prompt = _build_prompt(args.task, row)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad(), capture_residual(model, [args.layer]) as store:
            model(**enc)
        residual = store[args.layer][0]                       # [seq, d_model]
        pre = sae.pre_acts_subset(residual, targets)          # [seq, n_targets]
        peak = pre.max(dim=0).values                          # [n_targets]
        sid = str(row.get("id", row.get("sample_id", "")))
        for k, fid in enumerate(targets):
            hits[fid].append(Context(text=prompt, activation=float(peak[k]), sample_id=sid))

    labeler = _make_labeler(args.labeler_model, args.device)
    labels = load_label_cache(args.out) if args.out.exists() else {}
    for n, fid in enumerate(targets):
        ctxs = select_top_contexts(hits[fid], top_n=args.top_contexts)
        label = parse_label(labeler(build_label_prompt(ctxs)))
        if label:
            labels[fid] = label
        if (n + 1) % 25 == 0:
            print(f"  labeled {n + 1}/{len(targets)} ...")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_label_cache(labels, args.out)
    print(f"wrote {len(labels)} labels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
