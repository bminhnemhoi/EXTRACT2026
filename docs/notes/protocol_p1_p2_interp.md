# Protocol — P1 (capture) & P2 (alignment / RQ1)

> Companion to the research charter ([research_plan_solver_grounded_sae.md](research_plan_solver_grounded_sae.md)).
> Code lives in `src/exact_agent/interp/` (library, NOT deployed) and
> `scripts/interp/` (CLIs). Pure metric modules run anywhere; activation
> capture needs a GPU.

## 0. What's already verified (no GPU)

The pure path is unit-tested and runnable today:

```bash
export PYTHONPATH="$PWD/src"
python3 tests/unit/test_interp_alignment.py      # 12/12 pass
python3 scripts/interp/run_alignment.py --selfcheck
```

`tests/unit/test_interp_alignment.py` also runs under the normal suite
(`pytest -q`) once the env is set up.

## 1. Environment

* **Ground truth + metrics (CPU):** the repo's normal env — `uv sync --all-extras`
  (Python 3.11). Needs sympy/pint/z3 for the solver. No GPU.
* **Activation capture (GPU):** `pip install torch transformers huggingface_hub`
  (+ `bitsandbytes` only for `--load-in-4bit`). No `sae_lens` — Qwen-Scope SAEs
  are raw torch dicts (`W_enc/b_enc/W_dec/b_dec`), loaded by `interp/sae_loader.py`.

**Model choice (resolves charter §10 decision #1) + Colab VRAM:**

| Preset | Model | SAE | bf16 VRAM | Use |
|---|---|---|---|---|
| `qwen3.5-2b` | Qwen3.5-2B-Base | W32K | ~5 GB | **dev / de-risk** — fits any Colab GPU (T4/L4/A100); eligibility-safe (≤8B) |
| `qwen3-8b` | Qwen3-8B-Base | W64K (richest) | ~16–18 GB | **headline** — needs L4 24 GB / A100 40 GB; on a 16 GB T4 add `--load-in-4bit` (perturbs the residual stream → SAE less faithful; prefer bf16 on ≥24 GB) |
| `qwen3-1.7b` | Qwen3-1.7B-Base | W32K | ~4 GB | smallest fallback |

Recommended path: **debug the full pipeline end-to-end on `qwen3.5-2b`, then
produce headline numbers on `qwen3-8b`.** Both use the **Base** model (the SAE
is Base-trained); the agent's Instruct-vs-Base question (§2 below) does not
affect this RQ1 capture, which runs on Base for an exact SAE match.

## 2. Decisions to lock before running (charter §10)

1. **Model:** primary `Qwen/Qwen3-8B-Base` (richest SAE: W64K, layers 0–35,
   TopK-50). Eligibility-safe fallback `Qwen/Qwen3.5-2B-Base`. Confirm the ≤8B
   param convention with organizers (Qwen3-8B = 8.2B total / 6.95B non-emb).
2. **Base vs Instruct:** SAEs are trained on the **Base** model. Either run the
   *agent* on Base too (exact SAE, weaker instruction-following — gap-fill with
   few-shot), or run the agent on Instruct and capture on Base for the analysis
   pass (report it as approximate). Default = capture on Base.
3. **License:** Qwen-Scope SAE = custom "qwen" license; confirm competition use.

## 3. P1 — capture

**3a. CPU dry-run (validate the data path + ground-truth coverage first):**

```bash
export PYTHONPATH="$PWD/src"
python scripts/interp/capture_activations.py --task physics \
  --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \
  --layer 18 --out outputs/interp/records_physics_layer18.jsonl \
  --ground-truth-only
```

Produces records with `gold_concepts` filled and `active_*` empty. Check the
printed `solver_ok on N/total` — that is your usable-sample coverage. Repeat for
`--task logic --split .../logic_eval.jsonl`.

**3b. Full run (GPU):** drop `--ground-truth-only` and pass `--preset`. This
loads the model + SAE, hooks the residual stream at `--layer`, decodes the
active TopK features, maps them to concepts via `--labels`, and fills `active_*`:

```bash
python scripts/interp/capture_activations.py --task physics --preset qwen3.5-2b \
  --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \
  --layer 12 --out outputs/interp/records_physics_l12.jsonl \
  --labels outputs/interp/labels.json
```

The GPU path is **implemented** against the verified Qwen-Scope API
(`interp/{sae_loader,hooks,features,steering}.py`) — no code-filling needed.
Sweep `--layer` over a few mid-stack layers (8B: 12/18/24; 2B: 8/12/16).

**Autointerp labels (`--labels`):** a JSON `{feature_id: "natural language"}`.
Build it once per (model, layer) by feeding each feature's top-activating
contexts to an LLM (AutoInterp, arXiv 2410.13928) and caching the summary.
Unlabeled features contribute nothing to alignment (by design — honest).

## 4. P2 — alignment (RQ1)

**4.0 RQ1-v0 (label-free — run this FIRST, no autointerp needed):** uses only
the captured `active_feature_ids` + the solver `gold_label` (physics: formula_id;
logic: answer). Fastest honest read of whether features carry the solver class.

```bash
python scripts/interp/run_discriminance.py \
  --records outputs/interp/records_physics_l12.jsonl \
  --out outputs/interp/discriminance_physics_l12
```
Reports same-class vs different-class feature overlap, a separation AUC
(0.5 = no signal), and leave-one-out 1-NN accuracy vs majority. AUC ≫ 0.5 and
1-NN lift > 0 ⇒ go; ≈ 0.5 ⇒ honest negative (charter §8).

**4.1 Labels (prerequisite for the lexical alignment below):** the captured
`active_concepts` are empty until features are labeled. Build the cache with a
local open labeler (no external API):

```bash
python scripts/interp/build_labels.py --preset qwen3.5-2b --task physics \
  --split data/official_v20260515/eval_split/physics_eval_sft_unseen.jsonl \
  --records outputs/interp/records_physics_l12.jsonl --layer 12 \
  --out outputs/interp/labels.json --max-features 400   # --labeler-model to override
```
Then RE-RUN `capture_activations.py` with `--labels outputs/interp/labels.json`
so `active_concepts` is populated, and proceed:

**4.2 Lexical alignment (RQ1):**

```bash
python scripts/interp/run_alignment.py \
  --records outputs/interp/records_physics_layer18.jsonl \
  --out outputs/interp/alignment_physics_layer18 \
  --probe-records outputs/interp/probe_physics_layer18.jsonl   # optional baseline
```

**Metric (pure, `interp/alignment.py`):** per sample, precision/recall/F1 of the
SAE-derived concept set vs the solver gold set; macro/micro aggregates;
**random baseline** = drawing the same number of concepts uniformly from the
vocabulary (controls for set size); **lift** = observed − random; **win-rate** =
fraction of samples beating their own random baseline.

**Baselines (charter §8, mandatory):**
* random (built in);
* **linear probe** — train a logistic probe on the residual stream to predict
  gold concepts, decode its top concepts per sample into a `--probe-records`
  file of the same schema, and compare. (Probe ≥ SAE is the AxBench-style
  negative the paper must address.)

## 5. Success / kill criteria (charter §8)

* ✅ **Go:** RQ1 lift meaningfully > 0 **and** win-rate ≫ 50% **and** SAE not
  beaten by the linear probe; later RQ3 solver-loop > single-pass with
  over-correction ≈ 0.
* ⛔ **Pivot:** lift ≈ 0 or probe ≥ SAE → honest negative result (still
  publishable in a skeptical field); deployable lever reverts to richer
  solver-trace explanations (P2) per charter §8.

## 6. Outputs & reproducibility

Write everything under `outputs/interp/` (gitignored like `outputs/eval/`).
Record per run: model, SAE repo + revision, layer(s), label-cache hash, seed,
split file, `solver_ok` coverage. Every measured claim ships with an ADR
(`docs/decisions/00NN-...`) per the repo's iteration discipline.
