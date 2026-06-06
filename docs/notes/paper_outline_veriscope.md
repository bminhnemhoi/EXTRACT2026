# Paper Outline — VeriScope

> Companion to [research_plan_solver_grounded_sae.md](research_plan_solver_grounded_sae.md).
> Target venue: EXACT 2026 special session *"Explainable AI for Educational QA"*
> (CSoNet 2026) + competition solution description. Short/workshop-paper scope.
> Write-up must model the honesty the charter demands (esp. §5 anti-patterns).

**Title:** *VeriScope: A Solver-Grounded Agentic Interpretability Loop for Explainable Educational QA*

## Abstract (EN, draft)

> Large language models are unreliable judges of their own reasoning, and
> representation-level interpretability (sparse autoencoders, SAEs) is, on its
> own, a contested basis for explanation: recent work shows off-the-shelf SAE
> steering and probing are beaten by simple baselines and that "reasoning
> features" are often surface correlates. We study interpretability in a setting
> that supplies what this literature lacks — **ground truth**. In a solver-first
> neuro-symbolic educational QA agent, a deterministic symbolic solver
> (SymPy/pint for physics, forward-chaining/Z3 for logic) produces a faithful
> derivation for every answer. We use this derivation (1) as ground truth to
> measure whether the LLM's SAE features during quantity-extraction / NL→FOL
> translation align with the concepts actually used (RQ1); (2) as the in-loop
> **teacher** of a solver-grounded agentic loop that, offline, discovers
> question-conditioned steering + prompt guardrails and deploys them statically
> (RQ2–RQ3). The solver — never the SAE or the LLM — decides correctness, which
> sidesteps the failure mode where two unreliable signals reinforce each other.
> We report [lift], [error-reduction], and [loop vs baselines], plus an honest
> characterization of where SAE diagnosis agrees and disagrees with the
> symbolic ground truth.

## 1. Introduction
- Problem: explainability in educational QA (EXACT 2026 P2/P3); LLM free-text
  CoT is often unfaithful (Turpin 2305.04388; Chen/Anthropic 2505.05410).
- Tension: SAE interpretability is contested (AxBench 2501.17148; DeepMind
  deprioritization; Sanity Checks 2602.14111; Falsifying reasoning features
  2601.05679). Naively bolting SAE on is a liability.
- Our angle: a solver-first agent **has ground truth** (the derivation). Use it
  to *evaluate* interpretability and to *anchor* an agentic loop.
- Contributions (charter §3): C1 solver-as-ground-truth SAE-faithfulness
  testbed; C2 solver-grounded agentic loop (offline-discover → deploy-static);
  C3 honest finding + baselines.

## 2. Related Work (positioning — what is ours vs prior)
- **Open SAE suites:** Gemma Scope (2408.05147) → Qwen-Scope (2605.11887) is the
  Qwen analog. *We use, not invent, the SAE suite.*
- **Answer provenance / circuits:** Anthropic attribution graphs / circuit
  tracing (transformer-circuits 2025). *Stronger than SAE activation; we do not
  claim provenance from activation alone.*
- **SAE evaluation lacks ground truth:** SAGE (2410.07456), Principled Eval
  (2405.08366), CE-Bench (2509.00691). *Our gap: a realistic symbolic-solver
  ground truth, not toy/supervised-probe.*
- **Adaptive/conditional steering:** WAS (2505.20309), Adaptive Steering
  (2406.00034), Control-RL (2602.10437), conditional clamping (2503.11127),
  off-topic latents (2602.06941), topic alignment (2506.12576). *Per-input
  steering exists; ours is solver-judged + offline-distilled.*
- **Prompt-optimization loops:** ProTeGi, TextGrad, OPRO, DSPy, SIPDO
  (2505.19514), CriSPO; survey 2502.16923. *All judge with LLM-critique /
  synthetic data; we judge with a deterministic solver.*
- **Self-correction needs external feedback:** Huang (2310.01798), CRITIC
  (2305.11738), Reflexion (2303.11366); PRM (2305.20050), Math-Shepherd
  (2312.08935), RLVR/DeepSeek-R1 (2501.12948). *Justifies solver-as-teacher.*
- **Faithful-by-construction:** Faithful-CoT (2301.13379), SymbCoT (2405.18357),
  VeriCoT (2511.04662). *Our explanation is the solver trace.*

> **Novelty sentence (defend this):** prior loops judge with noisy signals and
> prior SAE evals lack ground truth; we close both gaps at once by making a
> sound symbolic solver the in-loop teacher AND the evaluation oracle.

## 3. System / Method
- 3.1 Solver-first agent (recap): LLM extracts/translates/phrases; SymPy/pint +
  forward-chaining/Z3 decide. The LLM never decides the answer.
- 3.2 Ground-truth concepts from the solver (physics: formula id/topic/inputs;
  logic: supporting premises + predicates) — `interp/ground_truth.py`.
- 3.3 SAE feature capture & concept mapping (Qwen-Scope, residual stream,
  TopK-50; autointerp labels) — `interp/{sae_loader,hooks,features}.py`.
- 3.4 Solver-grounded agentic loop: controller chooses question-conditioned
  steering + prompt guardrails; solver judges; monotone gate (keep only on
  score improvement, ≤2–3 rounds); **offline-discover → deploy-static**.
- 3.5 Architecture figure (charter §6).

## 4. Experiments
| RQ | Question | Metric | Baselines |
|---|---|---|---|
| RQ1 | SAE features ↔ solver concepts? | macro/micro P/R/F1, lift, win-rate (`interp/alignment.py`) | random draw; linear probe |
| RQ2 | Steering reduces an error class? | code-switch rate, distractor errors, FOL-malformed; coherence proxy; α-sweep | prompting; no-steer |
| RQ3 | Solver-loop > critique-loop > single? | P1/P2 lift; over-correction rate (must ≈0) | single-pass; LLM-critique loop (TextGrad/DSPy, no solver) |

- Data: official holdouts (physics 163 SFT-unseen, logic 81); models Qwen3-8B
  (vs Qwen2.5-3B baseline); analysis on Base, agent on Instruct (§ decision).
- Ablations: layer sweep; with/without solver gate; SAE vs probe.

## 5. Results (tables to fill)
- Table 1: RQ1 alignment per layer/task vs random + probe.
- Table 2: RQ2 error-class reduction + coherence at chosen α.
- Table 3: RQ3 loop comparison incl. over-correction.
- Agreement analysis: where SAE diagnosis matches / contradicts the solver.

## 6. Limitations & Threats to Validity (do NOT omit — this is the credibility)
- SAE steering/faithfulness is contested; report negatives honestly.
- Concept space is lexical (v1); curated ontology is future work.
- Base-vs-instruct SAE mismatch (if agent runs Instruct) — quantify the gap.
- No Qwen-Scope SAE for the originally-deployed Qwen2.5-3B (model switch needed).
- Solver ground truth covers *its* derivation, not the LLM's latent reasoning;
  RQ1 tests concept presence, not a full causal account.
- Steering can degrade safety/coherence (Rogue Scalpel 2509.22067).

## 7. Compliance & Ethics
- ≤8B open model; any FT dataset published (current plan uses no training);
  Qwen-Scope custom license; solver-first invariant preserved.

## 8. Conclusion
- A solver-first agent is a natural ground-truth testbed for interpretability;
  the contribution is the ground-truth anchor, not the use of SAEs.

---

## Citation corrections (apply before submission — verified)
- DeepSeek-R1-Zero AIME 2024 = **71.0% pass@1** (NOT 77.9%; 77.9%/86.7% are
  cons@64 majority-vote numbers).
- Do **not** cite Lanham (2307.13702) for "small model = less faithful" — Lanham
  reports *inverse scaling* (larger = less faithful, ~13B sweet spot). Motivate
  small-model unfaithfulness from Turpin (2305.04388) + Chen/Anthropic
  (2505.05410), which are size-agnostic.

## Bibliography (grouped — keys to fetch into .bib)
See charter §11 for the full keyed list (SAE tools; SAE limits; ground-truth
eval; steering controllers; prompt-opt; self-correction/verifier; faithfulness).
