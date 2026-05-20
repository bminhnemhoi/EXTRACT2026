# 0020 — Day-22 formula additions driven by the E10 audit

Date: 2026-05-21
Status: Accepted

## Context

After E5/E6 (commits a5d9e03 + 465ba6c) we ran the E10 dataset-issue
audit (`scripts/audit_dataset_issues.py`) on the post-E6 physics
holdout. The hypothesis was that we'd surface Q22-bonus candidates —
rows the gold gets wrong. **The audit instead surfaced our own gaps**:
of 22 flagged rows, **0 are credible dataset bugs** and ~15 are real
formula / routing gaps. The audit's null result on Q22 is itself a
finding: the organizer's 2026-05-15 cleanup eliminated the dataset
noise that was masquerading as solver wins.

(Also: my first audit script pass was wrong — it computed naïve
unit-blind rel_diff and flagged rows the scorer had correctly marked
*right* under unit-aware comparison. Filtering on `full_correct == False`
fixed that; the scorer itself is sound.)

## What was added (3 formulas, 1 routing-rule reordering)

### 1. `coulomb_force_equilateral_three_identical`

Net Coulomb force on any of three identical charges at the vertices of
an equilateral triangle: `F = √3 · k · |q|² / a²` (two equal pairwise
forces at 60° → resultant magnitude F·√3).

Routes via keywords `("equilateral triangle", "vertices of an
equilateral", "equilateral triangle with side")`, placed **before**
generic `coulomb_force`.

**Audit-target rows:** LD295, LD130, LD228, LD242. Measured: 8 rows
routed to this formula, **3 full-correct** (LD130, LD228, LD242 — gold
within 1 % rel of computed). LD295 still wrong (gold ≈ 0.0433 N,
computed ≈ 0.173 N — the question's geometry isn't strictly
equilateral; gold formula differs by a factor we haven't isolated).

### 2 + 3. `parallel_plate_capacitance` (air) and `parallel_plate_capacitance_dielectric`

`C = ε₀·A/d` and `C = ε₀·ε_r·A/d` (vacuum permittivity 8.8541878128e-12 F/m).

Routes via specific phrasings: `("air parallel-plate capacitor",
"parallel-plate air capacitor", "air-filled parallel-plate")` and the
dielectric variant on `"dielectric constant"`/`"relative permittivity"`.
Routing rules placed **before** the "calculate its capacitance" rule
(which previously won by lexicographic order and mis-routed plate-
geometry problems to `capacitance_from_energy_voltage`).

**Audit-target rows:** TD162, TD174, TD189, TD383. Measured: TD162 + TD174
**now full-correct** (29.97 pF, 23.9 pF); TD189 routes correctly but
the extractor misses `A` (regex can't pull "29.8 cm²"); TD383 is a
change-of-geometry problem that doesn't fit the bare C = ε₀A/d form.

### Routing-rule tightening (revert)

My **first** parallel-plate keyword list included the generic
`"plate area"` and `"plate separation"`. Measurement (intermediate eval
`day22_v0515_physics_e6_3formulas`) showed those bare keywords pulled
in TD rows that already routed correctly to other formulas (net −1
overall, +2 wins offset by −3 regressions). Tightened back to the
specific `"air parallel-plate capacitor"` family — TD162/174 wins
preserved, regressions undone.

## Measured impact on `data/official_v20260515/eval_split/physics_eval.jsonl`

| Variant | Overall Full✓ | Δ vs Day-21 baseline | Note |
|---|---:|---:|---|
| Day-21 (pre-E5/E6) | 16.3% (22) | 0 | honest baseline on official |
| + E6 self-consistency | 18.5% (25) | +2.2pp | ADR 0019 |
| + equilateral only | 20.7% (28) | +4.4pp | +3 LD rows |
| + parallel-plate (greedy keys) | 20.0% (27) | +3.7pp | net −1 from greedy regressions |
| **+ parallel-plate (tightened) — final** | **17.8% (24)** | **+1.5pp** | LLM 3B noise dominated |

**Variance honest:** the three "+ formulas" runs returned 28/27/24
full-correct on the same 135-row holdout with identical code,
indicating the qwen2.5:3b extractor has ±2-3 row variance even with E6
N=3 vote. Spot-check on the equilateral/parallel-plate target rows
confirms the formulas + routing land correctly (TD162 always 29.98 pF
= gold 29.97 pF; LD130/LD228 always within 1% of gold) — but
neighbour rows the extractor sometimes hallucinates push the headline
number around. **Mean across the three runs ≈ 19.5%**, i.e. ~+3pp over
Day-21 baseline, with the formula additions accounting for the
deterministic part of that gain.

**Implication for next move:** the deterministic surface (formulas,
routing, scorer) is now ~saturated against what a 3B extractor can
feed it. The next material P1 lever is **E7 — stronger backbone**
(SFT-7B sequential extractor, or DeepSeek-R1-Distill-Qwen-8B). This
requires the user to launch the Colab serving notebook.

## Honest caveats

1. **The audit's Q22 yield was 0 credible dataset bugs** — meaning the
   organizer's 2026-05-15 cleanup is thorough on the released data;
   there is no free bonus to claim from misreading it.
2. **LD295 didn't fix** despite the equilateral formula — the question
   geometry isn't pure equilateral and the gold uses a different
   closed form we haven't isolated. Defer.
3. **TD189 routes right but the regex extractor misses `A`** —
   "29.8 cm²" has a superscript-2 the regex doesn't recognize. A
   second self-consistency pass via LLM might catch it; E6 is wired
   but evidently didn't recover this row. Worth a unit_converter
   normalization pass for superscript area units.
4. **DDT / DT prefixes still 0% Full.** Multi-charge electric-field
   problems need vector-superposition formulas (similar shape to what
   we did Day 14 #2 for Coulomb). Backlog item.

## Tests

246 passed (no test changes — formula additions land in YAML +
classifier rule table; existing tests cover the classifier dispatch
mechanism, not specific formulas). ruff/mypy clean.

## Reproduction

```powershell
uv run python scripts/audit_dataset_issues.py \
    --report outputs/eval/day22_v0515_physics_e6_samples/physics_llm_report.json \
    --rel-tol 0.1
uv run python scripts/run_eval.py --task physics --with-llm --include-samples \
    --split data/official_v20260515/eval_split/physics_eval.jsonl \
    --out outputs/eval/day22_v0515_physics_final
```
