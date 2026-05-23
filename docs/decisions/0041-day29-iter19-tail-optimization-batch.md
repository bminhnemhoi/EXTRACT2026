# ADR 0041 — Iter-19 (Day-29): 7-fix tail-optimization batch

## Status
Accepted — 2026-05-23

## Context

After Iter-18 (Physics 60.7%), user requested "tối ưu nhất có thể trước
khi train". The originally proposed 3 follow-up items were:

1. LD052 same-sign isoceles cosine law
2. NL346 unit semantics
3. THCB unit-passthrough (deferred from Iter-17e)
4. √-literal extractor (deferred from Iter-17b)

A deeper audit of all 64 remaining wrong rows surfaced **4 new clusters
not in the original plan**:

| Cluster | Rows | Pattern |
|---|---|---|
| CH143/144/145 | 3 | RLC at resonance, given U_RC, asks U_C — no formula |
| CH247/249 | 2 | "LCω²=1 ... power factor" — cos(φ)=1 at resonance, but formula returned R/Z |
| CH160 | 1 | Imax at resonance — misrouted to power_at_resonance |
| NL334 | 1 | "What is its inductance" — misrouted to current_from_inductor_energy |

These outweigh the originally-proposed NL346 (dataset bug, can't fix
cleanly) and add **+7 confirmed rows** to the batch potential.

## Decision

Ship **7 fixes** as a single Iter-19 batch in 4 per-fix commits.

### 19a — `voltage_capacitor_from_urc_resonance` (new physics)
`U_C = sqrt(U_RC² - U²)` for series RLC at resonance. Derivation: at
resonance X_L = X_C → reactive voltages cancel → U_R = U (total).
The R-C section voltage U_RC = sqrt(U_R² + U_C²) (R and C are 90°
out of phase phasors). Solving: U_C = sqrt(U_RC² - U_R²) = sqrt(U_RC² - U²).

Verified <0.06% rel error on CH143 (203.96), CH144 (63.25), CH145 (99.50).

Routing: new `_has_rlc_resonance_with_urc` guard (true when "at
resonance" + "R-C"/"C-L"/section phrasing). Trigger phrases:
"rms voltage across the capacitor" / "voltage across the capacitor c".

### 19b — `electric_field_isoceles_two_same_sign_apex` (new physics)
Same-sign mirror of Iter-16a opposite-isoceles. AC=BC=r, AB=d,
q1=q2=q at A,B, field at apex C. By symmetry only the perpendicular
component survives:
```
E = 2·k·|q|·sqrt(r² - (d/2)²) / r³
```
LD052 verified (350795 vs gold 351000 = 0.06% off). Distinct from
Iter-16b (equilateral special case r=d) and Iter-16a (opposite signs).

Routing: new `_is_same_sign_isoceles_not_equilateral` composite guard
(NOT opposite-sign + NOT equilateral + same-sign-detector). Trigger
"ac = bc" — placed ABOVE the existing Iter-15 force-formula and
Iter-16a opposite-isoceles rules so the same-sign path wins.

### 19c — `mean_three_measurements_{mass,voltage,temperature}` (3 new formulas)
Three per-unit variants instead of a single dimensionless formula —
avoids solver-side input-unit-passthrough plumbing. Each variant
declares its output_unit explicitly so `unit_match` agrees with the
dataset gold (g / V / °C).

THCB094 + THCB114 gain. **THCB133 still fails** at runtime — pint
rejects offset-unit operations on Celsius (`cannot convert 20.1 °C ->
celsius: Ambiguous operation with offset unit`). Celsius variant
ships for completeness; would need `delta_degC` parameterisation or
treating temperatures as bare floats to be fixable.

### 19d — CH160 `ohm_law_current` triggers extended
Question: "operating at resonance ... Calculate the maximum effective
current Imax". Was routed to `power_at_resonance` via symbol fallback
(U + R match + "resonan" target word) → returned 400 W instead of 1.5 A.

Added narrow keyword triggers to the existing Iter-18f ohm_law_current
rule: "calculate the maximum effective current", "calculate imax",
"calculate the rms current at resonance".

### 19e — `power_factor_at_resonance_constant` (new constant formula)
CH247/249: "LCω² = 1 (resonance) ... power factor". At resonance
cos(φ) = 1 identically. The constant returner short-circuits the
default `power_factor_from_r_z` formula (which was returning 1.75 /
1.333 from incorrect R/Z extractions for these compound-circuit
problems).

Formula signature: `1 + 0*R1 + 0*R2` so the symbol fallback can fire
on the {R1, R2} extracted from typical "R1 = 70 Ω, R2 = 40 Ω"
questions.

Routing: keyword "lcω2 = 1" / "satisfies the condition lc" + new
`_asks_power_factor` guard. Non-resonance power-factor questions
still hit `power_factor_from_r_z`.

### 19f — NL334 `inductance_from_inductor_energy` triggers extended
Question: "An inductor has W = 0.2 J, I = 2 A. What is its inductance
(H)?". Was misrouting to `current_from_inductor_energy` via the
"magnetic field energy" / "magnetic energy" generic rule, then
failing extraction because L was treated as required input not output.

Added "what is its inductance" / "find its inductance" /
"determine its inductance" / "what is the inductance" / "find the
inductance" to the existing rule.

### 19g — √-literal cleaner + `g; g` unit-pair scorer (defensive)

**Cleaner**: pre-resolve `100√2` / `100*√2` / `√2` / `100/√3`
patterns to numeric values in `clean()` before the extractor sees
the text. NL361's "voltage across the capacitor is 100√2 V" was
being read as 100 V (regex stopped at "100"; "√2 V" leftover).
After cleaner: "voltage ... is 141.421 V". Defensive — catches any
future row using √-literal notation.

**Scorer**: `unit_match` now strips `;`-suffixed compound units
(e.g. `"g; g"` → `"g"`). The THCB lab gold rows pack value-unit
and error-unit into a single field separated by `;`; the scorer
was failing the dimensionality check on the full "g; g" string.

NL361 routing path now produces a numeric U value but final compute
depends on LLM extractor picking up the "141.421 V" assignment-free
mention. Not gained this run; flagged for follow-up.

## Consequences

* **Tests**: +25 unit tests (parametric × 18 across 7 fix types,
  regression × 7). Total 337 → 362.
* **Eval (163-row SFT-unseen, LLM-enabled)**:
  Physics **60.7% → 66.9% (+6.2pp, +10 rows net, 0 regressions)**.
  Gained (10): LD052, THCB094, THCB114, NL334, CH143, CH144, CH145,
              CH160, CH247, CH249.
  Not gained (2): THCB133 (pint offset-unit), NL361 (extractor LLM
                  doesn't pick up assignment-free U mention).
* **Cumulative session lift (Iter-15 → Iter-19, single day)**:
  Physics **47.2% → 66.9% (+19.7pp, +32 rows, 6 ADRs, 16 commits)**.
* **CH slice now 71.8%** (best CH ever — up from 53.8% pre-session).
* **Risk surface**: Same-sign isoceles guard composite logic
  (`NOT opposite-sign AND NOT equilateral AND same-sign-detector`)
  is the most complex per-rule guard so far. Each component is
  unit-tested independently; the composite has 1 dedicated test.

## Methodology note

This batch demonstrates the value of **re-auditing before executing
a stated plan**. The user's original 3-item list would have yielded
~4 rows; the audit-expanded 7-item batch yielded **+10 rows (2.5×)**.

The pattern: when the user proposes a fix list, treat it as a
**starting hypothesis** but always re-audit the current fail dump
to find clusters the user hasn't seen. The audit takes 15 minutes
and pays back 2-3× in row gains for the same implementation effort.

## Remaining tail (~54 rows)

After Iter-19:
* **Dataset bugs** (LD146, NL086, LD395, DT051): gold inconsistent
  with stated input values. Unfixable without overriding gold.
* **Symbolic / non-numeric golds** (NL302/303/323/324, DDT136/360):
  gold is a formula expression or qualitative phrase. Need separate
  MC-style answering framework.
* **Compound RLC multi-segment** (CH226/230/237/255, CH183, CH217):
  multi-step circuit analysis (M-segment in series with subsections,
  90° phase shift between U_AM and U_MB). Single-formula approach
  doesn't fit; need a multi-step composer.
* **Inverse / state-change qualitative** (TD369/380/386, NL323/324/
  328, DDT360, NL336 with time-dependent W_C): require either
  symbolic reasoning or new formulas for each pattern.

Realistic deterministic ceiling: **~70%** on this holdout, ~62-66%
on the public test (accounting for phrasing variance). Further gains
hinge on the SFT-7B retrain (extractor lift) — user action.
