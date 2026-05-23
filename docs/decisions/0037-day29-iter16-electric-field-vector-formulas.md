# ADR 0037 — Iter-16 (Day-29): Electric-field vector-composition formulas

## Status
Accepted — 2026-05-23

## Context

After Iter-15 (Physics 47.2% on the 163-row SFT-unseen holdout), an audit
of the remaining 86 wrong rows showed the **largest single cluster** was
`numeric_mismatch` on `electric_field_point_charge` — **8 rows / 35
numeric_mismatch / 86 total wrong** (≈23% of numeric_mismatch, ≈9% of
remaining wrong).

The 8 rows decompose into four sub-patterns; in all four the system
picked the single-charge formula `E = k·q/r²` when the actual physics is
**vector composition of two source-field vectors at known angles**:

| Sub-cluster | Rows | Geometry |
|---|---|---|
| Opposite-sign equidistant (perp-bisector special case) | LD053, LD090 | AC=BC or equilateral, q1 = −q2 |
| Same-sign equilateral, field at third vertex | DT051, LD319, LD392 | Equilateral, q1 = q2 (or all three identical), 2 source vectors @ 60° |
| Same-sign general isoceles, field at apex | LD052 | AC=BC ≠ AB, 2 source vectors @ arbitrary angle |
| Right-angle vertex, 3 identical | LD335 | Isoceles right triangle, 2 source vectors @ 90° |
| Inverse problem (find distance where E=0) | LD086 | Asks distance, not field |

This is a **modelling gap**, not an extractor failure — the extractor
correctly returns q, distance, but no formula in the registry knew how
to vector-sum two equal-magnitude source fields at a fixed angle.

## Decision

Ship **three new formulas + matching routing rules** in one iteration:

1. **16a — `electric_field_two_opposite_sources_isoceles_apex`**
   `E = k·|q|·d / r³` for q1 = −q2 equal-magnitude at A, B (AB=d) with
   field point at apex of AC=BC=r isoceles triangle. Equivalent to the
   `|q1|=|q2|` reduction of `electric_field_perp_bisector_two_opposite`,
   reparameterised in `r` so extraction needs only AC/BC (not the
   perpendicular distance `l`, which questions never state directly).
   Covers LD053 and LD090 (the equilateral case is r = d).

2. **16b — `electric_field_equilateral_three_identical`**
   `E = √3·k·|q|/a²` for two equal same-sign charges at two vertices of
   an equilateral triangle, field at the third vertex. Two source
   vectors at 60° resolve to a √3-scaled resultant. Covers DT051,
   LD319, LD392.

3. **16d — `electric_field_right_angle_two_identical`**
   `E = √2·k·|q|/r²` for two equal same-sign charges at the two acute
   vertices of an isoceles right triangle (legs `r`), field at the
   right-angle vertex. Two perpendicular source vectors resolve to
   √2-scaled resultant. Covers LD335.

### Routing strategy

The routing rules in `topic_classifier.py::_KEYWORD_RULES` are an
ordered list (first match wins). The new rules are inserted **above**
`electric_field_point_charge` and existing equilateral force routes:

```
... (existing perpendicular-bisector field/force pair) ...
16a-rule-1:  "ac = bc"            + opposite-sign guard  -> 16a
16a-rule-2:  "equilateral triangle" + opposite-sign guard  -> 16a
16b-rule  :  "equilateral triangle"                        -> 16b
16d-rule  :  "right-angle vertex" / "isosceles right triangle" -> 16d
... (existing coulomb_force_equilateral, F2-guarded)
... (electric_field_point_charge fallback)
```

The 16a rules are guarded by a new per-rule mechanism: a 3-tuple
`(keywords, formula_id, guard_name)` resolves the guard via
`_RULE_GUARDS[name]`. The current guard
`_has_opposite_sign_two_charges` scans the question text for any of
several `q1 = -q2` / `q2 = -…` patterns. This keeps the classifier a
pure function on question text and avoids coupling to the extractor.

### Defence in depth (audit-driven invariants from prior iterations)

* **F2 (Iter-4)** — all three new formulas are in `_FIELD_FORMULAS`,
  so a force-asking question CANNOT steal them; routing falls through
  to the corresponding Newton-output formula instead.
* **Dim guard (Iter-10)** — `_infer_expected_output_dim` reads the ask
  sentence; if it infers "newton" or "joule", a V/m formula is rejected
  in both keyword and symbol-fallback paths.
* **Target words (Iter-1+)** — symbol-only fallback for each new
  formula requires "electric field"/"field intensity"/"field strength"/
  "v/m" in the question, blocking a stray q+r overlap from committing.
* **Same-sign exclusion** — LD052 (same-sign isoceles, AC=BC) is
  intentionally NOT routed to 16a; the opposite-sign guard returns
  False, so it stays on `electric_field_point_charge` (wrong but
  unchanged) until a future Iter-16c general-isoceles formula lands.

### Per-fix isolation

Each fix is shipped as a separate commit so a per-row regression can
revert exactly one fix without losing the others:

* commit A — YAML formula additions (all three)
* commit B — `topic_classifier.py` routing + guard wiring
* commit C — tests + ADR

Per-row diff vs the Iter-15 holdout report is the acceptance gate; a
≥ 2-row regression on any prefix triggers per-fix rollback.

## Consequences

* **Tests**: +18 unit tests (routing × 7, numeric × 6, guards × 5);
  total 277 → 295. Existing 277 unaffected.
* **Expected lift**: +3 to +5 rows on the 163-holdout depending on
  tolerance behaviour for LD090 (gold "9 × 10³" is 1-sig-fig rounded;
  our 9.21e3 is 2.36% off the rounded gold and may not pass the
  default 1% `rel_tol`). Range: Physics **47.2% → 48.5–50.3%**.
* **Risk surface**: the new 16b "equilateral triangle" keyword could
  theoretically over-fire on a force-asked equilateral question, but
  F2 already guards that path; regression test
  `test_force_asked_equilateral_keeps_coulomb_formula` locks it in.
* **Deferred**:
  * **16c — `electric_field_isoceles_two_same_sign` (LD052)** — needs
    extraction of all three sides + cosine-law angle; deferred to a
    separate iter so its extractor risk doesn't contaminate this batch.
  * **16e — `zero_field_position_outside_opposite` (LD086)** — inverse
    problem (asks distance, not field); requires a new intent-detection
    path because the classifier currently keys on "calculate the field"
    style asks. Deferred as standalone iter.

## Methodology note

This iteration follows the same audit-driven loop that produced Iter-9
through Iter-15: triage failing rows, find a tight cluster, ship the
minimum code to close it, regression-test the existing slices.
Architecture is not built ahead of evidence.
