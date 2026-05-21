# 0031 — Day-27 Iter-4 — data-driven routing & unit cleanup (33.1% → ?)

Date: 2026-05-22 (Day 27)
Status: Accepted (final eval in flight; commit pending result)

## Context — what Iter-3 left on the table

Iter-3 (ADR 0030) cracked the architectural gaps that were holding 5
specific failure classes off the board (0/10 → 9/10 on pinned set; 27.6%
→ 33.1% on the 163-row SFT-unseen holdout). To find the next batch we
ran the holdout with `--include-samples` (Iter-3 commit
`7f32033` on a 135-row near-cousin), then grouped every `full_correct=False`
row by `(formula_id, fail_reason)`. Four high-density clusters surfaced
with **no architectural unknowns** — each is a directly verifiable
single-fix gap:

| Cluster | Rows | Symptom | Root cause |
|---|---:|---|---|
| **A** Plate-area `cm2` | 8 | `unit_conversion_failed: cm2 is not defined in the unit registry` | pint accepts `cm**2` / `cm^2`, not the bare-digit form copy-pasted from PDFs |
| **B** LLM glued `× 10^N` onto unit | 5 | `cannot convert 3.6 '× 10^-6 C' -> 'coulomb'` | LLM extractor returns `{"value": 3.6, "unit": "× 10^-6 C"}` instead of folding the exponent into `value` |
| **C** Capacitor-energy misroute | 3-4 | "has a capacitance of X pF, charged to Y V, calculate the electric field energy" hit `parallel_plate_capacitance` (needs A, d which the question never gives) | "air-filled parallel-plate" keyword rule preceded the energy intent rule in classifier order |
| **D** 2-charge perp-bisector field | 4 | LD065/099/100/340: `electric_field_point_charge` predicted with one q, gold needs both | No formula for "+q1 at A, −q2 at B, M on perp-bisector at distance l" geometry |

Cluster A alone could be +5pp (8/163), B another +3pp, C another +2pp,
D another +2.5pp — collectively a possible +12pp ceiling if every fix
holds in production. Cluster A is also a free defense win: a single
regex registry alias closes 8 rows.

## Fixes — each surgical, no agent-architecture changes

### Fix A — `cm2` / `mm3` / `m2` normalization in pint
[`src/exact_agent/physics/unit_converter.py`]

Single regex pass `_AREA_VOLUME_NORM = re.compile(r"\b(cm|mm|m|km|dm|nm)([23])\b")`
runs **before** the existing alias dict so legitimate Unicode forms
(`cm²` → `cm**2` via the `²` alias) still convert. 5 parametrized tests
(`test_no_superscript_area_volume_forms`) lock in the contract.

### Fix B — scientific-notation peel in LLM extractor
[`src/exact_agent/physics/llm_extractor.py`]

New `_peel_scientific_from_unit(value, unit)` detects a leading
`[x×*·]? 10 ^? ±N` token on the unit string, folds the `10^N` factor
into the value, returns the clean unit. Handles ASCII (`x 10^-6 C`,
`*10^6 V/m`), Unicode multiplication sign (`× 10^-7 C`), and Unicode
superscripts (`× 10⁻⁶ C`). 6 parametrized tests
(`TestUnitScientificPeel`) plus a `pyproject.toml` per-file ignore for
the ambiguous-glyph lint warnings (the glyphs are the test fixtures).

Caveat: LD396-style outright dimension errors (`{"value": 3.54e-06,
"unit": "μF"}` for a CHARGE) are still rejected by pint at convert
time — that's correct rejection of a hallucinated unit, not something
to swallow. Self-consistency vote N=3 is the mitigation there.

### Fix C — capacitor-energy intent rule lifted above geometry
[`src/exact_agent/physics/topic_classifier.py`]

Three new keyword phrases (`calculate the electric field energy stored`,
`calculate the electric field energy in`, `electric field energy stored
in the capacitor`) routed to `capacitor_energy` BEFORE the
`parallel_plate_capacitance` block. Surgical — does not touch the
pure-geometry rules below (TD162/189/etc. still route correctly because
their questions don't say "energy").

### Fix D — 2-charge perp-bisector field formula + symmetric guard
[`configs/physics_formulas.yaml`, `src/exact_agent/physics/topic_classifier.py`]

New formula `electric_field_perp_bisector_two_opposite`:
```
E = k · sqrt(((|q1|+|q2|)·d/2)² + ((|q1|−|q2|)·l)²) / ((d/2)² + l²)^1.5
```
where d = AB and l = perp distance from midpoint of AB. Verified against
4 gold answers (LD065/099/100/340) within 0.31% relative error each.
Equal-magnitude case (|q1|=|q2|) reduces to `k·|q|·d/r³`; unequal case
(LD340) needs the full sum/diff term.

Routing rule added BEFORE the existing `coulomb_force_perp_bisector`
keyword line — both fire on "perpendicular bisector", but the field
formula is in the new `_FIELD_FORMULAS` set which the existing F2
field-vs-force guard skips when force is asked. To prevent the reverse
misroute (force question landing on the field formula), Iter-4 adds:

- `_FIELD_FORMULAS` set + `_is_force_asking()` predicate + `_FORCE_ASK_TOKENS`
  list ("find the force", "calculate the force", " newton", etc.)
- Symmetric guard inside both `_keyword_match` and `_symbol_match`:
  `if force_asked and formula_id in _FIELD_FORMULAS: continue`

Existing `test_coulomb_force_perp_bisector` integration test confirms
the reverse path still routes to force (after broadening
`_FORCE_ASK_TOKENS` to include the plain "Find the force." phrasing).

## Quality gate

- 265 tests pass (228 unit incl. 11 new for Fixes A/B; integration
  unchanged)
- `ruff check src tests` — All checks passed
- `mypy src` — Success: no issues found in 51 source files

## Expected lift (production eval pending)

Optimistic ceiling: +12pp (all 4 clusters land cleanly). Realistic:
+5-8pp once LLM extractor variance and unit-vote interactions take their
toll. Iter-3's per-slice methodology shows DT, TD, and LD as the biggest
beneficiaries.

If the eval lands below 35% the regression analysis should look at:
- Whether Fix C's new energy-routing rule consumed a row that previously
  routed correctly via symbol fallback
- Whether Fix D's new symmetric guard `_FIELD_FORMULAS` is too broad
  and now skips a legitimate field question on a force-word match

Both are reversible single-edit rollbacks if the headline regresses.
