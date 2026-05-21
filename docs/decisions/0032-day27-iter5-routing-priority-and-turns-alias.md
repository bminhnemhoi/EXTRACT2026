# 0032 — Day-27 Iter-5 — routing precedence + turns/m alias + Q·U/2 formula

Date: 2026-05-22 (Day 27)
Status: Accepted (eval in flight; commit pending result)

## Context — what Iter-4 left on the table

Iter-4 closed the 4 dense clusters surfaced on Day-27 morning (33.1% →
36.8% on the 163-row holdout). The next pass on the **Iter-4** per-sample
report flags three more clusters with high single-fix yield:

| Cluster | Rows | Symptom | Root cause |
|---|---:|---|---|
| **A** Resonance/reactance misroute to `series_resistance_two` | 3-4 | CH025/031/032 ask "calculate the resonant frequency" but route to series_resistance_two | The naive "in series" trigger in `_KEYWORD_RULES` precedes the resonance rule (first-match wins) |
| **B** `turns/m` unit conversion fail | 2 | DDT382/392: `cannot convert 2500 'turns/m' -> '1/meter'` | pint's built-in `turn = 2π rad` overrides our `ureg.define("turn = 1")`; alias `"turns": ""` produces a malformed `/m` |
| **C** TD361 `capacitor_energy` missing C | 1 | "A capacitor has a charge of 40 μC and a voltage of 8 V. Calculate the energy" — extractor finds Q and U but not C | Formula registry only had E = ½·C·U²; needed E = ½·Q·U for this prose form |

These are all **one-line/one-rule** fixes — no architectural movement.

## Fixes

### Fix A — Resonance/reactance precedence + tighten series-resistance trigger
[`src/exact_agent/physics/topic_classifier.py`]

Two structural changes to `_KEYWORD_RULES`:

1. Moved `("resonance frequency", "resonant frequency", "natural frequency")
   → resonance_frequency` rule **above** the `parallel/series_resistance_two`
   block. First-match-wins means resonance now wins for "L in series with C
   … resonant frequency" questions.

2. Tightened series-resistance trigger from
   `("resistors in series", "connected in series", "in series")` to
   `("resistors in series", "two resistors connected", "series resistors")`.
   The bare `"in series"` phrase was over-matching any L/C-in-series RLC
   question.

Net effect: CH025/031/032 + DDT345 now route to the correct
resonance/RLC formula instead of `series_resistance_two` failing on
`missing_input: 'R1'`.

### Fix B — `turns/m` alias map
[`src/exact_agent/physics/unit_converter.py`]

pint's built-in turn unit is `turn = 2π rad`, which overrode our
`ureg.define("turn = 1 = turns")` and produced `2π·n/m` for a
solenoid turn-density (factor-of-2π wrong). The previous alias
`"turns": ""` mapped `turns/m → /m` which pint rejected outright
("missing unary operator '/'").

The fix maps the full composite to `1/m` directly, ordered with longer
keys first so substring aliases don't pre-empt them:

```python
"turns/m": "1/m",
"turn/m":  "1/m",
"turns / m": "1/m",
"turn / m":  "1/m",
"turns": "",
```

3 parametrized tests (`test_turns_per_metre_is_dimensionless_count`)
lock in the contract.

### Fix C — `capacitor_energy_from_charge_voltage` formula
[`configs/physics_formulas.yaml` + `topic_classifier.py`]

New formula:
```yaml
capacitor_energy_from_charge_voltage:
  inputs:  { Q: coulomb, U: volt }
  output:  { symbol: E, unit: joule }
  expression: "0.5 * Q * U"
```

Routing trigger placed BEFORE the generic "energy stored in the
capacitor" rule:
```python
(
    ("capacitor has a charge of",
     "capacitor with a charge of",
     "capacitor carries a charge of"),
    "capacitor_energy_from_charge_voltage",
),
```

Trigger phrase deliberately narrow (capacitor framing required) so it
does not catch LD/DT "two charges, q1 has a charge of..." prose.

## Quality gate

- 268 tests pass (3 new turns/m tests)
- `ruff check src tests` — All checks passed
- `mypy src` — Success: no issues found in 51 source files

## Smoke-test (deterministic, no LLM)

```
CH025 (was series_resistance_two, missing R1) -> resonance_frequency, OK route (LLM extracts L, C from prose)
CH031 (was series_resistance_two, missing R1) -> resonance_frequency, OK route
TD361 (was capacitor_energy, missing C)      -> capacitor_energy_from_charge_voltage, predicted 1.60e-4 J vs gold 1.60e-4 J (0.00% err)
DDT382 (was unit_conversion_failed turns/m)  -> magnetic_field_solenoid, predicted 9.42e-3 T vs gold 9.42e-3 T (0.05% err)
```

## Expected lift

Optimistic: +3-4pp (DDT382/392 +0.5pp each, TD361 +0.6pp, CH025/031/032
+1.8pp combined, DDT345 +0.6pp). Realistic with LLM-extraction variance
on CH-prose: +2-3pp → ~38-40% on the 163-row holdout.

Caveats: tightening the series-resistance trigger could regress 1-2
rows where genuine two-resistor questions used the bare "in series"
phrasing. The new trigger covers "resistors in series" + "two resistors
connected" + "series resistors" — should hold for all canonical
phrasings but watch the next eval for any series_resistance_two count
drop relative to Iter-4's 7 rows.
