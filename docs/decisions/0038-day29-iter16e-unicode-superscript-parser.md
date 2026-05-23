# ADR 0038 — Iter-16e (Day-29): Unicode superscript parser fix

## Status
Accepted — 2026-05-23

## Context

Post-Iter-16 (Physics 49.7%), per-row inspection of the remaining 82
wrong rows surfaced a non-physics bug: 9 gold answers in the SFT-unseen
holdout use **Unicode superscript digits** (U+2070..U+2079) rather than
ASCII `^N` for the exponent. Examples from the dataset:

```
LD392    gold = "8.48 × 10⁶"
LD394    gold = "14.03 × 10⁶"
DDT362   gold = "1.99 × 10⁻³"
DDT382   gold = "9.42×10⁻³"
DDT384   gold = "2.4×10⁻³"
DDT392   gold = "1.01×10⁻²"
DDT399   gold = "7.54×10⁻³"     (prediction missing — not recoverable)
NL302    gold = "The square..."  (non-numeric — N/A)
NL303    gold = "W = 1/2..."    (non-numeric — N/A)
```

`eval/metrics.py::parse_number` uses a regex
(`_NUMBER_RE`) that expects exponent forms like `10^6` / `10*6` /
`10x6` / `e6` — none of which match `10⁶`. The regex's leading
`[-+]?\d+(?:\.\d+)?` happily matches just the mantissa, so a gold of
`8.48 × 10⁶` was being parsed as `8.48`. The prediction `8.477e6`
correctly rounded to gold-precision, but the comparison ran against
the wrong gold magnitude and false-failed.

This is a **scorer bug** (parser), not a solver bug. The system's
numeric prediction was already correct on all 6 rows.

## Decision

Add a `str.maketrans` table that translates Unicode superscript digits
and signs to ASCII before the existing regex runs:

```python
_SUPERSCRIPT_TRANSLATE = str.maketrans({
    "⁰":"0", "¹":"1", "²":"2", "³":"3", "⁴":"4",
    "⁵":"5", "⁶":"6", "⁷":"7", "⁸":"8", "⁹":"9",
    "⁻":"-", "⁺":"+",
})
s = str(text).strip().translate(_SUPERSCRIPT_TRANSLATE)
```

Applied once at the top of `parse_number`. The existing regex then
matches `8.48 × 106` and the existing `re.sub(r"[*x×]10\^?", "e", ...)`
rewrites it to `8.48e6`. No regex change needed; no behavioural change
on any input that doesn't contain a superscript.

### Why not change the regex instead

The regex would need a new alternative `10[⁰-⁹⁻⁺]+` plus a separate
cleanup branch. The translate-then-match approach is one line, has
zero risk of altering existing scientific-notation parsing, and
mirrors how the upstream cleaner (`question_cleaner.py`) already
handles Unicode minus signs (`U+2212`, `U+2013`, …).

### Scope

Pure parser fix — does not touch solver, classifier, formulas, or any
LLM path. Six additional ASCII tests + six superscript tests added to
`TestParseNumber`.

## Consequences

* **Eval lift (163-row SFT-unseen holdout, LLM-enabled)**:
  Physics **49.7% → 53.4% (+3.7pp, +6 rows, 0 regressions)**.
  Gained: LD392, LD394 (both routed correctly by Iter-16b — parser
  was the blocker), DDT362 (natural_period_lc), DDT382 + DDT392
  (magnetic_field_solenoid), DDT384 (total_flux_linkage).
* **Tests**: +6 superscript cases in `TestParseNumber`; total 295 → 301.
* **Risk surface**: minimal. Any existing input without superscript
  characters is untouched by the translation table (translate on
  unrelated chars is a no-op). The 12 prior `parse_number`
  parametrised cases still pass.
* **Cumulative Iter-16 + Iter-16e lift**: Physics 47.2% → 53.4%
  (+6.2pp, +10 rows in one session: 3 vector formulas + 1 parser fix).
* **Not recoverable** by this fix: NL302 / NL303 (gold is a formula
  expression, not a number), DDT399 (prediction is missing — solver
  returned no answer). Those need future iterations targeted at the
  symbolic-answer and extraction-failure clusters respectively.

## Methodology note

Discovered during Iter-16's per-row diff: an inspection of the 8
target rows that didn't gain despite correct routing. Flagging the
parser bug as a side observation (rather than bundling into Iter-16)
keeps the attribution clean — each iteration's metric delta maps to
a single hypothesis. Same playbook as the per-fix commit structure in
Iter-15 and Iter-16.
