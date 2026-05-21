"""Pull ``name = value unit`` triples out of a free-form physics question.

Examples we need to handle (taken from ``physics_train_safe.jsonl``):

* ``"C = 100 μF and U = 30 V"``
* ``"q1 = 6 × 10^-8 C"``  (Unicode times + exponent)
* ``"q3 = 6 × 10⁻⁸ C"``   (superscript exponent)
* ``"Q = 3 mC"``
* ``"CA = 5 cm and CB = 3 cm"``

Day-25 (G3) additions:

* **Chained-equality broadcast**: ``q1 = q2 = q3 = 2.6 × 10^-6 C`` now
  yields three entries (q1, q2, q3), each with the shared value. Without
  this, equilateral-3-charge questions extracted only q3 and failed the
  formula's q1/q2 inputs.
* **"<noun> of <value> <unit>" prose pattern**: many questions describe
  the quantity in words ("plate area of 23.8 cm²", "capacitance of 19.75
  pF", "potential difference of 131.9 V") rather than equations. A second
  scan maps a small noun vocabulary to canonical variable names so the
  solver still sees them.

The extractor is intentionally regex-based and deterministic. It returns
multiple candidates per variable when the question repeats values
(``"q1 = 6 × 10^-8 C, q2 = -6 × 10^-8 C"`` → two entries). Downstream the
solver picks the symbols matching the chosen formula.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Superscript digit map (U+2070 ... U+2079) + supersript minus/plus.
_SUPERSCRIPT_MAP = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+",
    }
)


_ASCII_TIMES_RE = re.compile(r"(\d)\s*x\s*10\s*\^")
# G3 (Day-25): collapse "mantissa * 10^N" into "mantissaeN" so the main
# regex sees ONE scientific token instead of needing a chain branch.
# Eliminates a class of false negatives in chained-equality contexts
# (q1 = q2 = q3 = 2.6 * 10^-6 C, where the chain regex saw value=2.6 and
# unit=garbage). After this, standalone "10^N" (no mantissa multiplier)
# becomes "1eN" for the same reason.
_MANTISSA_TIMES_POWER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\*\s*10\s*\^?\s*([-+]?\d+)")
_STANDALONE_TEN_POWER_RE = re.compile(r"\b10\s*\^\s*([-+]?\d+)")


def _denormalize_exponents(text: str) -> str:
    """Replace Unicode superscript exponents and times glyphs with ASCII,
    then collapse ``mantissa * 10^N`` and standalone ``10^N`` into the
    scientific ``XeN`` form so a single regex catches every magnitude.
    """
    text = text.translate(_SUPERSCRIPT_MAP)
    text = text.replace("×", "*").replace("·", "*").replace("−", "-")
    text = _ASCII_TIMES_RE.sub(r"\1*10^", text)
    text = _MANTISSA_TIMES_POWER_RE.sub(r"\1e\2", text)
    text = _STANDALONE_TEN_POWER_RE.sub(r"1e\1", text)
    return text


# Magnitude regex — matches plain decimals, scientific, and the
# "<mantissa> × 10^<exp>" style common in the dataset.
_NUMBER = (
    r"[-+]?\d+(?:\.\d+)?"  # signed decimal
    r"(?:"
    r"\s*\*\s*10\s*\^?\s*[-+]?\d+"  # ' × 10^-8' (after _denormalize)
    r"|"
    r"[eE][-+]?\d+"  # '6e-8'
    r")?"
)

# Unit regex — allows SI symbols, micro prefix, slashes for V/m, dot for V·m,
# ohm spellings, superscripts have already been normalized to ascii.
_UNIT = r"[A-Za-zµμΩ°][\w·/*^.\-]*"

# Variable name: letter or letter+digits, plus optional subscript like q_1.
_VAR_NAME = r"[A-Za-z][A-Za-z0-9_]*"

_ASSIGNMENT_RE = re.compile(
    rf"(?P<name>{_VAR_NAME})\s*=\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})?",
    re.UNICODE,
)

# G3: chained equality "q1 = q2 = q3 = 2.6 × 10^-6 C" — broadcast value
# to all left-hand variables. Up to 4 vars in chain (equilateral, isosceles
# right, etc. — the dataset's common patterns).
_CHAIN_EQ_RE = re.compile(
    rf"(?P<n1>{_VAR_NAME})\s*=\s*(?P<n2>{_VAR_NAME})\s*=\s*"
    rf"(?:(?P<n3>{_VAR_NAME})\s*=\s*(?:(?P<n4>{_VAR_NAME})\s*=\s*)?)?"
    rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})?",
    re.UNICODE,
)

# G3 prose: "<noun> of <value> <unit>" — many questions describe the
# quantity in words rather than equations. The noun list maps to
# canonical variable names downstream formulas expect.
# Iter-3 (Day-25) — state-change ratio extractor. "distance ... is doubled" /
# tripled / halved / quadrupled — common phrasing for capacitor-state problems
# (TD010). Maps the verbal scaling word to a dimensionless `ratio` quantity.
_DISTANCE_RATIO_WORDS: dict[str, float] = {
    "doubled": 2.0,
    "tripled": 3.0,
    "quadrupled": 4.0,
    "halved": 0.5,
    "quartered": 0.25,
}
_DISTANCE_RATIO_RE = re.compile(
    r"(?:distance|separation|gap)[^.]*?\b(?P<word>"
    + "|".join(_DISTANCE_RATIO_WORDS) + r")\b",
    re.IGNORECASE,
)

# Iter-3 — "<N> V power source" / "connected to a <N> V source" → V1.
_POWER_SOURCE_RE = re.compile(
    r"(?:connected to|across)\s+(?:a\s+)?(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>V|volts?)\b",
    re.IGNORECASE,
)

# Iter-3 — "has resistance 8 Ω" / "has voltage 4 V" — noun WITHOUT "of".
# Limited to safe nouns to avoid catching "has plate area 30" which is
# already handled by _OF_PATTERN. The "has" prefix anchors intent.
_HAS_NOUN_RE = re.compile(
    r"\bhas\s+(?P<noun>resistance|capacitance|inductance|voltage|current)\s+"
    r"(?P<value>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
    rf"(?P<unit>{_UNIT})?",
    re.IGNORECASE,
)
_HAS_NOUN_MAP: dict[str, str] = {
    "resistance":  "R",
    "capacitance": "C",
    "inductance":  "L",
    "voltage":     "U",
    "current":     "I",
}

# Iter-3 — "is at 4V" / "is at 4 V" — voltage anchored after "is at".
_IS_AT_VOLT_RE = re.compile(
    r"\bis\s+at\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>V|volts?)\b",
    re.IGNORECASE,
)

# Iter-3 — "lamp D2 draws 0.5 A" / "draws X A" → I_remaining (THCB070).
_DRAWS_RE = re.compile(
    r"\bdraws\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>A|amps?|amperes?)\b",
    re.IGNORECASE,
)


_NOUN_TO_VAR: dict[str, str] = {
    "force":               "F",
    "voltage":             "U",
    "rms voltage":         "U",
    "potential difference": "U",
    "applied voltage":     "U",
    "source voltage":      "U",
    "current":             "I",
    "rms current":         "I",
    "resistance":          "R",
    "capacitance":         "C",
    "inductance":          "L",
    "charge":              "Q",
    "energy":              "W",
    "stored energy":       "W",
    "magnetic energy":     "W",
    "magnetic field energy": "W",
    "electric field energy": "W",
    "frequency":           "f",
    "angular frequency":   "omega",
    "plate area":          "A",
    "cross-sectional area": "A",
    "cross section":       "A",
    "area":                "A",
    "plate separation":    "d",
    "separation":          "d",
    "distance between the plates": "d",
    "distance between its plates": "d",
    "side length":         "a",
    "impedance":           "Z",
    "magnetic flux":       "phi",
    "flux density":        "B",
    "magnetic flux density": "B",
}

# Build a single alternation regex from the noun keys, longest first so
# "plate area" wins over "area".
_NOUN_ALT = "|".join(
    re.escape(n) for n in sorted(_NOUN_TO_VAR.keys(), key=len, reverse=True)
)
_OF_PATTERN_RE = re.compile(
    # Allow an optional parenthetical between noun and "of" — e.g.
    # "inductance (L) of 0.3 H" — common annotation in physics prose.
    rf"(?P<noun>{_NOUN_ALT})\s*(?:\([^)]+\))?\s+of\s+(?:magnitude\s+)?"
    rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})?",
    re.IGNORECASE | re.UNICODE,
)

# Iter-3 (Day-25): role-aware geometry extractors. The previous noun-of
# pattern catches scalar nouns ("voltage of X V") but loses GEOMETRY —
# "X cm away from q1" and "X cm apart" are role-bearing phrases. Without
# these, regex misses r1/r2 (the per-source distances for a 3-charge
# Coulomb question) and AB (the source separation in a 2-source isoceles/
# right-triangle question). LLM fallback then guesses and sometimes
# pre-converts units wrongly (the LD025 0.04 -> 0.004 bug). Catching them
# here keeps extraction in the deterministic regex path.

# "X cm/mm/m away from q1" / "from A" → role r1 (if ref∈{q1,A}) or r2 (if ref∈{q2,B}).
_AWAY_FROM_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|m)\s+(?:away\s+)?from\s+"
    r"(?P<ref>q[0-9A-D]|q_?\d+|[A-D]\b)",
    re.IGNORECASE,
)

# "X cm apart" / "X cm long" → source separation AB (no ref token).
_APART_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm|m)\s+(?:apart|long)\b",
    re.IGNORECASE,
)


def _ref_to_role(ref: str) -> str | None:
    """Map a reference token to a canonical role variable name."""
    r = ref.lower().replace("_", "")
    if r in ("q1", "a"):
        return "r1"
    if r in ("q2", "b"):
        return "r2"
    if r in ("q3", "c"):
        return "r3"
    return None


@dataclass(frozen=True)
class ExtractedQuantity:
    """A single ``name = value unit`` triple parsed from the question."""

    name: str
    value: float
    unit: str
    raw_value: str
    raw_unit: str


def _parse_value(raw: str) -> float:
    """Turn a (possibly ``6 * 10 ^ -8``) string into a float."""
    cleaned = raw.replace(" ", "")
    if "*10" in cleaned:
        mant, _, exp = cleaned.partition("*10")
        exp = exp.lstrip("^")
        return float(float(mant) * (10 ** int(exp)))
    return float(cleaned)


# Unit tokens we will refuse to treat as units — they're usually a stray
# word right after the number ("Calculate ... = 100 m apart"). The regex
# greedily grabs the next token; the filter trims false positives.
_UNIT_BLOCKLIST: frozenset[str] = frozenset(
    {
        "and",
        "in",
        "of",
        "the",
        "is",
        "at",
        "to",
        "with",
        "while",
        "apart",
        "respectively",
        "where",
        "when",
        "so",
        "if",
        "from",
        "between",
        "by",
    }
)


def extract_quantities(text: str) -> list[ExtractedQuantity]:  # noqa: PLR0912, PLR0915
    """Return all ``name = value unit`` triples found in ``text``.

    Three passes (in order; later passes don't overwrite earlier names):

    1. Standard ``name = value unit`` (the original Day-2 extractor).
    2. G3 (Day-25): **chained-equality broadcast** —
       ``q1 = q2 = q3 = X unit`` yields three entries.
    3. G3 (Day-25): **"<noun> of <value> <unit>" prose** — maps noun to
       a canonical variable name when not already extracted.

    Order is preserved (left-to-right). The same name can appear twice if
    the question genuinely re-defines it; callers pick the first match.
    """
    normalized = _denormalize_exponents(unicodedata.normalize("NFKC", text))
    results: list[ExtractedQuantity] = []
    seen_names: set[str] = set()
    chain_spans: list[tuple[int, int]] = []

    # Pass 2 (run first to claim multi-name spans before pass 1 sees them):
    # chained equality broadcasts the value to every left-hand variable.
    for match in _CHAIN_EQ_RE.finditer(normalized):
        raw_value = match.group("value")
        raw_unit = _clean_unit(match.group("unit"))
        try:
            value = _parse_value(raw_value)
        except ValueError:
            continue
        names = [
            match.group(f"n{i}") for i in (1, 2, 3, 4) if match.group(f"n{i}")
        ]
        if len(names) < 2:
            continue
        for name in names:
            if name in seen_names:
                continue
            results.append(
                ExtractedQuantity(
                    name=name, value=value, unit=raw_unit,
                    raw_value=raw_value, raw_unit=raw_unit,
                )
            )
            seen_names.add(name)
        chain_spans.append((match.start(), match.end()))

    # Pass 1: standard `name = value unit` — skip spans already claimed by
    # the chained-equality pass to avoid double-counting the tail variable.
    for match in _ASSIGNMENT_RE.finditer(normalized):
        if any(s <= match.start() < e for s, e in chain_spans):
            continue
        name = match.group("name")
        if name in seen_names:
            continue
        raw_value = match.group("value")
        raw_unit = _clean_unit(match.group("unit"))
        try:
            value = _parse_value(raw_value)
        except ValueError:
            continue
        results.append(
            ExtractedQuantity(
                name=name, value=value, unit=raw_unit,
                raw_value=raw_value, raw_unit=raw_unit,
            )
        )
        seen_names.add(name)

    # Pass 3: prose "<noun> of <value> <unit>" — fill in canonical names
    # the equation passes didn't catch. Lowercase the noun for the lookup
    # because the regex is IGNORECASE.
    for match in _OF_PATTERN_RE.finditer(normalized):
        canonical = _NOUN_TO_VAR.get(match.group("noun").lower())
        if canonical is None or canonical in seen_names:
            continue
        raw_value = match.group("value")
        raw_unit = _clean_unit(match.group("unit"))
        try:
            value = _parse_value(raw_value)
        except ValueError:
            continue
        results.append(
            ExtractedQuantity(
                name=canonical, value=value, unit=raw_unit,
                raw_value=raw_value, raw_unit=raw_unit,
            )
        )
        seen_names.add(canonical)

    # Pass 4 (Iter-3): role-aware geometry. "X cm away from q1" → r1,
    # "X cm apart" → AB. These are the role-bearing phrases the prior
    # passes lose. Done last so equation-form (`r1 = X`) wins if both
    # forms appear.
    for match in _AWAY_FROM_RE.finditer(normalized):
        role = _ref_to_role(match.group("ref"))
        if role is None or role in seen_names:
            continue
        raw_value = match.group("value")
        raw_unit = _clean_unit(match.group("unit"))
        try:
            value = _parse_value(raw_value)
        except ValueError:
            continue
        results.append(
            ExtractedQuantity(
                name=role, value=value, unit=raw_unit,
                raw_value=raw_value, raw_unit=raw_unit,
            )
        )
        seen_names.add(role)

    if "AB" not in seen_names:
        for match in _APART_RE.finditer(normalized):
            raw_value = match.group("value")
            raw_unit = _clean_unit(match.group("unit"))
            try:
                value = _parse_value(raw_value)
            except ValueError:
                continue
            results.append(
                ExtractedQuantity(
                    name="AB", value=value, unit=raw_unit,
                    raw_value=raw_value, raw_unit=raw_unit,
                )
            )
            seen_names.add("AB")
            break  # first "X cm apart" wins

    # Pass 5 (Iter-3) — state-change extractors. Capture the ratio word
    # ("doubled" / "tripled" / "halved") as a dimensionless `ratio` and
    # the prose voltage source ("connected to a 50 V power source") as V1.
    # Order matters: V1 / ratio are LAST so an explicit `V1 = ...` form
    # (rare but possible) wins via Pass 1.
    if "ratio" not in seen_names:
        m = _DISTANCE_RATIO_RE.search(normalized)
        if m:
            word = m.group("word").lower()
            value = _DISTANCE_RATIO_WORDS[word]
            results.append(
                ExtractedQuantity(
                    name="ratio", value=value, unit="",
                    raw_value=word, raw_unit="",
                )
            )
            seen_names.add("ratio")

    if "V1" not in seen_names:
        m = _POWER_SOURCE_RE.search(normalized)
        if m:
            raw_value = m.group("value")
            raw_unit = _clean_unit(m.group("unit"))
            try:
                value = _parse_value(raw_value)
                results.append(
                    ExtractedQuantity(
                        name="V1", value=value, unit=raw_unit,
                        raw_value=raw_value, raw_unit=raw_unit,
                    )
                )
                seen_names.add("V1")
            except ValueError:
                pass

    # Pass 6 (Iter-3) — "has resistance 8 Ω" / "is at 4 V" prose patterns
    # for THCB070-style circuit-state questions where the OF pattern isn't
    # used. Maps via _HAS_NOUN_MAP so "has resistance" → R, etc.
    for m in _HAS_NOUN_RE.finditer(normalized):
        canonical = _HAS_NOUN_MAP.get(m.group("noun").lower())
        if canonical is None or canonical in seen_names:
            continue
        raw_value = m.group("value")
        raw_unit = _clean_unit(m.group("unit"))
        try:
            value = _parse_value(raw_value)
            results.append(
                ExtractedQuantity(
                    name=canonical, value=value, unit=raw_unit,
                    raw_value=raw_value, raw_unit=raw_unit,
                )
            )
            seen_names.add(canonical)
        except ValueError:
            pass

    if "U" not in seen_names:
        m = _IS_AT_VOLT_RE.search(normalized)
        if m:
            raw_value = m.group("value")
            raw_unit = _clean_unit(m.group("unit"))
            try:
                value = _parse_value(raw_value)
                results.append(
                    ExtractedQuantity(
                        name="U", value=value, unit=raw_unit,
                        raw_value=raw_value, raw_unit=raw_unit,
                    )
                )
                seen_names.add("U")
            except ValueError:
                pass

    # Iter-3 (Day-25 THCB070) — "lamp D2 draws 0.5 A" → I_remaining.
    if "I_remaining" not in seen_names:
        m = _DRAWS_RE.search(normalized)
        if m:
            raw_value = m.group("value")
            raw_unit = _clean_unit(m.group("unit"))
            try:
                value = _parse_value(raw_value)
                results.append(
                    ExtractedQuantity(
                        name="I_remaining", value=value, unit=raw_unit,
                        raw_value=raw_value, raw_unit=raw_unit,
                    )
                )
                seen_names.add("I_remaining")
            except ValueError:
                pass

    return results


def _clean_unit(raw: str | None) -> str:
    out = (raw or "").strip(".,;:")
    if out.lower() in _UNIT_BLOCKLIST:
        return ""
    return out


def find_by_name(quantities: list[ExtractedQuantity], name: str) -> ExtractedQuantity | None:
    """First quantity whose name matches ``name`` (case-sensitive)."""
    for q in quantities:
        if q.name == name:
            return q
    return None
