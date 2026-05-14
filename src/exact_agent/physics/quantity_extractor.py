"""Pull ``name = value unit`` triples out of a free-form physics question.

Examples we need to handle (taken from ``physics_train_safe.jsonl``):

* ``"C = 100 μF and U = 30 V"``
* ``"q1 = 6 × 10^-8 C"``  (Unicode times + exponent)
* ``"q3 = 6 × 10⁻⁸ C"``   (superscript exponent)
* ``"Q = 3 mC"``
* ``"CA = 5 cm and CB = 3 cm"``

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


def _denormalize_exponents(text: str) -> str:
    """Replace Unicode superscript exponents and times glyphs with ASCII."""
    return (
        text.translate(_SUPERSCRIPT_MAP)
        .replace("×", "*")
        .replace("·", "*")
        .replace("−", "-")  # MINUS SIGN U+2212 → HYPHEN-MINUS
    )


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


def extract_quantities(text: str) -> list[ExtractedQuantity]:
    """Return all ``name = value unit`` triples found in ``text``.

    Order is preserved (left-to-right). The same name can appear twice if
    the question genuinely re-defines it (rare); callers usually pick the
    first match.
    """
    normalized = _denormalize_exponents(unicodedata.normalize("NFKC", text))
    results: list[ExtractedQuantity] = []
    for match in _ASSIGNMENT_RE.finditer(normalized):
        name = match.group("name")
        raw_value = match.group("value")
        raw_unit = (match.group("unit") or "").strip(".,;:")
        # Drop trailing punctuation glued onto the unit by NFKC.
        if raw_unit.lower() in _UNIT_BLOCKLIST:
            raw_unit = ""
        try:
            value = _parse_value(raw_value)
        except ValueError:
            continue
        results.append(
            ExtractedQuantity(
                name=name,
                value=value,
                unit=raw_unit,
                raw_value=raw_value,
                raw_unit=raw_unit,
            )
        )
    return results


def find_by_name(quantities: list[ExtractedQuantity], name: str) -> ExtractedQuantity | None:
    """First quantity whose name matches ``name`` (case-sensitive)."""
    for q in quantities:
        if q.name == name:
            return q
    return None
