"""Scoring primitives used by all evaluation harnesses.

The competition rubric uses three signals:

* **P1 (correctness)** — does the predicted answer agree with the gold?
  For physics the gold is numeric; we use a relative tolerance. For
  logic the gold is a label (``A``/``B``/``Yes``/``Unknown``); we do
  case-insensitive exact match.
* **P2 (explanation quality)** — surface-form similarity to the gold
  explanation. We approximate with token-overlap F1 for the Day-3
  baseline; embedding similarity can replace this later without
  touching the harness.
* **P3 (reasoning depth)** — proxy: number of CoT steps that quote a
  formula/unit/premise, plus presence of FOL / structured premises.

All metrics are deterministic and dependency-light. They are *not* the
final competition scorer — they are our local proxies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from exact_agent.physics.unit_converter import (
    UnitConversionError,
    convert,
    normalize_unit_string,
)

# ---------------------------------------------------------------------------
# Numeric comparison
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*[*x×]\s*10\s*\^?\s*[-+]?\d+|[eE][-+]?\d+)?")

# Iter-16e (Day-29): translate Unicode superscript digits + signs to ASCII so
# golds like "8.48 × 10⁶" or "1.99 × 10⁻³" parse as scientific notation
# instead of collapsing to the mantissa. 6 rows in the SFT-unseen holdout
# false-failed for exactly this reason (LD392, LD394, DDT362/382/384/392).
_SUPERSCRIPT_TRANSLATE = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
})

# Iter-18a (Day-29): dataset-specific notations the existing regex
# can't parse. The Vietnamese-style "X . 10^Y" dot-multiplier and the
# Latex "10^{N}" braces appear in 2+ holdout golds (DT046 "3 . 10^4",
# DT051 "1.22 . 10^{-3}"). Normalised away before _NUMBER_RE runs so
# no regex edit is needed.
_DOT_MULTIPLIER_RE = re.compile(r"\s+\.\s+10\s*\^")
_LATEX_BRACE_EXP_RE = re.compile(r"10\s*\^\s*\{\s*([-+]?\d+)\s*\}")


def parse_number(text: str | float | int | None) -> float | None:
    """Best-effort numeric coercion. Returns ``None`` on failure."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        v = float(text)
        return v if math.isfinite(v) else None
    s = str(text).strip().translate(_SUPERSCRIPT_TRANSLATE)
    if not s:
        return None
    # Iter-18a: normalise "X . 10^Y" -> "X * 10^Y" and "10^{N}" -> "10^N".
    s = _LATEX_BRACE_EXP_RE.sub(r"10^\1", s)
    s = _DOT_MULTIPLIER_RE.sub(" * 10^", s)
    # Try literal first.
    try:
        v = float(s.replace(",", ""))
        return v if math.isfinite(v) else None
    except ValueError:
        pass
    # Otherwise pick the first number-looking token.
    match = _NUMBER_RE.search(s.replace(" ", ""))
    if not match:
        return None
    token = match.group(0)
    if "*10" in token or "×10" in token or "x10" in token:
        # Rewrite "6×10^-8" / "6*10^-8" / "6x10^-8" to scientific form.
        cleaned = re.sub(r"[*x×]10\^?", "e", token)
        try:
            return float(cleaned)
        except ValueError:
            return None
    try:
        return float(token)
    except ValueError:
        return None


def numeric_match(
    predicted: str | float | None,
    expected: str | float | None,
    *,
    rel_tol: float = 0.01,
    abs_tol: float = 1e-9,
) -> bool:
    """Compare two raw numbers under a relative tolerance (unit-blind).

    Kept for backward compatibility and for logic-side numeric checks.
    Physics must use :func:`quantity_match` — comparing a solver SI float
    (e.g. ``5.99e-10`` coulomb) against a gold prefixed string
    (``"0.6"`` nC) without unit normalization is wrong.
    """
    p = parse_number(predicted)
    e = parse_number(expected)
    if p is None or e is None:
        return False
    return math.isclose(p, e, rel_tol=rel_tol, abs_tol=abs_tol)


# A plain fixed-point decimal gold ("0.09", "14.83"); scientific / messy
# golds ("1.6 × 10^-7", "0.3; 1.5 cm") are excluded from round-aware
# acceptance and rely on the relative-tolerance path only.
_FIXED_DECIMAL_RE = re.compile(r"^[-+]?\d+\.(\d+)$")


def _gold_decimals(exp_value: str | float | None) -> int | None:
    """Number of decimal places the gold is *stated* at, or None.

    The dataset frequently instructs "round the result to two decimal
    places" and ships the rounded value as gold. Returns the decimal
    count only for a clean fixed-point string so we don't loosen
    integer or scientific golds.
    """
    if not isinstance(exp_value, str):
        return None
    m = _FIXED_DECIMAL_RE.match(exp_value.strip().replace(",", ""))
    return len(m.group(1)) if m else None


def _values_agree(
    pred: float,
    gold: float,
    gold_raw: str | float | None,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    """Relative-tolerance match OR stated-precision (round-aware) match.

    Round-aware only fires when the gold is a fixed-point decimal with
    >=1 decimal place — i.e. an explicitly rounded value. A prediction
    that rounds to the gold at the gold's own precision is correct by
    construction (the true value is only known to that precision).
    """
    if math.isclose(pred, gold, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    d = _gold_decimals(gold_raw)
    if d is not None and d >= 1:
        return round(pred, d) == round(gold, d)
    return False


def quantity_match(
    pred_value: str | float | None,
    pred_unit: str | None,
    exp_value: str | float | None,
    exp_unit: str | None,
    *,
    rel_tol: float = 0.01,
    abs_tol: float = 1e-9,
) -> bool:
    """Unit-aware numeric comparison.

    When both sides carry a unit and the units are dimensionally
    convertible, the predicted magnitude is converted into the gold's
    unit before the tolerance check — so ``5.99e-10 C`` matches the gold
    ``0.6 nC``. When a unit is missing or the conversion fails, we fall
    back to the legacy raw comparison (:func:`numeric_match` semantics)
    so the metric never regresses on unit-free answers.

    Acceptance is relative-tolerance OR stated-precision (round-aware):
    a prediction that rounds to the gold at the gold's own decimal
    precision counts, since dataset golds are explicitly rounded
    (e.g. √(2·0.54e-3/0.12)=0.0949 vs gold "0.09").
    """
    p = parse_number(pred_value)
    e = parse_number(exp_value)
    if p is None or e is None:
        return False

    pu = normalize_unit_string(pred_unit or "")
    eu = normalize_unit_string(exp_unit or "")
    if pu and eu:
        try:
            converted = convert(p, pu, eu)
        except UnitConversionError:
            converted = None
        if converted is not None:
            return _values_agree(converted.value_si, e, exp_value, rel_tol=rel_tol, abs_tol=abs_tol)

    # No usable units (or inconvertible) → legacy raw compare.
    return _values_agree(p, e, exp_value, rel_tol=rel_tol, abs_tol=abs_tol)


# ---------------------------------------------------------------------------
# Unit comparison
# ---------------------------------------------------------------------------

# Tokens that all mean "no physical unit". A dimensionless solver answer
# (e.g. the resonance frequency factor k) carries unit "dimensionless"
# while the gold writes "-" or leaves it blank — these must compare equal.
_DIMENSIONLESS_TOKENS = frozenset({"", "-", "–", "—", "dimensionless", "none", "n/a", "ratio"})


def _is_dimensionless(token: str) -> bool:
    return token.strip().lower() in _DIMENSIONLESS_TOKENS


def _strip_compound_unit(raw: str) -> str:
    """Iter-19g (Day-29): some dataset rows pack the value-unit and the
    error-unit into a single field with a semicolon separator
    (e.g. ``"g; g"`` for the THCB lab questions). Take the FIRST unit
    token so the dimensionality compare doesn't fail on the trailing
    ``"; g"`` suffix."""
    if ";" in raw:
        return raw.split(";")[0].strip()
    return raw


def unit_match(predicted: str | None, expected: str | None) -> bool:
    """True when the two unit strings share dimensionality (V == volt, J == joule).

    Any pair of "no real unit" spellings (``""``, ``"-"``, ``"dimensionless"``,
    …) is treated as a match so a numerically-correct dimensionless answer
    isn't failed on a cosmetic unit-string difference.
    """
    p_raw = _strip_compound_unit((predicted or "").strip())
    e_raw = _strip_compound_unit((expected or "").strip())
    if _is_dimensionless(p_raw) and _is_dimensionless(e_raw):
        return True
    if not p_raw or not e_raw:
        return p_raw == e_raw
    pred_norm = normalize_unit_string(p_raw)
    exp_norm = normalize_unit_string(e_raw)
    if pred_norm.lower() == exp_norm.lower():
        return True
    try:
        # Convert 1 unit_pred -> unit_exp; success means same dimension.
        convert(1.0, pred_norm, exp_norm)
    except UnitConversionError:
        return False
    return True


# ---------------------------------------------------------------------------
# Label comparison (for logic answers)
# ---------------------------------------------------------------------------


def label_match(predicted: str | None, expected: str | None) -> bool:
    if predicted is None or expected is None:
        return predicted == expected
    return predicted.strip().lower() == expected.strip().lower()


# ---------------------------------------------------------------------------
# Token-overlap F1 (P2 proxy)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass(frozen=True)
class TokenOverlap:
    precision: float
    recall: float
    f1: float


def token_overlap_f1(predicted: str, expected: str) -> TokenOverlap:
    """Bag-of-words F1 — a quick stand-in for explanation similarity."""
    p = _tokenize(predicted)
    e = _tokenize(expected)
    if not p and not e:
        return TokenOverlap(precision=1.0, recall=1.0, f1=1.0)
    if not p or not e:
        return TokenOverlap(precision=0.0, recall=0.0, f1=0.0)

    p_set = set(p)
    e_set = set(e)
    common = p_set & e_set
    precision = len(common) / len(p_set)
    recall = len(common) / len(e_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return TokenOverlap(precision=precision, recall=recall, f1=f1)
