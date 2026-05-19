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


def parse_number(text: str | float | int | None) -> float | None:
    """Best-effort numeric coercion. Returns ``None`` on failure."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        v = float(text)
        return v if math.isfinite(v) else None
    s = str(text).strip()
    if not s:
        return None
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

    Comparing in the gold's natural unit (magnitude ~1) instead of raw
    SI (magnitude ~1e-10) also removes the ``abs_tol``-swamps-tiny-values
    failure mode of the old path.
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
            return math.isclose(converted.value_si, e, rel_tol=rel_tol, abs_tol=abs_tol)

    # No usable units (or inconvertible) → legacy raw compare.
    return math.isclose(p, e, rel_tol=rel_tol, abs_tol=abs_tol)


# ---------------------------------------------------------------------------
# Unit comparison
# ---------------------------------------------------------------------------


def unit_match(predicted: str | None, expected: str | None) -> bool:
    """True when the two unit strings share dimensionality (V == volt, J == joule)."""
    if not predicted or not expected:
        # An empty predicted unit is acceptable when the gold is also empty.
        return (predicted or "").strip() == (expected or "").strip()
    pred_norm = normalize_unit_string(predicted)
    exp_norm = normalize_unit_string(expected)
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
