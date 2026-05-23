"""Light-touch text normalization run before the extractor.

The dataset is mostly well-formed but contains a few systematic noise
patterns that hurt regex / keyword matching:

* Duplicated whitespace and stray ``"\\n"``.
* Imperative prefixes (``"Calculate"``, ``"Determine"``, ``"Find"``) that
  the classifier doesn't need to see.
* Non-breaking spaces (U+00A0) glued onto numbers.

Aggressive rewriting is out of scope — we keep the original wording so the
explanation can still quote it back accurately.
"""

from __future__ import annotations

import re
import unicodedata

_NBSP = " "
_WHITESPACE_RE = re.compile(r"\s+")
# Iter-6: dataset uses Unicode minus / en-dash / em-dash interchangeably
# with ASCII hyphen in expressions like ``C = 10−6 F`` or ``q = -3·10−8 C``.
# Mathematical regex stops at non-ASCII chars, dropping the exponent sign
# and giving e.g. C=10 instead of C=1e-6. NFKC does NOT normalize U+2212;
# we do it explicitly here so all downstream extractors see ASCII ``-``.
_UNICODE_MINUS_MAP = str.maketrans({
    "−": "-",  # U+2212 MINUS SIGN
    "–": "-",  # U+2013 EN DASH
    "—": "-",  # U+2014 EM DASH (used as minus in some rows)
    "‑": "-",  # U+2011 NON-BREAKING HYPHEN
    "‒": "-",  # U+2012 FIGURE DASH
})
# After unicode-minus normalization, ``C = 10−6 F`` becomes ``C = 10-6 F``.
# The extractor's scientific-notation regex needs ``10^-N`` form to fire;
# the bare ``10-N`` (no caret, no space) is read as ``C = 10`` and the
# exponent silently dropped. This regex catches ``10-N`` when it's flanked
# by space/equals on the left and a digit-then-space on the right, then
# rewrites to ``10^-N`` so the existing extractor catches it.
_BARE_TEN_NEG_EXPONENT_RE = re.compile(r"(?<=[\s=])10-(\d+)(?=\s)")

# Iter-19g (Day-29): √-literal handling — the dataset writes RMS-vs-peak
# voltages as "100√2 V" / "200√3 V" etc. The extractor's _NUMBER regex
# stops at the first non-digit, returning 100, and the √2 factor is lost
# (NL361 voltage 100√2 → 141.42 V was being read as 100 V).
#
# Pre-resolve the √ to a numeric multiplier before the extractor runs.
# Patterns covered (using Unicode U+221A √):
#   "100√2"    -> "141.42135624"
#   "100*√2"   -> "141.42135624"
#   "100 √2"   -> "141.42135624"
#   "√2"       -> "1.41421356"
#   "100/√3"   -> "57.7350269"
import math as _math
_SQRT_VALUES = {
    2: _math.sqrt(2),
    3: _math.sqrt(3),
    5: _math.sqrt(5),
    6: _math.sqrt(6),
    7: _math.sqrt(7),
    10: _math.sqrt(10),
}
_SQRT_DIV_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*√\s*(\d+)")
_SQRT_MUL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[\*·]?\s*√\s*(\d+)")
_SQRT_LEADING_RE = re.compile(r"(?<![\d.])√\s*(\d+)")


def _resolve_sqrt_literals(text: str) -> str:
    def _div(m: re.Match[str]) -> str:
        mant, n = float(m.group(1)), int(m.group(2))
        return f"{mant / _SQRT_VALUES.get(n, _math.sqrt(n)):.6g}"
    def _mul(m: re.Match[str]) -> str:
        mant, n = float(m.group(1)), int(m.group(2))
        return f"{mant * _SQRT_VALUES.get(n, _math.sqrt(n)):.6g}"
    def _lead(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return f"{_SQRT_VALUES.get(n, _math.sqrt(n)):.6g}"
    text = _SQRT_DIV_RE.sub(_div, text)
    text = _SQRT_MUL_RE.sub(_mul, text)
    text = _SQRT_LEADING_RE.sub(_lead, text)
    return text


def clean(question: str) -> str:
    """Return a normalized copy of ``question`` (idempotent)."""
    if not question:
        return ""
    text = unicodedata.normalize("NFKC", question)
    text = text.translate(_UNICODE_MINUS_MAP)
    text = _BARE_TEN_NEG_EXPONENT_RE.sub(r"10^-\1", text)
    text = _resolve_sqrt_literals(text)
    text = text.replace(_NBSP, " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
