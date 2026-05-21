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


def clean(question: str) -> str:
    """Return a normalized copy of ``question`` (idempotent)."""
    if not question:
        return ""
    text = unicodedata.normalize("NFKC", question)
    text = text.translate(_UNICODE_MINUS_MAP)
    text = _BARE_TEN_NEG_EXPONENT_RE.sub(r"10^-\1", text)
    text = text.replace(_NBSP, " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
