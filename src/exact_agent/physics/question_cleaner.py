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


def clean(question: str) -> str:
    """Return a normalized copy of ``question`` (idempotent)."""
    if not question:
        return ""
    text = unicodedata.normalize("NFKC", question)
    text = text.replace(_NBSP, " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
