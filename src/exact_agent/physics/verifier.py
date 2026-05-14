"""Plausibility checks on a solver answer.

The verifier is the first line of defense against silent numerical
nonsense (NaN, infinity, absurd magnitudes). It does **not** know what the
correct answer is; it only flags answers that physics rules out.

It accepts ``(value, unit)`` directly — not a ``SolverResult`` — to keep
this module free of a circular dependency with ``solver``. The pipeline
calls :func:`verify` after the solver has produced its provisional output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Per-output-unit sanity bounds (lower, upper). Anything outside this
# range is almost certainly an extraction / unit error. The bounds are
# deliberately broad — they reject "1e30 farad" without flagging the
# legitimate 100 μF.
_PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "joule": (1e-12, 1e9),
    "volt": (1e-9, 1e7),
    "ampere": (1e-9, 1e5),
    "ohm": (1e-3, 1e9),
    "watt": (1e-12, 1e9),
    "farad": (1e-15, 1e3),
    "newton": (1e-12, 1e9),
    "volt/meter": (1e-6, 1e12),
    "tesla": (1e-15, 1e3),
    "hertz": (1e-6, 1e15),
    "henry": (1e-9, 1e3),
}


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    warnings: tuple[str, ...]


def verify(value: float | None, unit: str) -> VerificationResult:
    """Run sanity checks on a numeric answer + its unit."""
    if value is None:
        return VerificationResult(ok=False, warnings=("solver did not produce an answer",))

    warnings: list[str] = []

    if not math.isfinite(value):
        warnings.append(f"non-finite result: {value}")

    bounds = _PLAUSIBLE_RANGE.get(unit.lower())
    if bounds is not None and math.isfinite(value):
        lo, hi = bounds
        if value != 0 and (abs(value) < lo or abs(value) > hi):
            warnings.append(
                f"{value} {unit} is outside the plausible range [{lo}, {hi}] for {unit}"
            )

    return VerificationResult(ok=not warnings, warnings=tuple(warnings))
