"""Unit normalization wrapping pint.

The competition data uses a mixture of unit notations:

* SI symbols: ``F``, ``V``, ``A``, ``Ω``, ``J``, ``W``, ``T``
* SI prefixes: ``μF``, ``mC``, ``mJ``, ``μC``, ``μH``
* ASCII fallbacks: ``uF`` for ``μF``, ``ohm`` for ``Ω``
* Composite: ``V/m``, ``V·m⁻¹``, ``turns/m``

We canonicalize everything to SI base units before the solver substitutes
values. The conversion *trace* is preserved so the explanation can quote
the conversion step ("convert C to 0.0001 F").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pint

# Recognises plain-digit area/volume forms like ``cm2``, ``mm3``, ``m2``
# (commonly emitted by LLMs and copy-paste from PDFs that strip the Unicode
# superscript). pint requires ``cm**2`` / ``cm^2``; this regex inserts the
# missing exponentiation marker without touching legitimate dimensionless
# tokens (``cm2per`` would only match the ``cm2`` head as desired).
_AREA_VOLUME_NORM = re.compile(r"\b(cm|mm|m|km|dm|nm)([23])\b")

# Lazily-initialised module-level registry. pint is heavy to construct.
# Held in a dict to avoid the `global` statement (ruff PLW0603).
_registry_holder: dict[str, pint.UnitRegistry] = {}


# Common Unicode/ASCII aliases the dataset uses. Mapped to pint-canonical
# spellings before the registry parses. Order matters — longer keys must
# precede their substrings (dict iteration is insertion order in 3.7+).
_UNIT_ALIASES: dict[str, str] = {
    "Ω": "ohm",
    "ohms": "ohm",
    "Ohm": "ohm",
    "OHM": "ohm",
    "μ": "u",  # pint accepts u as micro prefix
    "µ": "u",  # GREEK SMALL LETTER MU vs MICRO SIGN
    "°C": "degC",
    "°F": "degF",
    "·": "*",
    "⁻¹": "**-1",
    "⁻²": "**-2",
    "²": "**2",
    "³": "**3",
    # Iter-5 Fix B (DDT382/392): pint's built-in `turn` = 2π rad wins over
    # any re-definition, so naive "turns/m" → "/m" → 2π/m is WRONG for a
    # solenoid turn-density (which is dimensionless count per metre).
    # Map the whole composite to "1/m" directly. Longer keys come FIRST.
    "turns/m": "1/m",
    "turn/m": "1/m",
    "turns / m": "1/m",
    "turn / m": "1/m",
    "turns": "",      # bare "turns" (no /m) — dimensionless count
}


def get_registry() -> pint.UnitRegistry:
    """Return a process-wide UnitRegistry, creating it on first use."""
    if "ureg" not in _registry_holder:
        ureg: pint.UnitRegistry = pint.UnitRegistry()
        # Register a few convenience aliases that aren't standard in pint.
        ureg.define("turn = 1 = turns")
        _registry_holder["ureg"] = ureg
    return _registry_holder["ureg"]


def normalize_unit_string(raw: str) -> str:
    """Replace Unicode glyphs with pint-friendly ASCII before parsing."""
    text = raw.strip()
    # Normalise no-superscript area/volume forms first (``cm2`` -> ``cm**2``)
    # so the alias loop can still convert legitimate Unicode superscripts
    # without double-applying.
    text = _AREA_VOLUME_NORM.sub(r"\1**\2", text)
    for needle, replacement in _UNIT_ALIASES.items():
        text = text.replace(needle, replacement)
    return text.strip()


@dataclass(frozen=True)
class ConversionResult:
    """Outcome of converting a numeric quantity to a target unit."""

    value_si: float
    """Magnitude expressed in the requested target unit (typically SI base)."""

    original_value: float
    """Magnitude as originally extracted (before conversion)."""

    original_unit: str
    """Unit string as originally extracted."""

    target_unit: str
    """Target unit the value was converted into."""

    trace: str
    """Human-readable description, e.g. ``"Convert C to 0.0001 F"``."""


class UnitConversionError(ValueError):
    """Raised when a value cannot be parsed or converted."""


def convert(value: float, source_unit: str, target_unit: str) -> ConversionResult:
    """Convert ``value source_unit`` to ``target_unit``.

    Both unit strings are normalized for common Unicode glyphs (μ, Ω, ·, ⁻¹).
    On failure a :class:`UnitConversionError` is raised so callers can fall
    back to the LLM path without swallowing the underlying pint error.
    """
    ureg = get_registry()
    src = normalize_unit_string(source_unit)
    tgt = normalize_unit_string(target_unit)
    try:
        quantity = value * ureg.parse_expression(src)
        converted = quantity.to(ureg.parse_expression(tgt))
    except (
        pint.errors.UndefinedUnitError,
        pint.errors.DimensionalityError,
        AssertionError,
        AttributeError,
        ValueError,
        TypeError,
    ) as exc:
        # pint's parser can raise a surprising variety of exceptions on
        # malformed input; we treat them all as conversion failures so the
        # caller can fall back without crashing the eval harness.
        raise UnitConversionError(
            f"cannot convert {value} {source_unit!r} -> {target_unit!r}: {exc}"
        ) from exc

    si_value = float(converted.magnitude)
    trace = (
        f"Convert {_format_value(value)} {source_unit} to {_format_value(si_value)} {target_unit}"
    )
    return ConversionResult(
        value_si=si_value,
        original_value=value,
        original_unit=source_unit,
        target_unit=target_unit,
        trace=trace,
    )


def parse_quantity(text: str) -> tuple[float, str]:
    """Parse a free-form ``"100 μF"`` style string into ``(value, unit)``.

    Useful when the dataset glues value and unit together (e.g. ``"3 mC"``).
    """
    cleaned = normalize_unit_string(text)
    try:
        quantity = get_registry().parse_expression(cleaned)
    except Exception as exc:  # pint raises a number of subclasses
        raise UnitConversionError(f"cannot parse quantity {text!r}: {exc}") from exc
    if not hasattr(quantity, "magnitude"):
        raise UnitConversionError(f"{text!r} did not yield a numeric quantity")
    return float(quantity.magnitude), str(quantity.units)


def _format_value(value: float) -> str:
    """Compact numeric formatter — avoids gratuitous trailing zeros in the trace."""
    if value == 0:
        return "0"
    abs_v = abs(value)
    if abs_v >= 1e4 or abs_v < 1e-3:
        return f"{value:.4g}"
    # Drop trailing zeros for "100.0" → "100".
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"
