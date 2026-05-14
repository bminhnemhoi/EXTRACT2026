"""Formula registry — single source of truth for which equations the solver
knows about. Definitions live in ``configs/physics_formulas.yaml`` so adding
a new dataset coverage gap is a config edit, not a code change.

Each :class:`Formula` exposes:

* the SymPy expression used to compute the output magnitude,
* the input symbols expected (with their canonical SI units),
* a description used for the explanation trace,
* the topic prefix it belongs to (for the rule-based classifier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import sympy
import yaml

from exact_agent.config import get_settings


@dataclass(frozen=True)
class FormulaInput:
    """Specification for a formula input variable."""

    name: str
    unit: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Formula:
    """A single named equation, loaded from YAML."""

    id: str
    topic: str
    description: str
    inputs: dict[str, FormulaInput]
    output_symbol: str
    output_unit: str
    expression_text: str
    expression: sympy.Expr = field(repr=False)

    def required_symbols(self) -> tuple[str, ...]:
        return tuple(self.inputs.keys())

    def compute(self, values_si: dict[str, float]) -> float:
        """Evaluate the expression with SI-canonical magnitudes.

        Missing keys raise :class:`KeyError`. Numeric blow-ups (e.g. division
        by zero) surface SymPy's native errors; the caller can decide to fall
        back to the LLM.
        """
        missing = [s for s in self.inputs if s not in values_si]
        if missing:
            raise KeyError(f"missing inputs for formula {self.id!r}: {missing}")

        substitutions = {sympy.Symbol(name): values_si[name] for name in self.inputs}
        result = self.expression.evalf(subs=substitutions)
        if not result.is_real:
            raise ValueError(f"non-real result for formula {self.id!r}: {result}")
        return float(result)


class FormulaLibrary:
    """Loaded collection of :class:`Formula` objects keyed by id."""

    def __init__(self, formulas: dict[str, Formula]) -> None:
        self._by_id = formulas

    def __contains__(self, formula_id: str) -> bool:
        return formula_id in self._by_id

    def __getitem__(self, formula_id: str) -> Formula:
        return self._by_id[formula_id]

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id.keys())

    def by_topic(self, topic: str) -> tuple[Formula, ...]:
        return tuple(f for f in self._by_id.values() if f.topic == topic)

    def all(self) -> tuple[Formula, ...]:
        return tuple(self._by_id.values())


def _parse_input_spec(name: str, raw: dict[str, Any]) -> FormulaInput:
    unit = raw.get("unit")
    if not unit:
        raise ValueError(f"input {name!r} is missing required field 'unit'")
    aliases = tuple(raw.get("aliases") or ())
    return FormulaInput(name=name, unit=unit, aliases=aliases)


def _parse_formula(formula_id: str, payload: dict[str, Any]) -> Formula:
    inputs_raw = payload.get("inputs") or {}
    inputs = {name: _parse_input_spec(name, spec) for name, spec in inputs_raw.items()}
    output = payload.get("output") or {}
    expression_text = payload.get("expression")
    if not expression_text:
        raise ValueError(f"formula {formula_id!r} missing 'expression'")

    sym_locals = {name: sympy.Symbol(name) for name in inputs}
    try:
        expr = sympy.sympify(expression_text, locals=sym_locals)
    except (sympy.SympifyError, SyntaxError, TypeError) as exc:
        raise ValueError(f"could not parse expression for {formula_id!r}: {exc}") from exc

    return Formula(
        id=formula_id,
        topic=str(payload.get("topic") or ""),
        description=str(payload.get("description") or ""),
        inputs=inputs,
        output_symbol=str(output.get("symbol") or ""),
        output_unit=str(output.get("unit") or ""),
        expression_text=expression_text,
        expression=expr,
    )


def load_library(path: Path | None = None) -> FormulaLibrary:
    """Load formulas from YAML. Defaults to ``configs/physics_formulas.yaml``."""
    target = path or (get_settings().configs_dir / "physics_formulas.yaml")
    if not target.exists():
        raise FileNotFoundError(f"formula config not found at {target}")

    with target.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_formulas: dict[str, Any] = data.get("formulas") or {}
    formulas = {fid: _parse_formula(fid, payload) for fid, payload in raw_formulas.items()}
    if not formulas:
        raise ValueError(f"no formulas found in {target}")
    return FormulaLibrary(formulas)


@lru_cache(maxsize=1)
def default_library() -> FormulaLibrary:
    """Process-wide singleton built from the default config path."""
    return load_library()
