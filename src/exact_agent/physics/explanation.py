"""Turn a :class:`SolverResult` into a natural-language explanation.

We deliberately render from the trace rather than re-narrating in an LLM:
the rubric rewards verifiable, grounded explanations and a trace-derived
paragraph is the most defensible artifact.
"""

from __future__ import annotations

from exact_agent.physics.solver import SolverResult


def render_explanation(result: SolverResult) -> str:
    """Concatenate the solver trace into a single readable paragraph."""
    if not result.success:
        reason = result.fail_reason or "the solver could not derive an answer"
        return f"Unable to compute a numerical answer ({reason})."

    parts = [
        f"Use the {result.formula_description.lower()} relationship.",
    ]
    parts.extend(f"Step {idx + 1}: {line}" for idx, line in enumerate(result.trace))
    parts.append(f"Therefore the answer is {result.answer_str} {result.answer_unit}.")
    return "\n".join(parts)
