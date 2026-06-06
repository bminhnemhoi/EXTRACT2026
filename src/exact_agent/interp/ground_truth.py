"""Derive ground-truth concepts from the *symbolic solver* (charter §2/§5).

Two layers:

* **Pure derivations** (``physics_concepts`` / ``logic_concepts``) turn already
  extracted solver facts into a lexical concept set. Stdlib only -> unit-tested.
* **Lazy adapters** (``ground_truth_for_*``) actually run the solver. They
  import the heavy pipeline *inside* the function so importing this module
  stays cheap and GPU/pint/sympy-free.

GROUND-TRUTH RULE: only the solver decides what concepts a sample "uses".
For physics that is the matched formula (id, topic, input aliases, output) and
its description; for logic it is the supporting premises + predicate names.
"""

from __future__ import annotations

import re

from exact_agent.interp.concepts import (
    split_identifier,
    tokenize_concepts,
    topic_concepts,
)
from exact_agent.interp.records import GroundTruthConcepts

_PRED_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


# ---------------------------------------------------------------------------
# Pure derivations (no heavy imports)
# ---------------------------------------------------------------------------


def physics_concepts(
    *,
    description: str,
    input_aliases: list[str],
    output_symbol: str,
    topic: str | None,
) -> set[str]:
    """Lexical concept set for a solved physics sample.

    Built from the formula description, the input aliases (``capacitance``,
    ``voltage``, ...), the output symbol words, and the topic expansion.
    Bare 1–2 char symbols (C, U) are intentionally dropped by the tokenizer;
    their *aliases* carry the meaning.
    """
    out: set[str] = set()
    out |= tokenize_concepts(description)
    for alias in input_aliases:
        out |= split_identifier(alias)
    out |= split_identifier(output_symbol)
    out |= topic_concepts(topic)
    return out


def extract_predicates(fol_strings: list[str]) -> set[str]:
    """Pull predicate identifiers out of FOL strings -> word tokens."""
    out: set[str] = set()
    for s in fol_strings or []:
        for name in _PRED_RE.findall(s):
            out |= split_identifier(name)
    return out


def logic_concepts(
    *,
    supporting_premise_texts: list[str],
    fol_strings: list[str],
) -> set[str]:
    """Lexical concept set for a solved logic sample: supporting premise
    text tokens + predicate-name tokens from the FOL the solver used."""
    out: set[str] = set()
    for text in supporting_premise_texts:
        out |= tokenize_concepts(text)
    out |= extract_predicates(fol_strings)
    return out


# ---------------------------------------------------------------------------
# Lazy adapters (run the actual solver — heavy imports inside)
# ---------------------------------------------------------------------------


def ground_truth_for_physics(sample_id: str, question: str) -> GroundTruthConcepts:
    """Run the physics solver and reduce its result to ground-truth concepts."""
    from exact_agent.physics.formula_library import default_library  # noqa: PLC0415
    from exact_agent.physics.solver import PhysicsSolver  # noqa: PLC0415

    lib = default_library()
    result = PhysicsSolver(library=lib).solve(question)
    # The classifier sets formula_id *before* input extraction, so a routed
    # formula is known even when the (no-LLM) solve later fails on a missing
    # input. The ground-truth concept is the routed formula's domain; solver_ok
    # tracks whether the full numeric solve succeeded.
    if result.formula_id is None:
        return GroundTruthConcepts(sample_id, "physics", [], False, "no_formula")

    formula = lib[result.formula_id]
    aliases: list[str] = []
    for spec in formula.inputs.values():
        aliases.extend(getattr(spec, "aliases", []) or [])
    aliases.extend(formula.inputs.keys())
    concepts = physics_concepts(
        description=formula.description,
        input_aliases=aliases,
        output_symbol=getattr(formula, "output_symbol", "") or "",
        topic=getattr(formula, "topic", None),
    )
    return GroundTruthConcepts(
        sample_id, "physics", sorted(concepts), result.success, "formula+aliases+topic"
    )


def ground_truth_for_logic(
    sample_id: str,
    question: str,
    premises_nl: list[str],
    premises_fol: list[str] | None = None,
) -> GroundTruthConcepts:
    """Run the logic pipeline and reduce its result to ground-truth concepts.

    Uses the response's ``premises`` field (the supporting premise labels the
    verifier emitted) to recover which premise texts were actually used.
    """
    from exact_agent.logic.pipeline import LogicPipeline  # noqa: PLC0415
    from exact_agent.schemas import PredictRequest  # noqa: PLC0415

    payload = PredictRequest(
        question=question,
        **{"premises-NL": premises_nl, "premises-FOL": premises_fol},
    )
    resp = LogicPipeline().run(payload)
    used_texts: list[str] = []
    for label in resp.premises or []:
        # labels look like "P3: John completed the required courses."
        _, _, text = str(label).partition(":")
        used_texts.append(text.strip() or str(label))
    fol_strings = list(premises_fol or [])
    if resp.fol:
        fol_strings.append(resp.fol)
    concepts = logic_concepts(
        supporting_premise_texts=used_texts, fol_strings=fol_strings
    )
    ok = bool(resp.answer) and resp.answer.lower() not in {"unknown", "uncertain", ""}
    return GroundTruthConcepts(sample_id, "logic", sorted(concepts), ok, "supports+predicates")
