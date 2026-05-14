# Architecture

## High-level data flow

```
POST /predict
   │
   ▼
api/routes.py  ──▶  Orchestrator (agent/orchestrator.py)
                        │
                        │  route_task(payload)
                        ▼
              ┌──────────────────┬───────────────────┐
              │  task=logic       │  task=physics     │
              ▼                   ▼
        logic/pipeline.py    physics/pipeline.py
              │                   │
              │  premise_selector │  topic_classifier
              │  rule_parser      │  quantity_extractor
              │  forward_chainer  │  unit_converter
              │  z3_verifier      │  formula_library
              │  answer_verifier  │  solver
              │  explanation      │  verifier
              │                   │  explanation
              ▼                   ▼
             ────  agent/output_formatter.py  ────
                            │
                            ▼
                  PredictResponse (JSON)
```

## Module responsibilities

| Module | Role |
| --- | --- |
| `api/` | HTTP transport. Stateless; delegates to `Orchestrator`. |
| `router.py` | Decide `logic` vs `physics` from the payload. |
| `agent/orchestrator.py` | Run the chosen pipeline; one process-wide instance. |
| `agent/output_formatter.py` | Coerce internal dicts → `PredictResponse`. |
| `logic/` | Rule + FOL reasoning over premises-NL. Z3 fallback. |
| `physics/` | Quantity extraction, SymPy compute via `formula_library.yaml`. |
| `llm/` | vLLM client + Jinja2 prompts. Activated Phase 4. |
| `agent/self_correction.py` | Solver-grounded LLM revision loop. Phase 5. |
| `train/` | SFT (Unsloth) and GRPO (TRL) entry points. |
| `eval/` | Local metric harness over the 10% holdout split. |

## Key invariants

1. **Numbers come from the solver, not the LLM.** The physics pipeline
   never substitutes an LLM-emitted number into the final response.
2. **Premises are first-class citizens.** The logic pipeline always
   returns the IDs of the premises it used; the `premises` field is the
   primary P3 signal.
3. **`PredictResponse` is the only public schema.** Pipelines may return
   richer dicts internally, but the API answer is normalized through
   `output_formatter`.
