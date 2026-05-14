# exact-veriscope-agent

Explainable Educational QA agent for **EXACT 2026** (IEEE IJCNN 2026 challenge).

The system answers two kinds of educational queries — first-order-logic reasoning over university regulations and numerical physics problems on circuits / electrostatics — while always returning a verifiable explanation.

## Architecture

```
POST /predict
   ├─ router          decides logic vs physics
   ├─ logic pipeline  premise selector → rule parser → forward chainer → Z3 → explanation
   └─ physics pipeline classifier → quantity extractor → unit converter → SymPy solver → explanation
        │
        └─ optional: LLM draft generator (Qwen3-8B SFT) + self-correction loop
```

Output schema: `{answer, explanation, cot, premises, fol, confidence}`.

## Quickstart

```bash
uv sync
uv run pytest -x
uv run uvicorn exact_agent.api.app:app --reload --port 8000
```

Smoke test:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "Calculate the energy stored when C = 100 μF and U = 30 V."}'
```

## Project layout

See [docs/architecture.md](docs/architecture.md). The repo follows src-layout (`src/exact_agent/`) with config separated from code (`configs/*.yaml`).

## Implementation roadmap

7 phases over ~15 days — baseline solvers first, train the LLM only after the API and eval harness are working. See `C:\Users\Admin\.claude\plans\t-i-chu-n-b-tham-distributed-moore.md` for the full plan.

| Phase | Days | Deliverable |
| --- | --- | --- |
| 1. Baseline solver (no LLM) | 1–4 | Physics & logic baselines hitting ≥65% / ≥50% |
| 2. FastAPI wiring | 5 | `/predict` end-to-end |
| 3. Local eval | 5 | Metrics on 10% holdout split |
| 4. SFT Qwen3-8B | 6 | LoRA adapter for explanation formatting |
| 5. Self-correction loop | 7 | Solver-grounded LLM corrections |
| 6. GRPO (optional) | 8–10 | TRL multi-reward |
| 7. Qwen-Scope (optional) | 10–12 | XAI diagnostic for demo/paper |

## License

MIT
