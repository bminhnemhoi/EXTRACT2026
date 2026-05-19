# Deploying the EXACT 2026 submission endpoint

The submission is a single always-on HTTP endpoint: `POST /predict` (+ `GET
/healthz` for cron pre-warm). It is the deterministic neuro-symbolic solver;
the LLM is an OpenAI-compatible **Ollama sidecar** used only as a utility
extractor / NL→FOL translator / explanation stabiliser — never to compute
answers (ADR 0001/0006).

## Topology

```
              docker/docker-compose.yml
  ┌─────────────┐   http://ollama:11434/v1   ┌──────────────────────┐
  │   ollama     │◀───────────────────────────│  api (FastAPI)        │
  │  GPU, qwen   │   model-pull (one-shot)     │  SymPy+Z3+pint solver │
  │  2.5:3b-inst │                             │  :8000 /predict       │
  └─────────────┘                             └──────────────────────┘
```

Why an LLM at all if the solver is deterministic: the physics
quantity-extractor and the logic NL→FOL translator fall back to the LLM when
regex/rule parsing misses. **Rule-only collapses physics to 0.8%; with the
sidecar it is 28.6% (logic 37.5%).** The deploy env therefore sets
`EXACT_SELF_CORRECTION__ENABLED=true`, which is the switch that makes the
Orchestrator build the LLM client (ADR 0008/0016).

## 1. Provision a GPU host

Runpod / Vast.ai / Lambda — any box with **docker + docker compose v2 +
nvidia-container-toolkit** and ≥ 6 GB VRAM (qwen2.5:3b q4 ≈ 2.6 GB; an
A10/A40/RTX-4000-class instance is ample). The plan budgets ~150–300 USD for
15 dev days + the evaluation week on one always-on instance.

- Runpod: pick a **"RunPod PyTorch" / Docker-enabled** template, expose a
  public TCP port mapped to container `:8000`.
- Vast.ai: choose an image with docker; "Open Ports" → `8000`.

## 2. Bring it up

```bash
git clone <repo> && cd exact-veriscope-agent
bash scripts/start_runpod.sh          # build + ollama + model pull + health-gate
```

First run downloads the model into the `ollama-models` named volume (~2–4
min). Restarts reuse it — cold recovery is seconds. The script blocks until
`/healthz` is green and prints the smoke command.

CPU-only fallback: delete the `deploy.resources` block from
`docker/docker-compose.yml`. Functional but ~10× slower — fine for a smoke,
too slow for the timed evaluation.

## 3. Always-on + pre-warm

- `restart: unless-stopped` on `ollama` and `api` survives daemon/host
  reboots.
- `OLLAMA_KEEP_ALIVE=-1` pins the model in VRAM (no ~5-min idle eviction →
  no mid-evaluation cold start).
- Instance keep-alive + full-path warm-up via cron:

  ```bash
  crontab -e
  */4 * * * * /path/to/repo/scripts/prewarm_cron.sh >> /var/log/exact_prewarm.log 2>&1
  ```

## 4. Smoke test

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"question":"Calculate the energy stored when C = 100 uF and U = 30 V."}'
# → {"answer":"0.045 J", "explanation":"...E = 0.5·C·U²...", "cot":[...],
#    "confidence":0.x}

curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"premises-NL":["All students who pass the exam receive credit.",
        "Alice is a student.","Alice passed the exam."],
       "question":"Does Alice receive credit?"}'
# → {"answer":"Yes", "premises":["P1","P2","P3"], "fol":"...", ...}
```

`scripts/smoke_api.py` runs the bundled 5 logic + 5 physics cases.

## 5. Resilience

- **Backup snapshot**: after `start_runpod.sh` is green, snapshot the
  instance (Runpod "Save Template" / Vast snapshot) so a dead host is
  re-launchable in minutes during the evaluation week.
- **Health contract**: `/healthz` → `{"status":"ok","version":"…"}`. The
  Dockerfile `HEALTHCHECK` + compose `condition: service_healthy` gate
  traffic until the stack is ready.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `api` healthy but every physics answer ≈ rule-only (0.8%) | LLM unreachable → confirm `ollama` healthy & `EXACT_LLM__BASE_URL=http://ollama:11434/v1`; `docker compose logs api \| grep -i llm`. |
| `model-pull` never exits | No internet egress on host, or wrong `OLLAMA_HOST`. `docker compose logs model-pull`. |
| First `/predict` slow then fast | Expected if `OLLAMA_KEEP_ALIVE` unset — it is set to `-1` here; verify the env landed (`docker compose exec ollama env`). |
| 500 on `/predict` | Should never happen — fallbacks degrade to the deterministic answer. Capture `request_id` from the JSON log and the payload. |
