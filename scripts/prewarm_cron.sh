#!/usr/bin/env bash
# Keep the submission endpoint warm during the evaluation window.
#
# The Ollama model itself is pinned resident (OLLAMA_KEEP_ALIVE=-1 in
# docker-compose.yml), so this only guards against the *instance* idling
# and exercises the full solver→LLM path so nothing lazy-loads cold when
# the judges first call /predict.
#
# Install on the GPU host:
#   crontab -e
#   */4 * * * * /path/to/repo/scripts/prewarm_cron.sh >> /var/log/exact_prewarm.log 2>&1
set -euo pipefail

BASE="${EXACT_BASE_URL:-http://localhost:8000}"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Liveness — cheap, no LLM.
if ! curl -fsS --max-time 10 "${BASE}/healthz" >/dev/null 2>&1; then
  echo "${STAMP} UNHEALTHY ${BASE}/healthz"
  exit 1
fi

# 2. Real round-trip — keeps the physics solver + Ollama path hot.
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 \
  -X POST "${BASE}/predict" -H 'Content-Type: application/json' \
  -d '{"question":"Calculate the energy stored when C = 100 uF and U = 30 V."}')"

echo "${STAMP} healthz=ok predict=${code}"
[ "${code}" = "200" ] || exit 1
