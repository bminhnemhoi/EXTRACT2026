#!/usr/bin/env bash
# Bring the EXACT 2026 submission endpoint up on a rented GPU host
# (Runpod / Vast / Lambda — anything with docker + nvidia-container-toolkit).
#
#   bash scripts/start_runpod.sh
#
# Idempotent: re-running rebuilds the api image and reuses the warmed
# ollama-models volume (no model re-download).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker/docker-compose.yml"
PORT="${PORT:-8000}"
HEALTH_URL="http://localhost:${PORT}/healthz"

cd "${REPO_ROOT}"

echo "==> docker / compose preflight"
command -v docker >/dev/null 2>&1 || { echo "FATAL: docker not installed"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "FATAL: docker compose v2 required"; exit 1; }

echo "==> Building + starting stack (ollama → model-pull → api)"
docker compose -f "${COMPOSE_FILE}" up -d --build

echo "==> Waiting for the model pull to finish (first run ~2–4 min)"
# model-pull is a one-shot; block until the api dependency is satisfied.
for _ in $(seq 1 120); do
  state="$(docker compose -f "${COMPOSE_FILE}" ps -a --format '{{.Service}} {{.State}}' \
            | awk '/^model-pull/ {print $2}')"
  [ "${state}" = "exited" ] && break
  sleep 5
done

echo "==> Waiting for /healthz"
for _ in $(seq 1 60); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "OK  $(curl -fsS "${HEALTH_URL}")"
    break
  fi
  sleep 5
done

curl -fsS "${HEALTH_URL}" >/dev/null 2>&1 || {
  echo "FATAL: endpoint not healthy. Logs:"
  docker compose -f "${COMPOSE_FILE}" logs --tail=50 api
  exit 1
}

echo
echo "Endpoint is UP."
echo "  Local      : ${HEALTH_URL%/healthz}/predict"
echo "  Submit URL : map the host's public TCP \$PORT (Runpod 'Connect' /"
echo "               Vast 'Open Ports') to :${PORT}, then submit"
echo "               https://<public-host>/predict"
echo
echo "Smoke it:"
echo "  curl -s -X POST http://localhost:${PORT}/predict -H 'Content-Type: application/json' \\"
echo "    -d '{\"question\":\"Energy stored when C = 100 uF and U = 30 V.\"}'"
