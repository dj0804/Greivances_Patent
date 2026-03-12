#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_VENV_PYTHON="$BACKEND_DIR/venv/bin/python"

DEFAULT_BACKEND_PORT="${BACKEND_PORT:-8000}"
DEFAULT_FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"

if [[ ! -d "$BACKEND_DIR" || ! -d "$FRONTEND_DIR" ]]; then
  echo "Error: expected 'backend/' and 'frontend/' directories under $ROOT_DIR"
  exit 1
fi

if ! command -v lsof >/dev/null 2>&1; then
  echo "Error: 'lsof' is required but not found on PATH"
  exit 1
fi

if [[ -x "$BACKEND_VENV_PYTHON" ]]; then
  BACKEND_PYTHON="$BACKEND_VENV_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  BACKEND_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  BACKEND_PYTHON="$(command -v python)"
else
  echo "Error: no Python interpreter found."
  exit 1
fi

if ! "$BACKEND_PYTHON" -c "import fastapi" >/dev/null 2>&1; then
  echo "Error: FastAPI is not installed for backend interpreter: $BACKEND_PYTHON"
  echo "Install backend dependencies first (e.g., pip install -r backend/Requirements.txt)."
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Info: frontend dependencies missing, running npm install..."
  (
    cd "$FRONTEND_DIR"
    npm install
  )
fi

if ! "$BACKEND_PYTHON" -c "import fastapi, sentence_transformers" >/dev/null 2>&1; then
  echo "Info: backend dependencies missing, installing from backend/Requirements.txt..."
  "$BACKEND_PYTHON" -m pip install -r "$BACKEND_DIR/Requirements.txt"
fi

if ! "$BACKEND_PYTHON" -m fastapi --help >/dev/null 2>&1; then
  echo "Info: fastapi CLI extras missing, installing fastapi[standard]..."
  "$BACKEND_PYTHON" -m pip install "fastapi[standard]" >/dev/null
fi

if ! "$BACKEND_PYTHON" -m fastapi --help >/dev/null 2>&1; then
  echo "Error: FastAPI CLI is unavailable for backend interpreter: $BACKEND_PYTHON"
  echo "Try: $BACKEND_PYTHON -m pip install \"fastapi[standard]\""
  exit 1
fi

if ! (cd "$BACKEND_DIR" && "$BACKEND_PYTHON" -c "import api.main") >/dev/null 2>&1; then
  echo "Error: backend app import failed for entrypoint api.main:app"
  echo "Verify backend dependencies are installed and backend/api is importable."
  exit 1
fi

port_in_use() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

find_free_port() {
  local start_port="$1"
  local port="$start_port"
  while port_in_use "$port"; do
    port=$((port + 1))
  done
  echo "$port"
}

BACKEND_PORT_SELECTED="$(find_free_port "$DEFAULT_BACKEND_PORT")"
FRONTEND_PORT_SELECTED="$(find_free_port "$DEFAULT_FRONTEND_PORT")"

if [[ "$BACKEND_PORT_SELECTED" != "$DEFAULT_BACKEND_PORT" ]]; then
  echo "Info: backend port $DEFAULT_BACKEND_PORT busy, using $BACKEND_PORT_SELECTED"
fi

if [[ "$FRONTEND_PORT_SELECTED" != "$DEFAULT_FRONTEND_PORT" ]]; then
  echo "Info: frontend port $DEFAULT_FRONTEND_PORT busy, using $FRONTEND_PORT_SELECTED"
fi

BACKEND_BASE_URL="http://localhost:$BACKEND_PORT_SELECTED"
FRONTEND_LOCK_FILE="$FRONTEND_DIR/.next/dev/lock"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo
  echo "Stopping backend and frontend..."

  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi

  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi

  wait >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

echo "Starting backend on $BACKEND_BASE_URL"
(
  cd "$BACKEND_DIR"
  "$BACKEND_PYTHON" -m fastapi dev -e api.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT_SELECTED"
) &
BACKEND_PID=$!

sleep 1
if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
  echo "Error: backend failed to start"
  exit 1
fi

echo "Starting frontend on http://localhost:$FRONTEND_PORT_SELECTED"

if [[ -f "$FRONTEND_LOCK_FILE" ]]; then
  if lsof "$FRONTEND_LOCK_FILE" >/dev/null 2>&1; then
    echo "Error: another Next.js dev instance is already using $FRONTEND_LOCK_FILE"
    echo "Stop that process (or remove stale lock) and rerun run_dev.sh"
    exit 1
  else
    echo "Info: removing stale Next.js lock file"
    rm -f "$FRONTEND_LOCK_FILE"
  fi
fi

(
  cd "$FRONTEND_DIR"
  BACKEND_API_BASE_URL="$BACKEND_BASE_URL" npm run dev -- --port "$FRONTEND_PORT_SELECTED"
) &
FRONTEND_PID=$!

sleep 1
if ! kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
  echo "Error: frontend failed to start"
  exit 1
fi

echo

echo "✅ Dev stack is running"
echo "- Backend:  $BACKEND_BASE_URL"
echo "- Frontend: http://localhost:$FRONTEND_PORT_SELECTED"
echo
echo "Press Ctrl+C to stop both services"

while true; do
  if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
