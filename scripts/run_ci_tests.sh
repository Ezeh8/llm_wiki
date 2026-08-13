#!/usr/bin/env bash
# Step 18 CI gate — runs tests/test_endpoints.py, the ~52-test PRD Section 9 Step 18
# suite ("~52 tests, in-memory Postgres, mocked Qdrant. 3 per endpoint + specific
# failure tests. Failing test blocks deploy."). This is meant to be fast enough to run
# on every push/PR — nothing here downloads model weights or needs a live Qdrant.
#
# "In-memory Postgres": there is no real in-memory Postgres product (unlike SQLite's
# `:memory:` mode), and this schema leans on Postgres-specific types (native UUID,
# JSONB, `INSERT ... ON CONFLICT`) that a substitute engine like SQLite can't run
# unmodified — swapping the adapter layer's dialect just for tests would be a bigger,
# riskier change than Step 18 asks for. This script instead runs a real, throwaway
# postgres:16 container with its data directory on tmpfs (RAM-backed, never touches
# disk) and fsync/full_page_writes/synchronous_commit off — genuinely as fast as an
# in-memory database in practice, with zero schema compatibility risk. Same
# "set env vars before Python starts" pattern used for every Docker-backed test
# throughout this build (see BUILD_LOG Steps 8-17) — app.db.session's engine is a
# module-level singleton built from DATABASE_URL at import time, so the container must
# be up and the URL exported before pytest ever imports app code.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONTAINER_NAME="llm_wiki_ci_postgres_$$"
PORT="${CI_POSTGRES_PORT:-55432}"
ADMIN_USER="llm_wiki_admin"
ADMIN_PASSWORD="ci-admin-password"
APP_USER="llm_wiki_app"
APP_PASSWORD="ci-app-password"
DB_NAME="llm_wiki"
DEAD_LETTER_DIR="$(mktemp -d)"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  rm -rf "$DEAD_LETTER_DIR"
}
trap cleanup EXIT

echo "starting ephemeral tmpfs postgres ($CONTAINER_NAME on port $PORT)..."
docker run -d --name "$CONTAINER_NAME" \
  -e POSTGRES_USER="$ADMIN_USER" \
  -e POSTGRES_PASSWORD="$ADMIN_PASSWORD" \
  -e POSTGRES_DB="$DB_NAME" \
  -p "$PORT:5432" \
  --tmpfs /var/lib/postgresql/data \
  postgres:16 \
  -c fsync=off -c full_page_writes=off -c synchronous_commit=off \
  >/dev/null

echo "waiting for postgres to accept connections..."
for _ in $(seq 1 30); do
  if docker exec "$CONTAINER_NAME" pg_isready -U "$ADMIN_USER" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [ -z "${ready:-}" ]; then
  echo "postgres did not become ready in time" >&2
  docker logs "$CONTAINER_NAME" >&2 || true
  exit 1
fi

export DATABASE_ADMIN_URL="postgresql+asyncpg://${ADMIN_USER}:${ADMIN_PASSWORD}@localhost:${PORT}/${DB_NAME}"
export DATABASE_URL="postgresql+asyncpg://${APP_USER}:${APP_PASSWORD}@localhost:${PORT}/${DB_NAME}"
# Deliberately unreachable — this suite mocks Qdrant entirely (Step 18 spec) rather
# than requiring a live vector store; port 1 refuses instantly instead of timing out.
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:1}"
export LLM_API_KEY="${LLM_API_KEY:-ci-test-key-not-used}"
export DEAD_LETTER_PATH="$DEAD_LETTER_DIR"

echo "running alembic migrations (creates schema + restricted role + grants)..."
.venv/bin/python -m alembic upgrade head

echo "running test suite..."
.venv/bin/python -m pytest tests/test_endpoints.py -v "$@"
