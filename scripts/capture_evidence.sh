#!/usr/bin/env bash
# Run the full battery against real infrastructure and record the raw output.
#
# This exists because "the integration tests pass" is a claim, and a claim about
# a run that nobody can point at is worth about as much as no claim. The output
# lands in docs/evidence/, is committed, and carries the versions it ran
# against - so a reader six months from now can tell whether the run predates
# the change they are investigating.
#
#   docker compose up -d
#   ENGINE_DATABASE_URL=... uv run alembic upgrade head
#   ./scripts/capture_evidence.sh
#
# Deliberately not run by CI. CI proves the tests pass on every commit; this
# produces a durable artifact for a specific moment, and regenerating it on
# every push would make it noise.

set -euo pipefail

cd "$(dirname "$0")/.."

DATABASE_URL="${ENGINE_TEST_DATABASE_URL:-postgresql+asyncpg://engine:engine@localhost:5432/engine}"
OUTPUT="docs/evidence/battery-$(date -u +%Y-%m-%dT%H%M%SZ).txt"

mkdir -p docs/evidence

export PYTHONIOENCODING=utf-8
export ENGINE_DATABASE_URL="$DATABASE_URL"
# No colour. The file is read by people and diffed by tools; ANSI escapes
# make it unreadable in both.
export NO_COLOR=1
export TERM=dumb

{
  echo "OratorIA Multimodal Evidence Engine - battery evidence"
  echo "Captured (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Commit:         $(git rev-parse HEAD)"
  echo "Branch:         $(git rev-parse --abbrev-ref HEAD)"
  # A dirty tree makes the commit hash a lie about what actually ran, so it
  # is reported rather than assumed away.
  # Excludes docs/evidence: this file is being written by `tee` right now,
  # so counting it would report every clean capture as dirty.
  echo "Tree state:     $(git status --porcelain -- . ':!docs/evidence' | wc -l) uncommitted file(s) outside docs/evidence"
  echo "Python:         $(python --version 2>&1)"
  echo "Database:       ${DATABASE_URL%%://*}://... (credentials redacted)"
  echo

  echo "=== infrastructure ==="
  docker compose ps --format '{{.Service}}\t{{.Image}}\t{{.Status}}'
  echo

  echo "=== ruff check ==="
  uv run ruff check --no-cache --color never --output-format concise src tests scripts
  echo

  echo "=== ruff format --check ==="
  uv run ruff format --check --color never src tests scripts
  echo

  echo "=== mypy --strict ==="
  uv run mypy --no-color-output
  echo

  echo "=== import-linter ==="
  uv run lint-imports
  echo

  echo "=== alembic: upgrade -> downgrade -> upgrade ==="
  # The round-trip, not just the upgrade. A migration that cannot be reversed
  # is a deployment that cannot be rolled back, and the first time that matters
  # is the worst time to find out.
  uv run alembic upgrade head
  uv run alembic downgrade base
  uv run alembic upgrade head
  echo "round-trip completed"
  echo

  echo "=== pytest: full battery, warnings as errors ==="
  uv run pytest -v --tb=short --color=no
  echo

  echo "=== pytest: integration only ==="
  uv run pytest -m integration -v --tb=short --color=no
  echo

  echo "=== coverage: domain + application + corpus (NFR-021 floor is 80) ==="
  uv run pytest -q --color=no \
    --cov=evidence_engine.domain \
    --cov=evidence_engine.application \
    --cov=corpus \
    --cov-branch \
    --cov-report=term
} 2>&1 | tee "$OUTPUT"

echo
echo "written: $OUTPUT"
