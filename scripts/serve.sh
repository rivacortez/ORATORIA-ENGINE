#!/usr/bin/env bash
# Run the engine locally and open the documented API.
#
#   ./scripts/serve.sh                 # memory backend, deterministic runtimes
#   ./scripts/serve.sh --baseline      # the pinned Whisper research baseline
#   ./scripts/serve.sh --reload        # restart on edit - see the warning below
#
# Then:
#   http://127.0.0.1:8000/v1/docs      Swagger - call an endpoint
#   http://127.0.0.1:8000/v1/redoc     ReDoc - read the contract end to end
#   http://127.0.0.1:8000/v1/openapi.json
#
# The secrets below are development values and are visible on purpose. Real
# ones come from a secret manager (§15.2); a script that read them from a file
# in the repository would be teaching the wrong habit on the machine where it
# is cheapest to learn.

set -euo pipefail

cd "$(dirname "$0")/.."

export ENGINE_ENVIRONMENT=local
export ENGINE_BACKEND=memory
export ENGINE_API_KEY_PEPPER=local-development-pepper-at-least-32-characters
export ENGINE_STREAM_TOKEN_SIGNING_KEY=local-development-signing-key-32-chars
export PYTHONIOENCODING=utf-8

# The key the server will accept. Minted here rather than hard-coded, so the
# credential that opens a local instance is different on every machine and is
# not a string somebody can read off this file in the repository.
#
# Set ENGINE_BOOTSTRAP_API_KEY yourself to keep one across restarts - useful
# when a browser tab already has it in the Authorize box.
if [ -z "${ENGINE_BOOTSTRAP_API_KEY:-}" ]; then
  ENGINE_BOOTSTRAP_API_KEY="$(uv run evidence-engine issue-key --json | python -c 'import json,sys; print(json.load(sys.stdin)["secret"])')"
fi
export ENGINE_BOOTSTRAP_API_KEY

MODE=deterministic
RELOAD=no
for argument in "$@"; do
  case "$argument" in
    --baseline)
      MODE=baseline_whisper
      ;;
    --reload)
      RELOAD=yes
      ;;
    *)
      echo "unknown option: $argument" >&2
      exit 2
      ;;
  esac
done
export ENGINE_RUNTIME_MODE="$MODE"

echo
echo "  backend       memory (nothing persists; restart is a clean slate)"
echo "  runtimes      $MODE"
if [ "$MODE" = "baseline_whisper" ]; then
  echo "                the pinned Whisper checkpoint - first start loads ~3 GB."
  echo "                It is the frozen RESEARCH BASELINE, not the project"
  echo "                model: it detects no disfluency and measures no"
  echo "                prosody, and its environment is NOT pinned (Phase 3)."
fi
echo
echo "  Swagger       http://127.0.0.1:8000/v1/docs"
echo "  ReDoc         http://127.0.0.1:8000/v1/redoc"
echo "  schema        http://127.0.0.1:8000/v1/openapi.json"
echo
echo "  Everything but /health needs a key. This one is registered at startup;"
echo "  press Authorize in Swagger and paste it:"
echo
echo "      $ENGINE_BOOTSTRAP_API_KEY"
echo
echo "  It is a local development key: the server refuses to start with one set"
echo "  outside ENGINE_ENVIRONMENT=local (§15.2)."
echo

# Auto-reload is OFF by default, and that is a decision rather than an
# oversight. On this machine (Windows 11, uvicorn with WatchFiles) a reload was
# announced - "WatchFiles detected changes ... Reloading..." - and the old
# worker kept serving: the schema fetched afterwards was the pre-edit one, with
# no error anywhere. A server that says it reloaded and did not is worse than
# one that never offered to, because everything checked against it afterwards
# is unverified while looking verified.
#
# Recording a presentation does not need reload at all. If you are editing the
# engine, pass --reload and restart by hand when something looks stale.
if [ "$RELOAD" = "yes" ]; then
  echo "  ! --reload requested. If a change does not appear, restart: the"
  echo "    reloader has been observed announcing a reload and serving the"
  echo "    old code on Windows."
  echo
  exec uv run uvicorn "evidence_engine.bootstrap.app:create_app" \
    --factory --host 127.0.0.1 --port 8000 --reload
fi

exec uv run uvicorn "evidence_engine.bootstrap.app:create_app" \
  --factory --host 127.0.0.1 --port 8000
