#!/usr/bin/env bash
# Pilot A, mechanical half: drive the annotators' command line end to end.
#
# WHAT THIS IS NOT. Pilot A is a protocol step involving two people, ten minutes
# of real audio and a question only humans can answer - "can the annotators
# explain, in their own words, what each tier is for?". This script cannot
# answer that and does not try. It runs the checks in the protocol's Pilot A
# list that a machine can run, so that when two people sit down the tooling is
# already known to work and the session is spent on the taxonomy.
#
# WHY IT EXISTS. The six defects fixed in this branch were all in the path a
# real annotator walks: the template they open, the file they hand back, the
# report the methodologist reads. "The unit tests pass" does not establish that
# the four commands compose, and the round trip is exactly where they stopped
# composing before.
#
#   ./scripts/rehearse_pilot_a.sh
#
# Output lands in docs/evidence/ and is committed, for the same reason the
# battery capture is: a run nobody can point at is not evidence.

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT="docs/evidence/pilot-a-rehearsal-$(date -u +%Y-%m-%dT%H%M%SZ).txt"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p docs/evidence

export PYTHONIOENCODING=utf-8
export NO_COLOR=1
export TERM=dumb

# Every check reports pass or fail and the script keeps going, so one broken
# step does not hide the state of the other seven.
FAILURES=0
check() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "  PASS  $label"
  else
    echo "  FAIL  $label"
    FAILURES=$((FAILURES + 1))
  fi
}

# Inverted: the command is expected to fail. Used for the refusals, where a
# command that succeeds is the defect.
check_refuses() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "  FAIL  $label (the command succeeded; it must refuse)"
    FAILURES=$((FAILURES + 1))
  else
    echo "  PASS  $label"
  fi
}

corpus() { uv run python -m corpus.cli.main "$@"; }

# Everything `corpus template` now requires beyond the recording's identity.
# Kept as one array because that is how an operator will keep it: these values
# change per participant, not per command, and retyping nine flags is how a
# consent basis ends up wrong.
RECRUITMENT=(
  --variety es-PE
  --consent-basis written_informed
  --consent-policy 1.0.0
  --consent-granted 2026-09-01
  --microphone "Realtek(R) Audio - onboard array"
  --sample-rate-hz 16000
  --bit-depth 16
  --channels 1
  --virtual-audio-bypassed yes
)

{
  echo "OratorIA - Pilot A rehearsal (mechanical half)"
  echo "Captured (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Commit:         $(git rev-parse HEAD)"
  echo "Branch:         $(git rev-parse --abbrev-ref HEAD)"
  echo "Tree state:     $(git status --porcelain -- . ':!docs/evidence' | wc -l) uncommitted file(s) outside docs/evidence"
  echo "Python:         $(python --version 2>&1)"
  echo
  echo "This is not Pilot A. It is the part of Pilot A's checklist a machine"
  echo "can run. The human half - whether two annotators can explain the tiers"
  echo "in their own words - is unanswered and stays unanswered until they do."
  echo

  # -------------------------------------------------------------------------
  echo "=== 1. the template opens, and offers the taxonomy and nothing else ==="
  # -------------------------------------------------------------------------
  corpus template "$WORK/pilot-ana.eaf" \
    --recording-id pilot-a-001 --speaker P-001 --annotator ana --media pilot-a-001.wav \
    "${RECRUITMENT[@]}"
  corpus template "$WORK/pilot-beto.eaf" \
    --recording-id pilot-a-001 --speaker P-001 --annotator beto --media pilot-a-001.wav \
    "${RECRUITMENT[@]}"
  echo

  uv run python scripts/_pilot_a_checks.py inspect-template "$WORK/pilot-ana.eaf"
  echo

  # -------------------------------------------------------------------------
  echo "=== 2. the template refuses to overwrite a finished annotation ==="
  # -------------------------------------------------------------------------
  check_refuses "corpus template refuses an existing path" \
    corpus template "$WORK/pilot-ana.eaf" \
    --recording-id pilot-a-001 --speaker P-001 --annotator ana --media pilot-a-001.wav \
    "${RECRUITMENT[@]}"
  check "corpus template --force replaces it" \
    corpus template "$WORK/pilot-ana.eaf" \
    --recording-id pilot-a-001 --speaker P-001 --annotator ana --media pilot-a-001.wav \
    "${RECRUITMENT[@]}" --force
  echo

  # -------------------------------------------------------------------------
  echo "=== 3. two annotated files, as two annotators would hand them back ==="
  # -------------------------------------------------------------------------
  # Synthetic, and said so. Two people annotating the same ten minutes is what
  # Pilot A measures; this fixture only establishes that the format survives
  # the round trip and that the report renders over disagreement rather than
  # over an empty pair of files.
  uv run python scripts/_pilot_a_checks.py write-annotated "$WORK"
  echo

  echo "--- corpus validate ana ---"
  corpus validate "$WORK/annotated-ana.eaf"
  echo
  echo "--- corpus validate beto ---"
  corpus validate "$WORK/annotated-beto.eaf"
  echo

  # -------------------------------------------------------------------------
  echo "=== 4. the agreement report renders ==="
  # -------------------------------------------------------------------------
  corpus agreement "$WORK/annotated-ana.eaf" "$WORK/annotated-beto.eaf"
  echo

  # -------------------------------------------------------------------------
  echo "=== 5. the report does not depend on which file was named first ==="
  # -------------------------------------------------------------------------
  corpus agreement "$WORK/annotated-ana.eaf" "$WORK/annotated-beto.eaf" --json \
    > "$WORK/forward.json"
  corpus agreement "$WORK/annotated-beto.eaf" "$WORK/annotated-ana.eaf" --json \
    > "$WORK/backward.json"
  uv run python scripts/_pilot_a_checks.py compare-directions "$WORK"
  echo

  # -------------------------------------------------------------------------
  echo "=== 6. the adjudication worklist is produced ==="
  # -------------------------------------------------------------------------
  uv run python scripts/_pilot_a_checks.py show-worklist "$WORK/forward.json"
  echo

  # -------------------------------------------------------------------------
  echo "=== 7. what the tooling refuses ==="
  # -------------------------------------------------------------------------
  uv run python scripts/_pilot_a_checks.py write-refusable "$WORK"
  check_refuses "an adjudicated pass is refused" \
    corpus agreement "$WORK/annotated-ana.eaf" "$WORK/adjudicated.eaf"
  check_refuses "two files by one annotator are refused" \
    corpus agreement "$WORK/annotated-ana.eaf" "$WORK/annotated-ana.eaf"
  check_refuses "an incompatible schema version is refused" \
    corpus agreement "$WORK/annotated-ana.eaf" "$WORK/future-schema.eaf"
  check_refuses "a major taxonomy difference is refused" \
    corpus agreement "$WORK/annotated-ana.eaf" "$WORK/other-taxonomy.eaf"
  # The two the review found. A minor difference used to pass with a note, and
  # a missing version used to become `None` and skip the guard entirely.
  check_refuses "a minor taxonomy difference is refused too" \
    corpus agreement "$WORK/annotated-ana.eaf" "$WORK/minor-taxonomy.eaf"
  check_refuses "a file recording no taxonomy version is refused" \
    corpus agreement "$WORK/annotated-ana.eaf" "$WORK/no-taxonomy.eaf"
  check_refuses "a file with a validation error is refused" \
    corpus agreement "$WORK/annotated-ana.eaf" "$WORK/overlapping.eaf"
  check_refuses "an IoU threshold of 0 is refused" \
    corpus agreement "$WORK/annotated-ana.eaf" "$WORK/annotated-beto.eaf" --iou 0
  check_refuses "a negative boundary review threshold is refused" \
    corpus agreement "$WORK/annotated-ana.eaf" "$WORK/annotated-beto.eaf" \
    --boundary-review-ms -1
  echo

  # -------------------------------------------------------------------------
  echo "=== result ==="
  # -------------------------------------------------------------------------
  if [ "$FAILURES" -eq 0 ]; then
    echo "mechanical half: all checks passed"
  else
    echo "mechanical half: $FAILURES check(s) failed"
  fi
  echo
  echo "Still outstanding for Pilot A proper:"
  echo "  - ten minutes of real audio from a recording that will not enter the corpus"
  echo "  - two annotators, working independently"
  echo "  - both able to explain what each tier is for, in their own words"
  echo "  - the capture chain verified: no NVIDIA Broadcast, no Voicemeeter"
  echo "    (see docs/governance/REFERENCE_ENVIRONMENT.md)"
} 2>&1 | tee "$OUTPUT"

echo
echo "written: $OUTPUT"
exit "$FAILURES"
