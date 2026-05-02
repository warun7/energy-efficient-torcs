#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="${ROOT}/.venv"
PY="${PYTHON_BIN:-${VENV}/bin/python}"
ARTIFACTS="${ARTIFACTS:-${ROOT}/artifacts}"
MODEL_PREFIX="${MODEL_PREFIX:-sac_best_model}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-2500}"

TORCS_INST="${ROOT}/build/torcs-install"
export TORCS_BIN="${TORCS_BIN:-${TORCS_INST}/bin/torcs}"
export TORCS_PREFIX="${TORCS_PREFIX:-${TORCS_INST}}"
export TORCS_DATADIR="${TORCS_DATADIR:-${TORCS_INST}/share/games/torcs}"

MODEL_PATH="${ARTIFACTS}/${MODEL_PREFIX}_actor.pth"
if [ ! -f "$MODEL_PATH" ]; then
  printf 'ERROR: saved actor weights not found: %s\n' "$MODEL_PATH" >&2
  exit 1
fi

if [ ! -x "$PY" ]; then
  printf 'ERROR: python executable not found: %s\n' "$PY" >&2
  exit 1
fi

xvfb-run -a -s "-screen 0 1024x768x24" \
  env TORCS_BIN="$TORCS_BIN" TORCS_PREFIX="$TORCS_PREFIX" TORCS_DATADIR="$TORCS_DATADIR" \
  "$PY" sac.py \
    --train 0 \
    --episodes "$EVAL_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --artifact-dir "$ARTIFACTS" \
    --model-prefix "$MODEL_PREFIX" \
    --run-tag eval
