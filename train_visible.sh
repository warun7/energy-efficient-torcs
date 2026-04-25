#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export CUDA_VISIBLE_DEVICES="-1"

# Visible (non-headless) TORCS training run.
# NOTE: run this from a desktop session with X11 display available.
python ddpg_laptime.py \
  --train 1 \
  --episodes "${EPISODES:-2000}" \
  --max-steps "${MAX_STEPS:-100000}" \
  --artifact-dir "${ARTIFACT_DIR:-artifacts}" \
  --run-tag "${RUN_TAG:-laptime_visible}"
