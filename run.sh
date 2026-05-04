#!/usr/bin/env bash
# One-shot setup + train + evaluate for Ubuntu 22.04 (e.g. fresh Docker clone).
# All paths are relative to the repository root (this script's directory).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

TORCS_SRC="${ROOT}/gym_torcs/vtorcs-RL-color"
TORCS_INST="${ROOT}/build/torcs-install"
STAMP="${ROOT}/build/.torcs_built"
VENV="${ROOT}/.venv"
ARTIFACTS="${ROOT}/artifacts"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

# Short defaults keep CI/evaluation runtime reasonable; override when needed.
TRAIN_EPISODES="${TRAIN_EPISODES:-200}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-2500}"

APT_UPDATE="${APT_UPDATE:-1}"

log() { printf '%s\n' "$*"; }

if ! command -v apt-get >/dev/null 2>&1; then
  wget "https://sourceforge.net/projects/torcs/files/all-in-one/1.3.7/torcs-1.3.7.tar.bz2/download" -O torcs-1.3.7.tar.bz2
  log "ERROR: apt-get not found. This script targets Debian/Ubuntu (e.g. ubuntu:22.04)."
  exit 1
fi

if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo -E"
elif [ "$(id -u)" -ne 0 ]; then
  log "ERROR: need root or sudo to install packages."
  exit 1
else
  SUDO=""
fi

export DEBIAN_FRONTEND=noninteractive

if [ "$APT_UPDATE" != "0" ]; then
  $SUDO apt-get update -y
fi

$SUDO apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  procps \
  python3 \
  python3-pip \
  python3-venv \
  automake \
  autoconf \
  libtool \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  freeglut3-dev \
  libplib-dev \
  libopenal-dev \
  libalut-dev \
  libpng-dev \
  zlib1g-dev \
  libx11-dev \
  libxext-dev \
  libxi-dev \
  libxmu-dev \
  libxxf86vm-dev \
  libxrender-dev \
  libxrandr-dev \
  libice-dev \
  libsm-dev \
  libxt-dev \
  x11-utils \
  xbitmaps \
  xvfb \
  xautomation

mkdir -p "${ROOT}/build" "${ARTIFACTS}"

if [ ! -f "$STAMP" ]; then
  log "Building TORCS (vtorcs) into ${TORCS_INST} ..."
  if [ ! -x "${TORCS_SRC}/configure" ]; then
    log "ERROR: missing ${TORCS_SRC}/configure"
    exit 1
  fi
  (
    cd "$TORCS_SRC"
    find . -name ".depend" -delete
    # Wipe any host-compiled .o/.so artifacts so make does a full in-container build.
    make -k clean 2>/dev/null || true
    ./configure --prefix="$TORCS_INST"
    make -j1
    make install
    make datainstall
  )
  # Disable sound for headless docker execution to prevent ALSA crashes
  sed -i 's/val="openal"/val="disabled"/g' "$TORCS_INST/share/games/torcs/config/sound.xml"
  touch "$STAMP"
fi

if [ ! -x "${TORCS_INST}/bin/torcs" ]; then
  log "ERROR: TORCS binary missing after build: ${TORCS_INST}/bin/torcs"
  exit 1
fi

log "Creating Python venv..."
if [ ! -x "$PY" ] || ! "$PY" --version >/dev/null 2>&1 || ! "$PIP" --version >/dev/null 2>&1; then
  log "Recreating venv (existing interpreter missing or broken)..."
  rm -rf "$VENV"
  python3 -m venv "$VENV"
fi
"$PIP" install --upgrade pip
"$PIP" install -r "${ROOT}/requirements.txt"

export TORCS_BIN="${TORCS_INST}/bin/torcs"
export TORCS_PREFIX="$TORCS_INST"
export TORCS_DATADIR="${TORCS_INST}/share/games/torcs"

log "TORCS_BIN=${TORCS_BIN}"
log "Training (${TRAIN_EPISODES} episodes, max ${MAX_STEPS} steps/episode)..."
(
  cd "$ROOT"
  xvfb-run -a -s "-screen 0 1024x768x24" \
    env TORCS_BIN="$TORCS_BIN" TORCS_PREFIX="$TORCS_PREFIX" TORCS_DATADIR="$TORCS_DATADIR" \
    "$PY" sac.py \
      --train 1 \
      --episodes "$TRAIN_EPISODES" \
      --max-steps "$MAX_STEPS" \
      --fuel-lambda 0.2 \
      --artifact-dir "$ARTIFACTS" \
      --run-tag train \
    2>&1 | tee "${ARTIFACTS}/log_train.txt"
)

log "Evaluation (${EVAL_EPISODES} episodes)..."
(
  cd "$ROOT"
  xvfb-run -a -s "-screen 0 1024x768x24" \
    env TORCS_BIN="$TORCS_BIN" TORCS_PREFIX="$TORCS_PREFIX" TORCS_DATADIR="$TORCS_DATADIR" \
    "$PY" sac.py \
      --train 0 \
      --episodes "$EVAL_EPISODES" \
      --max-steps "$MAX_STEPS" \
      --fuel-lambda 0.2 \
      --artifact-dir "$ARTIFACTS" \
      --run-tag eval \
    2>&1 | tee "${ARTIFACTS}/log_eval.txt"
)

log "Done. Artifacts under ${ARTIFACTS}/"
ls -la "$ARTIFACTS"
