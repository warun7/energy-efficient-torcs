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
TORCS_ARCH_STAMP="${ROOT}/build/.torcs_arch"
TORCS_UPSTREAM="${ROOT}/build/gym_torcs-upstream"
VENV="${ROOT}/.venv"
ARTIFACTS="${ROOT}/artifacts"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"
TORCS_BUILD_ARCH="$(dpkg --print-architecture 2>/dev/null || uname -m)"

# Defaults are chosen to allow a full Aalborg lap.
# 500 steps is only about 10 seconds at the 50 Hz control rate, which is too
# short for lap completion even with competent driving.
TRAIN_EPISODES="${TRAIN_EPISODES:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-1200}"
FUEL_LAMBDA="${FUEL_LAMBDA:-0.2}"

APT_UPDATE="${APT_UPDATE:-1}"

log() { printf '%s\n' "$*"; }

package_has_candidate() {
  local pkg="$1"
  local candidate
  candidate="$(apt-cache policy "$pkg" 2>/dev/null | awk '/Candidate:/ {print $2; exit}')"
  [ -n "$candidate" ] && [ "$candidate" != "(none)" ]
}

apt_update_with_retries() {
  local attempt
  local max_attempts=5
  local check_packages=(
    build-essential
    automake
    freeglut3-dev
    libxrender-dev
    xautomation
  )

  for attempt in $(seq 1 "$max_attempts"); do
    log "Refreshing apt package lists (attempt ${attempt}/${max_attempts}) ..."
    $SUDO rm -rf /var/lib/apt/lists/*

    if $SUDO apt-get \
      -o Acquire::Retries=3 \
      -o Acquire::http::Timeout=30 \
      -o Acquire::https::Timeout=30 \
      update -y; then
      local pkg
      local missing=0
      for pkg in "${check_packages[@]}"; do
        if ! package_has_candidate "$pkg"; then
          missing=1
          break
        fi
      done

      if [ "$missing" -eq 0 ]; then
        return 0
      fi

      log "apt update completed with incomplete package indexes; retrying ..."
    else
      log "apt update failed; retrying ..."
    fi

    sleep $((attempt * 5))
  done

  log "ERROR: unable to fetch complete apt package indexes after ${max_attempts} attempts."
  exit 1
}

apt_install_with_retries() {
  local attempt
  local max_attempts=3

  for attempt in $(seq 1 "$max_attempts"); do
    log "Installing Ubuntu packages (attempt ${attempt}/${max_attempts}) ..."

    if $SUDO apt-get install -y --no-install-recommends "$@"; then
      return 0
    fi

    if [ "$attempt" -lt "$max_attempts" ]; then
      log "Package install failed; refreshing apt indexes before retry ..."
      apt_update_with_retries
    fi
  done

  log "ERROR: failed to install required Ubuntu packages after ${max_attempts} attempts."
  exit 1
}

clean_torcs_source() {
  log "Cleaning generated TORCS source artifacts ..."

  find "${TORCS_SRC}" -type f \( \
    -name '*.o' -o \
    -name '*.so' -o \
    -name '*.a' -o \
    -name '*.la' -o \
    -name '*.lo' -o \
    -name '.depend' -o \
    -name 'Make-config' -o \
    -name 'config.status' -o \
    -name 'config.log' -o \
    -name 'config.cache' -o \
    -name 'torcs-bin' -o \
    -name 'trackgen-bin' -o \
    -name 'texmapper-bin' -o \
    -name 'nfs2ac-bin' -o \
    -name 'nfsperf-bin' -o \
    -name 'accc-bin' -o \
    -path '*/src/linux/torcs' -o \
    -path '*/src/tools/accc/accc' -o \
    -path '*/src/tools/nfs2ac/nfs2ac' -o \
    -path '*/src/tools/nfsperf/nfsperf' -o \
    -path '*/src/tools/texmapper/texmapper' -o \
    -path '*/src/tools/trackgen/trackgen' \
  \) -delete

  rm -rf "${TORCS_SRC}/export"
}

ensure_torcs_source() {
  if [ -x "${TORCS_SRC}/configure" ]; then
    return
  fi

  log "TORCS source tree missing; fetching upstream gym_torcs into ${TORCS_UPSTREAM} ..."
  mkdir -p "${ROOT}/gym_torcs" "${ROOT}/build"

  if [ ! -d "${TORCS_UPSTREAM}/.git" ]; then
    rm -rf "${TORCS_UPSTREAM}"
    git clone --depth 1 https://github.com/ugo-nama-kun/gym_torcs.git "${TORCS_UPSTREAM}"
  fi

  if [ ! -d "${TORCS_UPSTREAM}/vtorcs-RL-color" ]; then
    log "ERROR: upstream clone is missing vtorcs-RL-color"
    exit 1
  fi

  rm -rf "${TORCS_SRC}"
  cp -R "${TORCS_UPSTREAM}/vtorcs-RL-color" "${TORCS_SRC}"
}

patch_torcs_source() {
  python3 - "$TORCS_SRC" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

patches = {
    "src/drivers/olethros/geometry.cpp": {
        "needle": "#include <cmath>\n",
        "insert": "#include <cmath>\nusing std::isnan;\n",
    },
    "src/drivers/olethros/driver.cpp": {
        "needle": "#include <math.h>\n",
        "insert": "#include <math.h>\n#include <cmath>\nusing std::isnan;\n",
    },
    "src/tools/accc/ac3dload.cpp": {
        "needle": "#include <math.h>\n",
        "insert": "#include <math.h>\n#include <cmath>\nusing std::isnan;\n",
    },
    "src/modules/simu/simuv2/simu.cpp": {
        "needle": "#include <math.h>\n",
        "insert": "#include <math.h>\n#include <cmath>\nusing std::isnan;\nusing std::isinf;\n",
    },
    "src/modules/simu/simuv2/carstruct.h": {
        "needle": "#include <SOLID/solid.h>\n",
        "insert": "#include <SOLID/solid.h>\n#include <cmath>\nusing std::isnan;\nusing std::isinf;\n",
    },
    "src/libs/learning/policy.cpp": {
        "needle": "#include <learning/MathFunctions.h>\n",
        "insert": "#include <learning/MathFunctions.h>\n#include <cmath>\nusing std::isnan;\n",
    },
}

for relative_path, patch in patches.items():
    path = root / relative_path
    text = path.read_text(encoding="utf-8")
    if patch["insert"] not in text:
        text = text.replace(patch["needle"], patch["insert"], 1)
    path.write_text(text, encoding="utf-8")
PY
}

disable_torcs_sound() {
  local sound_xml="${TORCS_INST}/share/games/torcs/config/sound.xml"

  mkdir -p "$(dirname "$sound_xml")"
  cat > "$sound_xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "params.dtd">


<params name="sound">
  <section name="Sound Settings">
    <attstr name="state" val="disabled"/>
    <attnum name="volume" unit="%" val="100"/>
  </section>

</params>
EOF
}

if ! command -v apt-get >/dev/null 2>&1; then
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
export MPLCONFIGDIR="${ROOT}/.cache/matplotlib"

if [ "$APT_UPDATE" != "0" ]; then
  apt_update_with_retries
fi

apt_install_with_retries \
  build-essential \
  ca-certificates \
  procps \
  git \
  python3 \
  python3-pip \
  python3-venv \
  automake \
  autoconf \
  libglib2.0-dev \
  libtool \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  freeglut3-dev \
  libplib-dev \
  libopenal-dev \
  libalut-dev \
  libpng-dev \
  libvorbis-dev \
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

mkdir -p "${ROOT}/build" "${ARTIFACTS}" "${MPLCONFIGDIR}"
ensure_torcs_source

if [ -f "$STAMP" ]; then
  PREV_TORCS_ARCH=""
  if [ -f "$TORCS_ARCH_STAMP" ]; then
    PREV_TORCS_ARCH="$(cat "$TORCS_ARCH_STAMP")"
  fi

  if [ "$PREV_TORCS_ARCH" != "$TORCS_BUILD_ARCH" ]; then
    log "Rebuilding TORCS for ${TORCS_BUILD_ARCH} (previous build was ${PREV_TORCS_ARCH:-unknown}) ..."
    rm -rf "$TORCS_INST" "$STAMP" "$TORCS_ARCH_STAMP"
  fi
fi

if [ ! -f "$STAMP" ]; then
  log "Building TORCS (vtorcs) into ${TORCS_INST} ..."
  if [ ! -x "${TORCS_SRC}/configure" ]; then
    log "ERROR: missing ${TORCS_SRC}/configure"
    exit 1
  fi
  (
    cd "$TORCS_SRC"
    clean_torcs_source
    patch_torcs_source
    ./configure --prefix="$TORCS_INST"
    make TORCS_BASE="$TORCS_SRC" MAKE_DEFAULT="$TORCS_SRC/Make-default.mk" -j1
    make TORCS_BASE="$TORCS_SRC" MAKE_DEFAULT="$TORCS_SRC/Make-default.mk" install
    make TORCS_BASE="$TORCS_SRC" MAKE_DEFAULT="$TORCS_SRC/Make-default.mk" datainstall
  )
  touch "$STAMP"
  printf '%s\n' "$TORCS_BUILD_ARCH" > "$TORCS_ARCH_STAMP"
fi

if [ ! -x "${TORCS_INST}/bin/torcs" ]; then
  log "ERROR: TORCS binary missing after build: ${TORCS_INST}/bin/torcs"
  exit 1
fi

disable_torcs_sound

log "Creating Python venv..."
if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV"
fi
"$PIP" install --upgrade pip
"$PIP" install -r "${ROOT}/requirements.txt"

export TORCS_BIN="${TORCS_INST}/bin/torcs"
export TORCS_PREFIX="$TORCS_INST"
export TORCS_DATADIR="${TORCS_INST}/share/games/torcs"
export TORCS_LAUNCH_LOG="${ARTIFACTS}/torcs_runtime.log"
export ALSOFT_DRIVERS="${ALSOFT_DRIVERS:-null}"
: > "${TORCS_LAUNCH_LOG}"

log "TORCS_BIN=${TORCS_BIN}"
log "Training (${TRAIN_EPISODES} episodes, max ${MAX_STEPS} steps/episode)..."
(
  cd "$ROOT"
  xvfb-run -a -s "-screen 0 1024x768x24" \
    env TORCS_BIN="$TORCS_BIN" TORCS_PREFIX="$TORCS_PREFIX" TORCS_DATADIR="$TORCS_DATADIR" \
    "$PY" sac.py \
      --train 1 \
      --resume 0 \
      --episodes "$TRAIN_EPISODES" \
      --max-steps "$MAX_STEPS" \
      --fuel-lambda "$FUEL_LAMBDA" \
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
      --fuel-lambda "$FUEL_LAMBDA" \
      --artifact-dir "$ARTIFACTS" \
      --run-tag eval \
    2>&1 | tee "${ARTIFACTS}/log_eval.txt"
)

log "Generating summary plots..."
"$PY" "${ROOT}/scripts/generate_sac_plots.py" --artifact-dir "$ARTIFACTS"

log "Done. Artifacts under ${ARTIFACTS}/"
ls -la "$ARTIFACTS"