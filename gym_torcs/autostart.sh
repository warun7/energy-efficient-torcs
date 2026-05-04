#!/usr/bin/env bash

set -euo pipefail

WINDOW_NAME_PATTERN="${TORCS_WINDOW_PATTERN:-torcs|TORCS|vtorcs}"
WINDOW_WAIT_SECONDS="${TORCS_WINDOW_WAIT_SECONDS:-30}"
STEP_DELAY_MS="${TORCS_AUTOSTART_DELAY_MS:-350}"
STARTUP_SETTLE_SECONDS="${TORCS_STARTUP_SETTLE_SECONDS:-2}"

sleep "${STARTUP_SETTLE_SECONDS}"

find_window_id() {
  local deadline torcs_pid binary_name window_id
  deadline=$((SECONDS + WINDOW_WAIT_SECONDS))
  binary_name="$(basename "${TORCS_BIN:-torcs}")"

  while (( SECONDS < deadline )); do
    if command -v xdotool >/dev/null 2>&1; then
      torcs_pid="$(pgrep -n -f "${TORCS_BIN:-torcs}" 2>/dev/null || true)"
      if [ -n "${torcs_pid}" ]; then
        while IFS= read -r window_id; do
          if [ -n "${window_id}" ]; then
            printf '%s\n' "${window_id}"
            return 0
          fi
        done < <(xdotool search --onlyvisible --pid "${torcs_pid}" 2>/dev/null || true)
      fi

      while IFS= read -r window_id; do
        if [ -n "${window_id}" ]; then
          printf '%s\n' "${window_id}"
          return 0
        fi
      done < <(xdotool search --onlyvisible --name "${WINDOW_NAME_PATTERN}" 2>/dev/null || true)

      while IFS= read -r window_id; do
        if [ -n "${window_id}" ]; then
          printf '%s\n' "${window_id}"
          return 0
        fi
      done < <(xdotool search --onlyvisible --name "${binary_name}" 2>/dev/null || true)
    fi
    sleep 1
  done

  return 1
}

send_key_sequence() {
  local window_id="$1"

  if command -v xdotool >/dev/null 2>&1; then
    xdotool windowactivate --sync "${window_id}"
    xdotool key --window "${window_id}" --delay "${STEP_DELAY_MS}" Return Return Up Up Return Return
    return 0
  fi

  xte 'key Return'
  xte 'usleep 350000'
  xte 'key Return'
  xte 'usleep 350000'
  xte 'key Up'
  xte 'usleep 350000'
  xte 'key Up'
  xte 'usleep 350000'
  xte 'key Return'
  xte 'usleep 350000'
  xte 'key Return'
}

if window_id="$(find_window_id)"; then
  send_key_sequence "${window_id}"
else
  echo "autostart.sh: TORCS window not found after ${WINDOW_WAIT_SECONDS}s" >&2
  exit 1
fi
