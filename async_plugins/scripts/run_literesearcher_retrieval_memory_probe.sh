#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID=${RUN_ID:?RUN_ID is required}
PYTHON_BIN=${PYTHON_BIN:-python3}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROBE_SCRIPT=${PROBE_SCRIPT:-$SCRIPT_DIR/probe_literesearcher_retrieval_memory.py}
FORMAL_MARKER=${FORMAL_MARKER:-/tmp/agentmemory-formal-cpu-active}
YIELD_MARKER=${YIELD_MARKER:-/tmp/crg-holder-yield}
child_pid=
child_start_tick=

same_process() {
  local pid=$1 expected_tick=$2
  [[ -r "/proc/$pid/stat" ]] || return 1
  [[ "$(awk '{print $22}' "/proc/$pid/stat")" == "$expected_tick" ]]
}

remove_owned_marker() {
  local path=$1
  if [[ -f "$path" ]] && [[ "$(cat "$path")" == "$RUN_ID" ]]; then
    rm -f -- "$path"
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${child_pid:-}" ]] && [[ -n "${child_start_tick:-}" ]] \
      && same_process "$child_pid" "$child_start_tick"; then
    kill -TERM "$child_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      same_process "$child_pid" "$child_start_tick" || break
      sleep 1
    done
    if same_process "$child_pid" "$child_start_tick"; then
      kill -KILL "$child_pid" 2>/dev/null || true
    fi
    wait "$child_pid" 2>/dev/null || true
  fi
  remove_owned_marker "$FORMAL_MARKER"
  remove_owned_marker "$YIELD_MARKER"
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

for marker in "$FORMAL_MARKER" "$YIELD_MARKER"; do
  if [[ -e "$marker" ]] && [[ "$(cat "$marker")" != "$RUN_ID" ]]; then
    printf 'refusing to overwrite foreign marker %s=%s\n' "$marker" "$(cat "$marker")" >&2
    exit 73
  fi
done
printf '%s\n' "$RUN_ID" >"$FORMAL_MARKER"
printf '%s\n' "$RUN_ID" >"$YIELD_MARKER"

"$PYTHON_BIN" "$PROBE_SCRIPT" "$@" &
child_pid=$!
child_start_tick=$(awk '{print $22}' "/proc/$child_pid/stat")
wait "$child_pid"
