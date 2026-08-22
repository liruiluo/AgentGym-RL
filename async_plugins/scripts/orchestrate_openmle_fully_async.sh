#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: orchestrate_openmle_fully_async.sh \
  --mode gate|formal --python PATH --verl-root DIR --endpoint-root DIR \
  --fixture-root DIR --launcher-root DIR --run-id ID --run-dir DIR \
  --persist-root DIR --expected-plugin-commit SHA --expected-verl-commit SHA \
  --expected-endpoint-commit SHA --known-cpu-owner ID \
  --known-cpu-session NAME --known-cpu-start-command COMMAND \
  --known-cpu-script PATH --known-cpu-script-sha256 SHA \
  --known-gpu-owner ID --known-gpu-script PATH \
  --known-gpu-script-sha256 SHA --known-gpu-process-command COMMAND \
  [--trainer-gpus 4|6 --standalone-rollout-gpus 4|2] \
  [--actor-use-fused-kernels] [--critic-use-fused-kernels] \
  [--actor-ppo-max-tokens-per-gpu N] [--critic-ppo-max-tokens-per-gpu N]
EOF
  exit 64
}

MODE=
PY=
VERL=
ENDPOINT_OUTER=
FIX=
LAUNCHER_ROOT=
RUN_ID=
RUN_DIR=
PERSIST_ROOT=
EXPECTED_PLUGIN_COMMIT=
EXPECTED_VERL_COMMIT=
EXPECTED_ENDPOINT_COMMIT=
KNOWN_CPU_OWNER=
KNOWN_CPU_SESSION=
KNOWN_CPU_START_COMMAND=
KNOWN_CPU_SCRIPT=
KNOWN_CPU_SCRIPT_SHA256=
KNOWN_GPU_OWNER=
KNOWN_GPU_SCRIPT=
KNOWN_GPU_SCRIPT_SHA256=
KNOWN_GPU_PROCESS_COMMAND=
PORT=65524
EXPECTED_CHECKPOINT_BYTES=108992339992
MEMORY_CGROUP_USAGE_PATH=/sys/fs/cgroup/memory/memory.usage_in_bytes
MEMORY_CGROUP_LIMIT_PATH=/sys/fs/cgroup/memory/memory.limit_in_bytes
MEMORY_CGROUP_RUNTIME_MARGIN_BYTES=274877906944
TRAINER_GPUS=6
STANDALONE_ROLLOUT_GPUS=2
ACTOR_USE_FUSED_KERNELS=0
CRITIC_USE_FUSED_KERNELS=0
ACTOR_PPO_MAX_TOKENS_PER_GPU=65536
CRITIC_PPO_MAX_TOKENS_PER_GPU=32768
CUDA_TOOLKIT_ROOT=/dev/shm/cuda-13-b300-toolkit

while (($#)); do
  case "$1" in
    --mode) MODE=${2:?}; shift 2 ;;
    --python) PY=${2:?}; shift 2 ;;
    --verl-root) VERL=${2:?}; shift 2 ;;
    --endpoint-root) ENDPOINT_OUTER=${2:?}; shift 2 ;;
    --fixture-root) FIX=${2:?}; shift 2 ;;
    --launcher-root) LAUNCHER_ROOT=${2:?}; shift 2 ;;
    --run-id) RUN_ID=${2:?}; shift 2 ;;
    --run-dir) RUN_DIR=${2:?}; shift 2 ;;
    --persist-root) PERSIST_ROOT=${2:?}; shift 2 ;;
    --expected-plugin-commit) EXPECTED_PLUGIN_COMMIT=${2:?}; shift 2 ;;
    --expected-verl-commit) EXPECTED_VERL_COMMIT=${2:?}; shift 2 ;;
    --expected-endpoint-commit) EXPECTED_ENDPOINT_COMMIT=${2:?}; shift 2 ;;
    --known-cpu-owner) KNOWN_CPU_OWNER=${2:?}; shift 2 ;;
    --known-cpu-session) KNOWN_CPU_SESSION=${2:?}; shift 2 ;;
    --known-cpu-start-command) KNOWN_CPU_START_COMMAND=${2:?}; shift 2 ;;
    --known-cpu-script) KNOWN_CPU_SCRIPT=${2:?}; shift 2 ;;
    --known-cpu-script-sha256) KNOWN_CPU_SCRIPT_SHA256=${2:?}; shift 2 ;;
    --known-gpu-owner) KNOWN_GPU_OWNER=${2:?}; shift 2 ;;
    --known-gpu-script) KNOWN_GPU_SCRIPT=${2:?}; shift 2 ;;
    --known-gpu-script-sha256) KNOWN_GPU_SCRIPT_SHA256=${2:?}; shift 2 ;;
    --known-gpu-process-command) KNOWN_GPU_PROCESS_COMMAND=${2:?}; shift 2 ;;
    --port) PORT=${2:?}; shift 2 ;;
    --expected-checkpoint-bytes) EXPECTED_CHECKPOINT_BYTES=${2:?}; shift 2 ;;
    --trainer-gpus) TRAINER_GPUS=${2:?}; shift 2 ;;
    --standalone-rollout-gpus) STANDALONE_ROLLOUT_GPUS=${2:?}; shift 2 ;;
    --actor-use-fused-kernels) ACTOR_USE_FUSED_KERNELS=1; shift ;;
    --critic-use-fused-kernels) CRITIC_USE_FUSED_KERNELS=1; shift ;;
    --actor-ppo-max-tokens-per-gpu) ACTOR_PPO_MAX_TOKENS_PER_GPU=${2:?}; shift 2 ;;
    --critic-ppo-max-tokens-per-gpu) CRITIC_PPO_MAX_TOKENS_PER_GPU=${2:?}; shift 2 ;;
    *) usage ;;
  esac
done

for required in MODE PY VERL ENDPOINT_OUTER FIX LAUNCHER_ROOT RUN_ID RUN_DIR \
  PERSIST_ROOT EXPECTED_PLUGIN_COMMIT EXPECTED_VERL_COMMIT \
  EXPECTED_ENDPOINT_COMMIT KNOWN_CPU_OWNER KNOWN_CPU_SESSION \
  KNOWN_CPU_START_COMMAND KNOWN_CPU_SCRIPT KNOWN_CPU_SCRIPT_SHA256 \
  KNOWN_GPU_OWNER KNOWN_GPU_SCRIPT KNOWN_GPU_SCRIPT_SHA256 \
  KNOWN_GPU_PROCESS_COMMAND; do
  [ -n "${!required:-}" ] || { echo "missing required option: $required" >&2; usage; }
done
case "$MODE" in
  gate)
    ENDPOINT_CONTRACT=gate1
    LAUNCH_MODE=gate
    SCHEDULE=$FIX/g64-gate-single-pass.jsonl
    VOLATILE_CHECKPOINT_COPIES=1
    MEMORY_CGROUP_CHECKPOINT_COPIES=1
    PERSISTENT_CHECKPOINT_COPIES=0
    ;;
  formal)
    ENDPOINT_CONTRACT=formal100
    LAUNCH_MODE=formal
    SCHEDULE=$FIX/formal100-schedule.jsonl
    VOLATILE_CHECKPOINT_COPIES=2
    MEMORY_CGROUP_CHECKPOINT_COPIES=2
    PERSISTENT_CHECKPOINT_COPIES=1
    ;;
  *) usage ;;
esac
case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'') echo "unsafe RUN_ID=$RUN_ID" >&2; exit 64 ;;
esac
case "$EXPECTED_CHECKPOINT_BYTES" in
  *[!0-9]*|'') echo "invalid checkpoint byte estimate" >&2; exit 64 ;;
esac
if [ "$MODE" = formal ] && [ "$TRAINER_GPUS:$STANDALONE_ROLLOUT_GPUS" != 6:2 ]; then
  echo "formal Hybrid + Standalone topology must be 6+2, got $TRAINER_GPUS+$STANDALONE_ROLLOUT_GPUS" >&2
  exit 64
fi
case "$TRAINER_GPUS:$STANDALONE_ROLLOUT_GPUS" in
  4:4|6:2) ;;
  *) echo "unsupported Hybrid + Standalone topology: $TRAINER_GPUS+$STANDALONE_ROLLOUT_GPUS" >&2; exit 64 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PLUGIN_OUTER=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
LIFECYCLE=$PLUGIN_OUTER/async_plugins/agentmemorygym_verl/orchestrator_lifecycle.py
LOCK=$FIX/source-lock.json
PUBLICATION_REGISTRY_ROOT=/dev/shm
PUBLICATION_REGISTRY_LOCK=$PUBLICATION_REGISTRY_ROOT/.openmle-fast-publication-registry.lock
TOOL=$FIX/launcher_contract.py
ENDPOINT_INNER=$ENDPOINT_OUTER/AgentGym
START_ENDPOINT=$LAUNCHER_ROOT/start_openmle_fast_endpoints.sh
PROCESS_GUARD=$LAUNCHER_ROOT/process_guard.py
ENV_ADDR=http://127.0.0.1:$PORT
PROCESS_OWNER=amg-verl-latest-fully-async
CPU_MARKER=/tmp/agentmemory-formal-cpu-active
GPU_MARKER=/tmp/crg-holder-yield
MARKER_LOCK=/tmp/amg-holder-marker-transaction.lock
MARKER_STATE=$RUN_DIR/marker-transaction/state.json
MARKER_WATCH_READY=$RUN_DIR/marker-transaction/watcher-ready.json
MARKER_WATCH_RECEIPT=$RUN_DIR/marker-transaction/watcher-exit.json

ENDPOINT_PID=
ENDPOINT_TICKS=
TRAIN_PID=
TRAIN_TICKS=
WATCHDOG_PID=
WATCHDOG_TICKS=
MARKER_WATCH_PID=
MARKER_WATCH_TICKS=
GPU_MONITOR_PID=
GPU_MONITOR_TICKS=
TEE_PID=
TEE_TICKS=
ORIGINAL_CPU_OWNER=
ORIGINAL_CPU_PID=0
ORIGINAL_CPU_TICKS=
ORIGINAL_GPU_OWNER=
ORIGINAL_GPU_PID=0
ORIGINAL_GPU_TICKS=
MARKER_STATE_PREPARED=0
RUNTIME_GUARD_ACTIVE=0
RUNTIME_CLEANED=0
CLEANUP_STATUS=pass
PUBLICATION_COMPLETE=0

[ ! -e "$RUN_DIR" ] || { echo "refusing to reuse $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR/process_guard/watchdog" "$RUN_DIR/process_guard/preflight" \
  "$RUN_DIR/process_guard/exit" "$RUN_DIR/marker-transaction" \
  "$RUN_DIR/gpu-monitor"
[ -d "$PERSIST_ROOT" ] && [ ! -L "$PERSIST_ROOT" ] || {
  echo "persistent root must already exist on the mounted durable filesystem: $PERSIST_ROOT" >&2
  exit 65
}
exec 3>&1 4>&2
LOG_FIFO=$RUN_DIR/.orchestrator-log.fifo
mkfifo -m 600 "$LOG_FIFO"
tee -a "$RUN_DIR/orchestrator.log" < "$LOG_FIFO" >&3 2>&4 &
TEE_PID=$!
exec > "$LOG_FIFO" 2>&1

process_ticks() {
  "$PY" "$TOOL" process-start-ticks --pid "$1"
}

process_alive_exact() {
  local pid=$1 ticks=$2
  [ -n "$pid" ] && [ -n "$ticks" ] || return 1
  "$PY" "$LIFECYCLE" process-identity-alive \
    --pid "$pid" --start-ticks "$ticks"
}

signal_exact() {
  "$PY" "$TOOL" signal-exact-process --pid "$1" --start-ticks "$2" --signal "$3"
}

capture_ticks() {
  local pid=$1 output_var=$2 ticks= index
  for index in $(seq 1 100); do
    ticks=$(process_ticks "$pid" 2>/dev/null || true)
    [ -n "$ticks" ] && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.05
  done
  [ -n "$ticks" ] || return 1
  printf -v "$output_var" '%s' "$ticks"
}

capture_ticks "$TEE_PID" TEE_TICKS || {
  echo "failed to identify orchestrator tee" >&2
  exit 65
}

wait_exact_exit() {
  local pid=$1 ticks=$2 timeout_seconds=$3 index loops
  loops=$((timeout_seconds * 10))
  for index in $(seq 1 "$loops"); do
    process_alive_exact "$pid" "$ticks" || return 0
    sleep 0.1
  done
  return 1
}

stop_exact_child() {
  local name=$1 pid_var=$2 ticks_var=$3 timeout_seconds=$4
  local pid=${!pid_var:-} ticks=${!ticks_var:-}
  [ -n "$pid" ] || return 0
  if process_alive_exact "$pid" "$ticks"; then
    signal_exact "$pid" "$ticks" TERM || CLEANUP_STATUS=fail
    if ! wait_exact_exit "$pid" "$ticks" "$timeout_seconds"; then
      echo "$name did not exit after TERM; sending KILL" >&2
      signal_exact "$pid" "$ticks" KILL || CLEANUP_STATUS=fail
      wait_exact_exit "$pid" "$ticks" 5 || CLEANUP_STATUS=fail
      CLEANUP_STATUS=fail
    fi
  fi
  wait "$pid" 2>/dev/null || true
  printf -v "$pid_var" ''
  printf -v "$ticks_var" ''
}

wait_trainer_with_marker_watcher() {
  local watcher_failure=0 trainer_rc=125
  while process_alive_exact "$TRAIN_PID" "$TRAIN_TICKS"; do
    if ! process_alive_exact "$MARKER_WATCH_PID" "$MARKER_WATCH_TICKS"; then
      echo "marker watcher died while trainer was active" >&2
      watcher_failure=1
      CLEANUP_STATUS=fail
      stop_exact_child trainer TRAIN_PID TRAIN_TICKS 30
      break
    fi
    sleep 0.5
  done
  if [ "$watcher_failure" -eq 1 ]; then
    return 125
  fi
  if ! process_alive_exact "$MARKER_WATCH_PID" "$MARKER_WATCH_TICKS"; then
    echo "marker watcher died before trainer completion was reaped" >&2
    watcher_failure=1
    CLEANUP_STATUS=fail
  fi
  if wait "$TRAIN_PID"; then trainer_rc=0; else trainer_rc=$?; fi
  TRAIN_PID=
  TRAIN_TICKS=
  if [ "$watcher_failure" -eq 1 ]; then return 125; fi
  return "$trainer_rc"
}

select_latest_publication() {
  local output=$1
  "$PY" "$LIFECYCLE" select-latest-publication \
    --registry-root "$PUBLICATION_REGISTRY_ROOT" \
    --receipt-glob '/dev/shm/openmle-fast-rich-v*-publication/publication-receipt.json' \
    --fixture-receipt "$FIX/publication-receipt.json" \
    --fixture-lock "$LOCK" \
    --fixture-certificate "$FIX/formal100-schedule-certificate.json" \
    --output "$output"
}

write_launcher_exit() {
  local trainer_rc=$1 publication_status=$2 temp
  temp=$RUN_DIR/.launcher-exit.env.$$.tmp
  {
    printf 'trainer_exit_code=%s\n' "$trainer_rc"
    printf 'cleanup_status=%s\n' "$CLEANUP_STATUS"
    printf 'publication_status=%s\n' "$publication_status"
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'utc=%s\n' "$(date -u +%FT%TZ)"
  } > "$temp"
  chmod 600 "$temp"
  mv -f "$temp" "$RUN_DIR/launcher-exit.env"
}

stop_endpoint() {
  local rc=0
  if [ -n "$ENDPOINT_PID" ]; then
    stop_exact_child endpoint ENDPOINT_PID ENDPOINT_TICKS 30
  fi
  [ -s "$RUN_DIR/endpoints/cleanup.json" ] || rc=1
  grep -q '"status":"pass"' "$RUN_DIR/endpoints/cleanup.json" 2>/dev/null || rc=1
  return "$rc"
}

restore_marker_transaction() {
  [ "$MARKER_STATE_PREPARED" -eq 1 ] || return 0
  if ! "$PY" "$LIFECYCLE" marker-restore \
    --state "$MARKER_STATE" --lock "$MARKER_LOCK"; then
    CLEANUP_STATUS=fail
    return 1
  fi
  "$PY" "$LIFECYCLE" marker-status --state "$MARKER_STATE" --require restored \
    || { CLEANUP_STATUS=fail; return 1; }
  if [ -n "$MARKER_WATCH_PID" ]; then
    if ! wait_exact_exit "$MARKER_WATCH_PID" "$MARKER_WATCH_TICKS" 15; then
      echo "marker restoration watcher did not acknowledge restore" >&2
      stop_exact_child marker-watcher MARKER_WATCH_PID MARKER_WATCH_TICKS 5
      CLEANUP_STATUS=fail
      return 1
    fi
    wait "$MARKER_WATCH_PID" 2>/dev/null || CLEANUP_STATUS=fail
    MARKER_WATCH_PID=
    MARKER_WATCH_TICKS=
  fi
  grep -q '"status": "pass"' "$MARKER_WATCH_RECEIPT" 2>/dev/null \
    || { CLEANUP_STATUS=fail; return 1; }
  return 0
}

cleanup_runtime() {
  [ "$RUNTIME_CLEANED" -eq 0 ] || return 0
  if [ -n "$TRAIN_PID" ]; then
    stop_exact_child trainer TRAIN_PID TRAIN_TICKS 30
  fi
  if [ -n "$GPU_MONITOR_PID" ]; then
    stop_exact_child gpu-monitor GPU_MONITOR_PID GPU_MONITOR_TICKS 10
    grep -q '"status": "pass"' "$RUN_DIR/gpu-monitor/exit.json" 2>/dev/null \
      || CLEANUP_STATUS=fail
  fi
  if [ -n "$ENDPOINT_PID" ]; then
    stop_endpoint || CLEANUP_STATUS=fail
  fi
  if [ "$RUNTIME_GUARD_ACTIVE" -eq 1 ]; then
    "$PY" "$PROCESS_GUARD" --mode cleanup --owner "$PROCESS_OWNER" \
      --run-id "$RUN_ID" --term-timeout 20 \
      --evidence-dir "$RUN_DIR/process_guard/exit" || CLEANUP_STATUS=fail
    "$PY" "$PROCESS_GUARD" --mode assert-clean --owner "$PROCESS_OWNER" \
      --run-id "$RUN_ID" --evidence-dir "$RUN_DIR/process_guard/exit" \
      || CLEANUP_STATUS=fail
  fi
  if [ -n "$WATCHDOG_PID" ]; then
    stop_exact_child process-watchdog WATCHDOG_PID WATCHDOG_TICKS 10
  fi
  restore_marker_transaction || true
  if [ "$CLEANUP_STATUS" = pass ]; then
    RUNTIME_CLEANED=1
    return 0
  fi
  return 1
}

cleanup_before_publication() {
  cleanup_runtime || return 125
  [ "$RUNTIME_CLEANED" -eq 1 ] || return 125
  [ "$CLEANUP_STATUS" = pass ] || return 125
  return 0
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if [ "$PUBLICATION_COMPLETE" -ne 1 ]; then
    cleanup_runtime
    write_launcher_exit "$rc" failed
  fi
  if [ "$CLEANUP_STATUS" != pass ] && [ "$rc" -eq 0 ]; then rc=125; fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

freeze_run_dir_logging() {
  exec 1>&3 2>&4
  if ! wait_exact_exit "$TEE_PID" "$TEE_TICKS" 10; then
    printf 'orchestrator tee failed to drain before publication\n' >&2
    CLEANUP_STATUS=fail
    return 1
  fi
  wait "$TEE_PID" 2>/dev/null || true
  TEE_PID=
  TEE_TICKS=
  "$PY" - "$LOG_FIFO" <<'PY'
from pathlib import Path
import sys
path=Path(sys.argv[1])
if path.is_symlink() or not path.exists():
    raise SystemExit(f'expected live logging FIFO: {path}')
if not path.is_fifo():
    raise SystemExit(f'logging path is not a FIFO: {path}')
path.unlink()
PY
}

[ -x "$CUDA_TOOLKIT_ROOT/bin/nvcc" ] || {
  echo "missing CUDA 13 toolkit: $CUDA_TOOLKIT_ROOT" >&2
  exit 66
}
RUNTIME_SITE_PACKAGES=$("$PY" - "$LOCK" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["training_runtime"]["site_packages"])
PY
)
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDA_PATH="$CUDA_TOOLKIT_ROOT"
export PATH="$CUDA_TOOLKIT_ROOT/bin:$(dirname -- "$PY"):$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib64:/usr/local/cuda/lib64/stubs:$RUNTIME_SITE_PACKAGES/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

printf '[async-%s] start=%s run_id=%s\n' "$MODE" "$(date -u +%FT%TZ)" "$RUN_ID"
for path in "$PY" "$LOCK" "$TOOL" "$FIX/publication-receipt.json" \
  "$FIX/formal100-schedule-certificate.json" "$SCHEDULE" "$START_ENDPOINT" \
  "$PROCESS_GUARD" "$LIFECYCLE" \
  "$PLUGIN_OUTER/async_plugins/scripts/launch_amg_fully_async.sh"; do
  [ -e "$path" ] || { echo "missing $path" >&2; exit 66; }
done
select_latest_publication "$RUN_DIR/latest-publication-selection.json"
[ "$(git -C "$VERL" rev-parse HEAD)" = "$EXPECTED_VERL_COMMIT" ]
[ -z "$(git -C "$VERL" status --porcelain)" ]
[ "$(git -C "$PLUGIN_OUTER" rev-parse HEAD)" = "$EXPECTED_PLUGIN_COMMIT" ]
[ -z "$(git -C "$PLUGIN_OUTER" status --porcelain)" ]
[ "$(git -C "$ENDPOINT_OUTER" rev-parse HEAD)" = "$EXPECTED_ENDPOINT_COMMIT" ]
[ -z "$(git -C "$ENDPOINT_OUTER" status --porcelain)" ]

"$PY" "$LIFECYCLE" capacity-check \
  --volatile-path /dev/shm --persistent-path "$PERSIST_ROOT" \
  --checkpoint-bytes "$EXPECTED_CHECKPOINT_BYTES" \
  --volatile-checkpoint-copies "$VOLATILE_CHECKPOINT_COPIES" \
  --persistent-checkpoint-copies "$PERSISTENT_CHECKPOINT_COPIES" \
  --volatile-margin-bytes 85899345920 --persistent-margin-bytes 34359738368 \
  --memory-cgroup-usage-path "$MEMORY_CGROUP_USAGE_PATH" \
  --memory-cgroup-limit-path "$MEMORY_CGROUP_LIMIT_PATH" \
  --memory-cgroup-checkpoint-copies "$MEMORY_CGROUP_CHECKPOINT_COPIES" \
  --memory-cgroup-margin-bytes "$MEMORY_CGROUP_RUNTIME_MARGIN_BYTES" \
  --require-distinct-filesystems --expected-persistent-fs-type nfs \
  --output "$RUN_DIR/capacity-preendpoint.json"

if [ "${ORCHESTRATOR_PREFLIGHT_ONLY:-0}" = 1 ]; then
  printf '[async-%s] preflight-only-pass=%s run_id=%s\n' \
    "$MODE" "$(date -u +%FT%TZ)" "$RUN_ID"
  exit 0
fi
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
  echo "foreign endpoint on $PORT" >&2
  exit 70
fi
INFERENCE_RESIDUE_PATTERN='ray::|raylet|gcs_server|VLLM::EngineCore|vllm.entrypoints|SGLang::|sglang::|python.*-m sglang\.(launch_server|serve)|sglang\.srt\.entrypoints'
if pgrep -af "$INFERENCE_RESIDUE_PATTERN" \
    | grep -vE '(^| )grep ' >/dev/null; then
  echo "foreign Ray/inference-engine residue detected before $MODE" >&2
  pgrep -af "$INFERENCE_RESIDUE_PATTERN" >&2 || true
  exit 71
fi
"$PY" "$PROCESS_GUARD" --mode assert-clean --owner "$PROCESS_OWNER" \
  --run-id "$RUN_ID" --evidence-dir "$RUN_DIR/process_guard/preflight"
RUNTIME_GUARD_ACTIVE=1

PREFLIGHT=$RUN_DIR/runtime-preflight.json
"$PY" "$TOOL" runtime-preflight --source-lock "$LOCK" \
  --contract "$ENDPOINT_CONTRACT" --outer-root "$ENDPOINT_OUTER" \
  --inner-root "$ENDPOINT_INNER" --output "$PREFLIGHT"

PARENT_PID=$$
PARENT_TICKS=$(process_ticks "$PARENT_PID")
export OPENMLE_FAST_SOURCE_LOCK=$LOCK
export OPENMLE_FAST_CONTRACT_TOOL=$TOOL
export OPENMLE_FAST_LAUNCHER_ROOT=$LAUNCHER_ROOT
export OPENMLE_FAST_SOURCE_ROOT=$ENDPOINT_OUTER
export OPENMLE_FAST_INNER_SOURCE_ROOT=$ENDPOINT_INNER
export OPENMLE_FAST_RUN_DIR=$RUN_DIR
export OPENMLE_FAST_PREFLIGHT_RECEIPT=$PREFLIGHT
export OPENMLE_FAST_PARENT_PID=$PARENT_PID
export OPENMLE_FAST_PARENT_START_TICKS=$PARENT_TICKS
export OPENMLE_FAST_PORT=$PORT
export OPENMLE_FAST_PYTHON=$PY
export AGENTMEMORY_PROCESS_OWNER=$PROCESS_OWNER
export AGENTMEMORY_RUN_ID=$RUN_ID

ORIGINAL_CPU_OWNER=$("$PY" "$LIFECYCLE" marker-read --path "$CPU_MARKER")
if [ -n "$ORIGINAL_CPU_OWNER" ]; then
  [ "$ORIGINAL_CPU_OWNER" = "$KNOWN_CPU_OWNER" ] || {
    echo "foreign CPU marker exists: $ORIGINAL_CPU_OWNER" >&2
    exit 74
  }
  [ -f "$KNOWN_CPU_SCRIPT" ] && [ ! -L "$KNOWN_CPU_SCRIPT" ]
  [ "$(sha256sum "$KNOWN_CPU_SCRIPT" | awk '{print $1}')" = "$KNOWN_CPU_SCRIPT_SHA256" ]
  mapfile -t cpu_panes < <(tmux list-panes -t "$KNOWN_CPU_SESSION" \
    -F '#{pane_pid}|#{pane_start_command}|#{pane_dead}' 2>/dev/null || true)
  [ "${#cpu_panes[@]}" -eq 1 ] || {
    echo "expected exactly one known CPU owner pane, got ${#cpu_panes[@]}" >&2
    exit 74
  }
  IFS='|' read -r ORIGINAL_CPU_PID cpu_start_command cpu_dead <<< "${cpu_panes[0]}"
  [ "$cpu_start_command" = "$KNOWN_CPU_START_COMMAND" ] && [ "$cpu_dead" = 0 ] || {
    echo "CPU marker owner pane identity mismatch: ${cpu_panes[0]}" >&2
    exit 74
  }
  ORIGINAL_CPU_TICKS=$(process_ticks "$ORIGINAL_CPU_PID")
fi
ORIGINAL_GPU_OWNER=$("$PY" "$LIFECYCLE" marker-read --path "$GPU_MARKER")
if [ -n "$ORIGINAL_GPU_OWNER" ]; then
  [ "$ORIGINAL_GPU_OWNER" = "$KNOWN_GPU_OWNER" ] || {
    echo "foreign GPU marker exists: $ORIGINAL_GPU_OWNER" >&2
    exit 75
  }
  [ -f "$KNOWN_GPU_SCRIPT" ] && [ ! -L "$KNOWN_GPU_SCRIPT" ]
  [ "$(sha256sum "$KNOWN_GPU_SCRIPT" | awk '{print $1}')" = "$KNOWN_GPU_SCRIPT_SHA256" ]
  mapfile -t gpu_pids < <(pgrep -f -x "$KNOWN_GPU_PROCESS_COMMAND" || true)
  [ "${#gpu_pids[@]}" -eq 1 ] || {
    echo "expected exactly one known GPU guard, got ${#gpu_pids[@]}" >&2
    exit 75
  }
  ORIGINAL_GPU_PID=${gpu_pids[0]}
  ORIGINAL_GPU_TICKS=$(process_ticks "$ORIGINAL_GPU_PID")
fi

"$PY" "$LIFECYCLE" marker-prepare \
  --state "$MARKER_STATE" --lock "$MARKER_LOCK" --run-id "$RUN_ID" \
  --parent-pid "$PARENT_PID" --parent-start-ticks "$PARENT_TICKS" \
  --cpu-path "$CPU_MARKER" --cpu-original-value "$ORIGINAL_CPU_OWNER" \
  --cpu-original-pid "$ORIGINAL_CPU_PID" \
  --cpu-original-start-ticks "$ORIGINAL_CPU_TICKS" \
  --gpu-path "$GPU_MARKER" --gpu-original-value "$ORIGINAL_GPU_OWNER" \
  --gpu-original-pid "$ORIGINAL_GPU_PID" \
  --gpu-original-start-ticks "$ORIGINAL_GPU_TICKS"
MARKER_STATE_PREPARED=1
nohup "$PY" "$LIFECYCLE" marker-watch \
  --state "$MARKER_STATE" --lock "$MARKER_LOCK" \
  --parent-pid "$PARENT_PID" --parent-start-ticks "$PARENT_TICKS" \
  --ready "$MARKER_WATCH_READY" --receipt "$MARKER_WATCH_RECEIPT" \
  --poll-seconds 0.1 --restore-timeout-seconds 300 \
  </dev/null >> "$RUN_DIR/marker-transaction/watcher.log" 2>&1 &
MARKER_WATCH_PID=$!
capture_ticks "$MARKER_WATCH_PID" MARKER_WATCH_TICKS
for _ in $(seq 1 100); do
  [ -s "$MARKER_WATCH_READY" ] && break
  process_alive_exact "$MARKER_WATCH_PID" "$MARKER_WATCH_TICKS"
  sleep 0.1
done
grep -q '"status": "ready"' "$MARKER_WATCH_READY"
grep -q '"signal_handlers_installed": true' "$MARKER_WATCH_READY"
process_alive_exact "$MARKER_WATCH_PID" "$MARKER_WATCH_TICKS" || {
  echo "marker watcher died after readiness handoff" >&2
  exit 125
}
"$PY" "$LIFECYCLE" marker-acquire --state "$MARKER_STATE" --lock "$MARKER_LOCK"
"$PY" "$LIFECYCLE" marker-status --state "$MARKER_STATE" --require acquired
process_alive_exact "$MARKER_WATCH_PID" "$MARKER_WATCH_TICKS" || {
  echo "marker watcher died during marker acquisition" >&2
  exit 125
}
# The 96-worker CPU holder materially delayed resident endpoint probes in
# r21/r22.  Acquire both canonical markers and observe both holder state
# machines before starting the endpoint; raw marker writes are forbidden.
CPU_HOLDER_STATE=
for _ in $(seq 1 120); do
  CPU_HOLDER_STATE=$("$PY" -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("state", ""))' \
    /tmp/amg-cpu-holder/status.json 2>/dev/null || true)
  [ "$CPU_HOLDER_STATE" = yielded ] && break
  sleep 0.25
done
[ "$CPU_HOLDER_STATE" = yielded ] || {
  echo "CPU holder did not reach state=yielded before endpoint startup" >&2
  exit 75
}
for _ in $(seq 1 120); do
  grep -q 'mode=yield' /tmp/crg-holder.state 2>/dev/null && break
  sleep 0.25
done
grep -q 'mode=yield' /tmp/crg-holder.state || {
  echo "GPU holder did not reach mode=yield before endpoint startup" >&2
  exit 75
}


"$START_ENDPOINT" "$ENDPOINT_CONTRACT" > "$RUN_DIR/endpoint-supervisor.log" 2>&1 &
ENDPOINT_PID=$!
capture_ticks "$ENDPOINT_PID" ENDPOINT_TICKS || {
  tail -n 100 "$RUN_DIR/endpoint-supervisor.log" >&2 || true
  exit 72
}
for _ in $(seq 1 1200); do
  [ -s "$RUN_DIR/endpoints/ready.json" ] && break
  process_alive_exact "$ENDPOINT_PID" "$ENDPOINT_TICKS" || {
    tail -n 100 "$RUN_DIR/endpoint-supervisor.log" >&2 || true
    exit 73
  }
  sleep 0.25
done
[ -s "$RUN_DIR/endpoints/ready.json" ]
grep -q '"status":"ready"' "$RUN_DIR/endpoints/ready.json"

eval "$("$PY" - "$LOCK" "$PREFLIGHT" <<'PY'
import json, shlex, sys
x=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); vals={
'MANIFEST':p['manifest_path'],
'MANIFEST_SHA':p['manifest_sha256'],
'OUTER_COMMIT':x['runtime_source']['outer_commit'],
'INNER_COMMIT':x['runtime_source']['inner_commit'],
'PROMPT_SHA':x['runtime_source']['policy_prompt_sha256']}
for k,v in vals.items(): print(f'{k}={shlex.quote(str(v))}')
PY
)"
if [ "$MODE" = gate ]; then
  PROBE_INDICES=0,63
else
  PROBE_INDICES=$("$PY" - "$FIX/formal100-schedule-certificate.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1])); indices=x['endpoint_probe_indices']
if not isinstance(indices,list) or len(indices)!=2 or any(not isinstance(i,int) or i<0 for i in indices):
    raise SystemExit('invalid endpoint_probe_indices')
print(','.join(str(i) for i in indices))
PY
)
fi
"$PY" "$ENDPOINT_OUTER/AgentGym-RL/scripts/agentmemory/verify_openmle_fast_resident_endpoint.py" \
  --base-url "$ENV_ADDR" --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
  --indices "$PROBE_INDICES" --expected-outer-commit "$OUTER_COMMIT" \
  --expected-inner-commit "$INNER_COMMIT" --expected-prompt-sha256 "$PROMPT_SHA" \
  --client-timeout-seconds 200 --timeout-margin-seconds 5 \
  --forbidden-canaries-file "$RUN_DIR/endpoints/private/forbidden-canaries.json" \
  --output "$RUN_DIR/resident-endpoint-probe.json" \
  | tee "$RUN_DIR/resident-endpoint-probe.log"


nohup "$PY" "$PROCESS_GUARD" --mode watch-parent --owner "$PROCESS_OWNER" \
  --run-id "$RUN_ID" --parent-pid "$PARENT_PID" --poll-interval 2 \
  --term-timeout 20 --evidence-dir "$RUN_DIR/process_guard/watchdog" \
  </dev/null >> "$RUN_DIR/process_guard/watchdog/watchdog.log" 2>&1 &
WATCHDOG_PID=$!
capture_ticks "$WATCHDOG_PID" WATCHDOG_TICKS
for _ in $(seq 1 100); do
  find "$RUN_DIR/process_guard/watchdog" -type f -name '*watch-parent-start*.json' \
    -print -quit | grep -q . && break
  process_alive_exact "$WATCHDOG_PID" "$WATCHDOG_TICKS"
  sleep 0.1
done
find "$RUN_DIR/process_guard/watchdog" -type f -name '*watch-parent-start*.json' \
  -print -quit | grep -q .

nohup "$PY" "$LIFECYCLE" gpu-monitor \
  --parent-pid "$PARENT_PID" --parent-start-ticks "$PARENT_TICKS" \
  --output "$RUN_DIR/nvidia-smi-timeseries.csv" \
  --stderr "$RUN_DIR/nvidia-smi-timeseries.stderr" \
  --ready "$RUN_DIR/gpu-monitor/ready.json" \
  --receipt "$RUN_DIR/gpu-monitor/exit.json" \
  --interval-seconds 2 --command-timeout-seconds 5 \
  </dev/null >> "$RUN_DIR/gpu-monitor/monitor.log" 2>&1 &
GPU_MONITOR_PID=$!
capture_ticks "$GPU_MONITOR_PID" GPU_MONITOR_TICKS
for _ in $(seq 1 100); do
  [ -s "$RUN_DIR/gpu-monitor/ready.json" ] && break
  process_alive_exact "$GPU_MONITOR_PID" "$GPU_MONITOR_TICKS"
  sleep 0.1
done
grep -q '"status": "ready"' "$RUN_DIR/gpu-monitor/ready.json"

# Re-check capacity, then make the publication check the final admission
# operation before trainer launch.  A newly sealed publication or consumed
# capacity aborts this lineage while the independent marker watcher restores
# holder ownership.
"$PY" "$LIFECYCLE" capacity-check \
  --volatile-path /dev/shm --persistent-path "$PERSIST_ROOT" \
  --checkpoint-bytes "$EXPECTED_CHECKPOINT_BYTES" \
  --volatile-checkpoint-copies "$VOLATILE_CHECKPOINT_COPIES" \
  --persistent-checkpoint-copies "$PERSISTENT_CHECKPOINT_COPIES" \
  --volatile-margin-bytes 85899345920 --persistent-margin-bytes 34359738368 \
  --memory-cgroup-usage-path "$MEMORY_CGROUP_USAGE_PATH" \
  --memory-cgroup-limit-path "$MEMORY_CGROUP_LIMIT_PATH" \
  --memory-cgroup-checkpoint-copies "$MEMORY_CGROUP_CHECKPOINT_COPIES" \
  --memory-cgroup-margin-bytes "$MEMORY_CGROUP_RUNTIME_MARGIN_BYTES" \
  --require-distinct-filesystems --expected-persistent-fs-type nfs \
  --output "$RUN_DIR/capacity-pretrainer.json"
export WANDB_MODE=disabled
printf '%s\n' "$(date -u +%FT%TZ)" > "$RUN_DIR/trainer-started-at"
LAUNCH_TUNING_ARGS=(
  --trainer-gpus "$TRAINER_GPUS"
  --standalone-rollout-gpus "$STANDALONE_ROLLOUT_GPUS"
  --actor-ppo-max-tokens-per-gpu "$ACTOR_PPO_MAX_TOKENS_PER_GPU"
  --critic-ppo-max-tokens-per-gpu "$CRITIC_PPO_MAX_TOKENS_PER_GPU"
)
if [ "$ACTOR_USE_FUSED_KERNELS" -eq 1 ]; then
  LAUNCH_TUNING_ARGS+=(--actor-use-fused-kernels)
fi
if [ "$CRITIC_USE_FUSED_KERNELS" -eq 1 ]; then
  LAUNCH_TUNING_ARGS+=(--critic-use-fused-kernels)
fi
# Re-select the live publication and exec the trainer from the same process.
# This removes the shell scheduling window between the final selection and the
# trainer process boundary; no custom queue/trainer logic is introduced.
"$PY" "$LIFECYCLE" exec-after-publication-check \
  --first "$RUN_DIR/latest-publication-selection.json" \
  --registry-root "$PUBLICATION_REGISTRY_ROOT" \
  --receipt-glob '/dev/shm/openmle-fast-rich-v*-publication/publication-receipt.json' \
  --fixture-receipt "$FIX/publication-receipt.json" \
  --fixture-lock "$LOCK" \
  --fixture-certificate "$FIX/formal100-schedule-certificate.json" \
  --selection-output "$RUN_DIR/latest-publication-pretrainer.json" \
  --check-output "$RUN_DIR/latest-publication-launch-race-check.json" \
  --registry-lock "$PUBLICATION_REGISTRY_LOCK" \
  --unset-env PYTHONPATH -- \
  "$PLUGIN_OUTER/async_plugins/scripts/launch_amg_fully_async.sh" \
  --mode "$LAUNCH_MODE" --verl-root "$VERL" --schedule "$SCHEDULE" \
  --env-addr "$ENV_ADDR" --run-dir "$RUN_DIR" --experiment-name "$RUN_ID" \
  --endpoint-source-lock "$LOCK" --endpoint-contract-tool "$TOOL" \
  --publication-receipt "$FIX/publication-receipt.json" \
  --formal-schedule-certificate "$FIX/formal100-schedule-certificate.json" \
  "${LAUNCH_TUNING_ARGS[@]}" \
  > "$RUN_DIR/train.log" 2>&1 &
TRAIN_PID=$!
capture_ticks "$TRAIN_PID" TRAIN_TICKS
printf '%s %s\n' "$TRAIN_PID" "$TRAIN_TICKS" > "$RUN_DIR/train.identity"
if wait_trainer_with_marker_watcher; then TRAIN_RC=0; else TRAIN_RC=$?; fi
printf '%s\n' "$TRAIN_RC" > "$RUN_DIR/trainer-exit-code"
[ "$TRAIN_RC" -eq 0 ] || exit "$TRAIN_RC"

curl -fsS -m 30 "$ENV_ADDR/metadata" > "$RUN_DIR/endpoint-metadata-after.json"
cleanup_before_publication || exit $?
[ "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$RUN_DIR/finalization.json")" = pass ]

PERSIST=$PERSIST_ROOT/$RUN_ID
[ ! -e "$PERSIST" ]
printf '%s\n' "$PERSIST" > "$RUN_DIR/persistent-evidence-path"
if [ "$MODE" = formal ]; then
  TRACKER=$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt
  [ -f "$TRACKER" ] && [ ! -L "$TRACKER" ]
  FINAL_PUBLICATION_STEP=$(cat "$TRACKER")
  case "$FINAL_PUBLICATION_STEP" in
    *[!0-9]*|'') echo 'invalid checkpoint tracker' >&2; exit 76 ;;
  esac
  EXPECTED_PUBLICATION_STEP=$("$PY" - "$RUN_DIR/finalization.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['counts']['publication_cycles'])
PY
)
  [ "$FINAL_PUBLICATION_STEP" = "$EXPECTED_PUBLICATION_STEP" ]
  "$PY" "$LIFECYCLE" capacity-check \
    --volatile-path /dev/shm --persistent-path "$PERSIST_ROOT" \
    --checkpoint-bytes "$EXPECTED_CHECKPOINT_BYTES" \
    --volatile-checkpoint-copies 0 --persistent-checkpoint-copies 1 \
    --volatile-margin-bytes 1073741824 --persistent-margin-bytes 34359738368 \
    --require-distinct-filesystems --expected-persistent-fs-type nfs \
    --output "$RUN_DIR/capacity-prepublication.json"
else
  FINAL_PUBLICATION_STEP=
  "$PY" "$LIFECYCLE" capacity-check \
    --volatile-path /dev/shm --persistent-path "$PERSIST_ROOT" \
    --checkpoint-bytes "$EXPECTED_CHECKPOINT_BYTES" \
    --volatile-checkpoint-copies 0 --persistent-checkpoint-copies 0 \
    --volatile-margin-bytes 1073741824 --persistent-margin-bytes 2147483648 \
    --require-distinct-filesystems --expected-persistent-fs-type nfs \
    --output "$RUN_DIR/capacity-prepublication.json"
fi
write_launcher_exit 0 ready_for_atomic_publication
freeze_run_dir_logging
[ "$CLEANUP_STATUS" = pass ]
# Runtime cleanup and holder restoration are already terminal.  Remove the
# shell fallback traps, then replace this launcher with the terminal publisher.
# Its hidden stage contains the complete launcher receipt and tree manifest;
# atomic rename is its final public action and is followed immediately by
# os._exit(0), so no cleanup or evidence work remains after publication.
trap - EXIT INT TERM
PUBLISH_ARGS=(
  --run-dir "$RUN_DIR" --persist-root "$PERSIST_ROOT"
  --run-id "$RUN_ID" --mode "$MODE"
)
if [ "$MODE" = formal ]; then
  PUBLISH_ARGS+=(--checkpoint-step "$FINAL_PUBLICATION_STEP")
else
  PUBLISH_ARGS+=(--discard-gate-checkpoints)
fi
exec "$PY" "$LIFECYCLE" terminal-publish "${PUBLISH_ARGS[@]}"
