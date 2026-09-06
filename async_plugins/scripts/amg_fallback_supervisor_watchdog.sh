#!/usr/bin/env bash
set -euo pipefail

# ``dist_train.py`` owns this wrapper and restarts it when it exits.  The
# wrapper execs the crash-safe Python supervisor so the platform entrance,
# rather than an unowned nohup process, remains the top-level watchdog.

# This interpreter bootstraps the allocation-level fallback holder, so it must
# survive reconstruction of the training runtime under /dev/shm.  Keep its
# override separate from the training launcher's PYTHON variable: the latter
# is intentionally allowed to point at the volatile, source-locked runtime.
PYTHON="${FALLBACK_SUPERVISOR_PYTHON:-/opt/conda/envs/py312/bin/python3}"
MODULE="${FALLBACK_SUPERVISOR_MODULE:-/export/App/training_platform/PinoModel/amg_fallback_supervisor_sao_v1.py}"
BOOTSTRAP="${FALLBACK_PROCESS_BOOTSTRAP:-/export/App/training_platform/PinoModel/amg_process_bootstrap_sao_v1.py}"
ORIGINAL="${FALLBACK_WATCHDOG_ORIGINAL:-/export/App/training_platform/PinoModel/non_yield_holder_watchdog.sh.original.8bb8a33b6c73e64f18dd53cf0307fd59f8049c4dc07d566c396a94d998819b0d}"
HOLDER="${HOLDER:-/export/App/training_platform/PinoModel/non_yield_gpu_cpu_fallback_holder.py}"
WRAPPER="${FALLBACK_WATCHDOG_WRAPPER:-/export/App/training_platform/PinoModel/non_yield_holder_watchdog.sh}"

ORIGINAL_SHA256="${FALLBACK_ORIGINAL_SHA256:-8bb8a33b6c73e64f18dd53cf0307fd59f8049c4dc07d566c396a94d998819b0d}"
HOLDER_SHA256="${FALLBACK_HOLDER_SHA256:-5f8dc879532937fae759fa9bfb99491387cc11c13d80629655442e0e99895d85}"
PID_FILE="${FALLBACK_PID_FILE:-/tmp/amg-non-yield-fallback-v4.pid}"
HOLDER_STATE_FILE="${FALLBACK_HOLDER_STATE_FILE:-/tmp/amg-non-yield-fallback-v4.state.json}"
PAUSE_MARKER="${FALLBACK_PAUSE_MARKER:-/tmp/amg-non-yield-fallback.pause}"
RELEASE_REQUEST_FILE="${FALLBACK_RELEASE_REQUEST_FILE:-/tmp/amg-non-yield-fallback.release.json}"
SUPERVISOR_STATE="${FALLBACK_SUPERVISOR_STATE:-/tmp/amg-non-yield-fallback-supervisor.json}"
SUPERVISOR_LOCK="${FALLBACK_SUPERVISOR_LOCK:-/tmp/amg-non-yield-fallback-supervisor.lock}"
TRANSACTION_LOCK="${FALLBACK_TRANSACTION_LOCK:-/tmp/amg-non-yield-fallback-transaction.lock}"
WATCHDOG_IDENTITY_FILE="${FALLBACK_WATCHDOG_IDENTITY_FILE:-/tmp/amg-non-yield-fallback-watchdog-process-identity.json}"
WATCHDOG_LOG="${FALLBACK_WATCHDOG_LOG:-/tmp/amg-non-yield-fallback-supervised.log}"
NVIDIA_SMI="${FALLBACK_NVIDIA_SMI:-/usr/bin/nvidia-smi}"
AUTO_GPU_HOLDER_STATE="${FALLBACK_AUTO_GPU_HOLDER_STATE:-/tmp/crg-holder.state}"
AUTO_GPU_HOLDER_COMMAND_FRAGMENT="${FALLBACK_AUTO_GPU_HOLDER_COMMAND_FRAGMENT:-platform_gpu_only_auto_yield_holder.py}"
AUTO_CPU_HOLDER_STATE="${FALLBACK_AUTO_CPU_HOLDER_STATE:-/tmp/amg-cpu-holder/status.json}"
AUTO_CPU_HOLDER_COMMAND_FRAGMENT="${FALLBACK_AUTO_CPU_HOLDER_COMMAND_FRAGMENT:-auto_yield_cpu_holder.py}"

# ``dist_train.py`` restarts this wrapper with its own clean environment, so
# every non-self component needs a reviewed default rather than an inherited
# shell variable.  ``migrate`` and ``formal`` independently pass and verify the
# exact wrapper digest; the restarted wrapper measures those same executable
# bytes so its published immutable contract can be matched by the owner.
FALLBACK_PYTHON_SHA256="${FALLBACK_PYTHON_SHA256:-2951874f3ba993b5ae678d80a7c7b30e0e47d166ad22e4e384474fbaaab52699}"
FALLBACK_MODULE_SHA256="${FALLBACK_MODULE_SHA256:-bd89f8f2705d0a3dae36764c521d5e2cf06b0e11a95525efb0b9efb6c34fab2f}"
FALLBACK_BOOTSTRAP_SHA256="${FALLBACK_BOOTSTRAP_SHA256:-d306492c257d309bf46a0915db961bf95e4811afd61fa6cbf5a683bab4cbe987}"
FALLBACK_NVIDIA_SMI_SHA256="${FALLBACK_NVIDIA_SMI_SHA256:-384896f294f80e8e5b59c11dbb04d3fdc028cf9bb01d7839154b96f94e966f4e}"
FALLBACK_WATCHDOG_WRAPPER_SHA256="${FALLBACK_WATCHDOG_WRAPPER_SHA256:-$(sha256sum "$WRAPPER" | awk '{print $1}')}"

if [[ ! -x "$PYTHON" ]]; then
  printf 'missing fallback Python interpreter: %s\n' "$PYTHON" >&2
  exit 89
fi
# Conda exposes ``bin/python3`` as a symlink on the 9N image.  The supervisor
# deliberately accepts only regular immutable files, so pin and pass the
# canonical interpreter rather than failing after the platform watchdog has
# already exec'd this wrapper.
PYTHON="$(readlink -f -- "$PYTHON")"
if [[ -z "$PYTHON" || ! -f "$PYTHON" || -L "$PYTHON" || ! -x "$PYTHON" ]]; then
  printf 'unsafe canonical fallback Python interpreter: %s\n' "$PYTHON" >&2
  exit 89
fi
for path in "$MODULE" "$BOOTSTRAP" "$ORIGINAL" "$HOLDER" "$WRAPPER"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    printf 'missing or unsafe fallback component: %s\n' "$path" >&2
    exit 90
  fi
done

observed_original="$(sha256sum "$ORIGINAL" | awk '{print $1}')"
observed_holder="$(sha256sum "$HOLDER" | awk '{print $1}')"
if [[ "$observed_original" != "$ORIGINAL_SHA256" ]]; then
  printf 'original watchdog sha256 mismatch: expected=%s observed=%s\n' \
    "$ORIGINAL_SHA256" "$observed_original" >&2
  exit 91
fi
if [[ "$observed_holder" != "$HOLDER_SHA256" ]]; then
  printf 'fallback holder sha256 mismatch: expected=%s observed=%s\n' \
    "$HOLDER_SHA256" "$observed_holder" >&2
  exit 92
fi

exec "$PYTHON" "$MODULE" supervise \
  --python "$PYTHON" \
  --python-sha256 "$FALLBACK_PYTHON_SHA256" \
  --original "$ORIGINAL" \
  --original-sha256 "$ORIGINAL_SHA256" \
  --holder "$HOLDER" \
  --holder-sha256 "$HOLDER_SHA256" \
  --watchdog-wrapper "$WRAPPER" \
  --watchdog-wrapper-sha256 "$FALLBACK_WATCHDOG_WRAPPER_SHA256" \
  --supervisor-script "$MODULE" \
  --supervisor-script-sha256 "$FALLBACK_MODULE_SHA256" \
  --bootstrap "$BOOTSTRAP" \
  --bootstrap-sha256 "$FALLBACK_BOOTSTRAP_SHA256" \
  --watchdog-identity-file "$WATCHDOG_IDENTITY_FILE" \
  --pid-file "$PID_FILE" \
  --holder-state-file "$HOLDER_STATE_FILE" \
  --pause-marker "$PAUSE_MARKER" \
  --release-request-file "$RELEASE_REQUEST_FILE" \
  --supervisor-state "$SUPERVISOR_STATE" \
  --supervisor-lock "$SUPERVISOR_LOCK" \
  --transaction-lock "$TRANSACTION_LOCK" \
  --watchdog-log "$WATCHDOG_LOG" \
  --nvidia-smi "$NVIDIA_SMI" \
  --nvidia-smi-sha256 "$FALLBACK_NVIDIA_SMI_SHA256" \
  --auto-gpu-holder-state "$AUTO_GPU_HOLDER_STATE" \
  --auto-gpu-holder-command-fragment "$AUTO_GPU_HOLDER_COMMAND_FRAGMENT" \
  --auto-cpu-holder-state "$AUTO_CPU_HOLDER_STATE" \
  --auto-cpu-holder-command-fragment "$AUTO_CPU_HOLDER_COMMAND_FRAGMENT"
