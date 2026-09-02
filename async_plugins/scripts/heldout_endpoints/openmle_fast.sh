#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$HERE/common.sh"

heldout_assert_base_contract openmle_fast 65123
heldout_assert_parent
heldout_assert_asset_env HELDOUT_MANIFEST "AutoResearch held-out manifest"
heldout_assert_asset_env PRIVATE_GRADER_BINDINGS "AutoResearch private grader bindings"
heldout_assert_asset_env ROUTING "AutoResearch held-out routing"
heldout_assert_asset_env RUNTIME_MANIFEST "AutoResearch runtime manifest"

PT=/home/ai-jingyan-train/luolirui.1/post-train
WS=$PT/agentmemorygym-rl-workspace
PYBIN=${HELDOUT_RUNTIME_PYTHON:-/dev/shm/qwen35-runtime-verl-main-sglang-fsdp-tf553-fla052-v2/bin/python3.12}
SUPERVISOR=$HERE/openmle_heldout_supervisor.py
RUNTIME_BASE=$HERE/openmle_runtime_base.py
SERVICE_ENTRYPOINT=$HERE/openmle_service_entrypoint.py
CONTRACT_TOOL=${OPENMLE_FAST_CONTRACT_TOOL_OVERRIDE:-$WS/runtime/amg-main-multitask400-20260823/tools/openmle_launcher_contract.py}
RUN_ID=${AMG_MULTITASK_RUN_ID:?missing run id}
RUN_DIR=${AMG_MULTITASK_ENDPOINT_RUN_DIR:?missing endpoint run dir}
PORT=${AMG_MULTITASK_ENDPOINT_PORT:?missing endpoint port}
EXPECTED_RUNTIME_SCHEMA=camg_openmle_fast_heldout_runtime_manifest_v1

heldout_assert_executable "$PYBIN" "held-out Python runtime"
heldout_assert_executable "$SUPERVISOR" "OpenMLE held-out supervisor"
heldout_assert_executable "$SERVICE_ENTRYPOINT" "OpenMLE service entrypoint"
heldout_assert_file "$RUNTIME_BASE" "$(heldout_sha256 "$RUNTIME_BASE")" "OpenMLE process-runtime donor"
[[ "$CONTRACT_TOOL" = /* && -f "$CONTRACT_TOOL" && ! -L "$CONTRACT_TOOL" ]] \
  || heldout_die "OpenMLE source-lock verifier is unavailable: $CONTRACT_TOOL"

SOURCE_LOCK=${OPENMLE_FAST_SOURCE_LOCK_OVERRIDE:-}
if [[ -z "$SOURCE_LOCK" ]]; then
  SOURCE_LOCK=$("$PYBIN" - "$CAMG_HELDOUT_ASSET_RUNTIME_MANIFEST_PATH" <<'PY_RUNTIME'
import json
import pathlib
import sys

document = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
bindings = document.get("source", {}).get("source_locks")
if not isinstance(bindings, list) or not bindings:
    raise SystemExit("held-out runtime manifest has no source-lock binding")
source_path = bindings[0].get("path")
if not isinstance(source_path, str) or not pathlib.Path(source_path).is_absolute():
    raise SystemExit("held-out source-lock path is not absolute")
print(source_path)
PY_RUNTIME
  )
fi
[[ "$SOURCE_LOCK" = /* && -f "$SOURCE_LOCK" && ! -L "$SOURCE_LOCK" ]] \
  || heldout_die "OpenMLE source lock is unavailable: $SOURCE_LOCK"

# These are the only public-manifest values admitted by the held-out path. The
# supervisor strips ambient OPENMLE_FAST_* values and reconstructs both child
# environments from the verified assets before either service starts.
export OPENMLE_FAST_MANIFEST_ROLE=heldout
export OPENMLE_FAST_TASK_MANIFEST="$CAMG_HELDOUT_ASSET_HELDOUT_MANIFEST_PATH"
export OPENMLE_FAST_TASK_MANIFEST_SHA256="$CAMG_HELDOUT_ASSET_HELDOUT_MANIFEST_SHA256"
export AGENTMEMORY_PROCESS_OWNER=amg-heldout-eval
export AGENTMEMORY_RUN_ID="amg_${RUN_ID}_openmle_fast"

exec "$PYBIN" "$SUPERVISOR" \
  --source-lock "$SOURCE_LOCK" \
  --contract-tool "$CONTRACT_TOOL" \
  --runtime-manifest "$CAMG_HELDOUT_ASSET_RUNTIME_MANIFEST_PATH" \
  --runtime-schema "$EXPECTED_RUNTIME_SCHEMA" \
  --heldout-manifest "$CAMG_HELDOUT_ASSET_HELDOUT_MANIFEST_PATH" \
  --private-grader-bindings "$CAMG_HELDOUT_ASSET_PRIVATE_GRADER_BINDINGS_PATH" \
  --routing "$CAMG_HELDOUT_ASSET_ROUTING_PATH" \
  --task-count "$CAMG_HELDOUT_TASK_COUNT" \
  --run-dir "$RUN_DIR" \
  --parent-pid "$AMG_MULTITASK_PARENT_PID" \
  --parent-start-ticks "$AMG_MULTITASK_PARENT_START_TICKS" \
  --outer-root "$CAMG_HELDOUT_SOURCE_OUTER_ROOT" \
  --inner-root "$CAMG_HELDOUT_SOURCE_INNER_ROOT" \
  --private-command "$PYBIN" "$SERVICE_ENTRYPOINT" private \
  --public-command "$PYBIN" "$SERVICE_ENTRYPOINT" public \
  --port "$PORT" \
  --owner "$AGENTMEMORY_PROCESS_OWNER" \
  --run-id "$AGENTMEMORY_RUN_ID"
