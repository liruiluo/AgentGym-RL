#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$HERE/common.sh"

heldout_assert_base_contract literesearcher 65122
heldout_assert_parent
heldout_assert_asset_env HELDOUT_MANIFEST "LiteResearcher held-out manifest"
heldout_assert_asset_env LOADER_RECEIPT "LiteResearcher loader receipt"
heldout_assert_asset_env RETRIEVAL_AND_GRADER_MANIFEST "LiteResearcher retrieval/grader manifest"
heldout_assert_asset_env ROUTING "LiteResearcher held-out routing"
heldout_assert_asset_env RUNTIME_ROWS "LiteResearcher held-out runtime rows"

PT=/home/ai-jingyan-train/luolirui.1/post-train
WS=$PT/agentmemorygym-rl-workspace
OLD=$WS/runtime/amg-main-multitask400-20260823
RUN_ID=${AMG_MULTITASK_RUN_ID:?missing AMG_MULTITASK_RUN_ID}
ENDPOINT_RUN_DIR=${AMG_MULTITASK_ENDPOINT_RUN_DIR:?missing AMG_MULTITASK_ENDPOINT_RUN_DIR}
PORT=${AMG_MULTITASK_ENDPOINT_PORT:?missing AMG_MULTITASK_ENDPOINT_PORT}
[[ "$PORT" == 65122 ]]
[[ "$CAMG_HELDOUT_TASK_COUNT" == 5319 ]]

HELDOUT_MANIFEST=$CAMG_HELDOUT_ASSET_HELDOUT_MANIFEST_PATH
HELDOUT_LOADER_RECEIPT=$CAMG_HELDOUT_ASSET_LOADER_RECEIPT_PATH
HELDOUT_RUNTIME_BINDING=$CAMG_HELDOUT_ASSET_RETRIEVAL_AND_GRADER_MANIFEST_PATH
HELDOUT_ROUTING=$CAMG_HELDOUT_ASSET_ROUTING_PATH
HELDOUT_RUNTIME_ROWS=$CAMG_HELDOUT_ASSET_RUNTIME_ROWS_PATH
OUTER_SOURCE_ROOT=$CAMG_HELDOUT_SOURCE_OUTER_ROOT
INNER_SOURCE_ROOT=$CAMG_HELDOUT_SOURCE_INNER_ROOT
OUTER_SOURCE_COMMIT=$CAMG_HELDOUT_SOURCE_OUTER_COMMIT
INNER_SOURCE_COMMIT=$CAMG_HELDOUT_SOURCE_INNER_COMMIT
ENV_PYTHON=$(heldout_python)
WORKSPACE_RG_BINARY=$WS/runtime/tools/ripgrep/15.1.0-x86_64-unknown-linux-musl/rg
WORKSPACE_RG_SHA256=ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2

TARGET=$WS/runtime/literesearcher/gaia-search-compacted-v2617-2g-serial1-20260825
STORE_MANIFEST=$TARGET/control/store-manifest.json
MATCHED=$OLD/runtime-controls/literesearcher-v2617-jemalloc-khr1-readratio025-readahead-random-r1
STACK_RESUME=$MATCHED/resume-stack.sh
STACK_INSPECTION_ROOT=$WS/runtime/literesearcher/gaia-search-recovery-20260817-lr1/receipts
LOAD_COLLECTION=$WS/runtime/literesearcher/gaia-search-recovery-20260817-lr1/scripts/lr_load_restored_collection.py
MIGRATED_PYTHON=$WS/runtime/literesearcher/gaia-search-recovery-20260818-migrated-r4/python-attempt-3-r2/bin/python3
ENDPOINT_SUPERVISOR=/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/amg-main-multitask400-20260823/runtime-controls/literesearcher-r62-qwentools-v1/lr_endpoint_supervisor_bounded_visitable_r62.sh
ENDPOINT_SUPERVISOR_SHA256=a104687c4b153baa5f78d8a2413a0267f49e2cf47419225c5e4bbb703bde11da
POSTGRES_RUNTIME_ROOT=$WS/runtime/literesearcher/portable-training/r8-runtime-live-p-20260821-r2
POSTGRES_SUPERVISOR=/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/amg-main-multitask400-20260823/runtime-controls/literesearcher-r62-qwentools-v1/postgres_supervisor_r10_readonly_toast_55434.py
POSTGRES_SUPERVISOR_SHA256=cbdeb9989dd189735a6d114f33fbb6e18b7fa98a9af850f0e0ba7b4529d4a384
DIVERSE_VISIT_VERIFIER=/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/amg-main-multitask400-20260823/runtime-controls/literesearcher-r62-qwentools-v1/verify_diverse_bounded_visit_c64_v3_55434.py
DIVERSE_VISIT_VERIFIER_SHA256=5184453941fa092b1344af7154977ed3a9bfdcf7d3c8c39a7cb8e13b23e31ad0
POSTGRES_ATTESTOR=/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/amg-main-multitask400-20260823/runtime-controls/literesearcher-r62-qwentools-v1/assert_colocated_postgres_v2_55434.py
POSTGRES_ATTESTOR_SHA256=77c264a32b0f3f51798ec4a7d726c5d7365241ae4480bfe62e293a480af72345
MILVUS_ASSET=$OLD/runtime-assets/milvus-v2.6.17-runtime-r1
MILVUS_ROOT=/dev/shm/amg-milvus-v2.6.17-full-eval-g12-r1
LR_PYTHON_ROOT=/dev/shm/lr-i-bge-r2
LR_PYTHON_ARCHIVE=$WS/runtime/literesearcher/upstream-runtime-20260816-i/service/runtime/lr-i-bge-r2.tar.gz
RUN_TMP=/dev/shm/amg-lr-heldout-$RUN_ID
POSTGRES_PYTHON=$LR_PYTHON_ROOT/bin/python3
POSTGRES_RUN_DIR=$ENDPOINT_RUN_DIR/local-upstream/postgres-r9
POSTGRES_OWNER=amg-literesearcher-postgres-$RUN_ID
POSTGRES_RUN_TAG=$(printf '%s' "$RUN_ID" | sha256sum | cut -c1-16)
SANDBOX_ROOTFS_PARENT=/tmp/agentmemorygym-sandbox-rootfs-$POSTGRES_RUN_TAG
POSTGRES_EXEC_ROOT=/dev/shm/amg-lr-pg-$POSTGRES_RUN_TAG
POSTGRES_SOCKET_DIR=$POSTGRES_EXEC_ROOT/socket
POSTGRES_READY_RECEIPT=$POSTGRES_RUN_DIR/ready.json
POSTGRES_ATTESTATION=$POSTGRES_RUN_DIR/attestation.json
POSTGRES_CANONICAL_PGDATA=$WS/runtime/literesearcher/upstream-runtime-20260815-e/postgres/data
POSTGRES_OVERLAY_ROOT=/dev/shm/amg-lr-pgov-$POSTGRES_RUN_TAG
POSTGRES_OVERLAY_LOWER=$POSTGRES_OVERLAY_ROOT/lower-readonly
POSTGRES_OVERLAY_UPPER=$POSTGRES_OVERLAY_ROOT/upper
POSTGRES_OVERLAY_WORK=$POSTGRES_OVERLAY_ROOT/work
POSTGRES_OVERLAY_MERGED=$POSTGRES_OVERLAY_ROOT/merged
POSTGRES_LOWER_BEFORE=$POSTGRES_RUN_DIR/canonical-lower-before.json
POSTGRES_LOWER_READY=$POSTGRES_RUN_DIR/canonical-lower-ready.json
POSTGRES_OVERLAY_RECEIPT=$POSTGRES_RUN_DIR/overlay-attestation.json
POSTGRES_READONLY_SHIM_SOURCE=$WS/runtime/literesearcher-formal100-20260826/endpoint-launchers/postgres_toast_readonly_open_shim_v1.c
POSTGRES_READONLY_SHIM_SOURCE_SHA256=c7d0fc44353fdadaaba5c23fce4ac28dbd7d949ceb561284399b753e5b96d303
POSTGRES_READONLY_SHIM_SO=$POSTGRES_EXEC_ROOT/postgres_toast_readonly_open_shim_v1.so
POSTGRES_READONLY_SHIM_LOAD_LOG=$POSTGRES_EXEC_ROOT/readonly-toast-shim-load.tsv
POSTGRES_READONLY_SHIM_LOAD_EVIDENCE=$POSTGRES_RUN_DIR/readonly-toast-shim-load.tsv
POSTGRES_READONLY_VERIFIER=/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/amg-main-multitask400-20260823/runtime-controls/literesearcher-r62-qwentools-v1/verify_postgres_toast_readonly_v1_55434.py
POSTGRES_READONLY_VERIFIER_SHA256=5ff68af10309c023e3377e9125e3561b3f91806a4393760caff6b0cefd50362b
OVERLAY_ROOT=$RUN_TMP/minio-overlay
OVERLAY_UPPER=$OVERLAY_ROOT/upper
OVERLAY_WORK=$OVERLAY_ROOT/work
OVERLAY_MERGED=$OVERLAY_ROOT/merged
MILVUS_LOCAL_STORAGE=$RUN_TMP/milvus-local-storage
STACK_RUN_ID=${RUN_ID}-local-stack
STACK_INSPECTION=$STACK_INSPECTION_ROOT/stack-resume-$STACK_RUN_ID/inspection.json
LOAD_RECEIPT=$ENDPOINT_RUN_DIR/local-upstream/load-receipt.json
UPSTREAM_RUN_DIR=$ENDPOINT_RUN_DIR/local-upstream/service
SERVICE_ID=amg-lr-heldout-${RUN_ID}
NFT_BIN=/usr/sbin/nft
NFT_TABLE=amg_lr_${POSTGRES_RUN_TAG}
NFT_OWNER_ROOT=/dev/shm/$NFT_TABLE
NFT_OWNER_FILE=$NFT_OWNER_ROOT/owner.json
NFT_RULESET_RECEIPT=$ENDPOINT_RUN_DIR/local-upstream/nft-ingress-guard.ruleset.txt
NFT_PORTS=(39091 39529 39530 41123 41124 42222 43100 43333 49531)
JUDGE_SERVICE_DIR=$PT/agentmemorygym-resident-services/literesearcher-kimi-k2.6-remote-20260815
JUDGE_API_BASE=$(cat "$JUDGE_SERVICE_DIR/api_base")
JUDGE_MODEL=$(cat "$JUDGE_SERVICE_DIR/model")
JUDGE_READINESS_PROBE=/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/amg-main-multitask400-20260823/runtime-controls/literesearcher-r82-local-sandbox-root-v1/probe_literesearcher_judge_readiness_v1.py
JUDGE_READINESS_PROBE_SHA256=336db363973e0204face03df86022e49af750eb792d788ba2a39a2cf4410ea0f
JUDGE_READINESS_RECEIPT=$ENDPOINT_RUN_DIR/judge-readiness.json

heldout_assert_executable "$ENV_PYTHON" "LiteResearcher runtime Python"
heldout_assert_executable "$WORKSPACE_RG_BINARY" "LiteResearcher workspace ripgrep"
[[ "$(heldout_sha256 "$WORKSPACE_RG_BINARY")" == "$WORKSPACE_RG_SHA256" ]] \
  || heldout_die "LiteResearcher workspace ripgrep SHA-256 mismatch"
[[ "$(basename -- "$OUTER_SOURCE_ROOT")" == AgentGym-RL && -d "$OUTER_SOURCE_ROOT/async_plugins" ]] \
  || heldout_die "LiteResearcher outer source is not the AgentGym-RL checkout"
[[ -d "$INNER_SOURCE_ROOT/agentenv" && -d "$INNER_SOURCE_ROOT/agentenv-agentmemory" ]] \
  || heldout_die "LiteResearcher inner source lacks endpoint packages"

# The upstream loader still calls this split ``train``.  The immutable CAMG
# binding is authoritative for its held-out role and maps runtime indices
# 0..5318 to the original source rows.  Verify all cross-file bindings before
# allocating any run-scoped service state.
LITERESEARCHER_SOURCE_ROOT=$(
  "$ENV_PYTHON" -B - \
    "$HELDOUT_MANIFEST" "$HELDOUT_LOADER_RECEIPT" \
    "$HELDOUT_RUNTIME_BINDING" "$HELDOUT_ROUTING" \
    "$HELDOUT_RUNTIME_ROWS" "$CAMG_HELDOUT_TASK_COUNT" <<'PY_LR_BINDING'
import hashlib
import json
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


manifest_path, loader_path, binding_path, routing_path, rows_path = map(
    pathlib.Path, sys.argv[1:6]
)
expected_count = int(sys.argv[6])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
loader = json.loads(loader_path.read_text(encoding="utf-8"))
binding = json.loads(binding_path.read_text(encoding="utf-8"))

assert manifest.get("schema") == "agentmemory_literesearcher_full_compatible_pool_v1"
protocol = manifest.get("evaluation_protocol", {})
assert protocol.get("camg_role") == "heldout", protocol
assert protocol.get("runtime_loader_split_alias") == "train", protocol
assert protocol.get("runtime_data_idx_range") == [0, expected_count - 1], protocol
assert int(manifest.get("pool", {}).get("contract_compatible_rows", -1)) == expected_count
rows_binding = manifest.get("artifacts", {}).get("pool_rows.jsonl", {})
assert sha256(rows_path) == rows_binding.get("sha256")
assert rows_path.stat().st_size == int(rows_binding.get("bytes", -1))

assert loader.get("schema") == "camg_literesearcher_heldout_loader_receipt_v1"
assert loader.get("status") == "pass"
assert int(loader.get("loaded_task_count", -1)) == expected_count
assert loader.get("rows", {}).get("sha256") == sha256(rows_path)
assert loader.get("manifest", {}).get("sha256") == sha256(manifest_path)

assert binding.get("schema") == "camg_literesearcher_heldout_runtime_binding_v1"
assert binding.get("status") == "ready"
assert binding.get("heldout_evaluation_run") is False
assert int(binding.get("test_items", -1)) == expected_count
heldout_pool = binding.get("heldout_pool", {})
for name, path in (
    ("manifest", manifest_path),
    ("loader_receipt", loader_path),
    ("routing", routing_path),
    ("rows", rows_path),
):
    item = heldout_pool.get(name, {})
    assert item.get("sha256") == sha256(path), (name, item, path)
    assert int(item.get("bytes", -1)) == path.stat().st_size, (name, item, path)

seen = set()
with routing_path.open(encoding="utf-8") as stream:
    routing_rows = [json.loads(line) for line in stream if line.strip()]
assert len(routing_rows) == expected_count
for expected, row in enumerate(routing_rows):
    assert int(row.get("data_idx", -1)) == expected, (expected, row)
    assert row.get("item_id") == f"literesearcher_{expected}", (expected, row)
    extra = row.get("extra_info", {})
    assert int(extra.get("index", -1)) == expected, (expected, extra)
    assert extra.get("component_role") == "heldout", (expected, extra)
    identity = str(extra.get("row_identity", ""))
    assert len(identity) == 64 and set(identity) <= set("0123456789abcdef")
    assert identity not in seen
    seen.add(identity)
assert len(seen) == expected_count

source_reports = manifest.get("source_reports")
assert isinstance(source_reports, list) and source_reports
source_paths = [pathlib.Path(str(item["parquet_path"])).resolve() for item in source_reports]
source_roots = {path.parent for path in source_paths}
assert len(source_roots) == 1, source_roots
source_root = next(iter(source_roots))
for report, path in zip(source_reports, source_paths):
    assert path.is_file() and not path.is_symlink(), path
    assert sha256(path) == report.get("parquet_sha256"), path
    assert path.name == report.get("parquet_relative_path"), (path, report)
print(source_root)
PY_LR_BINDING
)

mkdir -p "$ENDPOINT_RUN_DIR" "$ENDPOINT_RUN_DIR/local-upstream" "$POSTGRES_RUN_DIR" "$ENDPOINT_RUN_DIR/processes"
mkdir -p -m 700 "$SANDBOX_ROOTFS_PARENT"
chmod 0700 "$SANDBOX_ROOTFS_PARENT"
[[ ! -e "$RUN_TMP" ]]
mkdir -p "$OVERLAY_UPPER" "$OVERLAY_WORK" "$OVERLAY_MERGED" "$MILVUS_LOCAL_STORAGE" \
  "$POSTGRES_OVERLAY_LOWER" "$POSTGRES_OVERLAY_UPPER" "$POSTGRES_OVERLAY_WORK" "$POSTGRES_OVERLAY_MERGED"
printf '%s\n' "$RUN_ID" > "$RUN_TMP/owner-run-id"
printf '%s\n' "$RUN_ID" > "$POSTGRES_OVERLAY_ROOT/owner-run-id"
chmod 0755 "$POSTGRES_OVERLAY_ROOT" "$POSTGRES_OVERLAY_LOWER" "$POSTGRES_OVERLAY_MERGED"
chown 26:26 "$POSTGRES_OVERLAY_UPPER"
chmod 0700 "$POSTGRES_OVERLAY_UPPER" "$POSTGRES_OVERLAY_WORK"

postgres_supervisor_pid=
postgres_supervisor_tick=
postgres_pid=
postgres_tick=
stack_pid=
stack_tick=
upstream_pid=
upstream_tick=
env_pid=
env_tick=
nft_guard_pid=
nft_guard_tick=
mounted=0
postgres_lower_mounted=0
postgres_overlay_mounted=0

same_process() {
  local pid=${1:-} tick=${2:-}
  [[ -n "$pid" && -n "$tick" && -r /proc/$pid/stat ]] || return 1
  [[ "$(awk '{print $22}' /proc/$pid/stat)" == "$tick" ]]
}

capture_tick() {
  local pid=$1
  for _ in $(seq 1 100); do
    if [[ -r /proc/$pid/stat ]]; then awk '{print $22}' /proc/$pid/stat; return 0; fi
    sleep 0.02
  done
  return 1
}

persist_shim_load_log() {
  if [[ -f "$POSTGRES_READONLY_SHIM_LOAD_LOG" ]]; then
    local temporary="$POSTGRES_READONLY_SHIM_LOAD_EVIDENCE.tmp.$$"
    cp -- "$POSTGRES_READONLY_SHIM_LOAD_LOG" "$temporary"
    chmod 0600 "$temporary"
    mv -f -- "$temporary" "$POSTGRES_READONLY_SHIM_LOAD_EVIDENCE"
  fi
}

stop_owned() {
  local pid=${1:-} tick=${2:-} name=${3:-process}
  if same_process "$pid" "$tick"; then
    echo "[lr-fa100-local] TERM $name pid=$pid tick=$tick" >&2
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
      same_process "$pid" "$tick" || return 0
      sleep 1
    done
    if same_process "$pid" "$tick"; then
      echo "[lr-fa100-local] KILL $name pid=$pid tick=$tick" >&2
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
}

unmount_owned_retry() {
  local target=$1 name=${2:-mount}
  for _ in $(seq 1 120); do
    mountpoint -q "$target" || return 0
    if umount "$target" 2>/dev/null; then
      mountpoint -q "$target" || return 0
    fi
    sleep 1
  done
  echo "[lr-fa100-local] owned $name remained mounted after bounded cleanup: $target" >&2
  return 1
}

reap_postgres_mount_holders() {
  local target=$1 out="$POSTGRES_RUN_DIR/orphan-postgres-cleanup.json"
  setpriv --reuid=26 --regid=26 --clear-groups python3 - "$target" "$RUN_ID" <<'PY_POSTGRES_REAP' > "$out.tmp"
import json, os, pathlib, signal, sys, time

target = str(pathlib.Path(sys.argv[1])) + "/"
run_id = sys.argv[2]

def identity(pid):
    raw = pathlib.Path(f"/proc/{pid}/stat").read_text()
    tail = raw[raw.rfind(")") + 2:].split()
    return {
        "state": tail[0],
        "ppid": int(tail[1]),
        "pgrp": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
    }

def matching():
    found = []
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        try:
            status = (proc / "status").read_text()
            uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
            if int(uid_line.split()[2]) != 26:
                continue
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            if not cmdline.startswith("postgres:"):
                continue
            fds = []
            for fd in (proc / "fd").iterdir():
                try:
                    dest = os.readlink(fd)
                except OSError:
                    continue
                if dest.startswith(target):
                    fds.append(dest)
            if fds:
                found.append({"pid": pid, "cmdline": cmdline, "identity": identity(pid), "fds": sorted(fds)})
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            pass
    return sorted(found, key=lambda item: item["pid"])

before = matching()
after_term = before
for item in before:
    os.kill(item["pid"], signal.SIGTERM)
for _ in range(10):
    time.sleep(1)
    after_term = matching()
    if not after_term:
        break
for item in after_term:
    old = next(candidate for candidate in before if candidate["pid"] == item["pid"])
    if item["identity"]["start_ticks"] != old["identity"]["start_ticks"]:
        raise RuntimeError(("pid reuse during exact postgres cleanup", item, old))
    os.kill(item["pid"], signal.SIGKILL)
remaining = after_term
for _ in range(180):
    time.sleep(1)
    remaining = matching()
    if not remaining:
        break
payload = {
    "schema": "amg_literesearcher_exact_orphan_postgres_cleanup_v1",
    "status": "PASS" if not remaining else "INCOMPLETE",
    "run_id": run_id,
    "target_mount": target.rstrip("/"),
    "selection": "euid=26 AND cmdline postgres: AND fd under exact run overlay",
    "before": before,
    "after_term": after_term,
    "remaining": remaining,
    "created_at_epoch": time.time(),
}
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if not remaining else 2)
PY_POSTGRES_REAP
  local reap_rc=$?
  mv "$out.tmp" "$out"
  return "$reap_rc"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  set +e
  # Stop the parent-death firewall guard while this parent is still alive,
  # then remove only this run's named nft table after exact owner verification.
  stop_owned "$nft_guard_pid" "$nft_guard_tick" nft-ingress-guard
  if "$NFT_BIN" list table inet "$NFT_TABLE" >/dev/null 2>&1; then
    python3 - "$NFT_OWNER_FILE" "$RUN_ID" "$NFT_TABLE" <<'PY_NFT_OWNER'
import json,sys
p=json.load(open(sys.argv[1]))
assert p.get("schema")=="amg_run_scoped_nft_owner_v1",p
assert p.get("run_id")==sys.argv[2],p
assert p.get("table")==sys.argv[3],p
PY_NFT_OWNER
    "$NFT_BIN" delete table inet "$NFT_TABLE"
  fi
  if [[ "$NFT_OWNER_ROOT" == /dev/shm/amg_lr_* && ! -L "$NFT_OWNER_ROOT"       && -f "$NFT_OWNER_FILE" ]]; then
    owner_run=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_id",""))' "$NFT_OWNER_FILE" 2>/dev/null || true)
    [[ "$owner_run" == "$RUN_ID" ]] && rm -rf --one-file-system "$NFT_OWNER_ROOT"
  fi

  # Stop consumers before their backing stores.
  stop_owned "$env_pid" "$env_tick" environment-server
  stop_owned "$upstream_pid" "$upstream_tick" search-visit-supervisor
  stop_owned "$stack_pid" "$stack_tick" milvus-stack-supervisor
  # The canonical PostgreSQL supervisor owns and reaps its exact child.  The
  # second stop is only a PID/start-tick-guarded fallback if the supervisor died.
  stop_owned "$postgres_supervisor_pid" "$postgres_supervisor_tick" postgres-supervisor
  stop_owned "$postgres_pid" "$postgres_tick" postgres-fallback
  persist_shim_load_log || true
  cleanup_mount_failure=0
  # PostgreSQL backends can remain in uninterruptible NFS reads after the
  # postmaster exits. Reap only uid-26 postgres processes that still hold an
  # fd under this run's unique overlay before attempting to unmount it.
  reap_postgres_mount_holders "$POSTGRES_OVERLAY_MERGED" || cleanup_mount_failure=1
  if [[ "$postgres_overlay_mounted" == 1 ]] && mountpoint -q "$POSTGRES_OVERLAY_MERGED"; then
    unmount_owned_retry "$POSTGRES_OVERLAY_MERGED" postgres-overlay || cleanup_mount_failure=1
  fi
  if [[ "$postgres_lower_mounted" == 1 ]] && mountpoint -q "$POSTGRES_OVERLAY_LOWER"; then
    unmount_owned_retry "$POSTGRES_OVERLAY_LOWER" postgres-lower || cleanup_mount_failure=1
  fi
  if [[ "$POSTGRES_OVERLAY_ROOT" == /dev/shm/amg-lr-pgov-* \
      && ! -L "$POSTGRES_OVERLAY_ROOT" \
      && -f "$POSTGRES_OVERLAY_ROOT/owner-run-id" \
      && "$(cat "$POSTGRES_OVERLAY_ROOT/owner-run-id")" == "$RUN_ID" ]]; then
    if ! mountpoint -q "$POSTGRES_OVERLAY_MERGED" \
        && ! mountpoint -q "$POSTGRES_OVERLAY_LOWER"; then
      rm -rf --one-file-system "$POSTGRES_OVERLAY_ROOT"
    fi
  fi
  if [[ "$POSTGRES_EXEC_ROOT" == /dev/shm/amg-lr-pg-* \
      && ! -L "$POSTGRES_EXEC_ROOT" \
      && -f "$POSTGRES_EXEC_ROOT/owner-run-id" \
      && "$(cat "$POSTGRES_EXEC_ROOT/owner-run-id")" == "$RUN_ID" ]]; then
    rm -rf --one-file-system "$POSTGRES_EXEC_ROOT"
  fi
  if [[ "$mounted" == 1 ]] && mountpoint -q "$OVERLAY_MERGED"; then
    umount "$OVERLAY_MERGED"
  fi
  if [[ -f "$RUN_TMP/owner-run-id" && "$(cat "$RUN_TMP/owner-run-id")" == "$RUN_ID" ]]; then
    rm -rf --one-file-system "$RUN_TMP"
  fi
  if [[ "$SANDBOX_ROOTFS_PARENT" == /tmp/agentmemorygym-sandbox-rootfs-* ]]; then
    rmdir "$SANDBOX_ROOTFS_PARENT" 2>/dev/null || true
  fi
  printf '%s\n' "$rc" > "$ENDPOINT_RUN_DIR/local-upstream-wrapper.exit"
  if [[ "$cleanup_mount_failure" != 0 && "$rc" == 0 ]]; then rc=91; fi
  printf '{"schema":"amg_literesearcher_local_stack_cleanup_v3","run_id":"%s","exit_code":%d,"mount_cleanup_pass":%s,"cleaned_at":"%s"}\n' \
    "$RUN_ID" "$rc" "$([[ "$cleanup_mount_failure" == 0 ]] && echo true || echo false)" "$(date -u +%FT%TZ)" > "$ENDPOINT_RUN_DIR/local-upstream-cleanup.json"
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

# Cross-Pod service plumbing is prohibited; this launcher binds every owned service to loopback.

python3 - "$STORE_MANIFEST" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['status']=='READY',p
assert p['canonical_lower_modified'] is False,p
PY
[[ -r "$TARGET/control/etcd.db" && -r "$TARGET/control/etcd.db.sha256" ]]
[[ -d "$TARGET/minio_data" ]]
[[ -x "$STACK_RESUME" && -x "$LOAD_COLLECTION" && -x "$MIGRATED_PYTHON" && -x "$ENDPOINT_SUPERVISOR" ]]
[[ "$(sha256sum "$ENDPOINT_SUPERVISOR" | awk '{print $1}')" == "$ENDPOINT_SUPERVISOR_SHA256" ]]
[[ "$(sha256sum "$DIVERSE_VISIT_VERIFIER" | awk '{print $1}')" == "$DIVERSE_VISIT_VERIFIER_SHA256" ]]
[[ -x "$POSTGRES_SUPERVISOR" && -x "$POSTGRES_ATTESTOR" ]]
[[ "$(sha256sum "$POSTGRES_SUPERVISOR" | awk '{print $1}')" == "$POSTGRES_SUPERVISOR_SHA256" ]]
[[ "$(sha256sum "$POSTGRES_ATTESTOR" | awk '{print $1}')" == "$POSTGRES_ATTESTOR_SHA256" ]]
[[ -r "$POSTGRES_READONLY_SHIM_SOURCE" && -x "$POSTGRES_READONLY_VERIFIER" ]]
[[ "$(sha256sum "$POSTGRES_READONLY_SHIM_SOURCE" | awk '{print $1}')" == "$POSTGRES_READONLY_SHIM_SOURCE_SHA256" ]]
[[ "$(sha256sum "$POSTGRES_READONLY_VERIFIER" | awk '{print $1}')" == "$POSTGRES_READONLY_VERIFIER_SHA256" ]]
command -v gcc >/dev/null
[[ "$(sha256sum "$JUDGE_READINESS_PROBE" | awk '{print $1}')" == "$JUDGE_READINESS_PROBE_SHA256" ]]
python3 "$JUDGE_READINESS_PROBE" \
  --api-base "$JUDGE_API_BASE" \
  --model "$JUDGE_MODEL" \
  --output "$JUDGE_READINESS_RECEIPT"

# EmptyDir admission: preserve at least 200 GiB below the platform's 800 GiB cap.
shm_used=$(df -B1 --output=used /dev/shm | awk 'NR==2 {print $1}')
platform_limit=$((800 * 1024 * 1024 * 1024))
minimum_headroom=$((200 * 1024 * 1024 * 1024))
(( platform_limit - shm_used >= minimum_headroom )) || {
  echo "insufficient EmptyDir headroom: used=$shm_used limit=$platform_limit" >&2
  exit 70
}

# The 1.2-TiB canonical PostgreSQL tree is immutable shared lower data. Each
# Pod gets an explicit read-only bind lower plus a run-scoped tmpfs OverlayFS
# upper/work/merged. PostgreSQL never opens the canonical path as PGDATA.
[[ -d "$POSTGRES_CANONICAL_PGDATA" ]]
[[ ! -e "$POSTGRES_CANONICAL_PGDATA/postmaster.pid" ]]
python3 - "$POSTGRES_CANONICAL_PGDATA" "$POSTGRES_LOWER_BEFORE" "$RUN_ID" <<'PY'
import hashlib,json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2]); run_id=sys.argv[3]
files=("PG_VERSION","postgresql.conf","pg_hba.conf","global/pg_control")
def item(rel):
    p=root/rel; st=p.stat(); h=hashlib.sha256(p.read_bytes()).hexdigest()
    return {"path":rel,"size":st.st_size,"mtime_ns":st.st_mtime_ns,"sha256":h}
payload={"schema":"amg_postgres_shared_lower_snapshot_v1","status":"PASS","run_id":run_id,"canonical_pgdata":str(root),"captured_at_epoch":time.time(),"postmaster_pid_present":(root/"postmaster.pid").exists(),"files":[item(x) for x in files]}
tmp=out.with_name(out.name+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,out)
PY
mount --bind "$POSTGRES_CANONICAL_PGDATA" "$POSTGRES_OVERLAY_LOWER"
postgres_lower_mounted=1
mount -o remount,bind,ro "$POSTGRES_OVERLAY_LOWER"
findmnt -rn -T "$POSTGRES_OVERLAY_LOWER" -o OPTIONS | tr ',' '\n' | grep -qx ro
mount -t overlay overlay -o "lowerdir=$POSTGRES_OVERLAY_LOWER,upperdir=$POSTGRES_OVERLAY_UPPER,workdir=$POSTGRES_OVERLAY_WORK" "$POSTGRES_OVERLAY_MERGED"
postgres_overlay_mounted=1
[[ "$(findmnt -rn -T "$POSTGRES_OVERLAY_MERGED" -o FSTYPE)" == overlay ]]
[[ ! -e "$POSTGRES_OVERLAY_MERGED/postmaster.pid" ]]

python3 - "$MILVUS_ASSET/manifest.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p['status']=='READY' and p['milvus_version']=='2.6.17',p
PY
printf '%s  %s\n' e1f0b233707cf490c67e19be8990b33366d4c7dbd8f7078c13ea01c58a7192f3 "$MILVUS_ASSET/milvus-rootfs.tar.gz" | sha256sum -c -
if [[ ! -x "$MILVUS_ROOT/milvus/bin/milvus" ]]; then
  [[ ! -e "$MILVUS_ROOT" ]]
  stage=$RUN_TMP/milvus-rootfs-stage
  mkdir -p "$stage"
  tar -xzf "$MILVUS_ASSET/milvus-rootfs.tar.gz" -C "$stage"
  [[ -x "$stage/$(basename "$MILVUS_ROOT")/milvus/bin/milvus" ]]
  mv "$stage/$(basename "$MILVUS_ROOT")" "$MILVUS_ROOT"
  rmdir "$stage"
fi
printf '%s  %s\n' 870327298400681fe628a0169f4168d7de0d88f8c79606025d56e1d815054a92 "$MILVUS_ROOT/milvus/bin/milvus" | sha256sum -c -

if [[ ! -x "$LR_PYTHON_ROOT/bin/python3" ]]; then
  [[ ! -e "$LR_PYTHON_ROOT" ]]
  expected=$(awk 'NR==1 {print $1}' "$LR_PYTHON_ARCHIVE.sha256")
  [[ "$(sha256sum "$LR_PYTHON_ARCHIVE" | awk '{print $1}')" == "$expected" ]]
  stage=$RUN_TMP/lr-python-stage
  mkdir -p "$stage"
  tar -xzf "$LR_PYTHON_ARCHIVE" -C "$stage"
  [[ -x "$stage/$(basename "$LR_PYTHON_ROOT")/bin/python3" ]]
  mv "$stage/$(basename "$LR_PYTHON_ROOT")" "$LR_PYTHON_ROOT"
  rmdir "$stage"
fi

# The frozen object store is a lower layer. All runtime writes remain run-scoped.
mount -t overlay overlay -o "lowerdir=$TARGET/minio_data,upperdir=$OVERLAY_UPPER,workdir=$OVERLAY_WORK" "$OVERLAY_MERGED"
mounted=1
mountpoint -q "$OVERLAY_MERGED"

# Start the mature run-scoped PostgreSQL owner.  It takes the canonical NFS
# owner lock before opening PGDATA, stages only the executable runtime in tmpfs,
# binds loopback, and records exact supervisor/child identities.  A residual
# postmaster.pid is a hard failure here; one-time recovery must be performed
# separately while holding the same canonical lock with immutable evidence.
[[ -x "$POSTGRES_PYTHON" ]]
mkdir -p "$POSTGRES_RUN_DIR"
[[ ! -e "$POSTGRES_READY_RECEIPT" && ! -e "$POSTGRES_RUN_DIR/exit.json" && ! -e "$POSTGRES_RUN_DIR/supervisor.pid" ]]
[[ ! -e "$POSTGRES_EXEC_ROOT" ]]
mkdir -m 0755 "$POSTGRES_EXEC_ROOT"
printf '%s\n' "$RUN_ID" > "$POSTGRES_EXEC_ROOT/owner-run-id"
gcc -shared -fPIC -O2 -Wall -Wextra -Werror -ldl \
  -o "$POSTGRES_READONLY_SHIM_SO" "$POSTGRES_READONLY_SHIM_SOURCE"
chmod 0555 "$POSTGRES_READONLY_SHIM_SO"
: > "$POSTGRES_READONLY_SHIM_LOAD_LOG"
chown 26:26 "$POSTGRES_READONLY_SHIM_LOAD_LOG"
chmod 0600 "$POSTGRES_READONLY_SHIM_LOAD_LOG"
if ss -H -ltn 'sport = :55434' | grep -q .; then
  echo 'PostgreSQL port 55434 is already occupied' >&2
  exit 98
fi
env \
  LD_PRELOAD="$POSTGRES_READONLY_SHIM_SO" \
  AMG_POSTGRES_READONLY_RELATION_OPEN_SHIM=1 \
  AMG_POSTGRES_READONLY_SHIM_LOAD_LOG="$POSTGRES_READONLY_SHIM_LOAD_LOG" \
  AGENTMEMORY_WORKSPACE_ROOT="$WS" \
  LITERESEARCHER_POSTGRES_OWNER="$POSTGRES_OWNER" \
  LITERESEARCHER_POSTGRES_EXPECTED_HOST="$(hostname)" \
  LITERESEARCHER_POSTGRES_RUN_DIR="$POSTGRES_RUN_DIR" \
  LITERESEARCHER_POSTGRES_PGDATA="$POSTGRES_OVERLAY_MERGED" \
  LITERESEARCHER_POSTGRES_EXEC_ROOT="$POSTGRES_EXEC_ROOT" \
  LITERESEARCHER_POSTGRES_SOCKET_DIR="$POSTGRES_SOCKET_DIR" \
  "$POSTGRES_PYTHON" "$POSTGRES_SUPERVISOR" \
    > "$POSTGRES_RUN_DIR/supervisor.stdout.log" 2>&1 &
postgres_supervisor_pid=$!
postgres_supervisor_tick=$(capture_tick "$postgres_supervisor_pid")
for _ in $(seq 1 900); do
  [[ -s "$POSTGRES_READY_RECEIPT" ]] && break
  same_process "$postgres_supervisor_pid" "$postgres_supervisor_tick"
  sleep 1
done
[[ -s "$POSTGRES_READY_RECEIPT" ]]
[[ -S "$POSTGRES_SOCKET_DIR/.s.PGSQL.55434" ]]

"$POSTGRES_PYTHON" "$POSTGRES_ATTESTOR" \
  --ready-receipt "$POSTGRES_READY_RECEIPT" \
  --supervisor-pid-file "$POSTGRES_RUN_DIR/supervisor.pid" \
  --pgpass "$POSTGRES_RUN_DIR/pgpass" \
  --expected-owner "$POSTGRES_OWNER" \
  --expected-runtime-host "$(hostname)" \
  --expected-postgres-host "$(hostname)" \
  --expected-run-dir "$POSTGRES_RUN_DIR" \
  --expected-supervisor-script "$POSTGRES_SUPERVISOR" \
  --expected-supervisor-sha256 "$POSTGRES_SUPERVISOR_SHA256" \
  --postgres-socket "$POSTGRES_SOCKET_DIR" \
  --output "$POSTGRES_ATTESTATION" \
  > "$POSTGRES_RUN_DIR/attestation.stdout.log"

postgres_values=$(python3 - "$POSTGRES_READY_RECEIPT" "$postgres_supervisor_pid" "$postgres_supervisor_tick" "$POSTGRES_OVERLAY_MERGED" <<'PY'
import json,sys
from pathlib import Path
ready_path,expected_supervisor_pid,expected_supervisor_tick,expected_pgdata=sys.argv[1:]
p=json.load(open(ready_path, encoding='utf-8'))
assert p.get('schema') == 'amg_literesearcher_colocated_postgres_v1', p
assert p.get('status') == 'PASS', p
assert p.get('canonical_lock_held') is True, p
assert p.get('network_scope') == 'loopback_only', p
assert p.get('relay') is False, p
assert p.get('second_pgdata_copy_created') is False, p
assert p.get('canonical_pgdata') == expected_pgdata, p
supervisor_pid=int(expected_supervisor_pid)
raw=Path(f'/proc/{supervisor_pid}/stat').read_text(encoding='ascii')
supervisor_tick=int(raw[raw.rfind(')') + 2:].split()[19])
assert supervisor_tick == int(expected_supervisor_tick), (supervisor_tick, expected_supervisor_tick)
pid=int(p['postgres']['pid']); tick=int(p['postgres']['start_ticks'])
raw=Path(f'/proc/{pid}/stat').read_text(encoding='ascii')
observed=int(raw[raw.rfind(')') + 2:].split()[19])
assert observed == tick, (observed,tick,p)
print(pid, tick, sep='\t')
PY
)
IFS=$'\t' read -r postgres_pid postgres_tick <<< "$postgres_values"
same_process "$postgres_supervisor_pid" "$postgres_supervisor_tick"
same_process "$postgres_pid" "$postgres_tick"
"$POSTGRES_PYTHON" "$POSTGRES_READONLY_VERIFIER" \
  --postgres-pid "$postgres_pid" \
  --upper-root "$POSTGRES_OVERLAY_UPPER" \
  --postgres-log "$POSTGRES_RUN_DIR/postgres.log" \
  --shim-source "$POSTGRES_READONLY_SHIM_SOURCE" \
  --shim-binary "$POSTGRES_READONLY_SHIM_SO" \
  --shim-load-log "$POSTGRES_READONLY_SHIM_LOAD_LOG" \
  --pgpass "$POSTGRES_RUN_DIR/pgpass" \
  --phase post-readiness \
  --output "$POSTGRES_RUN_DIR/readonly-toast-post-readiness.json" \
  > "$POSTGRES_RUN_DIR/readonly-toast-post-readiness.stdout.log"
persist_shim_load_log
[[ ! -e "$POSTGRES_CANONICAL_PGDATA/postmaster.pid" ]]
[[ -s "$POSTGRES_OVERLAY_MERGED/postmaster.pid" ]]
[[ -s "$POSTGRES_OVERLAY_UPPER/postmaster.pid" ]]
python3 - "$POSTGRES_CANONICAL_PGDATA" "$POSTGRES_LOWER_BEFORE" "$POSTGRES_LOWER_READY" "$POSTGRES_OVERLAY_RECEIPT" "$POSTGRES_OVERLAY_LOWER" "$POSTGRES_OVERLAY_UPPER" "$POSTGRES_OVERLAY_WORK" "$POSTGRES_OVERLAY_MERGED" "$RUN_ID" <<'PY'
import hashlib,json,os,sys,time
from pathlib import Path
root,before_path,ready_path,out,*mounts,run_id=sys.argv[1:]
root=Path(root); before_path=Path(before_path); ready_path=Path(ready_path); out=Path(out)
files=("PG_VERSION","postgresql.conf","pg_hba.conf","global/pg_control")
def item(rel):
    p=root/rel; st=p.stat(); return {"path":rel,"size":st.st_size,"mtime_ns":st.st_mtime_ns,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
ready={"schema":"amg_postgres_shared_lower_snapshot_v1","status":"PASS","run_id":run_id,"canonical_pgdata":str(root),"captured_at_epoch":time.time(),"postmaster_pid_present":(root/"postmaster.pid").exists(),"files":[item(x) for x in files]}
tmp=ready_path.with_name(ready_path.name+".tmp"); tmp.write_text(json.dumps(ready,indent=2,sort_keys=True)+"\n"); os.replace(tmp,ready_path)
before=json.load(open(before_path))
assert before["postmaster_pid_present"] is False and ready["postmaster_pid_present"] is False,(before,ready)
assert before["files"]==ready["files"],(before,ready)
payload={"schema":"amg_postgres_per_pod_overlay_v1","status":"PASS","run_id":run_id,"canonical_lower":str(root),"canonical_lower_unchanged":True,"canonical_postmaster_pid_present":False,"lower_snapshot_before":str(before_path),"lower_snapshot_ready":str(ready_path),"mounts":{"lower_readonly":mounts[0],"upper":mounts[1],"work":mounts[2],"merged":mounts[3]},"postgres_pgdata":mounts[3],"network_scope":"loopback_only","cross_pod_forwarding":False,"created_at_epoch":time.time()}
tmp=out.with_name(out.name+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,out)
PY

# Warm the frozen database's generic exact-URL path, not only the readiness
# sample. A fresh Pod sees PostgreSQL 9.2's 1-TiB TOAST relation through NFS;
# page-cache the 7.1-GiB heap, 1.6-GiB URL B-tree, and 15.8-GiB TOAST B-tree,
# and touch every TOAST segment's first page so arbitrary URLs do not pay a
# metadata-open storm. This is read-only infrastructure and leaves task,
# Search/Visit semantics, rewards, and policy observations unchanged.
POSTGRES_GENERIC_PREWARM="$POSTGRES_RUN_DIR/generic-visit-path-prewarm.json"
python3 - "$POSTGRES_OVERLAY_MERGED" "$POSTGRES_GENERIC_PREWARM" <<'PY_GENERIC_PREWARM'
import json, os, re, sys, time
from pathlib import Path

pgdata = Path(sys.argv[1])
out = Path(sys.argv[2])
base = pgdata / "base" / "13017"
contracts = [
    ("documents_heap", "16387", "full", 8, 7_611_367_424),
    ("documents_url_key", "16397", "full", 2, 1_744_928_768),
    ("documents_toast_index", "16394", "full", 16, 17_012_252_672),
    ("documents_toast_heap_metadata", "16392", "first_page", 1104, 1_184_464_109_568),
]
started = time.monotonic()
items = []
for name, relfilenode, mode, expected_files, expected_bytes in contracts:
    pattern = re.compile(rf"^{re.escape(relfilenode)}(?:\.\d+)?$")
    files = [p for p in base.iterdir() if p.is_file() and pattern.fullmatch(p.name)]
    files.sort(key=lambda p: 0 if p.name == relfilenode else int(p.name.split(".", 1)[1]) + 1)
    sizes = [p.stat().st_size for p in files]
    assert len(files) == expected_files, (name, len(files), expected_files)
    assert sum(sizes) == expected_bytes, (name, sum(sizes), expected_bytes)
    read_started = time.monotonic()
    bytes_read = 0
    for path, size in zip(files, sizes):
        with path.open("rb", buffering=0) as handle:
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass
            remaining = size if mode == "full" else min(size, 8192)
            while remaining:
                data = handle.read(min(8 * 1024 * 1024, remaining))
                if not data:
                    break
                bytes_read += len(data)
                remaining -= len(data)
    items.append({
        "name": name,
        "relfilenode": relfilenode,
        "mode": mode,
        "files": len(files),
        "relation_bytes": sum(sizes),
        "bytes_read": bytes_read,
        "elapsed_seconds": time.monotonic() - read_started,
    })
payload = {
    "schema": "amg_literesearcher_postgres_generic_visit_path_prewarm_v1",
    "status": "PASS",
    "read_only": True,
    "pgdata": str(pgdata),
    "relations": items,
    "elapsed_seconds": time.monotonic() - started,
    "created_at_epoch": time.time(),
}
tmp = out.with_name(out.name + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out)
PY_GENERIC_PREWARM

# Attest the legacy frozen sample's index/heap identity without materializing
# its unrelated full TOAST body. Full-body readiness is decided below by the
# 64 distinct schedule-derived cold Visits plus Fulda and known-miss checks.
POSTGRES_VISIT_SAMPLE="$WS/runtime/literesearcher/upstream-runtime-20260815-e/receipts/browse-final-20260816/visit-sample-handoff.json"
POSTGRES_VISIT_PREWARM_SECONDS="$POSTGRES_RUN_DIR/visit-prewarm.seconds"
POSTGRES_VISIT_PREWARM_RESULT="$POSTGRES_RUN_DIR/visit-prewarm.result"
POSTGRES_VISIT_SAMPLE_URL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["url"])' "$POSTGRES_VISIT_SAMPLE")
PGPASSFILE="$POSTGRES_RUN_DIR/pgpass" \
POSTGRES_VISIT_SAMPLE_URL="$POSTGRES_VISIT_SAMPLE_URL" \
/usr/bin/time -f '%e' -o "$POSTGRES_VISIT_PREWARM_SECONDS" \
"$POSTGRES_PYTHON" - <<'PY_SAMPLE_QUERY' > "$POSTGRES_VISIT_PREWARM_RESULT"
import os
import psycopg2

conn = psycopg2.connect(
    host="127.0.0.1",
    port=55434,
    dbname="postgres",
    user="literesearch_visit",
    connect_timeout=10,
    options="-c statement_timeout=60000 -c default_transaction_read_only=on",
)
try:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT id,length(coalesce(title,'')),CASE WHEN text IS NULL THEN 0 ELSE 1 END "
            "FROM litesearch_sql.documents WHERE url=%s LIMIT 1",
            (os.environ["POSTGRES_VISIT_SAMPLE_URL"],),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("frozen Visit sample URL is absent from PostgreSQL")
        print("|".join(str(value) for value in row))
finally:
    conn.close()
PY_SAMPLE_QUERY
python3 - "$POSTGRES_VISIT_PREWARM_RESULT" "$POSTGRES_VISIT_PREWARM_SECONDS" "$POSTGRES_RUN_DIR/visit-prewarm.json" "$POSTGRES_VISIT_SAMPLE_URL" <<'PY_SAMPLE_PREWARM'
import hashlib,json,os,sys,time
from pathlib import Path
result_path,seconds_path,out=map(Path,sys.argv[1:4]); sample_url=sys.argv[4]
parts=result_path.read_text().strip().split('|')
assert len(parts)==3 and int(parts[0])>0 and int(parts[1])>=0 and int(parts[2])==1,parts
elapsed=float(seconds_path.read_text().strip())
payload={"schema":"amg_literesearcher_postgres_frozen_sample_presence_v3","status":"PASS","read_only":True,"full_body_materialized":False,"full_body_gate":"diverse-cold-c64-v2","sample_url_sha256":hashlib.sha256(sample_url.encode()).hexdigest(),"row":{"id":int(parts[0]),"title_chars":int(parts[1]),"text_pointer_present":bool(int(parts[2]))},"elapsed_seconds":elapsed,"created_at_epoch":time.time()}
tmp=out.with_name(out.name+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,out)
PY_SAMPLE_PREWARM

# Milvus 2.6.17 upstream records proxy.ip as its advertised address but its
# netutil.NewListener binds :port. Install a run-scoped kernel ingress guard
# before starting Milvus so those sockets are reachable only through loopback.
[[ -x "$NFT_BIN" ]]
[[ ! -e "$NFT_OWNER_ROOT" ]]
! "$NFT_BIN" list table inet "$NFT_TABLE" >/dev/null 2>&1
mkdir -p -m 700 "$NFT_OWNER_ROOT"
parent_tick=$(capture_tick $$)
python3 - "$NFT_OWNER_FILE" "$RUN_ID" "$NFT_TABLE" "$$" "$parent_tick" <<'PY_NFT_WRITE'
import json,os,sys,time
from pathlib import Path
out=Path(sys.argv[1])
p={"schema":"amg_run_scoped_nft_owner_v1","status":"ACTIVE","run_id":sys.argv[2],"table":sys.argv[3],"parent_pid":int(sys.argv[4]),"parent_start_tick":int(sys.argv[5]),"host":os.uname().nodename,"created_at_epoch":time.time()}
tmp=out.with_name(out.name+".tmp"); tmp.write_text(json.dumps(p,indent=2,sort_keys=True)+"\n"); tmp.replace(out)
PY_NFT_WRITE
"$NFT_BIN" add table inet "$NFT_TABLE"
"$NFT_BIN" "add chain inet $NFT_TABLE input { type filter hook input priority -200; policy accept; }"
for nft_port in "${NFT_PORTS[@]}"; do
  "$NFT_BIN" add rule inet "$NFT_TABLE" input iifname != lo tcp dport "$nft_port" reject
 done
"$NFT_BIN" list table inet "$NFT_TABLE" > "$NFT_RULESET_RECEIPT.tmp"
mv "$NFT_RULESET_RECEIPT.tmp" "$NFT_RULESET_RECEIPT"
cat > "$NFT_OWNER_ROOT/watch_parent.py" <<'PY_NFT_GUARD'
import json,os,pathlib,subprocess,sys,time
owner_path=pathlib.Path(sys.argv[1]); nft=sys.argv[2]; table=sys.argv[3]; parent=int(sys.argv[4]); tick=int(sys.argv[5])
def alive():
    try:
        raw=pathlib.Path(f"/proc/{parent}/stat").read_text()
        return int(raw[raw.rfind(")")+2:].split()[19])==tick
    except Exception:
        return False
while alive(): time.sleep(0.5)
try:
    p=json.loads(owner_path.read_text())
    if p.get("schema")=="amg_run_scoped_nft_owner_v1" and p.get("table")==table and int(p.get("parent_pid",-1))==parent and int(p.get("parent_start_tick",-1))==tick:
        subprocess.run([nft,"delete","table","inet",table],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
except Exception:
    pass
PY_NFT_GUARD
chmod 700 "$NFT_OWNER_ROOT/watch_parent.py"
setsid python3 "$NFT_OWNER_ROOT/watch_parent.py" "$NFT_OWNER_FILE" "$NFT_BIN" "$NFT_TABLE" "$$" "$parent_tick"   </dev/null >"$NFT_OWNER_ROOT/watch-parent.log" 2>&1 &
nft_guard_pid=$!
nft_guard_tick=$(capture_tick "$nft_guard_pid")
same_process "$nft_guard_pid" "$nft_guard_tick"

# Start loopback-only etcd/MinIO/Milvus from immutable snapshot + overlay.
env \
  CHECKPOINT_OVERRIDE="$TARGET/control/etcd.db" \
  MINIO_DATA_OVERRIDE="$OVERLAY_MERGED" \
  MILVUS_LOCAL_STORAGE_OVERRIDE="$MILVUS_LOCAL_STORAGE" \
  MILVUS_ROOT_OVERRIDE="$MILVUS_ROOT" \
  MALLOC_CONF_OVERRIDE='background_thread:true' \
  RUN_ID="$STACK_RUN_ID" HOST_IP=127.0.0.1 \
  "$STACK_RESUME" > "$ENDPOINT_RUN_DIR/local-upstream/stack-supervisor.log" 2>&1 &
stack_pid=$!
stack_tick=$(capture_tick "$stack_pid")
for _ in $(seq 1 600); do
  [[ -s "$STACK_INSPECTION" ]] && break
  same_process "$stack_pid" "$stack_tick"
  sleep 1
done
[[ -s "$STACK_INSPECTION" ]]
python3 - "$STACK_INSPECTION" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p['status']=='PASS',p
assert p['server_version']=='2.6.17',p
assert p['entity_count']==32127370,p
assert p['compaction_disabled'] is True,p
assert p['garbage_collection_disabled'] is True,p
PY

# Explicit load receipt consumed by the endpoint supervisor.
"$MIGRATED_PYTHON" "$LOAD_COLLECTION" \
  --host 127.0.0.1 --port 39530 --collection litesearch \
  --timeout-seconds 1200 --receipt "$LOAD_RECEIPT" \
  > "$ENDPOINT_RUN_DIR/local-upstream/load-collection.log" 2>&1

# Start Search/Visit in the same supervised process group, loopback-only.
env \
  WORKSPACE="$WS" \
  RESTORE_RECEIPT="$LOAD_RECEIPT" MILVUS_URI=http://127.0.0.1:39530 \
  RUN_STAMP="$RUN_ID" OWNER="$RUN_ID" RUN_DIR="$UPSTREAM_RUN_DIR" \
  LOCAL_ROOT="$RUN_TMP/endpoint-local" ENDPOINT_LOCK_PATH="$RUN_TMP/endpoint.lock" \
  SERVICE_BIND_HOST=127.0.0.1 SERVICE_PORT=18018 \
  LITERESEARCHER_ALLOWED_CLIENT_IPS=127.0.0.1,::1 \
  SQL_HOST=127.0.0.1 SQL_PORT=55434 \
  SQL_POOL_MIN=1 SQL_POOL_MAX=48 SQL_ROLE_CONNECTION_HEADROOM=8 \
  SQL_STATEMENT_TIMEOUT_MS=105000 \
  "$ENDPOINT_SUPERVISOR" > "$ENDPOINT_RUN_DIR/local-upstream/endpoint-supervisor.log" 2>&1 &
upstream_pid=$!
upstream_tick=$(capture_tick "$upstream_pid")
for _ in $(seq 1 480); do
  if curl -fsS -m 15 http://127.0.0.1:18018/readyz > "$ENDPOINT_RUN_DIR/local-upstream/readyz.json.tmp" 2>/dev/null; then
    mv "$ENDPOINT_RUN_DIR/local-upstream/readyz.json.tmp" "$ENDPOINT_RUN_DIR/local-upstream/readyz.json"
    break
  fi
  same_process "$upstream_pid" "$upstream_tick"
  sleep 1
done
[[ -s "$ENDPOINT_RUN_DIR/local-upstream/readyz.json" ]]

# Fail closed unless native loopback services are physically loopback-bound and
# every Milvus wildcard socket is covered by this run's pre-installed nft rule.
python3 - "$ENDPOINT_RUN_DIR/local-upstream/listener-scope-pre-environment.json" "$NFT_BIN" "$NFT_TABLE" "${NFT_PORTS[@]}" <<'PY_LISTENERS'
import json,re,subprocess,sys,time
from pathlib import Path
out=Path(sys.argv[1]); nft=sys.argv[2]; table=sys.argv[3]; guarded={int(x) for x in sys.argv[4:]}
core={39530:"milvus",18018:"search_visit",55434:"postgres"}
raw=subprocess.check_output(["ss","-H","-lntp"],universal_newlines=True)
rows=[]
for line in raw.splitlines():
    parts=line.split()
    if len(parts)<4: continue
    local=parts[3]
    m=re.match(r"^(?:\[([^]]+)\]|([^:]+)):(\d+)$",local)
    if not m: continue
    host=m.group(1) or m.group(2); port=int(m.group(3))
    if port in guarded or port in core:
        pm=re.search(r"pid=(\d+)",line)
        rows.append({"service":core.get(port,"milvus_internal"),"host":host,"port":port,"pid":int(pm.group(1)) if pm else None,"line":line})
by_port={p:[r for r in rows if r["port"]==p] for p in core}
missing=[p for p,v in by_port.items() if len(v)!=1]
physical_required=[r for r in rows if r["port"] in {18018,55434} and r["host"] not in {"127.0.0.1","::1"}]
rules=subprocess.check_output([nft,"list","table","inet",table],universal_newlines=True)
rule_lines=[x.strip() for x in rules.splitlines() if 'iifname != "lo"' in x and "tcp dport" in x and "reject" in x]
missing_rules=[p for p in guarded if not any(f"tcp dport {p} reject" in x for x in rule_lines)]
protected_wildcard=[r for r in rows if r["host"] not in {"127.0.0.1","::1"} and r["port"] in guarded]
unprotected_wildcard=[r for r in rows if r["host"] not in {"127.0.0.1","::1"} and r["port"] not in guarded]
status="PASS" if not missing and not physical_required and not missing_rules and not unprotected_wildcard else "FAIL"
payload={"schema":"amg_local_listener_and_nft_ingress_scope_v1","status":status,"checked_at_epoch":time.time(),"listeners":rows,"physical_loopback_required_failures":physical_required,"protected_wildcard":protected_wildcard,"unprotected_wildcard":unprotected_wildcard,"missing_or_ambiguous_core_ports":missing,"missing_nft_rules":missing_rules,"nft_table":table,"nft_rules":rule_lines,"security_semantics":"non-loopback ingress to every Milvus listener is rejected before service startup"}
tmp=out.with_name(out.name+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); tmp.replace(out)
if status!="PASS": raise SystemExit(json.dumps(payload,sort_keys=True))
PY_LISTENERS

# Exercise 64 distinct, schedule-derived PostgreSQL Visit rows before the
# trainer starts.  The previous same-URL C64 check warmed one TOAST row and did
# not expose pool starvation under 70-way fully-asynchronous rollout.
[[ -r "$DIVERSE_VISIT_VERIFIER" ]]
PREWARM_SCHEDULE=$ENDPOINT_RUN_DIR/local-upstream/heldout-prewarm-routing.jsonl
"$ENV_PYTHON" -B - "$HELDOUT_ROUTING" "$PREWARM_SCHEDULE" <<'PY_LR_PREWARM'
import json
import os
import pathlib
import sys

source, output = map(pathlib.Path, sys.argv[1:])
rows = []
with source.open(encoding="utf-8") as stream:
    for line in stream:
        if not line.strip():
            continue
        row = json.loads(line)
        data_idx = int(row["data_idx"])
        rows.append(
            {
                "data_idx": data_idx,
                "item_id": f"literesearcher:{data_idx}",
            }
        )
        if len(rows) == 16:
            break
assert len(rows) == 16
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    encoding="utf-8",
)
os.chmod(temporary, 0o600)
os.replace(temporary, output)
PY_LR_PREWARM
"$POSTGRES_PYTHON" "$DIVERSE_VISIT_VERIFIER" \
  --endpoint http://127.0.0.1:18018 \
  --schedule "$PREWARM_SCHEDULE" \
  --pool-rows "$CAMG_HELDOUT_ASSET_RUNTIME_ROWS_PATH" \
  --pgpass "$POSTGRES_RUN_DIR/pgpass" \
  --output "$ENDPOINT_RUN_DIR/local-upstream/diverse-bounded-cold-c64-visit-attestation.json" \
  --concurrency 64 --query-count 16 --search-limit 50 \
  --request-timeout-seconds 125 --maximum-latency-seconds 115
"$POSTGRES_PYTHON" "$POSTGRES_READONLY_VERIFIER" \
  --postgres-pid "$postgres_pid" \
  --upper-root "$POSTGRES_OVERLAY_UPPER" \
  --postgres-log "$POSTGRES_RUN_DIR/postgres.log" \
  --shim-source "$POSTGRES_READONLY_SHIM_SOURCE" \
  --shim-binary "$POSTGRES_READONLY_SHIM_SO" \
  --shim-load-log "$POSTGRES_READONLY_SHIM_LOAD_LOG" \
  --pgpass "$POSTGRES_RUN_DIR/pgpass" \
  --phase post-diverse-c64 \
  --output "$POSTGRES_RUN_DIR/readonly-toast-post-diverse-c64.json" \
  > "$POSTGRES_RUN_DIR/readonly-toast-post-diverse-c64.stdout.log"
persist_shim_load_log

# Launch the held-out LiteResearcher endpoint directly from the registry-pinned
# sources.  ``--split train`` is the upstream loader alias; the CAMG role and
# complete 5,319-row identity are bound by the verified held-out manifest.
mkdir -p -m 700 "$RUN_TMP/workspaces"
env \
  PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  PYTHONPATH="$OUTER_SOURCE_ROOT:$INNER_SOURCE_ROOT/agentenv:$INNER_SOURCE_ROOT/agentenv-agentmemory" \
  AGENTMEMORY_ENABLE_THINKING=0 AGENTMEMORY_ALLOW_REASONING=0 \
  AGENTMEMORY_RUN_ID="amg_${RUN_ID}_literesearcher" \
  AGENTMEMORY_SANDBOX_ROOTFS_PARENT="$SANDBOX_ROOTFS_PARENT" \
  LITERESEARCHER_CAMG_ROLE=heldout \
  "$ENV_PYTHON" -B -m agentenv_agentmemory.launch \
    --host 127.0.0.1 --port "$PORT" \
    --surface agentmemory_literesearcher_fullpool_upstream_hybrid_v1 \
    --run-id "amg_${RUN_ID}_literesearcher" --split train \
    --service-role formal \
    --runtime-source-id "${OUTER_SOURCE_COMMIT}_${INNER_SOURCE_COMMIT}" \
    --workspace-rg-binary "$WORKSPACE_RG_BINARY" \
    --workspace-rg-sha256 "$WORKSPACE_RG_SHA256" \
    --workspace-root-parent "$RUN_TMP/workspaces" \
    --literesearcher-full-pool-manifest "$CAMG_HELDOUT_ASSET_HELDOUT_MANIFEST_PATH" \
    --literesearcher-full-pool-rows "$CAMG_HELDOUT_ASSET_RUNTIME_ROWS_PATH" \
    --literesearcher-source-root "$LITERESEARCHER_SOURCE_ROOT" \
    --literesearcher-upstream-endpoint http://127.0.0.1:18018 \
    --literesearcher-filter-visitable \
    --literesearcher-backend-timeout-seconds 120 \
    --literesearcher-judge-api-base "$JUDGE_API_BASE" \
    --literesearcher-judge-model "$JUDGE_MODEL" \
    --literesearcher-judge-timeout-seconds 90 \
    --literesearcher-judge-max-retries 3 \
    --literesearcher-max-policy-steps 40 \
    --literesearcher-top-k 5 \
    --memory-first-add-reward 0 \
    --memory-first-later-retrieve-reward 0 \
    --memory-exact-repeat-reward 0 \
    --invalid-action-reward 0 \
    --memory-prompt-mode legacy \
  > "$ENDPOINT_RUN_DIR/environment-server.log" 2>&1 &
env_pid=$!
env_tick=$(capture_tick "$env_pid")
ready=0
for _ in $(seq 1 180); do
  if curl -fsS -m 30 "http://127.0.0.1:$PORT/metadata" \
      > "$ENDPOINT_RUN_DIR/environment-metadata.json.tmp" 2>/dev/null; then
    mv "$ENDPOINT_RUN_DIR/environment-metadata.json.tmp" \
      "$ENDPOINT_RUN_DIR/environment-metadata.json"
    ready=1
    break
  fi
  same_process "$env_pid" "$env_tick" || {
    tail -160 "$ENDPOINT_RUN_DIR/environment-server.log" >&2 || true
    exit 24
  }
  sleep 1
done
[[ "$ready" == 1 ]] || { echo "LiteResearcher endpoint did not become ready" >&2; exit 24; }
"$ENV_PYTHON" -B - "$ENDPOINT_RUN_DIR/environment-metadata.json" \
  "$CAMG_HELDOUT_TASK_COUNT" "$CAMG_HELDOUT_ASSET_HELDOUT_MANIFEST_SHA256" \
  "${OUTER_SOURCE_COMMIT}_${INNER_SOURCE_COMMIT}" \
  "$ENDPOINT_RUN_DIR/heldout-binding-attestation.json" <<'PY_LR_METADATA'
import json
import os
import pathlib
import sys

metadata_path, expected_count, manifest_sha, runtime_source_id, output = sys.argv[1:]
metadata = json.loads(pathlib.Path(metadata_path).read_text(encoding="utf-8"))
assert metadata.get("surface") == "agentmemory_literesearcher_fullpool_upstream_hybrid_v1"
assert metadata.get("domain_id") == "literesearcher"
assert metadata.get("split") == "train"
assert int(metadata.get("task_count", -1)) == int(expected_count)
assert int(metadata.get("train_count", -1)) == int(expected_count)
assert int(metadata.get("heldout_count", -1)) == 0
assert metadata.get("manifest_sha256") == manifest_sha
service = metadata.get("service", {})
assert service.get("role") == "formal"
assert service.get("runtime_source_id") == runtime_source_id
payload = {
    "schema": "camg_literesearcher_heldout_environment_binding_v1",
    "status": "pass",
    "camg_role": "heldout",
    "upstream_loader_split_alias": "train",
    "task_count": int(expected_count),
    "manifest_sha256": manifest_sha,
    "runtime_source_id": runtime_source_id,
    "environment_metadata": str(pathlib.Path(metadata_path).resolve()),
}
destination = pathlib.Path(output)
temporary = destination.with_name(destination.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.chmod(temporary, 0o600)
os.replace(temporary, destination)
PY_LR_METADATA

POSTGRES_SOCKET_DIR="$POSTGRES_SOCKET_DIR" POSTGRES_OVERLAY_RECEIPT="$POSTGRES_OVERLAY_RECEIPT" python3 - "$ENDPOINT_RUN_DIR/processes/local-stack.json" "$RUN_ID" \
  "$postgres_supervisor_pid" "$postgres_supervisor_tick" "$postgres_pid" "$postgres_tick" \
  "$stack_pid" "$stack_tick" "$upstream_pid" "$upstream_tick" \
  "$env_pid" "$env_tick" \
  "$POSTGRES_ATTESTATION" <<'PY'
import hashlib,json,os,sys,time
out,run_id,*raw=sys.argv[1:]
attestation=raw.pop()
labels=('postgres_supervisor','postgres','milvus_stack','search_visit','environment_server')
vals=list(zip(raw[0::2],raw[1::2]))
overlay=os.environ['POSTGRES_OVERLAY_RECEIPT']
p={'schema':'amg_literesearcher_local_stack_processes_v5','status':'PASS','run_id':run_id,'host':os.uname().nodename,'created_at':time.time(),'processes':{k:{'pid':int(v[0]),'start_tick':int(v[1])} for k,v in zip(labels,vals)},'postgres_attestation':{'path':attestation,'sha256':hashlib.sha256(open(attestation,'rb').read()).hexdigest()},'postgres_overlay_attestation':{'path':overlay,'sha256':hashlib.sha256(open(overlay,'rb').read()).hexdigest()},'endpoints':{'postgres':'127.0.0.1:55434','postgres_socket':os.environ['POSTGRES_SOCKET_DIR'],'milvus':'127.0.0.1:39530','search_visit':'127.0.0.1:18018','environment':'127.0.0.1:65122'}}
open(out,'w').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY

echo "LITERESEARCHER_LOCAL_STACK_READY run_id=$RUN_ID endpoint=http://127.0.0.1:$PORT server_pid=$env_pid"
while true; do
  same_process "$postgres_supervisor_pid" "$postgres_supervisor_tick" || { echo 'postgres supervisor exited' >&2; exit 21; }
  same_process "$postgres_pid" "$postgres_tick" || { echo 'postgres exited' >&2; exit 21; }
  same_process "$stack_pid" "$stack_tick" || { echo 'milvus stack exited' >&2; exit 22; }
  same_process "$upstream_pid" "$upstream_tick" || { echo 'Search/Visit supervisor exited' >&2; exit 23; }
  same_process "$env_pid" "$env_tick" || { echo 'LiteResearcher environment server exited' >&2; exit 24; }
  if find "$POSTGRES_OVERLAY_UPPER/base/13017" -maxdepth 1 -type f \
      \( -name '16392' -o -name '16392.*' -o -name '16394' -o -name '16394.*' \) \
      -print -quit 2>/dev/null | grep -q .; then
    echo 'immutable PostgreSQL TOAST relation copied into run upper' >&2
    exit 25
  fi
  if grep -Eqi 'bad file descriptor|could not write block|could not fsync|could not flush|PANIC:|invalid page in block' "$POSTGRES_RUN_DIR/postgres.log"; then
    echo 'PostgreSQL readonly-TOAST guard detected an I/O/write failure' >&2
    exit 25
  fi
  sleep 10
done
