#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$HERE/common.sh"

heldout_assert_base_contract swesmith 65124
heldout_assert_parent
heldout_assert_asset_env ADMISSION_CERTIFICATE "SWE-smith admission certificate"
heldout_assert_asset_env ADMITTED_POOL_MANIFEST "SWE-smith admitted-pool manifest"
heldout_assert_asset_env EXTENSION_POOL_MANIFEST "SWE-smith extension-pool manifest"
heldout_assert_asset_env FORMAL_EVAL_SELECTION "SWE-smith formal Eval selection"
heldout_assert_asset_env HELDOUT_MANIFEST "SWE-smith held-out manifest"
heldout_assert_asset_env IMAGE_BINDINGS "SWE-smith image bindings"
heldout_assert_asset_env IMAGE_MANIFEST "SWE-smith image manifest"
heldout_assert_asset_env MIRROR_BUNDLES_MANIFEST "SWE-smith mirror-bundles-manifest"
heldout_assert_asset_env OFFLINE_IMAGE_ASSETS "SWE-smith offline image assets"
heldout_assert_asset_env ROUTING "SWE-smith held-out routing"
heldout_assert_asset_env RUNTIME_MANIFEST "SWE-smith runtime manifest"
heldout_require_env SWESMITH_DETAIL_TOKEN

PT=/home/ai-jingyan-train/luolirui.1/post-train
WS=$PT/agentmemorygym-rl-workspace
SOURCE_OUTER=$CAMG_HELDOUT_SOURCE_OUTER_ROOT
SOURCE_ROOT=$CAMG_HELDOUT_SOURCE_INNER_ROOT
SOURCE_OUTER_COMMIT=$CAMG_HELDOUT_SOURCE_OUTER_COMMIT
SOURCE_COMMIT=$CAMG_HELDOUT_SOURCE_INNER_COMMIT
UPSTREAM_SOURCE_ROOT=$WS/runtime/swesmith/source/SWE-smith-9b74ac08
PREPARE_ROOTFS=$SOURCE_OUTER/AgentGym-RL/scripts/agentmemory/prepare_swesmith_oci_rootfs.py
CRANE=$WS/runtime/tools/crane/crane
RG_BINARY=$WS/runtime/tools/ripgrep/15.1.0-x86_64-unknown-linux-musl/rg
RG_SHA256=ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2
PYBIN=${HELDOUT_RUNTIME_PYTHON:-/opt/conda/envs/py312/bin/python3}
GRADER_SHIM=$WS/runtime/swesmith/grader-runtime/swebench-4.1.0-shim
GRADER_PYDEPS=$WS/runtime/swesmith/grader-pydeps/swebench-4.1.0
PORT=${AMG_MULTITASK_ENDPOINT_PORT:?missing endpoint port}
RUN_ID=${AMG_MULTITASK_RUN_ID:?missing run id}
RUN_DIR=${AMG_MULTITASK_ENDPOINT_RUN_DIR:?missing endpoint run dir}
SERVICE_ROOT=$RUN_DIR/service
RUN_KEY=$(printf '%s' "$RUN_ID" | sha256sum | awk '{print substr($1, 1, 16)}')
LOCAL_ROOT=/tmp/agentmemorygym-swesmith-$RUN_KEY
ROOTFS_RUN_ROOT=/tmp/agentmemorygym-swesmith-heldout-$RUN_KEY
EPISODES_ROOT=$LOCAL_ROOT/episodes
UID_LEASE_ROOT=$LOCAL_ROOT/uid-leases
MIRRORS_ROOT=$LOCAL_ROOT/mirrors
AUDIT_ROOT=$SERVICE_ROOT/audits
ROOTFS_CACHE_ROOT=$ROOTFS_RUN_ROOT/oci-rootfs
GENERATED_IMAGE_MANIFEST=$SERVICE_ROOT/image-manifest.generated.json
LAUNCH_RECEIPT=$SERVICE_ROOT/heldout-launch-contract.json

case "$LOCAL_ROOT" in
  /tmp/agentmemorygym-swesmith-[0-9a-f][0-9a-f]*) ;;
  *) heldout_die "unsafe SWE-smith local root: $LOCAL_ROOT" ;;
esac
case "$ROOTFS_RUN_ROOT" in
  /tmp/agentmemorygym-swesmith-heldout-[0-9a-f][0-9a-f]*) ;;
  *) heldout_die "unsafe SWE-smith rootfs root: $ROOTFS_RUN_ROOT" ;;
esac
for path in "$LOCAL_ROOT" "$ROOTFS_RUN_ROOT"; do
  [[ ! -e "$path" ]] || heldout_die "run-scoped path already exists: $path"
done
install -d -m 0700 \
  "$SERVICE_ROOT" "$AUDIT_ROOT" "$LOCAL_ROOT" "$EPISODES_ROOT" \
  "$UID_LEASE_ROOT" "$MIRRORS_ROOT" "$ROOTFS_CACHE_ROOT"

cleanup_local_roots() {
  local rc=$?
  set +e
  "$PYBIN" - "$LOCAL_ROOT" "$ROOTFS_RUN_ROOT" <<'PY_CLEANUP'
from pathlib import Path
import shutil
import sys

paths = [Path(value) for value in sys.argv[1:]]
expected = (
    (Path("/tmp"), "agentmemorygym-swesmith-"),
    (Path("/tmp"), "agentmemorygym-swesmith-heldout-"),
)
for path, (parent, prefix) in zip(paths, expected):
    if path.parent != parent or not path.name.startswith(prefix) or path.is_symlink():
        raise SystemExit(f"refusing unsafe SWE-smith cleanup path: {path}")
    if path.exists():
        shutil.rmtree(path)
PY_CLEANUP
  local cleanup_rc=$?
  if [[ $cleanup_rc -ne 0 ]]; then
    printf 'SWE-smith local cleanup failed\n' >&2
    [[ $rc -ne 0 ]] || rc=72
  fi
  exit "$rc"
}
trap cleanup_local_roots EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -x "$PYBIN" && -x "$CRANE" && -x "$RG_BINARY" ]]
[[ -r "$PREPARE_ROOTFS" ]] || heldout_die "missing SWE-smith OCI rootfs preparer: $PREPARE_ROOTFS"
[[ -d "$UPSTREAM_SOURCE_ROOT" && ! -L "$UPSTREAM_SOURCE_ROOT" ]]
[[ "$(sha256sum "$RG_BINARY" | awk '{print $1}')" == "$RG_SHA256" ]]

# Validate the formal Eval publication and its complete admission provenance
# before restoring any executable
# content. The same pass restores all frozen bare mirrors from the 36-repository
# mirror-bundles-manifest into this run's private /tmp root.
contract_values=$(
  "$PYBIN" -B - \
    "$CAMG_HELDOUT_ASSET_RUNTIME_MANIFEST_PATH" \
    "$CAMG_HELDOUT_ASSET_HELDOUT_MANIFEST_PATH" \
    "$CAMG_HELDOUT_ASSET_FORMAL_EVAL_SELECTION_PATH" \
    "$CAMG_HELDOUT_ASSET_ADMITTED_POOL_MANIFEST_PATH" \
    "$CAMG_HELDOUT_ASSET_EXTENSION_POOL_MANIFEST_PATH" \
    "$CAMG_HELDOUT_ASSET_IMAGE_BINDINGS_PATH" \
    "$CAMG_HELDOUT_ASSET_IMAGE_MANIFEST_PATH" \
    "$CAMG_HELDOUT_ASSET_MIRROR_BUNDLES_MANIFEST_PATH" \
    "$CAMG_HELDOUT_ASSET_ADMISSION_CERTIFICATE_PATH" \
    "$CAMG_HELDOUT_ASSET_ROUTING_PATH" \
    "$CAMG_HELDOUT_TASK_COUNT" "$MIRRORS_ROOT" "$LAUNCH_RECEIPT" <<'PY_CONTRACT'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

runtime_path = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
selection_path = Path(sys.argv[3])
admitted_pool_path = Path(sys.argv[4])
extension_pool_path = Path(sys.argv[5])
bindings_path = Path(sys.argv[6])
image_path = Path(sys.argv[7])
bundles_path = Path(sys.argv[8])
certificate_path = Path(sys.argv[9])
routing_path = Path(sys.argv[10])
expected_count = int(sys.argv[11])
mirrors_root = Path(sys.argv[12])
receipt_path = Path(sys.argv[13])

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def bound(base, record, label):
    path = Path(str(record["path"]))
    path = path if path.is_absolute() else (base / path).resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"{label} byte count drifted")
    if digest(path) != record["sha256"]:
        raise RuntimeError(f"{label} digest drifted")
    return path

runtime = load(runtime_path)
if runtime.get("schema") != "camg_swesmith_formal_eval_runtime_manifest_v5":
    raise RuntimeError("SWE-smith held-out runtime schema drifted")
if runtime.get("status") != "ready" or runtime.get("heldout_evaluation_run") is not False:
    raise RuntimeError("SWE-smith held-out runtime is not pre-evaluation ready")
if int(runtime.get("task_count", -1)) != expected_count:
    raise RuntimeError("SWE-smith held-out task count drifted")
runtime_base = runtime_path.parent
expected_files = {
    "manifest": dataset_path,
    "formal_eval_selection": selection_path,
    "admitted_pool_manifest": admitted_pool_path,
    "extension_pool_manifest": extension_pool_path,
    "image_bindings": bindings_path,
    "image_manifest": image_path,
    "routing": routing_path,
}
for name, expected_path in expected_files.items():
    if bound(runtime_base, runtime["files"][name], name) != expected_path.resolve():
        raise RuntimeError(f"runtime manifest binds a different {name}")

dataset = load(dataset_path)
selection = dataset.get("selection", {})
if (
    dataset.get("schema_version") != "swesmith_jsonl_manifest_v1"
    or dataset.get("role") != "formal_heldout"
    or selection.get("mode") != "instance_ids"
    or int(selection.get("count", -1)) != expected_count
):
    raise RuntimeError("SWE-smith held-out dataset contract drifted")
bound(dataset_path.parent, selection, "heldout instance ids")
dataset_revision = str(dataset["upstream"]["dataset_revision"])
source_revision = str(dataset["upstream"]["source_revision"])

formal_selection = load(selection_path)
admitted_pool = load(admitted_pool_path)
extension_pool = load(extension_pool_path)
complete_count = int(runtime.get("complete_admitted_pool_task_count", -1))
extension_count = int(runtime.get("extension_pool_task_count", -1))
if (
    runtime.get("selection")
    != "deterministic complete-repository subset of the exact-runtime-admitted held-out candidate pool"
    or runtime.get("active_training_inputs_modified") is not False
    or expected_count + extension_count != complete_count
    or formal_selection.get("schema") != "camg_swesmith_formal_eval_selection_v5"
    or formal_selection.get("status") != "frozen"
    or int(formal_selection.get("formal_eval_task_count", -1)) != expected_count
    or int(formal_selection.get("complete_admitted_heldout_pool_task_count", -1)) != complete_count
    or int(formal_selection.get("extension_pool_task_count", -1)) != extension_count
    or formal_selection.get("selection_depends_on_model_output_or_reward") is not False
    or formal_selection.get("active_training_inputs_modified") is not False
    or formal_selection.get("heldout_evaluation_run") is not False
    or admitted_pool.get("schema") != "camg_swesmith_admitted_heldout_pool_manifest_v5"
    or admitted_pool.get("status") != "complete"
    or int(admitted_pool.get("task_count", -1)) != complete_count
    or admitted_pool.get("formal_evaluation_role") is not False
    or admitted_pool.get("training_role") is not False
    or extension_pool.get("schema") != "camg_swesmith_extension_pool_manifest_v5"
    or extension_pool.get("status") != "frozen"
    or int(extension_pool.get("task_count", -1)) != extension_count
    or extension_pool.get("formal_evaluation_role") is not False
    or extension_pool.get("training_role") is not False
):
    raise RuntimeError("SWE-smith formal Eval selection/pool contract drifted")
selected_repositories = formal_selection.get("selected_repositories")
selected_counts = formal_selection.get("selected_repository_task_counts")
dataset_repositories = selection.get("repositories")
if (
    not isinstance(selected_repositories, list)
    or not selected_repositories
    or len(set(selected_repositories)) != len(selected_repositories)
    or selected_repositories != sorted(selected_repositories)
    or dataset_repositories != selected_repositories
    or int(formal_selection.get("formal_eval_repository_count", -1))
    != len(selected_repositories)
    or not isinstance(selected_counts, dict)
    or set(selected_counts) != set(selected_repositories)
    or sum(int(value) for value in selected_counts.values()) != expected_count
):
    raise RuntimeError("SWE-smith selected-repository contract drifted")

bindings = load(bindings_path)
records = bindings.get("records")
if bindings.get("schema") != "camg_swe_heldout_image_bindings_v1" or not isinstance(records, list):
    raise RuntimeError("SWE-smith image binding schema drifted")
if not records or any(item.get("status") != "pass" for item in records):
    raise RuntimeError("SWE-smith image bindings are incomplete")
binding_repositories = [str(item.get("base_repository", "")) for item in records]
if (
    any(not repository for repository in binding_repositories)
    or len(set(binding_repositories)) != len(binding_repositories)
    or not set(selected_repositories).issubset(binding_repositories)
):
    raise RuntimeError("SWE-smith image bindings do not cover the formal repositories")
materialized_records = [
    item for item in records if item["base_repository"] in selected_repositories
]
if len(materialized_records) != len(selected_repositories):
    raise RuntimeError("SWE-smith formal repositories do not map one-to-one to images")
images = load(image_path)
expected_images = sorted(
    (str(item["profile_image"]), str(item["digest"])) for item in records
)
actual_images = sorted(
    (str(item["image"]), str(item["digest"])) for item in images.get("images", [])
)
if images.get("schema_version") != "swesmith_oci_image_manifest_v1" or actual_images != expected_images:
    raise RuntimeError("SWE-smith image manifest differs from frozen bindings")

certificate = load(certificate_path)
if (
    certificate.get("schema") != "camg_swesmith_complete_certificate_index_v2"
    or certificate.get("status") != "pass"
    or certificate.get("heldout_evaluation_run") is not False
):
    raise RuntimeError("SWE-smith admission certificate drifted")

routing_count = 0
routing_repository_counts = {}
with routing_path.open(encoding="utf-8") as handle:
    for routing_count, line in enumerate(handle, start=1):
        row = json.loads(line)
        if row.get("data_idx") != routing_count - 1:
            raise RuntimeError("SWE-smith held-out routing is not contiguous")
        repository = str(row.get("extra_info", {}).get("base_repository", ""))
        if repository not in selected_repositories:
            raise RuntimeError(
                f"SWE-smith held-out routing escaped the formal repositories: {repository}"
            )
        routing_repository_counts[repository] = (
            routing_repository_counts.get(repository, 0) + 1
        )
if routing_count != expected_count:
    raise RuntimeError("SWE-smith held-out routing count drifted")
if routing_repository_counts != {
    key: int(value) for key, value in selected_counts.items()
}:
    raise RuntimeError("SWE-smith held-out repository counts drifted")

bundles = load(bundles_path)
bundle_records = bundles.get("records")
if (
    bundles.get("schema") != "camg_swesmith_mirror_bundles_v2"
    or bundles.get("status") != "pass"
    or bundles.get("heldout_evaluation_run") is not False
    or int(bundles.get("repository_count", -1)) != 36
    or not isinstance(bundle_records, list)
    or len(bundle_records) != 36
):
    raise RuntimeError("SWE-smith mirror bundle publication drifted")
bundle_base = bundles_path.parent
restored = []
for item in bundle_records:
    bundle = bound(bundle_base, item["bundle"], "repository bundle")
    bound(bundle_base, item["receipt"], "repository bundle receipt")
    exact_repository = str(item["exact_repository"])
    mirror_name = Path(exact_repository).name
    if not mirror_name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in mirror_name
    ):
        raise RuntimeError(f"unsafe mirror name: {mirror_name!r}")
    destination = mirrors_root / mirror_name
    if destination.exists():
        raise RuntimeError(f"duplicate mirror destination: {destination}")
    subprocess.run(
        ["git", "clone", "--mirror", "--quiet", str(bundle), str(destination)],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "git", "-c", f"safe.directory={destination}", "-C", str(destination),
            "fsck", "--no-dangling",
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    restored.append({"exact_repository": exact_repository, "mirror": str(destination)})

receipt = {
    "schema": "camg_swesmith_heldout_launch_contract_v1",
    "status": "pass",
    "heldout_evaluation_run": False,
    "task_count": expected_count,
    "complete_admitted_pool_task_count": complete_count,
    "extension_pool_task_count": extension_count,
    "dataset_revision": dataset_revision,
    "source_revision": source_revision,
    "frozen_image_count": len(records),
    "materialized_image_count": len(materialized_records),
    "materialized_repositories": selected_repositories,
    "materialized_profile_images": sorted(
        item["profile_image"] for item in materialized_records
    ),
    "mirror_count": len(restored),
    "restored_mirrors": restored,
    "created_at_unix_ns": time.time_ns(),
}
temporary = receipt_path.with_name(receipt_path.name + f".tmp-{os.getpid()}")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, receipt_path)
print(dataset_revision, source_revision, sep="\t")
PY_CONTRACT
)
IFS=$'\t' read -r DATASET_REVISION SOURCE_REVISION <<< "$contract_values"

image_binding_args=()
while IFS= read -r -d '' image_binding_arg; do
  image_binding_args+=("$image_binding_arg")
done < <(
  "$PYBIN" -B - \
    "$CAMG_HELDOUT_ASSET_IMAGE_BINDINGS_PATH" \
    "$CAMG_HELDOUT_ASSET_FORMAL_EVAL_SELECTION_PATH" <<'PY_BINDINGS'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
selection = json.load(open(sys.argv[2], encoding="utf-8"))
selected = set(selection["selected_repositories"])
for item in payload["records"]:
    value = f'{item["source_image"]}={item["profile_image"]}@{item["digest"]}'
    sys.stdout.buffer.write(b"--binding\0" + value.encode("utf-8") + b"\0")
    if item["base_repository"] in selected:
        profile = item["profile_image"].encode("utf-8")
        sys.stdout.buffer.write(b"--materialize-profile-image\0" + profile + b"\0")
PY_BINDINGS
)
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  "$PYBIN" -B "$PREPARE_ROOTFS" \
  "${image_binding_args[@]}" \
  --cache-root "$ROOTFS_CACHE_ROOT" \
  --crane "$CRANE" \
  --offline-image-asset-manifest "$CAMG_HELDOUT_ASSET_OFFLINE_IMAGE_ASSETS_PATH" \
  --dataset-revision "$DATASET_REVISION" \
  --source-revision "$SOURCE_REVISION" \
  --image-manifest-output "$GENERATED_IMAGE_MANIFEST" \
  --max-workers 4 \
  > "$SERVICE_ROOT/rootfs-prepare.log"
[[ "$(heldout_sha256 "$GENERATED_IMAGE_MANIFEST")" == "$CAMG_HELDOUT_ASSET_IMAGE_MANIFEST_SHA256" ]] \
  || heldout_die "generated SWE-smith image manifest differs from frozen manifest"
complete_count=$(find "$ROOTFS_CACHE_ROOT" -mindepth 2 -maxdepth 2 -type f -name .complete | wc -l | tr -d '[:space:]')
materialized_image_count=$("$PYBIN" -B - "$CAMG_HELDOUT_ASSET_IMAGE_BINDINGS_PATH" "$CAMG_HELDOUT_ASSET_FORMAL_EVAL_SELECTION_PATH" <<'PY_COUNT'
import json
import sys
bindings = json.load(open(sys.argv[1], encoding="utf-8"))["records"]
selected = set(json.load(open(sys.argv[2], encoding="utf-8"))["selected_repositories"])
print(sum(item["base_repository"] in selected for item in bindings))
PY_COUNT
)
[[ "$complete_count" == "$materialized_image_count" ]] \
  || heldout_die "SWE-smith OCI rootfs count drifted: $complete_count != $materialized_image_count"

export PYTHONPATH="$SOURCE_OUTER:$SOURCE_ROOT/agentenv-swesmith:$SOURCE_ROOT/agentenv-agentmemory:$SOURCE_ROOT/agentenv:$GRADER_SHIM:$GRADER_PYDEPS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export SWESMITH_HOST=127.0.0.1 SWESMITH_PORT="$PORT" SWESMITH_LOG_LEVEL=info
export SWESMITH_SERVICE_ID="amg_${RUN_ID}_swesmith"
export SWESMITH_DATASET_MANIFEST="$CAMG_HELDOUT_ASSET_HELDOUT_MANIFEST_PATH"
export SWESMITH_IMAGE_MANIFEST="$CAMG_HELDOUT_ASSET_IMAGE_MANIFEST_PATH"
export SWESMITH_MIRRORS_ROOT="$MIRRORS_ROOT"
export SWESMITH_SOURCE_ROOT="$UPSTREAM_SOURCE_ROOT"
export SWESMITH_SOURCE_REVISION="$SOURCE_REVISION"
export SWESMITH_OCI_CACHE_ROOT="$ROOTFS_CACHE_ROOT"
export SWESMITH_EPISODES_ROOT="$EPISODES_ROOT"
export SWESMITH_AUDIT_ROOT="$AUDIT_ROOT"
export SWESMITH_UID_LEASE_ROOT="$UID_LEASE_ROOT" SWESMITH_UID_LEASE_SLOTS=96
export SWESMITH_RG_BINARY="$RG_BINARY" SWESMITH_RG_SHA256="$RG_SHA256"
export SWESMITH_MAX_STEPS=30
export SWESMITH_MAX_OBSERVATION_TOKENS=8192
export SWESMITH_MAX_OBSERVATION_BYTES=6144
export SWESMITH_STDOUT_BYTES=8192 SWESMITH_STDERR_BYTES=3072
export SWESMITH_DEFAULT_TIMEOUT_MS=120000 SWESMITH_MAX_TIMEOUT_MS=120000
export SWESMITH_GRADER_TIMEOUT_MS=600000
export SWESMITH_RUNTIME_OUTER_COMMIT="$SOURCE_OUTER_COMMIT"
export SWESMITH_RUNTIME_INNER_COMMIT="$SOURCE_COMMIT"

env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$UPSTREAM_SOURCE_ROOT" \
  "$PYBIN" -m uvicorn agentenv_swesmith.server:app \
  --host 127.0.0.1 --port "$PORT" --log-level info
