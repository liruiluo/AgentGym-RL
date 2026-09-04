#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$HERE/common.sh"

heldout_assert_base_contract webshop 65121
heldout_assert_parent
heldout_assert_asset_env HELDOUT_EPISODES "WebShop held-out episodes"
heldout_assert_asset_env PRODUCT_POOL "WebShop product pool"
heldout_assert_asset_env ROUTING "WebShop routing"
heldout_assert_asset_env RUNTIME_MANIFEST "WebShop runtime manifest"

PT=/home/ai-jingyan-train/luolirui.1/post-train
WS=$PT/agentmemorygym-rl-workspace
WEB_PYTHON=${CAMG_WEBSHOP_PYTHON:-$WS/runtime/venvs/webshop-py310/bin/python}
WEB_OVERLAY=${CAMG_WEBSHOP_PYTHON_OVERLAY:-$WS/runtime/python-overlays/webshop-spacy-cp310}
MEMORYARENA_ROOT=${CAMG_WEBSHOP_MEMORYARENA_ROOT:-$WS/runtime/source-snapshots/memoryarena-6cd9de14}
DB=${CAMG_WEBSHOP_DB_ROOT:-$PT/data/memoryarena-product-db}
JAVA_HOME_PERSIST=${CAMG_WEBSHOP_JAVA_HOME:-$WS/runtime/jre11-conda}
LUCENE_MANIFEST=${CAMG_WEBSHOP_LUCENE_MANIFEST:-$WS/runtime/validation/procedural-memory-data-20260730/evidence/input-provenance/lucene-indexes-full.sha256}
RG=${CAMG_WEBSHOP_RG_BINARY:-$WS/runtime/tools/ripgrep/15.1.0-x86_64-unknown-linux-musl/rg}
RG_SHA256=${CAMG_WEBSHOP_RG_SHA256:-ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2}

heldout_assert_executable "$WEB_PYTHON" "WebShop Python"
heldout_assert_executable "$RG" "WebShop ripgrep"
[[ "$(heldout_sha256 "$RG")" == "$RG_SHA256" ]] \
  || heldout_die "WebShop ripgrep SHA-256 mismatch"
[[ -d "$WEB_OVERLAY" && ! -L "$WEB_OVERLAY" ]] \
  || heldout_die "WebShop Python overlay is missing"
[[ -d "$MEMORYARENA_ROOT" && ! -L "$MEMORYARENA_ROOT" ]] \
  || heldout_die "MemoryArena source root is missing"
[[ -d "$DB/search_engine" && -f "$DB/items_shuffle.json" && -f "$DB/items_ins_v2.json" ]] \
  || heldout_die "WebShop native catalog assets are incomplete"
[[ -x "$JAVA_HOME_PERSIST/bin/java" ]] || heldout_die "WebShop Java runtime is missing"
[[ -f "$LUCENE_MANIFEST" && ! -L "$LUCENE_MANIFEST" ]] \
  || heldout_die "WebShop Lucene manifest is missing"

RUNTIME_MANIFEST=$CAMG_HELDOUT_ASSET_RUNTIME_MANIFEST_PATH
ROUTING=$CAMG_HELDOUT_ASSET_ROUTING_PATH
HELDOUT_EPISODES=$CAMG_HELDOUT_ASSET_HELDOUT_EPISODES_PATH
PRODUCT_POOL=$CAMG_HELDOUT_ASSET_PRODUCT_POOL_PATH
RUN_DIR=$AMG_MULTITASK_ENDPOINT_RUN_DIR
mkdir -p "$RUN_DIR"

# The formal schedule contains 1,746 sparse global data_idx values.  The
# endpoint exposes the complete deterministic provider window; the evaluator
# schedule selects only the frozen held-out rows.
mapfile -t provider_contract < <("$(heldout_python)" - \
  "$RUNTIME_MANIFEST" "$ROUTING" "$HELDOUT_EPISODES" "$PRODUCT_POOL" \
  "$CAMG_HELDOUT_TASK_COUNT" "$RUN_DIR/heldout-launch-contract.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
routing_path = pathlib.Path(sys.argv[2])
episodes_path = pathlib.Path(sys.argv[3])
pool_path = pathlib.Path(sys.argv[4])
expected_count = int(sys.argv[5])
output = pathlib.Path(sys.argv[6])

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = load_json(manifest_path)
if manifest.get("schema") != "camg_shop_complete_heldout_runtime_manifest_v2":
    raise SystemExit("unexpected Shop held-out runtime manifest schema")
if manifest.get("status") != "ready" or manifest.get("heldout_evaluation_run") is not False:
    raise SystemExit("Shop held-out runtime manifest is not pre-eval ready")
task_count = int(manifest.get("task_count", 0))
if task_count != expected_count:
    raise SystemExit(f"Shop held-out count mismatch: {task_count} != {expected_count}")
provider = manifest.get("provider") or {}
if provider.get("mode") != "fixed_window_sparse_routing":
    raise SystemExit("Shop provider must use fixed_window_sparse_routing")
provider_task_count = int(provider.get("provider_task_count", 0))
generator_seed = int(provider.get("generator_seed", -1))
start_orbit = int(provider.get("global_orbit_index_start_inclusive", -1))
end_orbit = int(provider.get("global_orbit_index_end_exclusive", -1))
provider_split = str(provider.get("split", ""))
provider_capacity = (end_orbit - start_orbit) * 2
if (
    provider_task_count <= 0
    or end_orbit <= start_orbit
    or provider_task_count > provider_capacity
):
    raise SystemExit("Shop provider task count/window mismatch")
if generator_seed != 233 or start_orbit != 0 or provider_split != "dev":
    raise SystemExit("Shop held-out generator contract drifted")
pool_ref = manifest.get("product_pool") or {}
if pool_ref.get("file_sha256") != sha(pool_path):
    raise SystemExit("Shop product-pool binding mismatch")
routing = load_jsonl(routing_path)
episodes = load_jsonl(episodes_path)
if len(routing) != task_count or len(episodes) != task_count:
    raise SystemExit("Shop routing/episode cardinality mismatch")
routing_idx = [row.get("data_idx") for row in routing]
episode_idx = [row.get("data_idx") for row in episodes]
if routing_idx != episode_idx or len(set(routing_idx)) != task_count:
    raise SystemExit("Shop sparse routing identity mismatch")
if not routing_idx or min(routing_idx) < 0 or max(routing_idx) >= provider_task_count:
    raise SystemExit("Shop sparse data_idx escapes provider window")
receipt = {
    "schema": "camg_shop_heldout_endpoint_launch_contract_v1",
    "status": "pass",
    "heldout_task_count": task_count,
    "provider_task_count": provider_task_count,
    "provider_mode": "fixed_window",
    "provider_split": provider_split,
    "generator_seed": generator_seed,
    "start_orbit": start_orbit,
    "max_sparse_data_idx": max(routing_idx),
    "runtime_manifest_sha256": sha(manifest_path),
    "routing_sha256": sha(routing_path),
    "heldout_episodes_sha256": sha(episodes_path),
    "product_pool_sha256": sha(pool_path),
}
temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(provider_task_count)
print(generator_seed)
print(start_orbit)
print(provider_split)
print("fixed_window")
PY
)
[[ ${#provider_contract[@]} -eq 5 ]] \
  || heldout_die "Shop provider contract output is invalid"
provider_task_count=${provider_contract[0]}
generator_seed=${provider_contract[1]}
start_orbit=${provider_contract[2]}
provider_split=${provider_contract[3]}
provider_mode=${provider_contract[4]}

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export AGENTMEMORY_ENABLE_THINKING=0 AGENTMEMORY_ALLOW_REASONING=0
export AGENTMEMORY_WEBSHOP_POSITIVE_TASK_REWARD_SCALE=0.14285714285714285
export PYTHONPATH="$WEB_OVERLAY:$CAMG_HELDOUT_SOURCE_INNER_ROOT/agentenv-agentmemory:$CAMG_HELDOUT_SOURCE_INNER_ROOT/agentenv:$CAMG_HELDOUT_SOURCE_OUTER_ROOT"

exec "$WEB_PYTHON" -m agentenv_agentmemory.launch \
  --host 127.0.0.1 --port "$AMG_MULTITASK_ENDPOINT_PORT" \
  --surface agentmemory_webshop_procedural_natural_chain_filesystem_v2 \
  --memoryarena-root "$MEMORYARENA_ROOT" \
  --items-file "$DB/items_shuffle.json" \
  --attributes-file "$DB/items_ins_v2.json" \
  --search-root "$DB/search_engine" \
  --java-home "$JAVA_HOME_PERSIST" \
  --lucene-index-manifest "$LUCENE_MANIFEST" \
  --procedural-product-pool "$PRODUCT_POOL" \
  --procedural-product-pool-sha256 "$CAMG_HELDOUT_ASSET_PRODUCT_POOL_SHA256" \
  --procedural-task-count "$provider_task_count" \
  --procedural-generator-seed "$generator_seed" \
  --procedural-provider-mode "$provider_mode" \
  --procedural-start-orbit "$start_orbit" \
  --workspace-rg-binary "$RG" --workspace-rg-sha256 "$RG_SHA256" \
  --memoryarena-base-commit 6cd9de14b71915e39ac742a20dc33785e14b6aab \
  --run-id "amg_${AMG_MULTITASK_RUN_ID}_webshop" \
  --split "$provider_split" --price-seed 233 \
  --service-role formal \
  --runtime-source-id "${CAMG_HELDOUT_SOURCE_OUTER_COMMIT}_${CAMG_HELDOUT_SOURCE_INNER_COMMIT}" \
  --memory-first-add-reward 0 --memory-first-later-retrieve-reward 0 \
  --memory-exact-repeat-reward 0 --ltm-inventory-mode hidden \
  --ltm-transition-notice-mode none --memory-prompt-mode natural_filesystem \
  --action-listing-mode separate
