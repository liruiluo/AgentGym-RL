#!/usr/bin/env bash
set -euo pipefail

PT=/home/ai-jingyan-train/luolirui.1/post-train
WS=$PT/agentmemorygym-rl-workspace
SOURCE_WT=$WS/worktrees/AgentGym-RL-dataset-randomization-20260728
OUTER_COMMIT=d82e864c5cd359503effad825b321737afb00e37
INNER_COMMIT=cd51926f2f5ae0969d592af4b730169409d6a6f5
MODE=keys
NOTICE_MODE=state
PROMPT_MODE=neutral_horizon_responsibility
ACTION_MODE=unified
PRESENTATION_RANDOMIZATION=${AGENTMEMORY_PRESENTATION_RANDOMIZATION_MODE:-candidate_order_unique_v2}
PRESENTATION_SEED=${AGENTMEMORY_PRESENTATION_SEED:-20260728}
PROMPT_LENGTH=${AGENTMEMORY_MAX_PROMPT_LENGTH:-124928}
RESPONSE_LENGTH=${AGENTMEMORY_MAX_RESPONSE_LENGTH:-2048}
MODEL_LENGTH=${AGENTMEMORY_MAX_MODEL_LENGTH:-131072}
CONTEXT_SLACK=${AGENTMEMORY_CONTEXT_SLACK:-4096}

RUN_ID=${AGENTMEMORY_EVAL_RUN_ID:-amg_dataset_randomization_${PRESENTATION_RANDOMIZATION}_q35_1c_r30_n8_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR=$PT/agentmemorygym-evals/$RUN_ID
OUT=$RUN_DIR/results/$MODE
PANEL_DIR=$WS/manifests/heldout_panel_15_seed59_20260720
PANEL_DATA=$PANEL_DIR/data
PANEL_IDS=$PANEL_DIR/panel_task_ids.txt
PANEL_MANIFEST=$PANEL_DIR/panel_manifest.json
GATE=$RUN_DIR/annotation_gate.json
MODEL=$PT/models/Qwen3.5-4B
Q35_T=$WS/runtime/worktrees/AgentGym-RL-qwen35-upstream-verl-20260725
Q35_Q=/opt/conda/envs/amg-qwen35-packed-20260725
Q35_BRIDGE=/opt/conda/envs/amg-qwen35-vllm024-bridge-20260725
ENV_PORT=${AGENTMEMORY_EVAL_ENV_PORT:?set AGENTMEMORY_EVAL_ENV_PORT}
VENV=$WS/runtime/venvs/webshop-py310
ENV_PYTHON=${AGENTMEMORY_ENV_PYTHON:-$WS/runtime/venvs/webshop-py310/bin/python}
PYTHON_OVERLAY=$WS/runtime/python-overlays/webshop-spacy-cp310
DB=$PT/data/memoryarena-product-db
AUD=$WS/audits/upstream_whole_chain_20260718
RAW=$WS/audits/annotation_semantic_audit_20260718/inputs/raw_memoryarena_bundled_shopping_data.jsonl
MEMORYARENA=$WS/runtime/source-snapshots/memoryarena-6cd9de14
JAVA_HOME_PERSIST=$WS/runtime/jre11-conda
LUCENE_MANIFEST=$AUD/original_lucene_index_files.sha256
FROZEN_RUNNER=$WS/experiments/launcher_tools/run_frozen_launcher.sh
CLEANUP_TOOL=$WS/experiments/launcher_tools/cleanup_owned_ray_vllm.py
TOKEN_ANALYZER=$SOURCE_WT/experiments/analysis/analyze_rollout_prompt_tokens.py
BEHAVIOR_ANALYZER=$SOURCE_WT/experiments/analysis/analyze_webshop_rollouts.py
RUN_ANALYSIS_TOOLS=$RUN_DIR/analysis_tools

mkdir -p "$RUN_DIR"
if [ "${AGENTMEMORY_FROZEN_LAUNCHER:-0}" != "1" ]; then
  exec /usr/bin/env bash "$FROZEN_RUNNER" "$RUN_DIR" "${BASH_SOURCE[0]}" "$@"
fi
case "${BASH_SOURCE[0]}" in
  "$RUN_DIR"/launcher_snapshots/*/launcher.sh) ;;
  *) echo "FATAL frozen launcher flag set outside run-local snapshot" >&2; exit 70 ;;
esac

case "$ENV_PORT" in
  ''|*[!0-9]*) echo "FATAL invalid ENV_PORT=$ENV_PORT" >&2; exit 79 ;;
esac
if [ "$ENV_PORT" -lt 1 ] || [ "$ENV_PORT" -gt 65535 ]; then
  echo "FATAL invalid ENV_PORT=$ENV_PORT" >&2
  exit 79
fi
for context_var in PROMPT_LENGTH RESPONSE_LENGTH MODEL_LENGTH CONTEXT_SLACK; do
  context_value=${!context_var}
  case "$context_value" in
    ''|*[!0-9]*) echo "FATAL invalid $context_var=$context_value" >&2; exit 79 ;;
  esac
done
if [ "$((PROMPT_LENGTH + RESPONSE_LENGTH + CONTEXT_SLACK))" -gt "$MODEL_LENGTH" ]; then
  echo "FATAL prompt/response widths do not leave context slack" >&2
  exit 79
fi

exec 9>"/tmp/amg_eval_${ENV_PORT}.lock"
flock -n 9 || { echo "FATAL eval lock already held for port $ENV_PORT" >&2; exit 71; }

mkdir -p "$OUT/executer_logs"
ORCH_LOG=$RUN_DIR/orchestrator.log
exec > >(tee -a "$ORCH_LOG") 2>&1
echo "[eval] $(date -u +%FT%TZ) start run_id=$RUN_ID mode=$MODE notice=$NOTICE_MODE prompt=$PROMPT_MODE action_listing=$ACTION_MODE presentation_randomization=$PRESENTATION_RANDOMIZATION presentation_seed=$PRESENTATION_SEED"

ENV_PID=""
WATCH_PID=""
FORMAL_MARKER=/tmp/agentmemory-formal-cpu-active
GPU_YIELD_MARKER=/tmp/crg-holder-yield
EVIDENCE_DIR=$RUN_DIR/runtime_cleanup

stop_env() {
  if [ -n "$ENV_PID" ]; then
    kill "$ENV_PID" 2>/dev/null || true
    wait "$ENV_PID" 2>/dev/null || true
    ENV_PID=""
  fi
}

cleanup_exact_runtime() {
  /opt/conda/envs/py312/bin/python3 "$CLEANUP_TOOL" \
    --mode cleanup --owner root --run-id "$RUN_ID" \
    --evidence-dir "$EVIDENCE_DIR" || true
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  stop_env
  cleanup_exact_runtime
  [ -z "$WATCH_PID" ] || kill "$WATCH_PID" 2>/dev/null || true
  rm -f "$FORMAL_MARKER" "$GPU_YIELD_MARKER"
  for _ in $(seq 1 30); do
    grep -q 'mode=hold' /tmp/crg-holder.state 2>/dev/null && break
    sleep 2
  done
  cat /tmp/crg-holder.state > "$RUN_DIR/holder_state_after.txt" 2>/dev/null || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader > "$RUN_DIR/gpu_after.txt" 2>/dev/null || true
  printf '%s\n' "$rc" > "$RUN_DIR/eval_exit_code"
  date -u +%FT%TZ > "$RUN_DIR/finished_at"
  echo "[eval] $(date -u +%FT%TZ) cleanup rc=$rc"
  exit "$rc"
}
trap cleanup EXIT INT TERM

for path in \
  "$PANEL_MANIFEST" "$PANEL_IDS" "$PANEL_DATA/agentmemory_test.json" \
  "$MODEL/config.json" "$CLEANUP_TOOL" "$FROZEN_RUNNER" \
  "$TOKEN_ANALYZER" "$BEHAVIOR_ANALYZER"; do
  [ -s "$path" ] || { echo "FATAL missing input: $path" >&2; exit 72; }
done
NATIVE_MODEL_LENGTH=$(jq -r '.text_config.max_position_embeddings // .max_position_embeddings // empty' "$MODEL/config.json")
case "$NATIVE_MODEL_LENGTH" in
  ''|*[!0-9]*) echo "FATAL missing model native context length" >&2; exit 72 ;;
esac
if [ "$MODEL_LENGTH" -gt "$NATIVE_MODEL_LENGTH" ]; then
  echo "FATAL requested model context $MODEL_LENGTH exceeds native context $NATIVE_MODEL_LENGTH" >&2
  exit 79
fi
[ -d "$PYTHON_OVERLAY/spacy" ] && [ -d "$PYTHON_OVERLAY/en_core_web_lg" ] || {
  echo "FATAL missing Python dependency overlay: $PYTHON_OVERLAY" >&2
  exit 72
}

test -x "$Q35_Q/bin/python"
test -d "$Q35_BRIDGE"
test -d "$Q35_T/AgentGym-RL"
test "$(git -C "$SOURCE_WT" rev-parse --verify "$OUTER_COMMIT^{commit}")" = "$OUTER_COMMIT" || {
  echo "FATAL outer source base commit missing" >&2; exit 73;
}
test -z "$(git -C "$SOURCE_WT" diff --name-only "$OUTER_COMMIT" -- AgentGym AgentGym-RL)" || {
  echo "FATAL outer source code drift from pinned base" >&2; exit 73;
}
test "$(git -C "$SOURCE_WT/AgentGym" rev-parse HEAD)" = "$INNER_COMMIT" || {
  echo "FATAL inner source commit drift" >&2; exit 73;
}
test -z "$(git -C "$SOURCE_WT" status --porcelain)" || {
  echo "FATAL outer source worktree is dirty" >&2; exit 73;
}
test -z "$(git -C "$SOURCE_WT/AgentGym" status --porcelain)" || {
  echo "FATAL inner source worktree is dirty" >&2; exit 73;
}

mkdir -p "$RUN_ANALYSIS_TOOLS"
install -m 0755 "$TOKEN_ANALYZER" "$RUN_ANALYSIS_TOOLS/analyze_rollout_prompt_tokens.py"
install -m 0755 "$BEHAVIOR_ANALYZER" "$RUN_ANALYSIS_TOOLS/analyze_webshop_rollouts.py"
sha256sum "$RUN_ANALYSIS_TOOLS"/*.py > "$RUN_DIR/analysis_tools.sha256"

PANEL_MANIFEST="$PANEL_MANIFEST" PANEL_IDS="$PANEL_IDS" PANEL_DATA="$PANEL_DATA" \
  /opt/conda/envs/py312/bin/python3 - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["PANEL_MANIFEST"]).read_text())
protocol = manifest["protocol"]
expected = {
    "surface": "memoryarena_webshop_native_v1",
    "split_argument": "all",
    "annotation_gate": "exactly_panel_task_ids_in_raw_source_order",
    "price_seed": 233,
    "max_rounds": 30,
    "replicas_per_task": 8,
    "eval_batch_size": 16,
    "rollout_engine_n": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "memory_shaping": "off",
    "rollout_context_policy": "latest_observation_only",
}
for key, value in expected.items():
    if protocol.get(key) != value:
        raise SystemExit(f"panel protocol mismatch {key}: {protocol.get(key)!r} != {value!r}")
ids = [line.strip() for line in Path(os.environ["PANEL_IDS"]).read_text().splitlines() if line.strip()]
if len(ids) != 15 or ids != [row["task_id"] for row in manifest["panel"]["tasks"]]:
    raise SystemExit("panel task order/count mismatch")
rows = json.loads((Path(os.environ["PANEL_DATA"]) / "agentmemory_test.json").read_text())
if [row.get("item_id") for row in rows] != [f"agentmemory_{i}" for i in range(15)]:
    raise SystemExit("eval item IDs do not match positional panel order")
print("PANEL_PROTOCOL_OK", manifest["panel"]["panel_hash"])
PY

if [ ! -s "$GATE" ]; then
  PYTHONPATH="$SOURCE_WT/AgentGym/agentenv-agentmemory:$SOURCE_WT/AgentGym/agentenv" \
    "$VENV/bin/python" \
      "$SOURCE_WT/AgentGym/agentenv-agentmemory/scripts/audits/build_memoryarena_annotation_gate.py" \
      --run-id "$RUN_ID" --mode provisional --raw-data "$RAW" \
      --domain-data "$DB/domain_data.json" --items-file "$DB/items_shuffle.json" \
      --attributes-file "$DB/items_ins_v2.json" \
      --lucene-index-manifest "$LUCENE_MANIFEST" \
      --lucene-index-root "$DB/search_engine/indexes-full" \
      --audit-summary "$AUD/summary.json" --audit-chains "$AUD/chains.jsonl" \
      --manual-evidence "$AUD/manual_candidate_evidence.json" \
      --memoryarena-root "$MEMORYARENA" \
      --memoryarena-base-commit 6cd9de14b71915e39ac742a20dc33785e14b6aab \
      --price-seed 233 --requested-task-ids "$PANEL_IDS" --output "$GATE"
fi
GATE_SHA=$(sha256sum "$GATE" | awk '{print $1}')

RUN_ID="$RUN_ID" GATE="$GATE" PANEL_IDS="$PANEL_IDS" \
  /opt/conda/envs/py312/bin/python3 - <<'PY'
import json
import os
from pathlib import Path

gate = json.loads(Path(os.environ["GATE"]).read_text())
ids = [line.strip() for line in Path(os.environ["PANEL_IDS"]).read_text().splitlines() if line.strip()]
if gate["run"]["run_id"] != os.environ["RUN_ID"]:
    raise SystemExit("narrow gate run_id mismatch")
if gate["allowed_task_ids"] != ids:
    raise SystemExit("narrow gate task IDs/order mismatch")
print("PANEL_GATE_OK", len(ids), gate["allowed_task_ids_sha256"])
PY

sha256sum "$0" "$PANEL_MANIFEST" "$PANEL_IDS" "$GATE" \
  "$SOURCE_WT/AgentGym-RL/verl/agent_trainer/main_generation.py" \
  "$SOURCE_WT/AgentGym-RL/verl/workers/rollout/schemas.py" \
  "$SOURCE_WT/AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py" \
  "$SOURCE_WT/AgentGym/agentenv-agentmemory/agentenv_agentmemory/memoryarena_webshop_env.py" \
  "$SOURCE_WT/AgentGym/agentenv-agentmemory/agentenv_agentmemory/reward_hierarchy.py" \
  "$TOKEN_ANALYZER" "$BEHAVIOR_ANALYZER" \
  > "$RUN_DIR/launch_inputs.sha256"
sha256sum "$Q35_T/AgentGym-RL/verl/agent_trainer/main_generation.py" >> "$RUN_DIR/launch_inputs.sha256"
{
  sha256sum "$MODEL/config.json"
  find "$MODEL" -maxdepth 1 -type f -name 'model*.safetensors' -printf '%f %s bytes\n' | sort
} > "$RUN_DIR/model.identity"
SOURCE_OUTER_HEAD=$(git -C "$SOURCE_WT" rev-parse HEAD)
printf 'outer_base_commit=%s\nouter_head=%s\ninner_commit=%s\nouter_clean=1\ninner_clean=1\n' \
  "$OUTER_COMMIT" "$SOURCE_OUTER_HEAD" "$INNER_COMMIT" > "$RUN_DIR/source_identity.txt"
printf '%s\n' "cold_start_real_six_session_strict_memory_chain_qwen35_sopgate_unified_1card" > "$RUN_DIR/purpose"
printf '%s\n' "panel_hash=$(jq -r '.panel.panel_hash' "$PANEL_MANIFEST")" > "$RUN_DIR/protocol.txt"
printf '%s\n' \
  "model=Qwen3.5-4B prompt=react+generic-memory-timing+unified-listing reasoning=allowed thinking=off price_seed=233 replicas=8 max_rounds=30 temperature=1 top_p=1 n=1 memory_shaping=off ltm_inventory_mode=$MODE ltm_transition_notice_mode=$NOTICE_MODE memory_prompt_mode=$PROMPT_MODE action_listing_mode=$ACTION_MODE presentation_randomization=$PRESENTATION_RANDOMIZATION presentation_seed=$PRESENTATION_SEED prompt_length=$PROMPT_LENGTH response_length=$RESPONSE_LENGTH model_length=$MODEL_LENGTH context_slack=$CONTEXT_SLACK native_model_length=$NATIVE_MODEL_LENGTH retrieve_lookup_modes=query,memory_id" \
  >> "$RUN_DIR/protocol.txt"

mkdir -p "$EVIDENCE_DIR"
cleanup_exact_runtime
/opt/conda/envs/py312/bin/python3 "$CLEANUP_TOOL" \
  --mode assert-clean --owner root --run-id "$RUN_ID" \
  --evidence-dir "$EVIDENCE_DIR"
/opt/conda/envs/py312/bin/python3 "$CLEANUP_TOOL" \
  --mode watch-parent --owner root --run-id "$RUN_ID" --parent-pid "$$" \
  --evidence-dir "$EVIDENCE_DIR" > "$EVIDENCE_DIR/watch_parent.log" 2>&1 &
WATCH_PID=$!
kill -0 "$WATCH_PID"

printf 'eval_run_id=%s ts=%s\n' "$RUN_ID" "$(date -u +%FT%TZ)" > "$GPU_YIELD_MARKER"
printf 'eval_run_id=%s ts=%s\n' "$RUN_ID" "$(date -u +%FT%TZ)" > "$FORMAL_MARKER"
for _ in $(seq 1 60); do
  grep -q 'mode=yield' /tmp/crg-holder.state 2>/dev/null && break
  sleep 2
done
grep -q 'mode=yield' /tmp/crg-holder.state || {
  echo "FATAL auto-yield holder did not yield" >&2; exit 75;
}

NINJA_BIN_DIR=$(/opt/conda/envs/py312/bin/python3 -c 'import ninja; print(ninja.BIN_DIR)')
export PATH="$NINJA_BIN_DIR:$PATH"
export PYTHONPATH="$PYTHON_OVERLAY:$SOURCE_WT:$SOURCE_WT/AgentGym-RL:$SOURCE_WT/AgentGym/agentenv-agentmemory:$SOURCE_WT/AgentGym/agentenv"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true VLLM_ALLOW_INSECURE_SERIALIZATION=1
export AGENTMEMORY_ENABLE_THINKING=0 AGENTMEMORY_ALLOW_REASONING=1 AGENTMEMORY_ALLOW_RAW_HISTORY=0
export AGENTMEMORY_MEMORY_SHAPING=off AGENTMEMORY_BUY_SEMANTICS=terminate
export AGENTMEMORY_ROLLOUT_DEBUG=1 AGENTMEMORY_REQUIRE_FORMAL_RUNTIME_EVIDENCE=0
export AGENTMEMORY_RUN_ID="$RUN_ID" AGENTMEMORY_LTM_INVENTORY_MODE="$MODE"
export AGENTMEMORY_LTM_TRANSITION_NOTICE_MODE="$NOTICE_MODE"
export AGENTMEMORY_MEMORY_PROMPT_MODE="$PROMPT_MODE"
export AGENTMEMORY_ACTION_LISTING_MODE="$ACTION_MODE"

PYTHONPATH="$PYTHON_OVERLAY:$SOURCE_WT/AgentGym/agentenv-agentmemory:$SOURCE_WT/AgentGym/agentenv:$SOURCE_WT/AgentGym-RL" \
    "$ENV_PYTHON" -m agentenv_agentmemory.launch \
    --port "$ENV_PORT" --host 127.0.0.1 --surface memoryarena_webshop_native_v1 \
    --memoryarena-root "$MEMORYARENA" --raw-data "$RAW" \
    --items-file "$DB/items_shuffle.json" --attributes-file "$DB/items_ins_v2.json" \
    --search-root "$DB/search_engine" --java-home "$JAVA_HOME_PERSIST" \
    --domain-data-path "$DB/domain_data.json" --lucene-index-manifest "$LUCENE_MANIFEST" \
    --annotation-audit-summary "$AUD/summary.json" --annotation-audit-chains "$AUD/chains.jsonl" \
    --annotation-manual-evidence "$AUD/manual_candidate_evidence.json" \
    --memoryarena-base-commit 6cd9de14b71915e39ac742a20dc33785e14b6aab \
    --run-id "$RUN_ID" --split all --price-seed 233 \
    --presentation-randomization "$PRESENTATION_RANDOMIZATION" --presentation-seed "$PRESENTATION_SEED" \
    --memory-first-add-reward 0.0 --memory-first-later-retrieve-reward 0.0 \
    --ltm-inventory-mode "$MODE" --ltm-transition-notice-mode "$NOTICE_MODE" \
    --memory-prompt-mode "$PROMPT_MODE" --action-listing-mode "$ACTION_MODE" \
    --annotation-gate-mode provisional \
    --annotation-gate-manifest "$GATE" --annotation-gate-manifest-sha256 "$GATE_SHA" \
    > "$OUT/env.log" 2>&1 &
ENV_PID=$!
echo "$ENV_PID" > "$OUT/env.pid"
for i in $(seq 1 90); do
  if curl -fsS -m 8 "http://127.0.0.1:$ENV_PORT/metadata" > "$OUT/env_metadata.json"; then
    echo "[eval] env mode=$MODE ready after $((i * 5))s"
    break
  fi
  kill -0 "$ENV_PID" 2>/dev/null || { echo "FATAL env server died" >&2; exit 76; }
  [ "$i" -lt 90 ] || { echo "FATAL env readiness timeout" >&2; exit 77; }
  sleep 5
done
jq -e --arg mode "$MODE" --arg notice "$NOTICE_MODE" --arg prompt "$PROMPT_MODE" --arg action "$ACTION_MODE" --arg presentation "$PRESENTATION_RANDOMIZATION" --argjson presentation_seed "$PRESENTATION_SEED" '.surface == "memoryarena_webshop_native_v1"
  and .task_count == 15
  and .backend.price_seed == 233
  and .reward_contract.first_valid_add_reward == 0
  and .reward_contract.first_valid_later_session_retrieve_reward == 0
  and .ltm_inventory_mode == $mode
  and .ltm_transition_notice_mode == $notice
  and .memory_prompt_mode == $prompt
  and .action_listing_mode == $action
  and .presentation_randomization.mode == $presentation
  and .presentation_randomization.base_seed == $presentation_seed
  and .ltm_inventory_key_max_chars == 24' "$OUT/env_metadata.json" >/dev/null

export RAY_TMPDIR=/tmp/amg-ltm-transition-${MODE}-${NOTICE_MODE}-${ENV_PORT}
export TMPDIR=$RAY_TMPDIR
mkdir -p "$RAY_TMPDIR"
date -u +%FT%TZ > "$OUT/started_at"
echo "[eval] starting mode=$MODE model=$MODEL"
set +e
PYTHONPATH="$Q35_Q/lib/python3.12/site-packages:$Q35_BRIDGE:$Q35_T/AgentGym-RL:$Q35_T/AgentGym/agentenv:$Q35_T/AgentGym/agentenv-agentmemory:$PYTHONPATH" "$Q35_Q/bin/python" -m verl.agent_trainer.main_generation \
  data.path="$PANEL_DATA" data.prompt_key=item_id data.batch_size=16 \
  data.max_prompt_length="$PROMPT_LENGTH" data.max_response_length="$RESPONSE_LENGTH" data.n_samples=8 \
  agentgym.task_name=agentmemory agentgym.env_addr="http://127.0.0.1:$ENV_PORT" \
  agentgym.timeout=600 agentgym.max_retries=2 agentgym.max_rounds=30 \
  +agentgym.max_concurrent=16 model.path="$MODEL" +model.use_remove_padding=true \
  trainer.nnodes=1 trainer.n_gpus_per_node=1 \
  rollout.name=vllm rollout.tensor_model_parallel_size=1 rollout.gpu_memory_utilization=0.72 \
  +rollout.sync_weight_format=hf +rollout.vllm_init_load_format=dummy \
  rollout.load_format=dummy_hf rollout.max_model_len="$MODEL_LENGTH" \
  rollout.max_num_batched_tokens=32768 rollout.max_num_seqs=16 rollout.max_tokens=1024 \
  rollout.temperature=1.0 rollout.top_p=1.0 rollout.n=1 \
  +rollout.enable_sleep_mode=true \
  hydra.run.dir="$OUT/hydra" \
  rollout.rollout_log_dir="$OUT/executer_logs" \
  > "$OUT/generation.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$OUT/exit_code"
date -u +%FT%TZ > "$OUT/finished_at"
[ "$rc" -eq 0 ] || exit "$rc"
grep -E '^(Avg|Pass|Progress)@8:' "$OUT/generation.log" > "$OUT/metrics.txt" || true
"$Q35_Q/bin/python" "$RUN_ANALYSIS_TOOLS/analyze_rollout_prompt_tokens.py" \
  --run-dir "$RUN_DIR" --model "$MODEL" --runtime-source "$Q35_T/AgentGym-RL" \
  --reply-mode reasoning --expected-replicas 8 \
  --output "$RUN_DIR/prompt_token_telemetry.json"
"$Q35_Q/bin/python" "$RUN_ANALYSIS_TOOLS/analyze_webshop_rollouts.py" \
  --run-dir "$RUN_DIR" --expected-replicas 8 \
  --output "$RUN_DIR/webshop_rollout_analysis.json"
sha256sum "$RUN_DIR/prompt_token_telemetry.json" \
  "$RUN_DIR/webshop_rollout_analysis.json" > "$RUN_DIR/analysis_outputs.sha256"
echo "[eval] completed mode=$MODE"
