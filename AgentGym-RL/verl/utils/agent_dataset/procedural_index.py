from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence


PROCEDURAL_INDEX_SCHEMA = "agentmemory_procedural_index_source_v1"
PROCEDURAL_STREAM_IDENTITY_SCHEMA = "agentmemory_procedural_stream_identity_v1"
PROCEDURAL_STREAM_CHECKPOINT_SCHEMA = (
    "agentmemory_procedural_stream_checkpoint_v1"
)
ROLLOUT_REPLICA_INDEX_KEY = "agentmemory_replica_index"
PROVIDER_MODE_FIXED_WINDOW = "fixed_window"
PROVIDER_MODE_RESEEDED_STREAM = "reseeded_stream"
PROVIDER_MODES = (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM,
)
NEGATIVE_CONSTRAINT_SURFACE = (
    "agentmemory_webshop_negative_constraint_top1_train_v1"
)
FILESYSTEM_SURFACE = (
    "agentmemory_webshop_procedural_natural_chain_filesystem_v2"
)
RECENCY_OVERRIDE_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_recency_override_filesystem_v2"
)
COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_compositional_recall_filesystem_v2"
)
NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_negative_constraint_filesystem_v2"
)
FILESYSTEM_SURFACES = frozenset(
    {
        FILESYSTEM_SURFACE,
        RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
        COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
        NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
    }
)
FILESYSTEM_SURFACE_CONTRACTS = {
    FILESYSTEM_SURFACE: (
        "xor_lsb_within_orbit_v1",
        1,
        "natural_attribute_chain_filesystem_v2",
    ),
    RECENCY_OVERRIDE_FILESYSTEM_SURFACE: (
        "xor_lsb_within_orbit_v1",
        3,
        "recency_override_filesystem_v2",
    ),
    COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE: (
        "xor_lsb_within_orbit_v1",
        2,
        "compositional_recall_filesystem_v2",
    ),
    NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE: (
        "cyclic_next_within_orbit_v1",
        1,
        "negative_constraint_filesystem_v2",
    ),
}
SUPPORTED_SERVER_SURFACE_CONTRACTS = {
    "agentmemory_webshop_procedural_natural_chain_train_v1": (
        "agentmemory_verified_natural_chain_provider_v4",
        2,
    ),
    FILESYSTEM_SURFACE: (
        "agentmemory_verified_natural_chain_provider_v4",
        2,
    ),
    RECENCY_OVERRIDE_FILESYSTEM_SURFACE: (
        "agentmemory_verified_recency_override_provider_v1",
        2,
    ),
    COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE: (
        "agentmemory_verified_compositional_recall_provider_v1",
        4,
    ),
    NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE: (
        "agentmemory_verified_negative_constraint_provider_v1",
        3,
    ),
    "agentmemory_webshop_latent_preference_train_v1": (
        "agentmemory_verified_latent_preference_provider_v1",
        2,
    ),
    "agentmemory_webshop_recency_override_train_v1": (
        "agentmemory_verified_recency_override_provider_v1",
        2,
    ),
    "agentmemory_webshop_distractor_robustness_top1_train_v1": (
        "agentmemory_verified_distractor_robustness_provider_v1",
        2,
    ),
    "agentmemory_webshop_compositional_recall_top1_train_v1": (
        "agentmemory_verified_compositional_recall_provider_v1",
        4,
    ),
    "agentmemory_webshop_intent_clarification_train_v1": (
        "agentmemory_verified_intent_clarification_provider_v1",
        2,
    ),
    "agentmemory_webshop_selective_memory_use_top1_train_v1": (
        "agentmemory_verified_selective_memory_use_provider_v1",
        4,
    ),
    NEGATIVE_CONSTRAINT_SURFACE: (
        "agentmemory_verified_negative_constraint_provider_v1",
        3,
    ),
}


class ProceduralIndexError(ValueError):
    pass


def generation_non_tensor_keys(
    non_tensor_batch: Mapping[str, Any],
) -> list[str]:
    """Return generation fields while preserving an explicit environment index."""

    keys = ["item_id", "raw_prompt"]
    if "data_idx" in non_tensor_batch:
        keys.append("data_idx")
    return keys


def promote_data_idx_for_rollout(
    non_tensor_batch: MutableMapping[str, Any],
) -> bool:
    """Move a dataset index to the rollout field consumed by env reset."""

    if "data_idx" not in non_tensor_batch:
        return False
    if "rollout_data_indices" in non_tensor_batch:
        raise ProceduralIndexError(
            "generation batch contains both data_idx and rollout_data_indices"
        )
    non_tensor_batch["rollout_data_indices"] = non_tensor_batch.pop("data_idx")
    return True


def validate_orbit_batch_indices(
    indices: Sequence[Any],
    *,
    tasks_per_orbit: int,
) -> None:
    """Require a contiguous batch made of complete procedural task orbits."""

    if (
        isinstance(tasks_per_orbit, bool)
        or not isinstance(tasks_per_orbit, int)
        or tasks_per_orbit <= 0
    ):
        raise ProceduralIndexError("tasks_per_orbit must be a positive integer")

    normalized = []
    for position, value in enumerate(indices):
        if isinstance(value, bool):
            raise ProceduralIndexError(
                f"procedural batch index {position} must be an integer, got bool"
            )
        try:
            normalized.append(operator.index(value))
        except TypeError as exc:
            raise ProceduralIndexError(
                f"procedural batch index {position} is not an integer: {value!r}"
            ) from exc
    if not normalized or len(normalized) % tasks_per_orbit:
        raise ProceduralIndexError(
            "procedural batch must contain a positive number of complete orbits"
        )
    if normalized[0] < 0 or normalized[0] % tasks_per_orbit:
        raise ProceduralIndexError(
            "procedural batch must start at a non-negative orbit boundary"
        )
    expected = list(range(normalized[0], normalized[0] + len(normalized)))
    if normalized != expected:
        raise ProceduralIndexError(
            "procedural batch indices must be contiguous and preserve adjacent "
            "task orbits"
        )


def validate_paired_batch_indices(indices: Sequence[Any]) -> None:
    """Compatibility wrapper for the original two-task orbit contract."""

    validate_orbit_batch_indices(indices, tasks_per_orbit=2)


def validate_rollout_parent_coverage(
    non_tensor_batch: Mapping[str, Any],
    *,
    expected_parent_count: int,
    expected_replicas: int,
) -> None:
    """Fail if an infrastructure exclusion removed any source trajectory."""

    if expected_parent_count <= 0 or expected_replicas <= 0:
        raise ProceduralIndexError(
            "expected_parent_count and expected_replicas must be positive"
        )
    parent_indices = non_tensor_batch.get("rollout_parent_indices")
    replica_indices = non_tensor_batch.get(ROLLOUT_REPLICA_INDEX_KEY)
    if parent_indices is None or replica_indices is None:
        raise ProceduralIndexError(
            "procedural rollout is missing parent or replica identity metadata"
        )
    if len(parent_indices) != len(replica_indices):
        raise ProceduralIndexError(
            "procedural rollout parent and replica metadata lengths disagree"
        )

    observed = set()
    for row, (raw_parent, raw_replica) in enumerate(
        zip(parent_indices, replica_indices)
    ):
        if isinstance(raw_parent, bool) or isinstance(raw_replica, bool):
            raise ProceduralIndexError(
                f"procedural rollout identity row {row} contains bool"
            )
        try:
            pair = (operator.index(raw_parent), operator.index(raw_replica))
        except TypeError as exc:
            raise ProceduralIndexError(
                f"procedural rollout identity row {row} is not integral"
            ) from exc
        observed.add(pair)

    expected = {
        (parent, replica)
        for parent in range(expected_parent_count)
        for replica in range(expected_replicas)
    }
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ProceduralIndexError(
            "procedural rollout lost or invented source trajectories; refusing "
            f"a partial PPO update: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )


def stream_identity_sha256(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_stream_checkpoint(
    sampler: "StatefulProceduralStreamSampler",
    stream_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _canonical_json_mapping(stream_identity, field="stream identity")
    return {
        "schema": PROCEDURAL_STREAM_CHECKPOINT_SCHEMA,
        "stream_identity": identity,
        "stream_identity_sha256": stream_identity_sha256(identity),
        "sampler_state": sampler.state_dict(),
    }


def restore_stream_checkpoint(
    sampler: "StatefulProceduralStreamSampler",
    current_stream_identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    if checkpoint.get("schema") != PROCEDURAL_STREAM_CHECKPOINT_SCHEMA:
        raise ProceduralIndexError(
            "procedural resume requires a versioned stream checkpoint"
        )
    saved_identity = checkpoint.get("stream_identity")
    if not isinstance(saved_identity, Mapping):
        raise ProceduralIndexError(
            "procedural stream checkpoint is missing stream identity"
        )
    saved_identity = _canonical_json_mapping(
        saved_identity,
        field="checkpoint stream identity",
    )
    saved_digest = checkpoint.get("stream_identity_sha256")
    if saved_digest != stream_identity_sha256(saved_identity):
        raise ProceduralIndexError(
            "procedural stream checkpoint identity digest is invalid"
        )
    current_identity = _canonical_json_mapping(
        current_stream_identity,
        field="current stream identity",
    )
    if saved_identity != current_identity:
        raise ProceduralIndexError(
            "procedural stream checkpoint identity does not match the current "
            "dataset, server, or training geometry"
        )
    sampler_state = checkpoint.get("sampler_state")
    if not isinstance(sampler_state, Mapping):
        raise ProceduralIndexError(
            "procedural stream checkpoint is missing sampler state"
        )
    sampler.load_state_dict(sampler_state)


def _canonical_json_mapping(
    value: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        canonical = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise ProceduralIndexError(f"{field} must be JSON serializable") from exc
    if not isinstance(canonical, dict):
        raise ProceduralIndexError(f"{field} must be a mapping")
    return canonical


@dataclass(frozen=True)
class ProceduralIndexSource:
    """Map lightweight positions to deterministic AgentMemoryGym task indices.

    The source stores no task rows. The environment server materializes and
    verifies a task only when the resulting absolute index is reset.
    """

    task_count: int
    provider_mode: str
    tasks_per_orbit: int = 2
    start_index: int = 0
    item_id_prefix: str = "agentmemory"

    def __post_init__(self) -> None:
        if (
            isinstance(self.tasks_per_orbit, bool)
            or not isinstance(self.tasks_per_orbit, int)
            or self.tasks_per_orbit <= 0
        ):
            raise ProceduralIndexError(
                "tasks_per_orbit must be a positive integer"
            )
        if (
            isinstance(self.task_count, bool)
            or not isinstance(self.task_count, int)
            or self.task_count <= 0
            or self.task_count % self.tasks_per_orbit
        ):
            raise ProceduralIndexError(
                "task_count must be a positive multiple of tasks_per_orbit"
            )
        if self.provider_mode not in PROVIDER_MODES:
            raise ProceduralIndexError(
                f"provider_mode must be one of {PROVIDER_MODES}, got "
                f"{self.provider_mode!r}"
            )
        if (
            isinstance(self.start_index, bool)
            or not isinstance(self.start_index, int)
            or self.start_index < 0
            or self.start_index % self.tasks_per_orbit
        ):
            raise ProceduralIndexError(
                "start_index must be a non-negative orbit boundary"
            )
        if self.provider_mode == PROVIDER_MODE_FIXED_WINDOW and self.start_index != 0:
            raise ProceduralIndexError("fixed_window requires start_index=0")
        if self.item_id_prefix != "agentmemory":
            raise ProceduralIndexError(
                "the current rollout parser requires item_id_prefix='agentmemory'"
            )

    def __len__(self) -> int:
        return self.task_count

    def row_for_position(self, position: int) -> dict[str, Any]:
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 0
        ):
            raise IndexError(f"invalid procedural dataset position {position!r}")
        if (
            self.provider_mode == PROVIDER_MODE_FIXED_WINDOW
            and position >= self.task_count
        ):
            raise IndexError(
                f"procedural dataset position {position} is outside [0, "
                f"{self.task_count})"
            )
        absolute_index = self.start_index + position
        return {
            "item_id": f"{self.item_id_prefix}_{absolute_index}",
            "data_idx": absolute_index,
            "extra_info": {"index": absolute_index},
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": PROCEDURAL_INDEX_SCHEMA,
            "provider_mode": self.provider_mode,
            "task_count": self.task_count,
            "start_index": self.start_index,
            "item_id_prefix": self.item_id_prefix,
            "materialized_rows": 0,
            "tasks_per_orbit": self.tasks_per_orbit,
            "counterfactual_pair_size": (
                2 if self.tasks_per_orbit == 2 else None
            ),
        }

    def validate_training_batch_size(self, batch_size: int) -> None:
        if self.provider_mode != PROVIDER_MODE_RESEEDED_STREAM:
            raise ProceduralIndexError(
                "PPO procedural training requires provider_mode='reseeded_stream'"
            )
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
            or batch_size % self.tasks_per_orbit
        ):
            raise ProceduralIndexError(
                "procedural train_batch_size must contain a positive number of "
                "complete orbits"
            )
        if self.task_count % batch_size:
            raise ProceduralIndexError(
                "procedural task_count must be divisible by train_batch_size; "
                "drop_last would otherwise consume untrained stream indices"
            )

    def validate_server_metadata(self, metadata: Mapping[str, Any]) -> None:
        surface = metadata.get("surface")
        expected_contract = SUPPORTED_SERVER_SURFACE_CONTRACTS.get(surface)
        if expected_contract is None:
            raise ProceduralIndexError(
                "procedural index source received an unsupported server surface"
            )
        expected_provider_schema, expected_tasks_per_orbit = expected_contract
        if self.tasks_per_orbit != expected_tasks_per_orbit:
            raise ProceduralIndexError(
                "dataset tasks_per_orbit does not match the server surface"
            )
        provider = metadata.get("provider")
        if not isinstance(provider, Mapping):
            raise ProceduralIndexError("server metadata is missing provider")
        if provider.get("schema") != expected_provider_schema:
            raise ProceduralIndexError(
                "server surface and provider schema are not an approved pair"
            )
        if provider.get("tasks_per_orbit") != self.tasks_per_orbit:
            raise ProceduralIndexError(
                "dataset/server tasks_per_orbit metadata disagrees"
            )
        if provider.get("provider_mode") != self.provider_mode:
            raise ProceduralIndexError(
                "dataset/server provider modes disagree: "
                f"{self.provider_mode!r} != {provider.get('provider_mode')!r}"
            )
        if provider.get("task_count") != self.task_count:
            raise ProceduralIndexError(
                "dataset/server task counts disagree: "
                f"{self.task_count!r} != {provider.get('task_count')!r}"
            )
        expected_candidate_count = 3 if surface in {
            NEGATIVE_CONSTRAINT_SURFACE,
            NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
        } else 2
        if provider.get("candidate_count_per_phase") != expected_candidate_count:
            raise ProceduralIndexError(
                "server candidate count per phase disagrees with the approved "
                f"surface contract: expected {expected_candidate_count}"
            )
        if provider.get("phase_count_per_task") != 6:
            raise ProceduralIndexError("server no longer provides six-phase tasks")
        if provider.get("human_review_required") is not False:
            raise ProceduralIndexError("server unexpectedly requires human review")
        if provider.get("llm_judge_required") is not False:
            raise ProceduralIndexError("server unexpectedly requires an LLM judge")
        if provider.get("task_prompt_product_identity") != "complete_native_title":
            raise ProceduralIndexError("server task prompt identity is unsupported")
        if provider.get("target_asin_in_task_prompt") is not False:
            raise ProceduralIndexError("server task prompt leaks the target ASIN")
        if provider.get("native_search_result_asin_handles_visible") is not True:
            raise ProceduralIndexError(
                "server no longer exposes native search-result ASIN handles"
            )
        if provider.get("native_click_action_uses_asin_handle") is not True:
            raise ProceduralIndexError("server no longer uses native click[ASIN]")
        if surface in FILESYSTEM_SURFACES:
            source_pairing, boundary, prompt_family = (
                FILESYSTEM_SURFACE_CONTRACTS[surface]
            )
            if metadata.get("source_pairing") != source_pairing:
                raise ProceduralIndexError(
                    "filesystem source-pairing metadata disagrees"
                )
            if metadata.get("tasks_per_orbit") != self.tasks_per_orbit:
                raise ProceduralIndexError(
                    "filesystem top-level tasks_per_orbit metadata disagrees"
                )
            if metadata.get("workspace_prompt_family") != prompt_family:
                raise ProceduralIndexError(
                    "filesystem prompt-family metadata disagrees"
                )
            control = metadata.get("workspace_intervention_control")
            expected_arms = ["correct", "blank", "swapped", "no_workspace"]
            if surface == RECENCY_OVERRIDE_FILESYSTEM_SURFACE:
                expected_arms.insert(3, "stale")
            if (
                not isinstance(control, Mapping)
                or control.get("allowed_arms") != expected_arms
                or control.get("boundary_session_index") != boundary
                or control.get("source_state")
                != "policy_authored_workspace_only"
                or control.get("hidden_answer_injection") is not False
            ):
                raise ProceduralIndexError(
                    "filesystem intervention-boundary metadata disagrees"
                )
            if metadata.get("memory_prompt_mode") != "natural_filesystem":
                raise ProceduralIndexError(
                    "filesystem surface requires natural_filesystem prompt mode"
                )
            if metadata.get("workspace_surface") != "codex_workspace_v2":
                raise ProceduralIndexError(
                    "filesystem surface workspace contract is missing"
                )
            if (
                metadata.get("workspace_tool_contract")
                != "codex_shell_command_apply_patch_v1"
            ):
                raise ProceduralIndexError(
                    "filesystem surface Codex tool contract is missing"
                )
            if metadata.get("workspace_shell_enabled") is not True:
                raise ProceduralIndexError(
                    "filesystem surface must expose shell_command"
                )
            if metadata.get("workspace_apply_patch_enabled") is not True:
                raise ProceduralIndexError(
                    "filesystem surface must expose apply_patch"
                )
            if metadata.get("workspace_host_path_exposed") is not False:
                raise ProceduralIndexError(
                    "filesystem surface must not expose a host path"
                )
            observed_ops = metadata.get("workspace_tool_ops")
            if not isinstance(observed_ops, (list, tuple)) or {
                str(value).upper() for value in observed_ops
            } != {"SHELL_COMMAND", "APPLY_PATCH"}:
                raise ProceduralIndexError(
                    "filesystem surface Codex workspace tool contract is invalid"
                )
            reward_contract = metadata.get("reward_contract")
            if not isinstance(reward_contract, Mapping) or any(
                float(reward_contract.get(field, float("nan"))) != 0.0
                for field in (
                    "workspace_action_reward",
                    "shell_command_reward",
                    "apply_patch_reward",
                )
            ):
                raise ProceduralIndexError(
                    "filesystem surface must use zero workspace-action shaping"
                )
        elif metadata.get("memory_prompt_mode") == "natural_filesystem":
            raise ProceduralIndexError(
                "natural_filesystem prompt mode is bound to the filesystem surface"
            )
        semantic_period_orbits = provider.get("semantic_period_orbits")
        semantic_period_tasks = provider.get("semantic_period_tasks")
        if (
            isinstance(semantic_period_orbits, bool)
            or not isinstance(semantic_period_orbits, int)
            or semantic_period_orbits <= 0
            or isinstance(semantic_period_tasks, bool)
            or not isinstance(semantic_period_tasks, int)
            or semantic_period_tasks
            != semantic_period_orbits * self.tasks_per_orbit
        ):
            raise ProceduralIndexError("server semantic period metadata is invalid")
        if self.provider_mode == PROVIDER_MODE_RESEEDED_STREAM:
            if provider.get("accepted_index_domain") != "all_nonnegative_integers":
                raise ProceduralIndexError(
                    "server stream does not accept every non-negative index"
                )
            stream = provider.get("reseeded_stream")
            if not isinstance(stream, Mapping):
                raise ProceduralIndexError("server stream metadata is missing")
            orbit_boundary_field = {
                4: "factorial_orbit_never_crosses_seed_epoch",
                3: "counterfactual_orbit_never_crosses_seed_epoch",
                2: "counterfactual_pair_never_crosses_seed_epoch",
            }[self.tasks_per_orbit]
            expected_stream_values = {
                "tasks_per_seed_epoch": semantic_period_tasks,
                "orbits_per_seed_epoch": semantic_period_orbits,
                orbit_boundary_field: True,
                "seed_epoch_zero_uses_base_seed": True,
                "collision_free_within_complete_seed_epoch": True,
                "semantic_uniqueness_guaranteed_through_task_index": (
                    semantic_period_tasks - 1
                ),
                "cross_seed_epoch_semantic_uniqueness_guaranteed": False,
            }
            if any(
                stream.get(key) != value
                for key, value in expected_stream_values.items()
            ):
                raise ProceduralIndexError(
                    "server stream epoch metadata is inconsistent"
                )

    def training_identity(
        self,
        *,
        server_metadata: Mapping[str, Any],
        train_batch_size: int,
    ) -> dict[str, Any]:
        self.validate_training_batch_size(train_batch_size)
        self.validate_server_metadata(server_metadata)
        return {
            "schema": PROCEDURAL_STREAM_IDENTITY_SCHEMA,
            "index_source": self.metadata(),
            "training_geometry": {
                "train_batch_size": train_batch_size,
                "shuffle": False,
                "drop_last": True,
                "tasks_per_orbit": self.tasks_per_orbit,
            },
            "server_metadata": _canonical_json_mapping(
                server_metadata,
                field="server metadata",
            ),
        }


class StatefulProceduralStreamSampler:
    """Yield disjoint contiguous windows and preserve the cursor in checkpoints."""

    def __init__(
        self,
        data_source: ProceduralIndexSource,
        *,
        next_position: int = 0,
    ) -> None:
        if data_source.provider_mode != PROVIDER_MODE_RESEEDED_STREAM:
            raise ProceduralIndexError(
                "StatefulProceduralStreamSampler requires reseeded_stream"
            )
        if (
            isinstance(next_position, bool)
            or not isinstance(next_position, int)
            or next_position < 0
            or next_position % data_source.tasks_per_orbit
        ):
            raise ProceduralIndexError(
                "next_position must be a non-negative orbit boundary"
            )
        self.data_source = data_source
        self.next_position = next_position
        self.samples_per_epoch = len(data_source)

    def __iter__(self):
        stop = self.next_position + self.samples_per_epoch
        while self.next_position < stop:
            position = self.next_position
            self.next_position += 1
            yield position

    def __len__(self) -> int:
        return self.samples_per_epoch

    def state_dict(self) -> dict[str, int | str]:
        return {
            "schema": "agentmemory_procedural_stream_sampler_state_v1",
            "next_position": self.next_position,
            "samples_per_epoch": self.samples_per_epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "agentmemory_procedural_stream_sampler_state_v1":
            raise ProceduralIndexError("unsupported procedural sampler state schema")
        next_position = state.get("next_position")
        if (
            isinstance(next_position, bool)
            or not isinstance(next_position, int)
            or next_position < 0
            or next_position % self.data_source.tasks_per_orbit
        ):
            raise ProceduralIndexError("invalid sampler next_position")
        if state.get("samples_per_epoch") != self.samples_per_epoch:
            raise ProceduralIndexError("sampler state uses a different epoch length")
        self.next_position = next_position


def procedural_index_source_from_config(
    config: Mapping[str, Any] | Any,
) -> ProceduralIndexSource | None:
    raw = config.get("procedural_index")
    if raw is None:
        return None
    enabled = raw.get("enabled", False)
    if enabled is False:
        return None
    if enabled is not True:
        raise ProceduralIndexError("procedural_index.enabled must be a boolean")
    return ProceduralIndexSource(
        task_count=raw.get("task_count"),
        provider_mode=raw.get("provider_mode"),
        tasks_per_orbit=raw.get("tasks_per_orbit", 2),
        start_index=raw.get("start_index", 0),
        item_id_prefix=raw.get("item_id_prefix", "agentmemory"),
    )
