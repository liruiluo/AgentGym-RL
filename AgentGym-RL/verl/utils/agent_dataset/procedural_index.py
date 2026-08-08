from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, MutableMapping, Sequence


PROCEDURAL_INDEX_SCHEMA = "agentmemory_procedural_index_source_v1"
PROCEDURAL_STREAM_IDENTITY_SCHEMA = "agentmemory_procedural_stream_identity_v1"
PROCEDURAL_STREAM_CHECKPOINT_SCHEMA = (
    "agentmemory_procedural_stream_checkpoint_v1"
)
ROLLOUT_REPLICA_INDEX_KEY = "agentmemory_replica_index"
MULTITASK_SURFACE_SLOT_KEY = "agentmemory_surface_slot"
MULTITASK_LOCAL_DATA_INDEX_KEY = "agentmemory_local_data_idx"
MULTITASK_ROUTE_KIND_KEY = "agentmemory_multitask_route_kind"
MULTITASK_SAMPLING_SEED_KEY = "agentmemory_multitask_sampling_seed"
MULTITASK_LOCAL_TASK_COUNT_KEY = "agentmemory_multitask_local_task_count"
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
LATENT_PREFERENCE_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_latent_preference_filesystem_v2"
)
RECENCY_OVERRIDE_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_recency_override_filesystem_v2"
)
DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_distractor_robustness_filesystem_v2"
)
COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_compositional_recall_filesystem_v2"
)
NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_negative_constraint_filesystem_v2"
)
INTENT_CLARIFICATION_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_intent_clarification_filesystem_v2"
)
SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_selective_memory_use_filesystem_v2"
)
FILESYSTEM_SURFACES = frozenset(
    {
        FILESYSTEM_SURFACE,
        LATENT_PREFERENCE_FILESYSTEM_SURFACE,
        RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
        DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
        COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
        INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
        SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
        NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
    }
)
FILESYSTEM_MULTITASK_KIND = "filesystem_task_balanced_v1"
FILESYSTEM_MULTITASK_UNIFORM_KIND = "filesystem_surface_uniform_v2"
FILESYSTEM_MULTITASK_FIXED_BATCH_SIZE = 64
FILESYSTEM_MULTITASK_DEFAULT_SAMPLING_SEED = 233
FILESYSTEM_MULTITASK_SURFACE_ORDER = (
    FILESYSTEM_SURFACE,
    LATENT_PREFERENCE_FILESYSTEM_SURFACE,
    RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
    DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
    COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
    NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
    INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
    SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
)
FILESYSTEM_MULTITASK_ROUTE_ORBIT_SIZES = (2, 2, 2, 2, 4, 3, 2, 4)
FILESYSTEM_MULTITASK_ROWS_PER_SURFACE = 12
FILESYSTEM_MULTITASK_CYCLE_SIZE = (
    len(FILESYSTEM_MULTITASK_SURFACE_ORDER)
    * FILESYSTEM_MULTITASK_ROWS_PER_SURFACE
)
FILESYSTEM_SURFACE_CONTRACTS = {
    FILESYSTEM_SURFACE: (
        "xor_lsb_within_orbit_v1",
        1,
        "natural_attribute_chain_filesystem_v2",
        "directional_counterfactual_separation_v1",
    ),
    LATENT_PREFERENCE_FILESYSTEM_SURFACE: (
        "xor_lsb_within_orbit_v1",
        1,
        "latent_preference_filesystem_v2",
        "directional_counterfactual_separation_v1",
    ),
    RECENCY_OVERRIDE_FILESYSTEM_SURFACE: (
        "xor_lsb_within_orbit_v1",
        3,
        "recency_override_filesystem_v2",
        "directional_counterfactual_separation_v1",
    ),
    DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE: (
        "xor_distractor_condition_within_orbit_v1",
        1,
        "distractor_robustness_filesystem_v2",
        "paired_distractor_robustness_v1",
    ),
    COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE: (
        "xor_lsb_within_orbit_v1",
        2,
        "compositional_recall_filesystem_v2",
        "directional_counterfactual_separation_v1",
    ),
    NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE: (
        "cyclic_next_within_orbit_v1",
        1,
        "negative_constraint_filesystem_v2",
        "directional_counterfactual_separation_v1",
    ),
    INTENT_CLARIFICATION_FILESYSTEM_SURFACE: (
        "xor_lsb_within_orbit_v1",
        1,
        "intent_clarification_filesystem_v2",
        "directional_counterfactual_separation_v1",
    ),
    SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE: (
        "xor_preference_coordinate_within_factorial_v1",
        1,
        "selective_memory_use_filesystem_v2",
        "selective_required_separation_not_required_invariance_v1",
    ),
}
FILESYSTEM_SEEDED_WORKSPACE_CONTRACTS = {
    DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE: (
        "policy_authored_current_record_plus_branch_distractors",
        "branch_conditioned_ordinary_profile_files_v1",
    ),
    SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE: (
        "harness_seeded_branch_profile_with_optional_policy_edits",
        "branch_conditioned_initial_profile_files_v1",
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
    DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE: (
        "agentmemory_verified_distractor_robustness_provider_v1",
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
    LATENT_PREFERENCE_FILESYSTEM_SURFACE: (
        "agentmemory_verified_latent_preference_provider_v1",
        2,
    ),
    INTENT_CLARIFICATION_FILESYSTEM_SURFACE: (
        "agentmemory_verified_intent_clarification_provider_v1",
        2,
    ),
    SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE: (
        "agentmemory_verified_selective_memory_use_provider_v1",
        4,
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
    for key in (
        MULTITASK_SURFACE_SLOT_KEY,
        MULTITASK_LOCAL_DATA_INDEX_KEY,
        MULTITASK_ROUTE_KIND_KEY,
        MULTITASK_SAMPLING_SEED_KEY,
        MULTITASK_LOCAL_TASK_COUNT_KEY,
    ):
        if key in non_tensor_batch:
            keys.append(key)
    return keys


def _normalize_nonnegative_index(field: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ProceduralIndexError(f"{field} must be an integer, got bool")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ProceduralIndexError(
            f"{field} must be an integer, got {value!r}"
        ) from exc
    if normalized < 0:
        raise ProceduralIndexError(f"{field} must be non-negative")
    return normalized


@lru_cache(maxsize=16)
def _uniform_multitask_pool_permutation(
    sampling_seed: int,
    pool_epoch: int,
    pool_size: int,
) -> tuple[int, ...]:
    """Return a deterministic random-key permutation of one source-row pool."""

    if pool_size <= 0:
        raise ProceduralIndexError("multitask source-row pool must be non-empty")
    prefix = (
        f"{FILESYSTEM_MULTITASK_UNIFORM_KIND}:"
        f"{sampling_seed}:{pool_epoch}:"
    ).encode("ascii")
    return tuple(
        sorted(
            range(pool_size),
            key=lambda flat_index: (
                hashlib.sha256(
                    prefix + str(flat_index).encode("ascii")
                ).digest(),
                flat_index,
            ),
        )
    )


def uniform_multitask_route_for_position(
    global_data_idx: Any,
    *,
    sampling_seed: Any,
    local_task_count: Any,
) -> tuple[int, int, int]:
    """Map a stream position to one uniformly shuffled surface/source row."""

    normalized_global = _normalize_nonnegative_index(
        "global_data_idx",
        global_data_idx,
    )
    normalized_seed = _normalize_nonnegative_index(
        MULTITASK_SAMPLING_SEED_KEY,
        sampling_seed,
    )
    normalized_local_count = _normalize_nonnegative_index(
        MULTITASK_LOCAL_TASK_COUNT_KEY,
        local_task_count,
    )
    if normalized_local_count == 0:
        raise ProceduralIndexError(
            f"{MULTITASK_LOCAL_TASK_COUNT_KEY} must be positive"
        )
    pool_size = (
        len(FILESYSTEM_MULTITASK_SURFACE_ORDER) * normalized_local_count
    )
    pool_epoch, pool_offset = divmod(normalized_global, pool_size)
    flat_index = _uniform_multitask_pool_permutation(
        normalized_seed,
        pool_epoch,
        pool_size,
    )[pool_offset]
    surface_slot, local_data_idx = divmod(
        flat_index,
        normalized_local_count,
    )
    return normalized_global, surface_slot, local_data_idx


def validate_multitask_route_triplet(
    global_data_idx: Any,
    surface_slot: Any,
    local_data_idx: Any,
    *,
    route_kind: Any = None,
    sampling_seed: Any = None,
    local_task_count: Any = None,
) -> tuple[int, int, int]:
    """Validate and normalize one frozen multitask routing row."""

    normalized_global = _normalize_nonnegative_index(
        "global_data_idx",
        global_data_idx,
    )
    normalized_slot = _normalize_nonnegative_index(
        MULTITASK_SURFACE_SLOT_KEY,
        surface_slot,
    )
    normalized_local = _normalize_nonnegative_index(
        MULTITASK_LOCAL_DATA_INDEX_KEY,
        local_data_idx,
    )
    if route_kind == FILESYSTEM_MULTITASK_UNIFORM_KIND:
        expected = uniform_multitask_route_for_position(
            normalized_global,
            sampling_seed=sampling_seed,
            local_task_count=local_task_count,
        )
        if (normalized_slot, normalized_local) != expected[1:]:
            raise ProceduralIndexError(
                "uniform multitask route identity disagrees with the seeded "
                "global-index mapping: "
                f"global={normalized_global} expected_slot={expected[1]} "
                f"observed_slot={normalized_slot} expected_local={expected[2]} "
                f"observed_local={normalized_local}"
            )
        return normalized_global, normalized_slot, normalized_local
    if route_kind not in {None, FILESYSTEM_MULTITASK_KIND}:
        raise ProceduralIndexError(
            f"unsupported multitask route kind: {route_kind!r}"
        )
    if sampling_seed is not None or local_task_count is not None:
        raise ProceduralIndexError(
            "legacy task-balanced routes must not carry uniform-sampler fields"
        )
    cycle_index, cycle_offset = divmod(
        normalized_global,
        FILESYSTEM_MULTITASK_CYCLE_SIZE,
    )
    expected_slot, within_surface = divmod(
        cycle_offset,
        FILESYSTEM_MULTITASK_ROWS_PER_SURFACE,
    )
    expected_local = (
        cycle_index * FILESYSTEM_MULTITASK_ROWS_PER_SURFACE
        + within_surface
    )
    if normalized_slot != expected_slot or normalized_local != expected_local:
        raise ProceduralIndexError(
            "multitask route identity disagrees with the frozen global-index "
            "mapping: "
            f"global={normalized_global} expected_slot={expected_slot} "
            f"observed_slot={normalized_slot} expected_local={expected_local} "
            f"observed_local={normalized_local}"
        )
    return normalized_global, normalized_slot, normalized_local


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


def resolve_rollout_reset_index(handler: Any) -> int:
    """Return the explicit dataset index used for environment reset.

    ``item_id`` is an opaque source identity for coding datasets.  Reset must
    therefore fail closed when the rollout handler is missing its separately
    routed integer ``data_idx`` instead of parsing or falling back to item_id.
    A routed multitask row additionally carries the endpoint-local index.  The
    stream ``data_idx`` identifies the global source position, while the
    endpoint must reset using that row's local position within the selected
    surface.
    """

    if not hasattr(handler, "data_idx"):
        raise ProceduralIndexError(
            "rollout handler is missing the authoritative data_idx"
        )
    reset_field = (
        MULTITASK_LOCAL_DATA_INDEX_KEY
        if hasattr(handler, MULTITASK_LOCAL_DATA_INDEX_KEY)
        else "data_idx"
    )
    candidate = getattr(handler, reset_field)
    if isinstance(candidate, bool):
        raise ProceduralIndexError(
            f"rollout handler {reset_field} must not be bool"
        )
    try:
        resolved = operator.index(candidate)
    except TypeError as exc:
        raise ProceduralIndexError(
            f"rollout handler {reset_field} is not an integer: {candidate!r}"
        ) from exc
    if resolved < 0:
        raise ProceduralIndexError(
            f"rollout handler {reset_field} must be non-negative: {resolved}"
        )
    return int(resolved)


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
            source_pairing, boundary, prompt_family, evaluation_contract = (
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
            if (
                metadata.get("workspace_evaluation_contract")
                != evaluation_contract
            ):
                raise ProceduralIndexError(
                    "filesystem evaluation-contract metadata disagrees"
                )
            control = metadata.get("workspace_intervention_control")
            expected_arms = ["correct", "blank", "swapped", "no_workspace"]
            if surface == DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE:
                expected_arms = ["correct", "blank", "no_workspace"]
            elif surface == RECENCY_OVERRIDE_FILESYSTEM_SURFACE:
                expected_arms.insert(3, "stale")
            expected_source_state, expected_seed_contract = (
                FILESYSTEM_SEEDED_WORKSPACE_CONTRACTS.get(
                    surface,
                    ("policy_authored_workspace_only", "none"),
                )
            )
            if (
                not isinstance(control, Mapping)
                or control.get("allowed_arms") != expected_arms
                or control.get("boundary_session_index") != boundary
                or control.get("source_state")
                != expected_source_state
                or control.get("hidden_answer_injection") is not False
                or metadata.get("workspace_seed_contract", "none")
                != expected_seed_contract
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


@dataclass(frozen=True)
class TaskBalancedMultitaskIndexSource(ProceduralIndexSource):
    """Emit complete, equally sized task blocks for eight filesystem surfaces."""

    kind: str = FILESYSTEM_MULTITASK_KIND

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != FILESYSTEM_MULTITASK_KIND:
            raise ProceduralIndexError(
                f"unsupported multitask procedural kind: {self.kind!r}"
            )
        if self.tasks_per_orbit != FILESYSTEM_MULTITASK_CYCLE_SIZE:
            raise ProceduralIndexError(
                "filesystem multitask tasks_per_orbit must equal the frozen "
                f"cycle size {FILESYSTEM_MULTITASK_CYCLE_SIZE}"
            )
        if self.start_index % FILESYSTEM_MULTITASK_CYCLE_SIZE:
            raise ProceduralIndexError(
                "filesystem multitask start_index must begin at a complete "
                "task-balanced cycle"
            )

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
        cycle_index, cycle_offset = divmod(
            absolute_index,
            FILESYSTEM_MULTITASK_CYCLE_SIZE,
        )
        surface_slot, within_surface = divmod(
            cycle_offset,
            FILESYSTEM_MULTITASK_ROWS_PER_SURFACE,
        )
        local_data_idx = (
            cycle_index * FILESYSTEM_MULTITASK_ROWS_PER_SURFACE
            + within_surface
        )
        return {
            "item_id": f"{self.item_id_prefix}_m{surface_slot}_{absolute_index}",
            "data_idx": absolute_index,
            MULTITASK_SURFACE_SLOT_KEY: surface_slot,
            MULTITASK_LOCAL_DATA_INDEX_KEY: local_data_idx,
            "extra_info": {"index": absolute_index},
        }

    @property
    def required_local_task_count(self) -> int:
        """Return the exclusive local index bound every routed server needs."""

        global_stop = self.start_index + self.task_count
        complete_cycles, remainder = divmod(
            global_stop,
            FILESYSTEM_MULTITASK_CYCLE_SIZE,
        )
        if remainder:
            raise ProceduralIndexError(
                "filesystem multitask stream must end at a complete cycle"
            )
        return complete_cycles * FILESYSTEM_MULTITASK_ROWS_PER_SURFACE

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "kind": self.kind,
                "task_balanced": True,
                "surface_order": list(FILESYSTEM_MULTITASK_SURFACE_ORDER),
                "route_orbit_sizes": list(
                    FILESYSTEM_MULTITASK_ROUTE_ORBIT_SIZES
                ),
                "rows_per_surface_per_cycle": (
                    FILESYSTEM_MULTITASK_ROWS_PER_SURFACE
                ),
                "cycle_size": FILESYSTEM_MULTITASK_CYCLE_SIZE,
                "required_local_task_count": self.required_local_task_count,
            }
        )
        return metadata

    def validate_server_metadatas(
        self,
        server_metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        if len(server_metadatas) != len(FILESYSTEM_MULTITASK_SURFACE_ORDER):
            raise ProceduralIndexError(
                "filesystem multitask requires one server metadata record for "
                f"each of {len(FILESYSTEM_MULTITASK_SURFACE_ORDER)} surfaces"
            )
        for slot, (metadata, expected_surface, route_orbit_size) in enumerate(
            zip(
                server_metadatas,
                FILESYSTEM_MULTITASK_SURFACE_ORDER,
                FILESYSTEM_MULTITASK_ROUTE_ORBIT_SIZES,
            )
        ):
            if not isinstance(metadata, Mapping):
                raise ProceduralIndexError(
                    f"filesystem multitask route {slot} metadata must be a mapping"
                )
            if metadata.get("surface") != expected_surface:
                raise ProceduralIndexError(
                    "filesystem multitask route order drifted: "
                    f"slot={slot} expected={expected_surface!r} "
                    f"observed={metadata.get('surface')!r}"
                )
            provider = metadata.get("provider")
            if not isinstance(provider, Mapping):
                raise ProceduralIndexError(
                    f"filesystem multitask route {slot} is missing provider metadata"
                )
            route_source = ProceduralIndexSource(
                task_count=provider.get("task_count"),
                provider_mode=self.provider_mode,
                tasks_per_orbit=route_orbit_size,
                start_index=0,
                item_id_prefix=self.item_id_prefix,
            )
            route_source.validate_server_metadata(metadata)
            if route_source.task_count < self.required_local_task_count:
                raise ProceduralIndexError(
                    "filesystem multitask route cannot cover the frozen stream: "
                    f"slot={slot} surface={expected_surface!r} "
                    f"required_local_task_count={self.required_local_task_count} "
                    f"server_task_count={route_source.task_count}"
                )

    def training_identity(
        self,
        *,
        server_metadata: Sequence[Mapping[str, Any]],
        train_batch_size: int,
    ) -> dict[str, Any]:
        self.validate_training_batch_size(train_batch_size)
        self.validate_server_metadatas(server_metadata)
        return {
            "schema": PROCEDURAL_STREAM_IDENTITY_SCHEMA,
            "index_source": self.metadata(),
            "training_geometry": {
                "train_batch_size": train_batch_size,
                "shuffle": False,
                "drop_last": True,
                "tasks_per_orbit": self.tasks_per_orbit,
                "task_balanced": True,
            },
            "server_metadata": [
                _canonical_json_mapping(
                    metadata,
                    field=f"server metadata route {slot}",
                )
                for slot, metadata in enumerate(server_metadata)
            ],
        }


@dataclass(frozen=True)
class UniformMultitaskIndexSource(ProceduralIndexSource):
    """Sample independent certified rows with equal surface probability.

    The underlying providers may organize rows into factual/counterfactual
    semantic orbits for certification and held-out analysis.  Those semantic
    relationships do not constrain this learner-side sampler: an orbit member
    is just another independently sampled row.
    """

    kind: str = FILESYSTEM_MULTITASK_UNIFORM_KIND
    sampling_seed: int = FILESYSTEM_MULTITASK_DEFAULT_SAMPLING_SEED
    local_task_count: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != FILESYSTEM_MULTITASK_UNIFORM_KIND:
            raise ProceduralIndexError(
                f"unsupported uniform multitask procedural kind: {self.kind!r}"
            )
        if self.tasks_per_orbit != 1:
            raise ProceduralIndexError(
                "uniform multitask stream positions are independent rows; "
                "tasks_per_orbit must be 1"
            )
        if (
            isinstance(self.sampling_seed, bool)
            or not isinstance(self.sampling_seed, int)
            or self.sampling_seed < 0
        ):
            raise ProceduralIndexError(
                "uniform multitask sampling_seed must be a non-negative integer"
            )
        if (
            isinstance(self.local_task_count, bool)
            or not isinstance(self.local_task_count, int)
            or self.local_task_count <= 0
        ):
            raise ProceduralIndexError(
                "uniform multitask local_task_count must be a positive integer"
            )
    def validate_training_batch_size(self, batch_size: int) -> None:
        if batch_size != FILESYSTEM_MULTITASK_FIXED_BATCH_SIZE:
            raise ProceduralIndexError(
                "filesystem uniform multitask train_batch_size is frozen at "
                f"{FILESYSTEM_MULTITASK_FIXED_BATCH_SIZE}, got {batch_size!r}"
            )
        super().validate_training_batch_size(batch_size)

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
        _, surface_slot, local_data_idx = uniform_multitask_route_for_position(
            absolute_index,
            sampling_seed=self.sampling_seed,
            local_task_count=self.local_task_count,
        )
        return {
            "item_id": f"{self.item_id_prefix}_u{surface_slot}_{absolute_index}",
            "data_idx": absolute_index,
            MULTITASK_SURFACE_SLOT_KEY: surface_slot,
            MULTITASK_LOCAL_DATA_INDEX_KEY: local_data_idx,
            MULTITASK_ROUTE_KIND_KEY: self.kind,
            MULTITASK_SAMPLING_SEED_KEY: self.sampling_seed,
            MULTITASK_LOCAL_TASK_COUNT_KEY: self.local_task_count,
            "extra_info": {"index": absolute_index},
        }

    @property
    def required_local_task_count(self) -> int:
        return self.local_task_count

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "kind": self.kind,
                "task_balanced_in_expectation": True,
                "surface_order": list(FILESYSTEM_MULTITASK_SURFACE_ORDER),
                "source_semantic_orbit_sizes": list(
                    FILESYSTEM_MULTITASK_ROUTE_ORBIT_SIZES
                ),
                "sampling_contract": (
                    "uniform_independent_source_rows_without_replacement_v2"
                ),
                "sampling_unit": "independent_certified_source_row",
                "counterfactual_members_are_independent_rows": True,
                "window_coverage_required": False,
                "counterfactual_window_coverage_required": False,
                "coverage_audit": "posthoc_distribution_only",
                "sampling_seed": self.sampling_seed,
                "local_task_count": self.local_task_count,
                "source_pool_size": (
                    len(FILESYSTEM_MULTITASK_SURFACE_ORDER)
                    * self.local_task_count
                ),
                "orbit_members_coupled": False,
                "fixed_train_batch_size": (
                    FILESYSTEM_MULTITASK_FIXED_BATCH_SIZE
                ),
                "required_local_task_count": self.required_local_task_count,
            }
        )
        return metadata

    def validate_server_metadatas(
        self,
        server_metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        if len(server_metadatas) != len(FILESYSTEM_MULTITASK_SURFACE_ORDER):
            raise ProceduralIndexError(
                "filesystem multitask requires one server metadata record for "
                f"each of {len(FILESYSTEM_MULTITASK_SURFACE_ORDER)} surfaces"
            )
        for slot, (metadata, expected_surface, route_orbit_size) in enumerate(
            zip(
                server_metadatas,
                FILESYSTEM_MULTITASK_SURFACE_ORDER,
                FILESYSTEM_MULTITASK_ROUTE_ORBIT_SIZES,
            )
        ):
            if not isinstance(metadata, Mapping):
                raise ProceduralIndexError(
                    f"filesystem multitask route {slot} metadata must be a mapping"
                )
            if metadata.get("surface") != expected_surface:
                raise ProceduralIndexError(
                    "filesystem multitask route order drifted: "
                    f"slot={slot} expected={expected_surface!r} "
                    f"observed={metadata.get('surface')!r}"
                )
            provider = metadata.get("provider")
            if not isinstance(provider, Mapping):
                raise ProceduralIndexError(
                    f"filesystem multitask route {slot} is missing provider metadata"
                )
            route_source = ProceduralIndexSource(
                task_count=provider.get("task_count"),
                provider_mode=self.provider_mode,
                tasks_per_orbit=route_orbit_size,
                start_index=0,
                item_id_prefix=self.item_id_prefix,
            )
            route_source.validate_server_metadata(metadata)
            if route_source.task_count < self.required_local_task_count:
                raise ProceduralIndexError(
                    "filesystem multitask route cannot cover the uniform source "
                    "pool: "
                    f"slot={slot} surface={expected_surface!r} "
                    f"required_local_task_count={self.required_local_task_count} "
                    f"server_task_count={route_source.task_count}"
                )

    def training_identity(
        self,
        *,
        server_metadata: Sequence[Mapping[str, Any]],
        train_batch_size: int,
    ) -> dict[str, Any]:
        self.validate_training_batch_size(train_batch_size)
        self.validate_server_metadatas(server_metadata)
        return {
            "schema": PROCEDURAL_STREAM_IDENTITY_SCHEMA,
            "index_source": self.metadata(),
            "training_geometry": {
                "train_batch_size": train_batch_size,
                "shuffle": False,
                "drop_last": True,
                "tasks_per_orbit": 1,
                "task_balanced_in_expectation": True,
                "orbit_members_coupled": False,
                "counterfactual_window_coverage_required": False,
                "coverage_audit": "posthoc_distribution_only",
                "fixed_compute_budget": True,
            },
            "server_metadata": [
                _canonical_json_mapping(
                    metadata,
                    field=f"server metadata route {slot}",
                )
                for slot, metadata in enumerate(server_metadata)
            ],
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
    kind = raw.get("kind")
    source_class = {
        FILESYSTEM_MULTITASK_KIND: TaskBalancedMultitaskIndexSource,
        FILESYSTEM_MULTITASK_UNIFORM_KIND: UniformMultitaskIndexSource,
    }.get(kind, ProceduralIndexSource)
    return source_class(
        task_count=raw.get("task_count"),
        provider_mode=raw.get("provider_mode"),
        tasks_per_orbit=raw.get(
            "tasks_per_orbit",
            (
                FILESYSTEM_MULTITASK_CYCLE_SIZE
                if source_class is TaskBalancedMultitaskIndexSource
                else 1 if source_class is UniformMultitaskIndexSource else 2
            ),
        ),
        start_index=raw.get("start_index", 0),
        item_id_prefix=raw.get("item_id_prefix", "agentmemory"),
        **(
            {"kind": raw.get("kind")}
            if source_class is TaskBalancedMultitaskIndexSource
            else {}
        ),
        **(
            {
                "kind": kind,
                "sampling_seed": raw.get(
                    "sampling_seed",
                    FILESYSTEM_MULTITASK_DEFAULT_SAMPLING_SEED,
                ),
                "local_task_count": raw.get("local_task_count"),
            }
            if source_class is UniformMultitaskIndexSource
            else {}
        ),
    )
