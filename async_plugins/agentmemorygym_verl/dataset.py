"""Schedule-preserving dataset adapter for upstream veRL AgentLoop rollouts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

from .env_client import create_env_client
from .routes import (
    canonical_policy_framing_sha256,
    normalize_policy_framing,
    route_registry_from_agentgym_config,
)


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    try:
        normalized = int(value)
        exact = float(value) == float(normalized)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if not exact or normalized < 0:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return normalized


class AMGTrajectoryDataset(RLHFDataset):
    """Attach wrapper-owned policy framing to the frozen AMG task schedule.

    The JSONL schedule remains authoritative for ``item_id``/``data_idx`` and
    ordering. Task observations are fetched only after AgentLoop reset; they
    are never materialized into the dataset or leaked into the prompt file.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        agentgym_config = self.config.get("agentgym")
        if agentgym_config is None:
            raise ValueError("AMGTrajectoryDataset requires data.agentgym config")
        self._route_registry = route_registry_from_agentgym_config(agentgym_config)
        self._policy_framing_by_route: dict[str, tuple[dict[str, str], ...]] = {}
        for route in self._route_registry.routes:
            client = create_env_client(route.client_config)
            try:
                policy_framing_method = getattr(client, "policy_framing", None)
                if not callable(policy_framing_method):
                    raise TypeError(
                        f"AMG route {route.route_id!r} wrapper must expose "
                        "policy_framing()"
                    )
                policy_framing = normalize_policy_framing(policy_framing_method())
                observed_digest = canonical_policy_framing_sha256(policy_framing)
                if (
                    route.policy_framing_sha256 is not None
                    and observed_digest != route.policy_framing_sha256
                ):
                    raise ValueError(
                        f"AMG route {route.route_id!r} policy framing sha256 "
                        "does not match its immutable route registry"
                    )
                self._policy_framing_by_route[route.route_id] = tuple(
                    dict(message) for message in policy_framing
                )
            finally:
                client.close()

    def maybe_filter_out_long_prompts(self, dataframe=None):
        # Schedule rows contain opaque task IDs, not task observations. The
        # AgentLoop performs the exact post-reset prompt-width check.
        return self.dataframe if dataframe is None else dataframe

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = dict(self.dataframe[item])
        route = self._route_registry.resolve_row(row)
        if "item_id" not in row or "data_idx" not in row:
            raise ValueError("AMG schedule row must contain item_id and data_idx")
        data_idx = _nonnegative_int(
            row["data_idx"], field="AMG schedule data_idx"
        )
        row["data_idx"] = data_idx
        row["item_id"] = str(row["item_id"])
        row["route_id"] = route.route_id
        row["raw_prompt"] = deepcopy(self._policy_framing_by_route[route.route_id])
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        configured_agent = row.get("agent_name")
        if (
            configured_agent is not None
            and str(configured_agent) != self._route_registry.agent_name
        ):
            raise ValueError(
                "AMG schedule row agent_name must select the shared task-neutral loop"
            )
        row["agent_name"] = self._route_registry.agent_name

        extra_info = dict(row.get("extra_info") or {})
        top_index = row.get("index")
        nested_index = extra_info.get("index")
        if top_index is None and nested_index is None:
            if len(self._route_registry.route_ids) > 1:
                raise ValueError(
                    "AMG multi-environment schedule row is missing a global index"
                )
            schedule_index = data_idx
        elif top_index is None:
            schedule_index = _nonnegative_int(
                nested_index, field="AMG schedule global index"
            )
        elif nested_index is None:
            schedule_index = _nonnegative_int(
                top_index, field="AMG schedule global index"
            )
        else:
            normalized_top = _nonnegative_int(
                top_index, field="AMG schedule global index"
            )
            normalized_nested = _nonnegative_int(
                nested_index, field="AMG schedule global index"
            )
            if normalized_top != normalized_nested:
                raise ValueError(
                    "AMG schedule global index drift: "
                    f"row={normalized_top} extra_info={normalized_nested}"
                )
            schedule_index = normalized_top
        extra_info["index"] = schedule_index
        extra_info["route_id"] = route.route_id
        if route.route_attestation_sha256 is not None:
            if "route_attestation_sha256" in extra_info:
                if (
                    extra_info["route_attestation_sha256"]
                    != route.route_attestation_sha256
                ):
                    raise ValueError(
                        "AMG schedule route_attestation_sha256 drift from registry"
                    )
            else:
                extra_info["route_attestation_sha256"] = (
                    route.route_attestation_sha256
                )
        if self._route_registry.sha256 is not None:
            if "route_registry_sha256" in extra_info:
                if extra_info["route_registry_sha256"] != self._route_registry.sha256:
                    raise ValueError(
                        "AMG schedule route_registry_sha256 drift from registry"
                    )
            else:
                extra_info["route_registry_sha256"] = self._route_registry.sha256
        row["extra_info"] = extra_info
        row["index"] = schedule_index

        configured_source = row.get("data_source")
        if (
            len(self._route_registry.route_ids) > 1
            and configured_source is not None
            and str(configured_source) != route.route_id
        ):
            raise ValueError(
                "AMG multi-environment schedule data_source must equal route_id"
            )
        row["data_source"] = route.route_id
        row.setdefault("tools_kwargs", {})
        row.setdefault("interaction_kwargs", {})
        return row
