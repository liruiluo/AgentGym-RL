"""Schedule-preserving dataset adapter for upstream veRL AgentLoop rollouts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

from .env_client import create_env_client


class AMGTrajectoryDataset(RLHFDataset):
    """Attach wrapper-owned policy framing to the frozen AMG task schedule.

    The JSONL schedule remains authoritative for ``item_id``/``data_idx`` and
    ordering.  Task observations are fetched only after AgentLoop reset; they
    are never materialized into the dataset or leaked into the prompt file.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        agentgym_config = self.config.get("agentgym")
        if agentgym_config is None:
            raise ValueError("AMGTrajectoryDataset requires data.agentgym config")
        client = create_env_client(agentgym_config)
        try:
            policy_framing = client.policy_framing()
            if not isinstance(policy_framing, list) or not policy_framing:
                raise ValueError("AMG environment returned an empty policy framing")
            self._policy_framing = deepcopy(policy_framing)
        finally:
            client.close()

    def maybe_filter_out_long_prompts(self, dataframe=None):
        # Schedule rows contain opaque task IDs, not task observations.  The
        # AgentLoop performs the exact post-reset prompt-width check.
        return self.dataframe if dataframe is None else dataframe

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = dict(self.dataframe[item])
        if "item_id" not in row or "data_idx" not in row:
            raise ValueError("AMG schedule row must contain item_id and data_idx")
        raw_data_idx = row["data_idx"]
        if isinstance(raw_data_idx, bool):
            raise TypeError("AMG schedule data_idx must be an integer, not bool")
        try:
            data_idx = int(raw_data_idx)
            exact_data_idx = float(raw_data_idx) == float(data_idx)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"AMG schedule data_idx must be an integer, got {raw_data_idx!r}"
            ) from exc
        if not exact_data_idx or data_idx < 0:
            raise ValueError(
                f"AMG schedule data_idx must be an integer, got {raw_data_idx!r}"
            )
        row["data_idx"] = data_idx
        row["item_id"] = str(row["item_id"])
        row["raw_prompt"] = deepcopy(self._policy_framing)
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        extra_info = dict(row.get("extra_info") or {})
        raw_index = extra_info.get("index", data_idx)
        if isinstance(raw_index, bool):
            raise TypeError("AMG schedule index must be an integer, not bool")
        try:
            schedule_index = int(raw_index)
            exact_index = float(raw_index) == float(schedule_index)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"AMG schedule index must be an integer, got {raw_index!r}"
            ) from exc
        if not exact_index or schedule_index != data_idx:
            raise ValueError(
                "AMG schedule index differs from data_idx: "
                f"index={raw_index!r} data_idx={data_idx}"
            )
        extra_info["index"] = data_idx
        row["extra_info"] = extra_info
        row["index"] = data_idx
        row.setdefault("data_source", str(self.config.agentgym.task_name))
        row.setdefault("tools_kwargs", {})
        row.setdefault("interaction_kwargs", {})
        return row
