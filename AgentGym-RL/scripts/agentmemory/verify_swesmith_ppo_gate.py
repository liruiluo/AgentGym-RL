#!/usr/bin/env python3
"""Verify the first real eight-rank SWE-smith PPO update."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any


STEP_SCHEMA = "task_neutral_policy_step_v1"
AUDIT_SCHEMA = "agentmemory_swesmith_private_episode_audit_v1"
NATIVE_EVENT = "native_action"
COMPACTION_EVENT = "context_compaction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--endpoint-probe", type=Path, required=True)
    parser.add_argument("--metadata-after", type=Path, required=True)
    parser.add_argument("--global-step", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"{label} is not finite: {value!r}")
    return result


def verify_row_evidence(
    payload: dict[str, Any], expected_indices: set[int]
) -> dict[str, Any]:
    evidence = payload["row_evidence"]
    assert evidence["task_name"] == "swesmith"
    schema = evidence["schema"]
    if schema == "agentmemory_formal_step_records_v1":
        rows = evidence["rows"]
        assert rows
        assert payload["formal_step_records"] == rows
        indices = [int(row["parent_index"]) for row in rows]
        assert set(indices) == expected_indices
        for row, index in zip(rows, indices):
            assert row["schema_version"] == STEP_SCHEMA
            assert row["item_id"] == f"swesmith_{index}"
        return {
            "schema": schema,
            "dataset_indices": sorted(set(indices)),
            "row_count": len(rows),
        }
    if schema == "generic_task_dataset_rows_v1":
        assert evidence["index_field"] in {
            "rollout_data_indices",
            "data_idx",
            "index",
        }
        indices = [int(value) for value in evidence["dataset_indices"]]
        assert set(indices) == expected_indices
        return {
            "schema": schema,
            "dataset_indices": sorted(set(indices)),
            "row_count": len(indices),
        }
    raise AssertionError(f"unexpected row evidence schema: {schema!r}")


def verify_readback(
    run_dir: Path, global_step: int, expected_indices: set[int]
) -> dict[str, Any]:
    path = run_dir / "diagnostics" / f"formal_update_readback_step{global_step}.json"
    payload = load_json(path)
    assert int(payload["global_step"]) == global_step
    assert payload["role"] == "same_batch_post_optimizer_readback"
    result: dict[str, Any] = {
        "row_evidence": verify_row_evidence(payload, expected_indices)
    }
    for role in ("actor", "critic"):
        role_payload = payload[role]
        max_abs_delta = finite(
            role_payload["summary"]["max_abs_delta"],
            f"{role}.max_abs_delta",
        )
        delta_l2 = finite(
            role_payload["parameter_delta_l2"],
            f"{role}.parameter_delta_l2",
        )
        changed = int(
            role_payload["parameter_probe"]["parameter_probe_changed_count"]
        )
        assert max_abs_delta > 0
        assert delta_l2 > 0
        assert changed > 0
        result[role] = {
            "max_abs_delta": max_abs_delta,
            "parameter_delta_l2": delta_l2,
            "parameter_probe_changed_count": changed,
        }
    return result


def verify_audits(
    audit_root: Path,
    endpoint_probe: dict[str, Any],
    expected_indices: set[int],
) -> dict[str, Any]:
    probe_audit_ids = set(str(value) for value in endpoint_probe["audit_ids"])
    trainer_audits: list[dict[str, Any]] = []
    for path in sorted(audit_root.glob("episode-*.json")):
        payload = load_json(path)
        if str(payload["audit_id"]) in probe_audit_ids:
            continue
        assert payload["schema"] == AUDIT_SCHEMA
        assert payload["close_reason"] == "client_close"
        assert payload["done"] is True
        assert float(payload["reward"]) in {0.0, 1.0}
        assert payload["grade"] is not None
        assert int(payload["step_count"]) > 0
        trainer_audits.append(payload)
    assert len(trainer_audits) == len(expected_indices)
    indices = [int(value["data_idx"]) for value in trainer_audits]
    assert set(indices) == expected_indices
    assert len(indices) == len(set(indices))
    assert len({int(value["slot_id"]) for value in trainer_audits}) == len(indices)
    return {
        "audit_count": len(trainer_audits),
        "dataset_indices": sorted(indices),
        "resolved_count": sum(float(value["reward"]) == 1.0 for value in trainer_audits),
        "audit_ids": sorted(str(value["audit_id"]) for value in trainer_audits),
    }


def main() -> None:
    args = parse_args()
    assert args.global_step == 1
    assert args.train_batch_size == 8
    expected_indices = set(range(args.train_batch_size))
    endpoint_probe = load_json(args.endpoint_probe)
    assert endpoint_probe["status"] == "pass"
    assert set(int(value) for value in endpoint_probe["indices"]) == expected_indices

    batch_path = (
        args.run_dir
        / "diagnostics"
        / f"ppo_batch_step{args.global_step}_post_adv.json"
    )
    payload = load_json(batch_path)
    assert int(payload["global_step"]) == args.global_step
    assert payload["stage"] == "post_adv"
    assert payload["prompt_attestation_passed"] is True
    assert int(payload["valid_rows"]) > 0

    rows_by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    event_counts: Counter[str] = Counter()
    action_kind_counts: Counter[str] = Counter()
    nonzero_advantage_rows = 0
    nonzero_return_rows = 0
    truncated_rows = []

    for row in payload["rows"]:
        if not row["ppo_valid_sample"]:
            continue
        parent_index = int(row["parent_index"])
        assert parent_index in expected_indices
        record = row["formal_step_record"]
        assert record["schema_version"] == STEP_SCHEMA
        assert int(record["parent_index"]) == parent_index
        assert int(record["task_round"]) == int(row["task_round"])
        assert record["item_id"] == f"swesmith_{parent_index}"
        assert record["action"] == row["agentmemory_action_text"]
        assert record["action_submission"]["raw_policy_output"] == record["action"]
        assert record["generation_token_ids_are_exact"] is True
        assert record["backend_token_ids_are_exact"] is True
        assert record["response_token_ids"]
        assert int(record["response_token_count"]) == len(record["response_token_ids"])
        assert int(row["response_mask_sum"]) == len(record["response_token_ids"])
        assert record["generation_prompt_digest"] == row[
            "agentmemory_generation_prompt_digest"
        ]
        assert record["packed_prompt_digest"] == row[
            "agentmemory_packed_prompt_digest"
        ]
        assert abs(
            finite(row["score_sum"], "score_sum")
            - finite(record["immediate_reward"], "immediate_reward")
        ) < 1e-6
        finite(row["old_logprob_mean"], "old_logprob_mean")
        nonzero_advantage_rows += int(row.get("adv_nonzero", 0) > 0)
        nonzero_return_rows += int(row.get("return_nonzero", 0) > 0)
        if record["truncated"]:
            truncated_rows.append(
                {
                    "parent_index": parent_index,
                    "task_round": int(record["task_round"]),
                    "response_token_count": int(record["response_token_count"]),
                }
            )
        rows_by_parent[parent_index].append(record)

    assert set(rows_by_parent) == expected_indices
    assert not truncated_rows
    assert nonzero_advantage_rows > 0
    assert nonzero_return_rows > 0

    workspace_ids: dict[int, int] = {}
    for parent_index, records in sorted(rows_by_parent.items()):
        records.sort(key=lambda value: int(value["task_round"]))
        assert [int(value["task_round"]) for value in records] == list(
            range(1, len(records) + 1)
        )
        previous: dict[str, int] | None = None
        for record in records:
            evidence = record["wrapper_evidence"]
            event = str(evidence["event"])
            assert event in {NATIVE_EVENT, COMPACTION_EVENT}
            event_counts[event] += 1
            workspace_id = int(evidence["workspace_continuity_id"])
            workspace_ids.setdefault(parent_index, workspace_id)
            assert workspace_ids[parent_index] == workspace_id
            counts = {
                "native_before": int(evidence["native_call_count_before"]),
                "native_after": int(evidence["native_call_count_after"]),
                "policy_before": int(evidence["policy_step_before"]),
                "policy_after": int(evidence["policy_step_after"]),
                "context_before": int(evidence["context_epoch_before"]),
                "context_after": int(evidence["context_epoch_after"]),
            }
            assert counts["policy_after"] == counts["policy_before"] + 1
            assert counts["policy_after"] == int(record["task_round"])
            if previous is not None:
                assert counts["native_before"] == previous["native_after"]
                assert counts["policy_before"] == previous["policy_after"]
                assert counts["context_before"] == previous["context_after"]
            previous = counts
            if event == COMPACTION_EVENT:
                assert counts["native_after"] == counts["native_before"]
                assert counts["context_after"] == counts["context_before"] + 1
                assert record["context_transition"]["operation"] == "replace_messages"
                assert record["action_submission"]["submitted_action"] is None
                assert record["action_submission"]["parser_status"] == (
                    "policy_context_compaction"
                )
                continue

            assert counts["native_after"] == counts["native_before"] + 1
            assert counts["context_after"] == counts["context_before"]
            action_kind = str(record["env_info_after"].get("action_kind", ""))
            assert action_kind in {
                "shell_command",
                "apply_patch",
                "final",
                "parser_error",
                "policy_turn_horizon",
            }
            action_kind_counts[action_kind] += 1
            if action_kind in {"shell_command", "apply_patch", "parser_error"}:
                assert float(record["immediate_reward"]) == 0.0

    assert len(set(workspace_ids.values())) == args.train_batch_size
    assert event_counts[NATIVE_EVENT] > 0
    assert event_counts[COMPACTION_EVENT] > 0
    assert action_kind_counts["shell_command"] + action_kind_counts["apply_patch"] > 0

    metadata_after = load_json(args.metadata_after)
    for key in (
        "active_slot_count",
        "active_environment_count",
        "active_workspace_count",
    ):
        assert int(metadata_after[key]) == 0

    readback = verify_readback(args.run_dir, args.global_step, expected_indices)
    audits = verify_audits(args.audit_root, endpoint_probe, expected_indices)
    evidence = {
        "schema": "agentmemory_swesmith_ppo_gate_attestation_v1",
        "status": "pass",
        "global_step": args.global_step,
        "train_batch_size": args.train_batch_size,
        "valid_rows": int(payload["valid_rows"]),
        "parent_row_counts": {
            str(parent): len(records) for parent, records in sorted(rows_by_parent.items())
        },
        "event_counts": dict(event_counts),
        "action_kind_counts": dict(action_kind_counts),
        "workspace_continuity_ids": workspace_ids,
        "truncated_rows": truncated_rows,
        "readback": readback,
        "private_audits": audits,
        "metadata_after": metadata_after,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
