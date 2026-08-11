#!/usr/bin/env python3
"""Verify an eight-slot LiteResearcher resident endpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping


EXPECTED_SURFACE = "agentmemory_literesearcher_stage1_rag_only_v1"
EXPECTED_BACKEND = "literesearcher_frozen_search_page_backend_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--coverage-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runtime-source-id", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def parse_probe_indices(raw: str) -> list[int]:
    try:
        indices = [int(value) for value in raw.split(",") if value != ""]
    except ValueError as exc:
        raise ValueError("probe indices must be comma-separated integers") from exc
    if len(indices) != 8:
        raise ValueError("the eight-slot endpoint probe requires exactly 8 indices")
    if any(index < 0 for index in indices):
        raise ValueError("probe indices must be nonnegative")
    if len(set(indices)) != len(indices):
        raise ValueError("probe indices must be distinct")
    return indices


def load_probe_tasks(
    path: Path,
    indices: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("LiteResearcher coverage manifest must be an object")
    train = manifest.get("train")
    if not isinstance(train, list) or len(train) != 64:
        raise ValueError("LiteResearcher coverage manifest requires 64 train rows")
    try:
        tasks = [train[index] for index in indices]
    except IndexError as exc:
        raise ValueError("probe index is outside the 64-row train coverage") from exc
    required = {
        "index",
        "question",
        "targets",
        "mask_url",
        "public_url",
        "page_title",
        "resolved_url",
    }
    for task in tasks:
        if not isinstance(task, dict) or not required.issubset(task):
            raise ValueError("LiteResearcher probe task is missing required fields")
        targets = task["targets"]
        if not isinstance(targets, list) or not targets or not all(
            isinstance(value, str) and value.strip() for value in targets
        ):
            raise ValueError("LiteResearcher probe task requires answer targets")
    return manifest, tasks


class Endpoint:
    def __init__(self, base_url: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                f"LiteResearcher {method} {path} failed: HTTP {exc.code}: {detail}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(
                f"LiteResearcher {method} {path} returned non-object JSON"
            )
        return value


def parallel_map(function: Callable[[Any], Any], values: list[Any]) -> list[Any]:
    with ThreadPoolExecutor(max_workers=len(values)) as executor:
        return list(executor.map(function, values))


def require_idle(metadata: Mapping[str, Any], *, label: str) -> None:
    for key in ("active_environment_count", "active_workspace_count"):
        if int(metadata[key]) != 0:
            raise AssertionError(f"{label} {key} is not zero: {metadata[key]!r}")


def require_workspace_root_empty(path: Path, *, label: str) -> None:
    residue = sorted(str(child) for child in path.iterdir())
    if residue:
        raise AssertionError(f"{label} workspace residue remains: {residue!r}")


def close_slots(endpoint: Endpoint, slot_ids: list[int]) -> list[str]:
    errors: list[str] = []
    for slot_id in slot_ids:
        try:
            endpoint.request("POST", "close", {"id": slot_id})
        except Exception as exc:  # Preserve every cleanup failure for diagnosis.
            errors.append(f"slot {slot_id}: {type(exc).__name__}: {exc}")
    return errors


def shell_action(command: str) -> str:
    return "shell_command " + json.dumps(
        {"command": command, "workdir": ".", "timeout_ms": 120000},
        separators=(",", ":"),
    )


def tool_action(name: str, arguments: Mapping[str, Any]) -> str:
    return "<tool_call>" + json.dumps(
        {"name": name, "arguments": dict(arguments)},
        ensure_ascii=True,
        separators=(",", ":"),
    ) + "</tool_call>"


def assert_common_metadata(
    metadata: Mapping[str, Any],
    *,
    manifest_sha256: str,
    runtime_source_id: str,
    run_id: str,
) -> None:
    assert metadata["surface"] == EXPECTED_SURFACE
    assert metadata["domain_id"] == "literesearcher"
    assert metadata["split"] == "train"
    assert int(metadata["task_count"]) == 64
    assert metadata["manifest_sha256"] == manifest_sha256
    assert metadata["backend"]["backend_contract"] == EXPECTED_BACKEND
    assert metadata["backend"]["coverage_manifest_sha256"] == manifest_sha256
    assert metadata["reward_contract"] == "terminal_answer_only_binary_v1"
    assert metadata["workspace_tool_contract"] == (
        "codex_shell_command_apply_patch_v1"
    )
    assert metadata["compaction_contract"] == (
        "task_neutral_client_replace_messages_v1"
    )
    assert metadata["compaction_counts_as_env_step"] is True
    assert metadata["compaction_calls_backend"] is False
    service = metadata["service"]
    assert service["schema"] == "agentmemory_service_identity_v1"
    assert service["runtime_source_id"] == runtime_source_id
    assert service["instance_run_id"] == run_id


def assert_receipt(
    value: Mapping[str, Any],
    *,
    action_kind: str,
    done: bool = False,
    reward: float = 0.0,
) -> None:
    assert bool(value["done"]) is done
    assert float(value["reward"]) == reward
    info = value["info"]
    assert info["domain_id"] == "literesearcher"
    assert info["formal_schema_version"] == "agentmemory_formal_step_v3"
    submission = info["action_submission"]
    if action_kind in {"search", "visit"}:
        assert submission["tool"] == action_kind
        assert int(info["wrapper_evidence"]["native_environment_call_count"]) == 1
    elif action_kind == "workspace":
        assert submission["kind"] == "workspace"
        assert submission["op"] == "SHELL_COMMAND"
        assert info["wrapper_evidence"]["workspace_op"] == "SHELL_COMMAND"
        assert float(info["wrapper_evidence"]["workspace_reward"]) == 0.0
        assert int(info["wrapper_evidence"]["native_environment_call_count"]) == 0
    elif action_kind == "answer":
        assert submission["kind"] == "answer"
        assert info["wrapper_evidence"]["terminal_answer_only"] is True
        assert info["wrapper_evidence"]["answer_correct"] is (reward == 1.0)
    else:
        raise AssertionError(f"unsupported receipt action kind: {action_kind}")


def main() -> None:
    args = parse_args()
    indices = parse_probe_indices(args.indices)
    manifest_path = args.coverage_manifest.resolve()
    manifest, tasks = load_probe_tasks(manifest_path, indices)
    manifest_sha256 = str(manifest["manifest_sha256"])
    endpoint = Endpoint(args.base_url, args.timeout_seconds)
    workspace_root = args.workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata_before = endpoint.request("GET", "metadata")
    assert_common_metadata(
        metadata_before,
        manifest_sha256=manifest_sha256,
        runtime_source_id=args.expected_runtime_source_id,
        run_id=args.expected_run_id,
    )
    require_idle(metadata_before, label="before")
    require_workspace_root_empty(workspace_root, label="before")

    slot_ids: list[int] = []
    markers: list[str] = []
    metadata_active: dict[str, Any] | None = None
    try:
        for _ in indices:
            created = endpoint.request("POST", "create")
            slot_ids.append(int(created["id"]))
        assert len(set(slot_ids)) == len(indices)
        pairs = list(zip(slot_ids, indices, strict=True))
        reset = parallel_map(
            lambda pair: endpoint.request(
                "POST", "reset", {"id": pair[0], "data_idx": pair[1]}
            ),
            pairs,
        )
        for task, value in zip(tasks, reset, strict=True):
            assert value["observation"] == task["question"]
            assert bool(value["done"]) is False
            assert float(value["reward"]) == 0.0
            assert int(value["info"]["source_data_idx"]) == int(task["index"])

        markers = [
            hashlib.sha256(f"slot={slot}:index={index}".encode()).hexdigest()
            for slot, index in pairs
        ]
        writes = parallel_map(
            lambda item: endpoint.request(
                "POST",
                "step",
                {
                    "id": item[0],
                    "action": shell_action(
                        "mkdir -p .agent_memory && printf %s "
                        + json.dumps(item[1])
                        + " > .agent_memory/MEMORY.md"
                    ),
                },
            ),
            list(zip(slot_ids, markers, strict=True)),
        )
        for value in writes:
            assert_receipt(value, action_kind="workspace")

        reads = parallel_map(
            lambda slot_id: endpoint.request(
                "POST",
                "step",
                {
                    "id": slot_id,
                    "action": shell_action("cat .agent_memory/MEMORY.md"),
                },
            ),
            slot_ids,
        )
        for marker, value in zip(markers, reads, strict=True):
            assert_receipt(value, action_kind="workspace")
            assert marker in value["observation"]
            assert sum(other in value["observation"] for other in markers) == 1

        searches = parallel_map(
            lambda item: endpoint.request(
                "POST",
                "step",
                {
                    "id": item[0],
                    "action": tool_action("search", {"query": item[1]["page_title"]}),
                },
            ),
            list(zip(slot_ids, tasks, strict=True)),
        )
        for task, value in zip(tasks, searches, strict=True):
            assert_receipt(value, action_kind="search")
            observation = json.loads(value["observation"])
            assert observation["tool"] == "search"
            urls = [record["url"] for record in observation["results"]]
            assert task["public_url"] in urls
            assert task["mask_url"] not in value["observation"]
            assert task["resolved_url"] not in value["observation"]

        visits = parallel_map(
            lambda item: endpoint.request(
                "POST",
                "step",
                {
                    "id": item[0],
                    "action": tool_action(
                        "visit",
                        {
                            "url": item[1]["public_url"],
                            "goal": item[1]["question"],
                            "page": 1,
                        },
                    ),
                },
            ),
            list(zip(slot_ids, tasks, strict=True)),
        )
        for task, value in zip(tasks, visits, strict=True):
            assert_receipt(value, action_kind="visit")
            observation = json.loads(value["observation"])
            assert observation["tool"] == "visit"
            assert observation["page"]["url"] == task["public_url"]
            assert int(observation["page"]["page"]) == 1
            assert task["mask_url"] not in value["observation"]
            assert task["resolved_url"] not in value["observation"]

        answers = parallel_map(
            lambda item: endpoint.request(
                "POST",
                "step",
                {"id": item[0], "action": f"<answer>{item[1]['targets'][0]}</answer>"},
            ),
            list(zip(slot_ids, tasks, strict=True)),
        )
        for value in answers:
            assert_receipt(value, action_kind="answer", done=True, reward=1.0)

        details = parallel_map(
            lambda slot_id: endpoint.request("GET", f"detail?id={slot_id}"),
            slot_ids,
        )
        for task, value in zip(tasks, details, strict=True):
            assert bool(value["done"]) is True
            assert float(value["reward"]) == 1.0
            assert int(value["info"]["source_data_idx"]) == int(task["index"])

        metadata_active = endpoint.request("GET", "metadata")
        assert int(metadata_active["active_environment_count"]) == len(indices)
        assert int(metadata_active["active_workspace_count"]) == len(indices)
        assert metadata_active["service"]["fingerprint_sha256"] == (
            metadata_before["service"]["fingerprint_sha256"]
        )
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors = close_slots(endpoint, slot_ids)
        if cleanup_errors:
            message = "LiteResearcher slot cleanup failed: " + "; ".join(cleanup_errors)
            if active_error is not None:
                active_error.add_note(message)
            else:
                raise RuntimeError(message)

    metadata_after = endpoint.request("GET", "metadata")
    require_idle(metadata_after, label="after")
    require_workspace_root_empty(workspace_root, label="after")
    assert metadata_after["service"]["fingerprint_sha256"] == (
        metadata_before["service"]["fingerprint_sha256"]
    )

    evidence = {
        "schema": "agentmemory_literesearcher_resident_endpoint_probe_v1",
        "status": "pass",
        "base_url": args.base_url,
        "indices": indices,
        "source_data_indices": [int(task["index"]) for task in tasks],
        "task_ids": [f"stage1:{int(task['index']):05d}" for task in tasks],
        "public_urls": [str(task["public_url"]) for task in tasks],
        "slot_ids": slot_ids,
        "marker_sha256": markers,
        "manifest_sha256": manifest_sha256,
        "runtime_source_id": args.expected_runtime_source_id,
        "metadata_before": metadata_before,
        "metadata_active": metadata_active,
        "metadata_after": metadata_after,
        "receipt_counts": {
            "workspace": len(indices) * 2,
            "search": len(indices),
            "visit": len(indices),
            "answer": len(indices),
        },
        "workspace_residue_count": 0,
    }
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "output": str(args.output)}))


if __name__ == "__main__":
    main()
