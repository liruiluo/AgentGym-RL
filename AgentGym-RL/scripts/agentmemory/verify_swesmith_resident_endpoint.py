#!/usr/bin/env python3
"""Verify an eight-slot SWE-smith resident endpoint without grading tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any, Callable


EXPECTED_SCHEMA = "agentmemory_swesmith_native_episode_v1"
AUDIT_SCHEMA = "agentmemory_swesmith_private_episode_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detail-token-env", default="SWESMITH_DETAIL_TOKEN")
    parser.add_argument("--expected-outer-commit", required=True)
    parser.add_argument("--expected-inner-commit", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


class Endpoint:
    def __init__(self, base_url: str, timeout: int, detail_token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.detail_token = detail_token

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        private: bool = False,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if private:
            headers["X-SWESMITH-Detail-Token"] = self.detail_token
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                f"SWE-smith {method} {path} failed: HTTP {exc.code}: {detail}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"SWE-smith {method} {path} returned non-object JSON")
        return value


def parallel_map(function: Callable[[Any], Any], values: list[Any]) -> list[Any]:
    with ThreadPoolExecutor(max_workers=len(values)) as executor:
        return list(executor.map(function, values))


def require_idle(metadata: dict[str, Any], *, label: str) -> None:
    for key in (
        "active_slot_count",
        "active_environment_count",
        "active_workspace_count",
    ):
        if int(metadata[key]) != 0:
            raise AssertionError(f"{label} {key} is not zero: {metadata[key]!r}")


def main() -> None:
    args = parse_args()
    indices = [int(value) for value in args.indices.split(",") if value != ""]
    if indices != list(range(8)):
        raise ValueError("the first formal gate must bind exact indices 0..7")
    detail_token = os.environ.get(args.detail_token_env, "")
    if not detail_token:
        raise RuntimeError(f"private detail token is unset: {args.detail_token_env}")
    endpoint = Endpoint(args.base_url, args.timeout_seconds, detail_token)
    args.episodes_root = args.episodes_root.resolve()
    args.audit_root = args.audit_root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata_before = endpoint.request("GET", "metadata")
    assert metadata_before["schema"] == EXPECTED_SCHEMA
    assert metadata_before["tool_contract"] == "codex_shell_command_apply_patch_v1"
    assert metadata_before["tool_serialization"] == (
        "qwen35_native_single_function_v1"
    )
    assert metadata_before["reward_contract"] == "terminal_full_resolution_binary_v1"
    assert metadata_before["context_contract"] == "one_native_issue_continuous_episode_v1"
    assert metadata_before["private_audit_contract"] == AUDIT_SCHEMA
    runtime_source = metadata_before["runtime_source"]
    assert runtime_source["outer_commit"] == args.expected_outer_commit
    assert runtime_source["inner_commit"] == args.expected_inner_commit
    assert runtime_source["source_id"] == (
        f"{args.expected_outer_commit}_{args.expected_inner_commit}"
    )
    assert int(metadata_before["task_count"]) >= len(indices)
    require_idle(metadata_before, label="before")
    if any(args.episodes_root.iterdir()):
        raise AssertionError("episodes root is not empty before the isolation probe")

    created = parallel_map(lambda _: endpoint.request("POST", "create", {}), indices)
    slot_ids = [int(value["id"]) for value in created]
    assert len(set(slot_ids)) == len(indices)

    pairs = list(zip(slot_ids, indices, strict=True))
    reset = parallel_map(
        lambda pair: endpoint.request(
            "POST", "reset", {"id": pair[0], "data_idx": pair[1]}
        ),
        pairs,
    )
    for value in reset:
        assert value["done"] is False
        assert float(value["reward"]) == 0.0
        assert value["info"]["schema"] == EXPECTED_SCHEMA

    details = parallel_map(
        lambda slot_id: endpoint.request(
            "GET", f"detail?id={slot_id}", private=True
        ),
        slot_ids,
    )
    audit_ids = [value["audit_id"] for value in details]
    episode_roots = [Path(value["workspace"]["episode_root"]) for value in details]
    policy_roots = [Path(value["workspace"]["policy_root"]) for value in details]
    model_uids = [int(value["workspace"]["model_uid"]) for value in details]
    assert [int(value["data_idx"]) for value in details] == indices
    assert [int(value["slot_id"]) for value in details] == slot_ids
    assert len(set(audit_ids)) == len(indices)
    assert len(set(episode_roots)) == len(indices)
    assert len(set(policy_roots)) == len(indices)
    assert len(set(model_uids)) == len(indices)
    for episode_root, policy_root in zip(episode_roots, policy_roots, strict=True):
        assert episode_root.parent == args.episodes_root
        assert policy_root.parent == episode_root

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
                "action": "shell_command "
                + json.dumps(
                    {
                        "command": (
                            "mkdir -p .agent_memory && printf %s "
                            + json.dumps(item[1])
                            + " > .agent_memory/MEMORY.md"
                        ),
                        "workdir": ".",
                        "timeout_ms": 120000,
                    },
                    separators=(",", ":"),
                ),
            },
        ),
        list(zip(slot_ids, markers, strict=True)),
    )
    for value in writes:
        assert value["done"] is False
        assert float(value["reward"]) == 0.0
        assert value["info"]["action_kind"] == "shell_command"

    reads = parallel_map(
        lambda slot_id: endpoint.request(
            "POST",
            "step",
            {
                "id": slot_id,
                "action": "shell_command "
                + json.dumps(
                    {
                        "command": "cat .agent_memory/MEMORY.md",
                        "workdir": ".",
                        "timeout_ms": 120000,
                    },
                    separators=(",", ":"),
                ),
            },
        ),
        slot_ids,
    )
    for marker, value in zip(markers, reads, strict=True):
        assert value["done"] is False
        assert float(value["reward"]) == 0.0
        assert marker in value["observation"]
        assert sum(other in value["observation"] for other in markers) == 1

    metadata_active = endpoint.request("GET", "metadata")
    assert int(metadata_active["active_slot_count"]) == len(indices)
    assert int(metadata_active["active_environment_count"]) == len(indices)
    assert int(metadata_active["active_workspace_count"]) == len(indices)

    parallel_map(
        lambda slot_id: endpoint.request("POST", "close", {"id": slot_id}),
        slot_ids,
    )
    metadata_after = endpoint.request("GET", "metadata")
    require_idle(metadata_after, label="after")
    if any(args.episodes_root.iterdir()):
        raise AssertionError("episode workspace residue remains after close")

    audits = []
    for audit_id, slot_id, index, marker in zip(
        audit_ids, slot_ids, indices, markers, strict=True
    ):
        path = args.audit_root / f"episode-{audit_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == AUDIT_SCHEMA
        assert payload["audit_id"] == audit_id
        assert int(payload["data_idx"]) == index
        assert int(payload["slot_id"]) == slot_id
        assert payload["close_reason"] == "client_close"
        assert payload["done"] is False
        assert float(payload["reward"]) == 0.0
        assert int(payload["step_count"]) == 2
        assert payload["evidence"][1]["action"]["kind"] == "shell_command"
        assert payload["evidence"][2]["result"]["stdout"] == marker
        audits.append(str(path))

    evidence = {
        "schema": "agentmemory_swesmith_resident_endpoint_probe_v1",
        "status": "pass",
        "base_url": args.base_url,
        "indices": indices,
        "slot_ids": slot_ids,
        "audit_ids": audit_ids,
        "episode_roots": [str(path) for path in episode_roots],
        "policy_roots": [str(path) for path in policy_roots],
        "model_uids": model_uids,
        "audit_paths": audits,
        "metadata_before": metadata_before,
        "metadata_active": metadata_active,
        "metadata_after": metadata_after,
        "workspace_residue_count": 0,
    }
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "output": str(args.output)}))


if __name__ == "__main__":
    main()
