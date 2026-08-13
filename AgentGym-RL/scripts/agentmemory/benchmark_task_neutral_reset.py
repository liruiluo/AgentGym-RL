#!/usr/bin/env python3
"""Benchmark ordered task-neutral reset with formal rank-local topology."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

from agentenv.controller import bind_initial_policy_context
from agentenv.envs import SwesmithEnvClient

from verl.utils.agentgym.task_neutral_parallel_reset import (
    reset_task_neutral_policy_contexts,
)
from verl.utils.agent_dataset.procedural_index import (
    resolve_rollout_reset_index,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _tree_digest(root: Path) -> str:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        stat_result = path.lstat()
        if path.is_symlink():
            entries.append(
                {
                    "kind": "symlink",
                    "path": relative,
                    "target": os.readlink(path),
                    "mode": stat_result.st_mode & 0o7777,
                }
            )
        elif path.is_dir():
            entries.append(
                {
                    "kind": "directory",
                    "path": relative,
                    "mode": stat_result.st_mode & 0o7777,
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "kind": "file",
                    "path": relative,
                    "mode": stat_result.st_mode & 0o7777,
                    "size": stat_result.st_size,
                    "sha256": _sha256_bytes(path.read_bytes()),
                }
            )
        else:
            raise RuntimeError(f"unsupported workspace entry: {path}")
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def _rank_reset(
    *,
    rank: int,
    data_indices: list[int],
    env_addr: str,
    detail_token: str,
    max_workers: int,
    client_timeout_seconds: int,
) -> dict[str, Any]:
    clients: list[SwesmithEnvClient] = []
    try:
        clients = [
            SwesmithEnvClient(
                env_server_base=env_addr,
                data_len=None,
                timeout=client_timeout_seconds,
            )
            for _ in data_indices
        ]
        handlers = [
            SimpleNamespace(
                messages=client.conversation_start,
                item_id=f"swesmith_{data_idx}",
                data_idx=data_idx,
                done=False,
            )
            for client, data_idx in zip(clients, data_indices, strict=True)
        ]
        result = reset_task_neutral_policy_contexts(
            handlers,
            clients,
            resolve_reset_index=resolve_rollout_reset_index,
            bind_initial_policy_context=bind_initial_policy_context,
            max_workers=max_workers,
        )
        rows: list[dict[str, Any]] = []
        for handler, client, messages in zip(
            handlers, clients, result.policy_messages, strict=True
        ):
            detail = client.detail(private_token=detail_token)
            reset_evidence = detail["evidence"][0]
            workspace = detail["workspace"]
            policy_root = Path(workspace["policy_root"])
            profile = reset_evidence["profile"]
            hidden_tests = tuple(
                dict.fromkeys(
                    (*profile["f2p_test_paths"], *profile["p2p_test_paths"])
                )
            )
            hidden_root = Path(workspace["episode_root"]) / "private/pristine-tests"
            hidden_digests = {
                path: _sha256_bytes((hidden_root / path).read_bytes())
                for path in hidden_tests
            }
            rows.append(
                {
                    "rank": rank,
                    "data_idx": int(handler.data_idx),
                    "item_id": handler.item_id,
                    "instance_id": detail["instance_id"],
                    "physical_index": detail["physical_index"],
                    "observation_sha256": _sha256_text(client.observe()),
                    "policy_messages_sha256": _sha256_text(
                        json.dumps(messages, sort_keys=True, separators=(",", ":"))
                    ),
                    "workspace_tree_sha256": _tree_digest(policy_root),
                    "hidden_test_sha256": hidden_digests,
                    "model_uid": workspace["model_uid"],
                    "model_gid": workspace["model_gid"],
                    "workspace_contract": reset_evidence["workspace_contract"],
                    "server_workspace_initial": reset_evidence["workspace_initial"],
                    "sandbox": reset_evidence["sandbox"],
                }
            )
        return {
            "rank": rank,
            "data_indices": data_indices,
            "wall_seconds": result.wall_seconds,
            "item_timings": [asdict(item) for item in result.item_timings],
            "rows": rows,
        }
    finally:
        for client in clients:
            try:
                client.close()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-addr", required=True)
    parser.add_argument("--detail-token-file", type=Path, required=True)
    parser.add_argument("--first-data-idx", type=int, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--max-workers-per-rank", type=int, required=True)
    parser.add_argument("--client-timeout-seconds", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.world_size <= 0:
        raise ValueError("count and world size must be positive")
    if args.count % args.world_size:
        raise ValueError("count must divide evenly across ranks")
    per_rank = args.count // args.world_size
    all_indices = list(range(args.first_data_idx, args.first_data_idx + args.count))
    shards = [
        all_indices[rank * per_rank : (rank + 1) * per_rank]
        for rank in range(args.world_size)
    ]
    detail_token = args.detail_token_file.read_text(encoding="ascii").strip()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.world_size) as executor:
        futures = [
            executor.submit(
                _rank_reset,
                rank=rank,
                data_indices=shard,
                env_addr=args.env_addr,
                detail_token=detail_token,
                max_workers=args.max_workers_per_rank,
                client_timeout_seconds=args.client_timeout_seconds,
            )
            for rank, shard in enumerate(shards)
        ]
        rank_results = [future.result() for future in futures]
    rank_results.sort(key=lambda value: value["rank"])
    rows = [row for rank in rank_results for row in rank["rows"]]
    rows.sort(key=lambda value: value["data_idx"])
    if [row["data_idx"] for row in rows] != all_indices:
        raise RuntimeError("reset result does not cover the requested indices")
    payload = {
        "schema": "task_neutral_reset_benchmark_v1",
        "env_addr": args.env_addr,
        "first_data_idx": args.first_data_idx,
        "count": args.count,
        "world_size": args.world_size,
        "items_per_rank": per_rank,
        "max_workers_per_rank": args.max_workers_per_rank,
        "wall_seconds": time.perf_counter() - started,
        "rank_results": rank_results,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in (
        "schema", "count", "world_size", "items_per_rank",
        "max_workers_per_rank", "wall_seconds",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
