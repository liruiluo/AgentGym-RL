#!/usr/bin/env python3
"""Certify private SWE-smith grading on the first formal schedule batches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import urllib.error
import urllib.request
from typing import Any

ARMS = ("gold", "wrong", "tamper")
EXPECTED_REWARD = {"gold": 1.0, "wrong": 0.0, "tamper": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--routing-file", type=Path, required=True)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--gate-steps", type=int, default=3)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detail-token-env", default="SWESMITH_DETAIL_TOKEN")
    parser.add_argument("--expected-outer-commit", required=True)
    parser.add_argument("--expected-inner-commit", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_schedule_indices(
    path: Path, *, train_batch_size: int, gate_steps: int
) -> list[int]:
    if train_batch_size <= 0 or gate_steps <= 0:
        raise ValueError("train batch size and gate steps must be positive")
    required = train_batch_size * gate_steps
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"routing row {line_number} is invalid JSON") from exc
        position = len(rows)
        if row.get("item_id") != f"swesmith_{position}":
            raise ValueError(f"routing row {line_number} has the wrong item_id")
        data_idx = row.get("data_idx")
        if not isinstance(data_idx, int) or isinstance(data_idx, bool) or data_idx < 0:
            raise ValueError(f"routing row {line_number} has an invalid data_idx")
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            raise ValueError(f"routing row {line_number} has invalid extra_info")
        if int(extra_info.get("index", -1)) != data_idx:
            raise ValueError(f"routing row {line_number} index does not match data_idx")
        if int(extra_info.get("schedule_position", -1)) != position:
            raise ValueError(f"routing row {line_number} has wrong schedule position")
        rows.append(row)
    if len(rows) < required:
        raise ValueError(f"routing file has {len(rows)} rows but {required} are required")
    indices = [int(row["data_idx"]) for row in rows[:required]]
    if len(set(indices)) != len(indices):
        raise ValueError("the certified schedule prefix contains duplicate data_idx values")
    return indices


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


def action_for(instance: dict[str, Any], arm: str) -> str:
    if arm == "gold":
        from agentenv_swesmith.patch_conversion import unified_to_codex_patch

        return "apply_patch\n" + unified_to_codex_patch(
            str(instance["patch"]), reverse=True
        )
    if arm == "wrong":
        return (
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Add File: agentmemory_wrong_probe.txt\n"
            "+This change does not fix the issue.\n"
            "*** End Patch"
        )
    if arm == "tamper":
        fail_to_pass = list(instance["FAIL_TO_PASS"])
        if not fail_to_pass:
            raise ValueError("tamper arm requires at least one FAIL_TO_PASS test")
        test_path = str(fail_to_pass[0]).split("::", 1)[0]
        command = (
            "printf 'def test_tampered(): pass\\n' > " + shlex.quote(test_path)
        )
        return "shell_command " + json.dumps(
            {"command": command, "workdir": "."}, separators=(",", ":")
        )
    raise ValueError(f"unsupported certificate arm: {arm}")


def run_arm(
    endpoint: Endpoint,
    dataset: Any,
    *,
    data_idx: int,
    arm: str,
) -> dict[str, Any]:
    slot_id: int | None = None
    errors: list[str] = []
    result: dict[str, Any] = {"arm": arm, "data_idx": data_idx, "errors": errors}
    try:
        slot_id = int(endpoint.request("POST", "create", {})["id"])
        reset = endpoint.request(
            "POST", "reset", {"id": slot_id, "data_idx": data_idx}
        )
        record = dataset[data_idx]
        action = endpoint.request(
            "POST",
            "step",
            {"id": slot_id, "action": action_for(dict(record.instance), arm)},
        )
        terminal = endpoint.request(
            "POST", "step", {"id": slot_id, "action": "Implemented the fix."}
        )
        detail = endpoint.request("GET", f"detail?id={slot_id}", private=True)
        grade = dict(detail.get("grade") or {})
        f2p = dict(grade.get("f2p_run") or {})
        full = dict(grade.get("full_run") or {})
        result.update(
            {
                "instance_id": record.instance_id,
                "repo": record.instance.get("repo"),
                "reset_done": reset.get("done"),
                "action_kind": dict(action.get("info") or {}).get("action_kind"),
                "reward": terminal.get("reward"),
                "episode_success": dict(terminal.get("info") or {}).get(
                    "episode_success"
                ),
                "resolution_status": grade.get("resolution_status"),
                "f2p_exit_code": f2p.get("exit_code"),
                "f2p_output_truncated": f2p.get("output_truncated"),
                "f2p_status_source": f2p.get("status_source"),
                "full_exit_code": full.get("exit_code"),
                "full_output_truncated": full.get("output_truncated"),
                "full_status_source": full.get("status_source"),
                "grader_error": grade.get("error"),
            }
        )
    except Exception as exc:  # Preserve job-local evidence, then fail globally.
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if slot_id is not None:
            try:
                endpoint.request("POST", "close", {"id": slot_id})
            except Exception as exc:
                errors.append(f"close {type(exc).__name__}: {exc}")
    return result


def result_passes(result: dict[str, Any]) -> bool:
    arm = str(result["arm"])
    if result.get("errors") or float(result.get("reward", -1.0)) != EXPECTED_REWARD[arm]:
        return False
    if result.get("grader_error") not in (None, ""):
        return False
    if arm == "gold":
        return (
            result.get("episode_success") is True
            and result.get("resolution_status") == "RESOLVED_FULL"
        )
    return result.get("episode_success") is False


def main() -> int:
    from agentenv_swesmith.dataset import SwesmithDataset

    args = parse_args()
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not arms or any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError(f"arms must be a unique nonempty subset of {ARMS!r}")
    detail_token = os.environ.get(args.detail_token_env, "").strip()
    if not detail_token:
        raise RuntimeError(f"private detail token is unset: {args.detail_token_env}")
    indices = load_schedule_indices(
        args.routing_file,
        train_batch_size=args.train_batch_size,
        gate_steps=args.gate_steps,
    )
    dataset = SwesmithDataset(args.dataset_manifest)
    if max(indices) >= len(dataset):
        raise ValueError("certified schedule data_idx exceeds dataset length")
    endpoint = Endpoint(args.base_url, args.timeout_seconds, detail_token)
    metadata_before = endpoint.request("GET", "metadata")
    source = metadata_before["runtime_source"]
    if source["outer_commit"] != args.expected_outer_commit:
        raise AssertionError("endpoint outer commit does not match the certificate")
    if source["inner_commit"] != args.expected_inner_commit:
        raise AssertionError("endpoint inner commit does not match the certificate")
    for key in ("active_slot_count", "active_environment_count", "active_workspace_count"):
        if int(metadata_before[key]) != 0:
            raise AssertionError(f"endpoint is not idle before certificate: {key}")

    jobs = [(data_idx, arm) for data_idx in indices for arm in arms]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_arm, endpoint, dataset, data_idx=index, arm=arm): (
                index,
                arm,
            )
            for index, arm in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (indices.index(int(row["data_idx"])), ARMS.index(str(row["arm"]))))

    metadata_after = endpoint.request("GET", "metadata")
    for key in ("active_slot_count", "active_environment_count", "active_workspace_count"):
        if int(metadata_after[key]) != 0:
            raise AssertionError(f"endpoint is not idle after certificate: {key}")
    failures = [row for row in results if not result_passes(row)]
    artifact = {
        "schema": "swesmith_fullpool_schedule_private_grader_certificate_v1",
        "status": "pass" if not failures else "fail",
        "runtime_source": {
            "outer_commit": args.expected_outer_commit,
            "inner_commit": args.expected_inner_commit,
        },
        "dataset_manifest": str(args.dataset_manifest),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "routing_file": str(args.routing_file),
        "routing_file_sha256": sha256_file(args.routing_file),
        "train_batch_size": args.train_batch_size,
        "gate_steps": args.gate_steps,
        "schedule_indices": indices,
        "arms": list(arms),
        "job_count": len(jobs),
        "pass_count": len(jobs) - len(failures),
        "failure_count": len(failures),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "job_count": len(jobs),
                "pass_count": len(jobs) - len(failures),
                "failure_count": len(failures),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
