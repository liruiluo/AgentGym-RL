#!/usr/bin/env python3
"""Run the frozen OpenMLE services against the verified held-out manifest.

The historical endpoint supervisor only accepts training contracts
(`gate1`/`formal100`).  Held-out evaluation must not masquerade as either one,
so this adapter validates the held-out publication explicitly and then reuses
only the donor's process, credential, Unix-socket, and cleanup primitives.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, TextIO

import openmle_runtime_base as runtime


RUNTIME_SCHEMA = "camg_openmle_fast_heldout_runtime_manifest_v1"
PUBLIC_MANIFEST_SCHEMA = "openmle_fast_public_manifest_v1"
PRIVATE_MANIFEST_SCHEMA = "openmle_fast_fullpool_private_grader_manifest_v1"
HELDOUT_ROLE = "heldout"
SERVER_SELECTED_PREFIXES = (
    "inner:agentenv-openmle-fast/",
)
SERVER_SELECTED_FILES = frozenset(
    {
        "inner:agentenv/agentenv/controller/policy_turn.py",
        "inner:agentenv/agentenv/controller/types.py",
        "inner:agentenv/agentenv/envs/filesystem_checkpoint.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not an absolute regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not an absolute regular file: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"{label} has a blank line at {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{label} line {line_number} is invalid JSON") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"{label} line {line_number} is not an object")
        records.append(record)
    return records


def _digest(value: Any, label: str) -> str:
    rendered = str(value or "").lower()
    if len(rendered) != 64 or any(c not in "0123456789abcdef" for c in rendered):
        raise RuntimeError(f"{label} is not a SHA-256 digest")
    return rendered


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _workspace_root(outer_root: Path, inner_root: Path) -> Path:
    """Return the parent expected by the imported runtime donor.

    The evaluator registry binds each git checkout at its repository root.
    ``build_service_environments`` is a narrowly copied donor that still
    receives the historical workspace parent and appends ``AgentGym-RL``.
    Validate that relationship explicitly instead of silently appending a
    second ``AgentGym-RL`` component.
    """

    outer = outer_root.resolve()
    inner = inner_root.resolve()
    if outer.name != "AgentGym-RL":
        raise RuntimeError("outer source root must be the AgentGym-RL checkout")
    if inner != outer / "AgentGym":
        raise RuntimeError("inner source root must be the outer checkout's AgentGym submodule")
    return outer.parent


def _bound_file(
    runtime_manifest_path: Path,
    runtime_document: Mapping[str, Any],
    field: str,
    supplied_path: Path,
) -> None:
    binding = runtime_document.get(field)
    if not isinstance(binding, Mapping):
        raise RuntimeError(f"runtime manifest lacks {field} binding")
    bound_path = Path(str(binding.get("path", "")))
    if not bound_path.is_absolute():
        bound_path = runtime_manifest_path.parent / bound_path
    if bound_path.resolve() != supplied_path.resolve():
        raise RuntimeError(f"runtime manifest binds a different {field} file")
    if _positive_int(binding.get("bytes"), f"{field} byte count") != supplied_path.stat().st_size:
        raise RuntimeError(f"{field} byte count drifted")
    if _digest(binding.get("sha256"), f"{field} digest") != _sha256(supplied_path):
        raise RuntimeError(f"{field} SHA-256 drifted")


def _verify_selected_server_source(document: Mapping[str, Any], inner_root: Path) -> None:
    selected = document["runtime_source"]["selected_files"]
    checked = 0
    for name, expected in selected.items():
        if not (
            name.startswith(SERVER_SELECTED_PREFIXES)
            or name in SERVER_SELECTED_FILES
        ):
            continue
        prefix, relative = name.split(":", 1)
        if prefix != "inner":
            raise RuntimeError(f"unexpected endpoint source owner: {name}")
        path = (inner_root / relative).resolve()
        try:
            path.relative_to(inner_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"selected endpoint source escaped inner root: {name}") from exc
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"selected endpoint source is missing: {name}")
        if _sha256(path) != _digest(expected, f"selected source {name}"):
            raise RuntimeError(f"selected endpoint source digest drifted: {name}")
        checked += 1
    if checked < 10:
        raise RuntimeError("source lock did not bind the complete OpenMLE endpoint surface")


def _verify_public_bindings(
    public_manifest: Mapping[str, Any],
    bindings_path: Path,
    routing_path: Path,
    private_manifest: Mapping[str, Any],
    task_count: int,
) -> None:
    public_records = public_manifest.get("records")
    private_records = private_manifest.get("records")
    if not isinstance(public_records, list) or len(public_records) != task_count:
        raise RuntimeError("held-out public manifest record count drifted")
    if not isinstance(private_records, list) or not private_records:
        raise RuntimeError("private grader manifest has no records")
    bindings = _jsonl(bindings_path, "private grader bindings")
    routing = _jsonl(routing_path, "held-out routing")
    if len(bindings) != task_count or len(routing) != task_count:
        raise RuntimeError("held-out routing/binding cardinality drifted")

    public_by_id = {str(record.get("task_id")): record for record in public_records}
    private_by_id = {str(record.get("task_id")): record for record in private_records}
    binding_by_id = {str(record.get("task_id")): record for record in bindings}
    if len(public_by_id) != task_count or len(binding_by_id) != task_count:
        raise RuntimeError("held-out task identifiers are not unique")
    if set(public_by_id) != set(binding_by_id):
        raise RuntimeError("public manifest and private bindings cover different tasks")
    if not set(public_by_id).issubset(private_by_id):
        raise RuntimeError("private grader manifest does not cover every held-out task")

    seen_indices: set[int] = set()
    for route_record in routing:
        index = route_record.get("data_idx")
        extra = route_record.get("extra_info")
        if (
            type(index) is not int
            or not isinstance(extra, Mapping)
            or extra.get("index") != index
            or extra.get("role") != HELDOUT_ROLE
            or route_record.get("item_id") != f"openmle_fast_{index}"
        ):
            raise RuntimeError("held-out routing identity drifted")
        task_id = str(extra.get("task_id", ""))
        public = public_records[index] if 0 <= index < task_count else None
        if not isinstance(public, Mapping) or public.get("task_id") != task_id:
            raise RuntimeError("held-out routing no longer follows manifest order")
        if extra.get("source_family") != public.get("source_family"):
            raise RuntimeError("held-out routing source family drifted")
        seen_indices.add(index)
    if seen_indices != set(range(task_count)):
        raise RuntimeError("held-out routing indices are not contiguous")

    for task_id, binding in binding_by_id.items():
        public = public_by_id[task_id]
        private = private_by_id[task_id]
        for key in (
            "grader_binding",
            "grader_binding_sha256",
            "answer_sha256",
            "metric_sha256",
        ):
            if binding.get(key) != private.get(key):
                raise RuntimeError(f"private binding {key} drifted for {task_id}")
        for key in (
            "grader_binding",
            "grader_binding_sha256",
            "package_identity_sha256",
            "task_spec_sha256",
        ):
            if public.get(key) != private.get(key):
                raise RuntimeError(f"public/private identity {key} drifted for {task_id}")


def validate_and_overlay(args: argparse.Namespace, contract_module) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.runtime_schema != RUNTIME_SCHEMA:
        raise RuntimeError("launcher expected-runtime schema drifted")
    runtime_document = _json(args.runtime_manifest, "OpenMLE held-out runtime manifest")
    if runtime_document.get("schema") != RUNTIME_SCHEMA:
        raise RuntimeError("OpenMLE held-out runtime manifest schema mismatch")
    if runtime_document.get("status") != "ready":
        raise RuntimeError("OpenMLE held-out runtime manifest is not ready")
    if runtime_document.get("heldout_evaluation_run") is not False:
        raise RuntimeError("held-out runtime manifest was already consumed")
    if _positive_int(runtime_document.get("task_count"), "runtime task count") != args.task_count:
        raise RuntimeError("runtime task count differs from verified endpoint count")

    _bound_file(args.runtime_manifest, runtime_document, "heldout_manifest", args.heldout_manifest)
    _bound_file(
        args.runtime_manifest,
        runtime_document,
        "private_grader_bindings",
        args.private_grader_bindings,
    )
    _bound_file(args.runtime_manifest, runtime_document, "routing", args.routing)

    source_lock_sha = _sha256(args.source_lock)
    source_bindings = runtime_document.get("source", {}).get("source_locks")
    if not isinstance(source_bindings, list) or not any(
        isinstance(binding, Mapping)
        and _digest(binding.get("sha256"), "source lock digest") == source_lock_sha
        for binding in source_bindings
    ):
        raise RuntimeError("source lock is not bound by the held-out runtime manifest")
    document = contract_module.load_source_lock(
        args.source_lock, require_final_runtime=True
    )
    source = runtime_document.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError("runtime manifest source binding is missing")
    for key in ("outer_commit", "inner_commit"):
        if source.get(key) != document["runtime_source"][key]:
            raise RuntimeError(f"runtime manifest {key} differs from source lock")

    heldout_document = _json(args.heldout_manifest, "OpenMLE held-out manifest")
    if (
        heldout_document.get("schema") != PUBLIC_MANIFEST_SCHEMA
        or heldout_document.get("role") != HELDOUT_ROLE
        or _positive_int(heldout_document.get("task_count"), "held-out task count")
        != args.task_count
    ):
        raise RuntimeError("OpenMLE public held-out manifest contract drifted")
    heldout_sha = _sha256(args.heldout_manifest)
    source_heldout = document["integration"]["manifests"][HELDOUT_ROLE]
    if (
        source_heldout.get("role") != HELDOUT_ROLE
        or source_heldout.get("task_count") != args.task_count
        or source_heldout.get("sha256") != heldout_sha
    ):
        raise RuntimeError("source lock held-out binding differs from evaluator asset")
    if heldout_document.get("release_revision") != document["runtime_source"]["openmle_tasks_revision"]:
        raise RuntimeError("held-out release revision differs from source lock")

    integration = document["integration"]
    private_binding = integration["private_manifest"]
    private_manifest_path = Path(integration["pod_root"]) / private_binding["relpath"]
    private_document = _json(private_manifest_path, "installed private grader manifest")
    if (
        private_document.get("schema") != PRIVATE_MANIFEST_SCHEMA
        or _sha256(private_manifest_path) != private_binding["sha256"]
        or private_document.get("runtime_digest") != document["exact_runtime"]["runtime_digest"]
    ):
        raise RuntimeError("installed private grader manifest is not the frozen runtime")
    _verify_public_bindings(
        heldout_document,
        args.private_grader_bindings,
        args.routing,
        private_document,
        args.task_count,
    )
    _verify_selected_server_source(document, args.inner_root)

    overlay = copy.deepcopy(document)
    overlay["launch_contracts"]["heldout_eval"] = {
        "manifest_role": HELDOUT_ROLE,
        "task_count": args.task_count,
    }
    overlay["integration"]["manifests"][HELDOUT_ROLE] = {
        "role": HELDOUT_ROLE,
        "task_count": args.task_count,
        "source_family_count": heldout_document.get("source_family_count"),
        "relpath": str(args.heldout_manifest),
        "sha256": heldout_sha,
    }
    receipt = {
        "schema": "camg_openmle_fast_heldout_preflight_v1",
        "status": "pass",
        "role": HELDOUT_ROLE,
        "task_count": args.task_count,
        "runtime_manifest": {"path": str(args.runtime_manifest), "sha256": _sha256(args.runtime_manifest)},
        "heldout_manifest": {"path": str(args.heldout_manifest), "sha256": heldout_sha},
        "private_grader_bindings": {"path": str(args.private_grader_bindings), "sha256": _sha256(args.private_grader_bindings)},
        "routing": {"path": str(args.routing), "sha256": _sha256(args.routing)},
        "source_lock": {"path": str(args.source_lock), "sha256": source_lock_sha},
        "private_manifest": {"path": str(private_manifest_path), "sha256": _sha256(private_manifest_path)},
        "endpoint_server_selected_file_count": sum(
            1
            for name in document["runtime_source"]["selected_files"]
            if name.startswith(SERVER_SELECTED_PREFIXES) or name in SERVER_SELECTED_FILES
        ),
    }
    return overlay, receipt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--contract-tool", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-schema", required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--private-grader-bindings", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-start-ticks", type=int, required=True)
    parser.add_argument("--outer-root", type=Path, required=True)
    parser.add_argument("--inner-root", type=Path, required=True)
    parser.add_argument("--private-command", nargs="+", required=True)
    parser.add_argument("--public-command", nargs="+", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    contract_module = runtime.load_contract_module(args.contract_tool)
    runtime.set_parent_death_signal()
    if contract_module.process_start_ticks(args.parent_pid) != args.parent_start_ticks:
        raise RuntimeError("endpoint supervisor parent identity drifted at startup")
    if not 1024 <= args.port <= 65535:
        raise ValueError("endpoint port is outside the unprivileged TCP range")
    if not args.owner or not args.run_id:
        raise ValueError("endpoint ownership identity is empty")
    document, preflight = validate_and_overlay(args, contract_module)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    contract_module.atomic_write_json(
        args.run_dir / "openmle-heldout-preflight.json", preflight
    )

    workspace_root = _workspace_root(args.outer_root, args.inner_root)
    private_environment, public_environment = runtime.build_service_environments(
        document,
        "heldout_eval",
        args.run_dir,
        workspace_root,
        args.inner_root,
        args.port,
        args.owner,
        args.run_id,
    )
    private_socket = Path(private_environment["OPENMLE_FAST_GRADER_ENDPOINT"])
    public_episodes = Path(public_environment["OPENMLE_FAST_EPISODES_ROOT"])
    credential = Path(private_environment["OPENMLE_FAST_GRADER_CREDENTIAL"])
    credential_value = runtime.write_credential(credential)
    runtime.write_forbidden_canaries(
        credential.parent / "forbidden-canaries.json",
        credential_value,
        private_environment,
    )
    processes = []
    logs: list[TextIO] = []
    records: list[dict[str, Any]] = []
    stopping = False
    unexpected_exit: Optional[str] = None

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        private_process, private_log, private_record = runtime.start_private_grader(
            args.private_command,
            private_environment,
            args.run_dir,
            contract_module,
            args.owner,
            args.run_id,
        )
        processes.append(private_process)
        logs.append(private_log)
        records.append(private_record)
        public_process, public_log, public_record, metadata = runtime.start_public_endpoint(
            args.public_command,
            public_environment,
            args.run_dir,
            contract_module,
            args.owner,
            args.run_id,
            HELDOUT_ROLE,
            _sha256(args.heldout_manifest),
            args.task_count,
        )
        processes.append(public_process)
        logs.append(public_log)
        records.append(public_record)
        contract_module.atomic_write_json(
            args.run_dir / "endpoints/ready.json",
            {
                "schema": "camg_openmle_fast_heldout_endpoint_supervision_v1",
                "status": "ready",
                "process_owner": args.owner,
                "run_id": args.run_id,
                "parent_pid": args.parent_pid,
                "parent_start_ticks": args.parent_start_ticks,
                "startup_order": list(runtime.PROCESS_ROLES),
                "processes": records,
                "metadata": metadata,
                "preflight_sha256": _sha256(args.run_dir / "openmle-heldout-preflight.json"),
            },
        )
        while not stopping:
            if contract_module.process_start_ticks(args.parent_pid) != args.parent_start_ticks:
                unexpected_exit = "exact parent identity disappeared"
                stopping = True
                break
            for process, record in zip(processes, records):
                if process.poll() is not None:
                    unexpected_exit = f"{record['role']} exited before supervisor shutdown"
                    stopping = True
                    break
            time.sleep(0.5)
    finally:
        exit_codes: dict[str, int] = {}
        termination_requested: dict[str, bool] = {}
        for process, record in reversed(list(zip(processes, records))):
            exit_code, requested = runtime.stop_exact_process(
                process, record, contract_module
            )
            exit_codes[record["role"]] = exit_code
            termination_requested[record["role"]] = requested
        for log_handle in logs:
            log_handle.close()
        if credential.exists() and not credential.is_symlink():
            credential.unlink()
        socket_cleanup_error: Optional[str] = None
        try:
            runtime.cleanup_private_socket_path(private_socket)
        except Exception as exc:  # cleanup receipt must retain the exact error
            socket_cleanup_error = f"{type(exc).__name__}: {exc}"
        episodes_cleanup_error: Optional[str] = None
        try:
            runtime.cleanup_public_episodes_path(public_episodes)
        except Exception as exc:  # cleanup receipt must retain the exact error
            episodes_cleanup_error = f"{type(exc).__name__}: {exc}"
        cleanup_processes = [
            {
                **record,
                "alive": contract_module.exact_process_alive(
                    record["pid"], record["start_ticks"]
                ),
                "exit_code": exit_codes.get(record["role"], 127),
                "termination_requested": termination_requested.get(
                    record["role"], False
                ),
            }
            for record in records
        ]
        clean = (
            len(cleanup_processes) == 2
            and all(not record["alive"] for record in cleanup_processes)
            and all(
                runtime.managed_exit_is_clean(
                    record["exit_code"], record["termination_requested"]
                )
                for record in cleanup_processes
            )
            and socket_cleanup_error is None
            and episodes_cleanup_error is None
        )
        contract_module.atomic_write_json(
            args.run_dir / "endpoints/cleanup.json",
            {
                "schema": "camg_openmle_fast_heldout_endpoint_cleanup_v1",
                "status": "pass" if clean and unexpected_exit is None else "fail",
                "process_owner": args.owner,
                "run_id": args.run_id,
                "unexpected_exit": unexpected_exit,
                "private_socket_path": str(private_socket),
                "private_socket_cleanup_error": socket_cleanup_error,
                "public_episodes_root": str(public_episodes),
                "public_episodes_cleanup_error": episodes_cleanup_error,
                "processes": cleanup_processes,
                "residual_pids": [
                    record["pid"] for record in cleanup_processes if record["alive"]
                ],
            },
        )
    return 0 if unexpected_exit is None and clean else 1


if __name__ == "__main__":
    sys.exit(main())
