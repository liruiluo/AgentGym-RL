"""Run-scoped owner for CAMG native held-out evaluation.

The evaluator itself owns only task-neutral model serving and AgentLoop
sampling.  This module supplies the missing outer lifecycle: holder yield and
restore, four same-Pod loopback endpoints, an exact evaluator process lease,
and fail-closed cleanup evidence.  It intentionally reuses the mature
multitask orchestrator primitives instead of creating another process model.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from .heldout_eval import HeldoutEvalPlan, derive_eval_config, run_contract
from .heldout_eval_contract import (
    atomic_write_json,
    canonical_json_bytes,
    finalize_run_metrics,
    inspect_heldout_schedule,
    inspect_resume_state,
    sha256_bytes,
    sha256_file,
)
from .heldout_endpoints import (
    HeldoutEndpointSpec,
    load_heldout_endpoint_registry,
    probe_heldout_reset_identity,
)
from .multitask_orchestrator import (
    HolderLease,
    LocalBackend,
    OrchestratorError,
    ProcessLease,
    _signal_process_identity,
    _HolderHandle,
    assert_ports_available,
    load_holder_lease,
)
from .orchestrator_lifecycle import process_start_ticks
from .routes import load_route_registry

ORCHESTRATOR_SCHEMA = "camg_heldout_eval_orchestrator_contract_v1"
ATTEMPT_SCHEMA = "camg_heldout_eval_orchestrator_attempt_v1"
EXPECTED_EVALUATOR_SCRIPT = "run_amg_heldout_eval.py"
_ATTEMPT_SUFFIX = re.compile(r"\.attempt-(\d{6})$")
_INFERENCE_EXECUTABLES = frozenset(
    {
        "gcs_server",
        "raylet",
        "plasma_store_server",
        "vllm",
        "sglang",
    }
)
_INFERENCE_MODULE_PREFIXES = (
    "ray.",
    "sglang.",
    "vllm.",
)


@dataclass(frozen=True)
class EvalOrchestratorPlan:
    evaluation: HeldoutEvalPlan
    orchestration_dir: Path
    outer_root: Path
    inner_root: Path
    verl_root: Path
    evaluator_script: Path
    evaluator_script_sha256: str
    endpoint_registry_path: Path
    endpoint_registry_sha256: str
    endpoints: tuple[HeldoutEndpointSpec, ...]
    endpoint_report: Mapping[str, Any]
    holder_lease: HolderLease
    holder_lock_path: Path
    resolve_only: bool = False


@dataclass(frozen=True)
class EvalAttempt:
    index: int
    directory: Path
    runtime_plan: Any
    already_complete: bool
    owner_id: str


@dataclass(frozen=True)
class WatchParentLease:
    pid: int
    start_ticks: str
    process: Any
    log_handle: Any
    ready_path: Path
    receipt_path: Path


def _attempt_owner_id(eval_run_id: str, index: int) -> str:
    if not eval_run_id or any(character in eval_run_id for character in ("/", "\0", "\n", "\r")):
        raise OrchestratorError("held-out eval run id is unsafe for attempt ownership")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise OrchestratorError("held-out attempt index must be non-negative")
    return f"{eval_run_id}.attempt-{index:06d}"


def _absolute_directory(path: str | os.PathLike[str], *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise OrchestratorError(f"{field} must be an absolute non-symlink directory")
    return candidate.resolve()


def _absolute_regular(path: str | os.PathLike[str], *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise OrchestratorError(f"{field} must be an absolute regular file")
    return candidate.resolve()


def _verify_clean_git_source(root: Path, expected_commit: str, *, label: str) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise OrchestratorError(
            f"cannot verify {label} git source: {exc.output.strip()}"
        ) from exc
    if commit != expected_commit:
        raise OrchestratorError(
            f"{label} source commit mismatch: {commit} != {expected_commit}"
        )
    if status:
        raise OrchestratorError(f"{label} source tree is dirty: {status}")


def _verify_endpoint_task_counts(
    evaluation: HeldoutEvalPlan,
    endpoints: Sequence[HeldoutEndpointSpec],
) -> dict[str, int]:
    endpoint_counts = {endpoint.route_id: endpoint.task_count for endpoint in endpoints}
    if set(endpoint_counts) != set(evaluation.route_counts):
        raise OrchestratorError(
            "held-out endpoint route set differs from the eval schedule: "
            f"endpoints={endpoint_counts!r}, schedule={evaluation.route_counts!r}"
        )
    undersized = {
        route_id: {
            "endpoint": endpoint_counts[route_id],
            "scheduled": scheduled_count,
        }
        for route_id, scheduled_count in evaluation.route_counts.items()
        if endpoint_counts[route_id] < scheduled_count
    }
    if undersized:
        raise OrchestratorError(
            "held-out endpoint task pools cannot cover the eval schedule: "
            f"{undersized!r}"
        )
    return endpoint_counts


def load_eval_orchestrator_plan(
    *,
    evaluation: HeldoutEvalPlan,
    orchestration_dir: str | os.PathLike[str],
    outer_root: str | os.PathLike[str],
    inner_root: str | os.PathLike[str],
    verl_root: str | os.PathLike[str],
    evaluator_script: str | os.PathLike[str],
    expected_evaluator_script_sha256: str,
    endpoint_registry_path: str | os.PathLike[str],
    expected_endpoint_registry_sha256: str,
    holder_lease_path: str | os.PathLike[str],
    expected_holder_lease_sha256: str,
    holder_lock_path: str | os.PathLike[str],
    resolve_only: bool = False,
) -> EvalOrchestratorPlan:
    """Validate every source/runtime identity before any holder mutation."""

    root = _absolute_directory(outer_root, field="evaluator outer root")
    inner = _absolute_directory(inner_root, field="evaluator inner root")
    verl = _absolute_directory(verl_root, field="evaluator veRL root")
    script = _absolute_regular(evaluator_script, field="evaluator script")
    if script.name != EXPECTED_EVALUATOR_SCRIPT or root not in script.parents:
        raise OrchestratorError("evaluator script is outside the pinned outer source")
    observed_script_sha256 = sha256_file(script)
    if observed_script_sha256 != expected_evaluator_script_sha256:
        raise OrchestratorError("evaluator script sha256 mismatch")
    _verify_clean_git_source(
        root, evaluation.evaluator_outer_commit, label="evaluator outer"
    )
    _verify_clean_git_source(
        inner, evaluation.evaluator_inner_commit, label="evaluator inner"
    )
    _verify_clean_git_source(
        verl, evaluation.evaluator_verl_commit, label="evaluator veRL"
    )

    destination = Path(orchestration_dir)
    if not destination.is_absolute() or destination.is_symlink():
        raise OrchestratorError(
            "orchestration_dir must be an absolute non-symlink path"
        )
    if evaluation.run_dir != destination / "evaluation":
        raise OrchestratorError(
            "evaluation run_dir must equal orchestration_dir/evaluation"
        )
    lock = Path(holder_lock_path)
    if not lock.is_absolute() or lock.parent != Path("/tmp") or lock.is_symlink():
        raise OrchestratorError("holder lock must be a direct /tmp path")

    route_registry = load_route_registry(
        evaluation.route_registry_path,
        expected_sha256=evaluation.route_registry_sha256,
        expected_route_ids=tuple(evaluation.route_max_rounds),
    )
    observed_max_rounds = {
        route.route_id: route.max_rounds for route in route_registry.routes
    }
    if observed_max_rounds != evaluation.route_max_rounds:
        raise OrchestratorError("route max-rounds changed after evaluator preflight")
    endpoints, endpoint_report = load_heldout_endpoint_registry(
        Path(endpoint_registry_path),
        expected_sha256=expected_endpoint_registry_sha256,
        route_registry=route_registry,
    )
    if len(endpoints) != len(evaluation.route_max_rounds):
        raise OrchestratorError("held-out evaluator requires exactly four endpoints")
    _verify_endpoint_task_counts(evaluation, endpoints)
    holder = load_holder_lease(
        Path(holder_lease_path), expected_sha256=expected_holder_lease_sha256
    )
    return EvalOrchestratorPlan(
        evaluation=evaluation,
        orchestration_dir=destination,
        outer_root=root,
        inner_root=inner,
        verl_root=verl,
        evaluator_script=script,
        evaluator_script_sha256=observed_script_sha256,
        endpoint_registry_path=Path(endpoint_registry_path).resolve(),
        endpoint_registry_sha256=expected_endpoint_registry_sha256,
        endpoints=endpoints,
        endpoint_report=endpoint_report,
        holder_lease=holder,
        holder_lock_path=lock,
        resolve_only=bool(resolve_only),
    )


def evaluator_cli_arguments(plan: HeldoutEvalPlan) -> list[str]:
    """Render the exact child CLI from a previously verified evaluation plan."""

    return [
        "--run-id",
        plan.run_id,
        "--run-dir",
        str(plan.run_dir),
        "--resolved-config",
        str(plan.resolved_config_path),
        "--resolved-config-sha256",
        plan.resolved_config_sha256,
        "--schedule",
        str(plan.schedule_path),
        "--schedule-sha256",
        plan.schedule_sha256,
        "--schedule-manifest",
        str(plan.schedule_manifest_path),
        "--schedule-manifest-sha256",
        plan.schedule_manifest_sha256,
        "--route-registry",
        str(plan.route_registry_path),
        "--route-registry-sha256",
        plan.route_registry_sha256,
        "--agent-loop-config",
        str(plan.agent_loop_config_path),
        "--agent-loop-config-sha256",
        plan.agent_loop_config_sha256,
        "--model-manifest",
        str(plan.model.manifest_path),
        "--model-manifest-sha256",
        plan.model.manifest_sha256,
        "--method-id",
        plan.method_id,
        "--model-kind",
        plan.model_kind,
        "--training-run-id",
        plan.model.training_run_id,
        "--training-outer-commit",
        plan.model.source_commits.get("outer", ""),
        "--training-inner-commit",
        plan.model.source_commits.get("inner", ""),
        "--training-verl-commit",
        plan.model.source_commits.get("verl", ""),
        "--evaluator-outer-commit",
        plan.evaluator_outer_commit,
        "--evaluator-inner-commit",
        plan.evaluator_inner_commit,
        "--evaluator-verl-commit",
        plan.evaluator_verl_commit,
        "--checkpoint-step",
        str(plan.model.checkpoint_step),
        "--batch-size",
        str(plan.batch_size),
        "--num-gpus",
        str(plan.num_gpus),
        "--gpu-memory-utilization",
        str(plan.gpu_memory_utilization),
    ]


def _orchestrator_contract(plan: EvalOrchestratorPlan) -> dict[str, Any]:
    config = derive_eval_config(plan.evaluation)
    eval_config_sha256 = sha256_bytes(canonical_json_bytes(config))
    return {
        "schema": ORCHESTRATOR_SCHEMA,
        "run_id": plan.evaluation.run_id,
        "evaluation_run_dir": str(plan.evaluation.run_dir),
        "evaluation_contract": run_contract(plan.evaluation, eval_config_sha256),
        "sources": {
            "outer": {
                "root": str(plan.outer_root),
                "commit": plan.evaluation.evaluator_outer_commit,
            },
            "inner": {
                "root": str(plan.inner_root),
                "commit": plan.evaluation.evaluator_inner_commit,
            },
            "verl": {
                "root": str(plan.verl_root),
                "commit": plan.evaluation.evaluator_verl_commit,
            },
        },
        "evaluator_script": {
            "path": str(plan.evaluator_script),
            "sha256": plan.evaluator_script_sha256,
        },
        "endpoint_registry": {
            "path": str(plan.endpoint_registry_path),
            "sha256": plan.endpoint_registry_sha256,
        },
        "holder_lease": {
            "path": str(plan.holder_lease.source_path),
            "sha256": plan.holder_lease.sha256,
            "lock_path": str(plan.holder_lock_path),
        },
        "lifecycle": {
            "same_pod_loopback_endpoints": True,
            "exact_process_leases": True,
            "watch_parent": True,
            "holder_transaction": True,
            "foreign_inference_processes": "fail_closed",
        },
    }


def _initialize_orchestration(plan: EvalOrchestratorPlan) -> EvalAttempt:
    root = plan.orchestration_dir
    root.mkdir(parents=True, exist_ok=True)
    contract = _orchestrator_contract(plan)
    contract_path = root / "orchestrator-contract.json"
    if contract_path.exists():
        if contract_path.is_symlink() or not contract_path.is_file():
            raise OrchestratorError("orchestrator contract is not a regular file")
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise OrchestratorError("existing held-out orchestrator contract differs")
    else:
        unexpected = [child.name for child in root.iterdir()]
        if unexpected:
            raise OrchestratorError(
                f"new orchestration directory is not empty: {unexpected!r}"
            )
        atomic_write_json(contract_path, contract)

    schedule_rows = inspect_heldout_schedule(
        plan.evaluation.schedule_path,
        expected_sha256=plan.evaluation.schedule_sha256,
        expected_count=plan.evaluation.episode_count,
    )
    already_complete = False
    if plan.evaluation.run_dir.exists():
        resume = inspect_resume_state(
            plan.evaluation.run_dir, schedule_rows=schedule_rows
        )
        already_complete = resume["next_schedule_position"] == len(schedule_rows)
        if already_complete:
            finalize_run_metrics(
                plan.evaluation.run_dir,
                expected_episode_count=len(schedule_rows),
            )

    attempts_root = root / "attempts"
    attempts_root.mkdir(exist_ok=True)
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise OrchestratorError("held-out attempts root is not a regular directory")
    attempts: list[Path] = []
    for candidate in sorted(attempts_root.iterdir()):
        match = _ATTEMPT_SUFFIX.search(candidate.name)
        expected_prefix = f"{plan.evaluation.run_id}.attempt-"
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or match is None
            or not candidate.name.startswith(expected_prefix)
        ):
            raise OrchestratorError(
                f"unexpected held-out attempt entry: {candidate.name}"
            )
        if int(match.group(1)) != len(attempts):
            raise OrchestratorError("held-out attempt directories are not contiguous")
        attempts.append(candidate)
    index = len(attempts)
    owner_id = _attempt_owner_id(plan.evaluation.run_id, index)
    attempt_dir = attempts_root / owner_id
    attempt_dir.mkdir()
    runtime_plan = SimpleNamespace(
        config=SimpleNamespace(holder_lock_path=plan.holder_lock_path),
        run_dir=attempt_dir,
        experiment_name=owner_id,
        outer_root=plan.outer_root,
        endpoints=plan.endpoints,
        holder_lease=plan.holder_lease,
    )
    return EvalAttempt(
        index=index,
        directory=attempt_dir,
        runtime_plan=runtime_plan,
        already_complete=already_complete,
        owner_id=owner_id,
    )


def _child_environment(
    plan: EvalOrchestratorPlan, attempt: EvalAttempt
) -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    roots = [str(plan.outer_root / "async_plugins"), str(plan.verl_root)]
    if existing:
        roots.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["AGENTMEMORY_PROCESS_OWNER"] = (
        f"camg-heldout-eval:{attempt.owner_id}"
    )
    environment["AGENTMEMORY_RUN_ID"] = plan.evaluation.run_id
    environment["AGENTMEMORY_ATTEMPT_ID"] = attempt.owner_id
    return environment


def _read_argv(path: Path) -> tuple[str, ...]:
    try:
        return tuple(
            value.decode("utf-8", "replace")
            for value in path.read_bytes().split(b"\0")
            if value
        )
    except OSError:
        return ()


def classify_inference_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Classify executable/module identities without scanning data arguments."""

    if not argv:
        return ()
    first = Path(str(argv[0])).name.lower()
    markers: list[str] = []
    if first in _INFERENCE_EXECUTABLES:
        markers.append(f"executable:{first}")
    if first.startswith(("ray::", "sglang::", "vllm::")):
        markers.append(f"process_title:{first}")
    try:
        module_index = list(argv).index("-m")
    except ValueError:
        module = ""
    else:
        module = (
            str(argv[module_index + 1]).lower()
            if module_index + 1 < len(argv)
            else ""
        )
    if module and any(
        module == prefix[:-1] or module.startswith(prefix)
        for prefix in _INFERENCE_MODULE_PREFIXES
    ):
        markers.append(f"python_module:{module}")
    return tuple(sorted(set(markers)))


def _ancestor_pids() -> set[int]:
    ancestors = {os.getpid()}
    current = os.getpid()
    while current > 1:
        try:
            fields = (
                Path(f"/proc/{current}/stat")
                .read_text(encoding="utf-8")
                .rsplit(")", 1)[1]
                .split()
            )
            parent = int(fields[1])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            break
        if parent <= 1 or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def foreign_inference_processes() -> list[dict[str, Any]]:
    """List, but never signal, pre-existing Ray/SGLang/vLLM processes."""

    proc = Path("/proc")
    if not proc.is_dir():
        raise OrchestratorError("held-out runtime ownership requires Linux /proc")
    findings: list[dict[str, Any]] = []
    ignored = _ancestor_pids()
    for candidate in proc.iterdir():
        if not candidate.name.isdigit() or int(candidate.name) in ignored:
            continue
        argv = _read_argv(candidate / "cmdline")
        markers = classify_inference_argv(argv)
        if markers:
            findings.append(
                {
                    "pid": int(candidate.name),
                    "markers": list(markers),
                    "command_sha256": sha256_bytes(
                        b"\0".join(value.encode("utf-8", "replace") for value in argv)
                    ),
                }
            )
    return sorted(findings, key=lambda item: item["pid"])


def run_owned_processes(run_id: str) -> list[dict[str, Any]]:
    """Find exact run-owned residue using inherited environment identities."""

    proc = Path("/proc")
    if not proc.is_dir():
        raise OrchestratorError("held-out runtime ownership requires Linux /proc")
    needles = {
        f"AGENTMEMORY_RUN_ID={run_id}",
        f"AGENTMEMORY_ATTEMPT_ID={run_id}",
        f"AMG_MULTITASK_RUN_ID={run_id}",
    }
    findings: list[dict[str, Any]] = []
    ignored = _ancestor_pids()
    for candidate in proc.iterdir():
        if not candidate.name.isdigit() or int(candidate.name) in ignored:
            continue
        try:
            values = {
                value.decode("utf-8", "replace")
                for value in (candidate / "environ").read_bytes().split(b"\0")
                if value
            }
        except OSError:
            continue
        matched = sorted(needles & values)
        if matched:
            findings.append({"pid": int(candidate.name), "identities": matched})
    return sorted(findings, key=lambda item: item["pid"])


def mounts_below(root: Path) -> list[str]:
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise OrchestratorError("held-out cleanup audit requires Linux mountinfo")
    prefix = str(root.resolve()) + os.sep
    findings: list[str] = []
    for line in mountinfo.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        mount_point = fields[4]
        for encoded, decoded in (
            ("\\040", " "),
            ("\\011", "\t"),
            ("\\012", "\n"),
            ("\\134", "\\"),
        ):
            mount_point = mount_point.replace(encoded, decoded)
        if mount_point == str(root.resolve()) or mount_point.startswith(prefix):
            findings.append(mount_point)
    return sorted(set(findings))


def _attempt_runtime_roots(owner_id: str) -> tuple[Path, Path]:
    """Return the only endpoint scratch roots authorized for one attempt."""

    if not owner_id or "/" in owner_id:
        raise OrchestratorError("held-out attempt owner id is unsafe")
    return (
        Path("/tmp") / f"agentmemorygym-swesmith-{owner_id}",
        Path("/dev/shm") / f"amg-lr-{owner_id}",
    )


def _remove_empty_runtime_root(root: Path) -> bool:
    """Remove one known run root only when its complete tree is directories."""

    if root.is_symlink():
        raise OrchestratorError(
            f"held-out runtime root is not a real directory: {root}"
        )
    if not root.exists():
        return False
    if not root.is_dir():
        raise OrchestratorError(
            f"held-out runtime root is not a real directory: {root}"
        )
    descendants = sorted(
        root.rglob("*"), key=lambda path: len(path.parts), reverse=True
    )
    unsafe = [
        str(path)
        for path in descendants
        if path.is_symlink() or not path.is_dir()
    ]
    if unsafe:
        raise OrchestratorError(
            "held-out runtime root contains non-directory residue: "
            + ", ".join(unsafe[:8])
        )
    for path in descendants:
        try:
            path.rmdir()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise OrchestratorError(
                f"held-out runtime directory is not empty: {path}"
            ) from exc
    try:
        root.rmdir()
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise OrchestratorError(
            f"held-out runtime root is not empty: {root}"
        ) from exc
    return True


class EvalBackend(Protocol):
    evaluator: ProcessLease | None
    endpoint_leases: tuple[ProcessLease, ...]
    holder_handle: _HolderHandle | None
    watch_parent: WatchParentLease | None

    def resolve(self, plan: EvalOrchestratorPlan) -> EvalAttempt: ...

    def prepare_runtime(self, plan: EvalOrchestratorPlan, attempt: EvalAttempt) -> None: ...

    def acquire_holders(self, plan: EvalOrchestratorPlan, attempt: EvalAttempt) -> Any: ...

    def start_watch_parent(self, plan: EvalOrchestratorPlan, attempt: EvalAttempt) -> Any: ...

    def start_endpoints(self, plan: EvalOrchestratorPlan, attempt: EvalAttempt) -> Any: ...

    def start_evaluator(self, plan: EvalOrchestratorPlan, attempt: EvalAttempt) -> Any: ...

    def wait_evaluator(self, plan: EvalOrchestratorPlan, attempt: EvalAttempt, evaluator: Any, endpoints: Any, holder: Any) -> int: ...

    def stop_evaluator(self, plan: EvalOrchestratorPlan, evaluator: Any) -> None: ...

    def stop_watch_parent(self, plan: EvalOrchestratorPlan, attempt: EvalAttempt, watcher: Any) -> None: ...

    def stop_endpoints(self, plan: EvalOrchestratorPlan, attempt: EvalAttempt, endpoints: Any) -> None: ...

    def restore_holders(self, plan: EvalOrchestratorPlan, attempt: EvalAttempt, holder: Any) -> None: ...

    def cleanup_audit(self, plan: EvalOrchestratorPlan, attempt: EvalAttempt) -> Mapping[str, Any]: ...


class HeldoutEvalLocalBackend:
    def __init__(self) -> None:
        self.runtime = LocalBackend()
        self.evaluator: ProcessLease | None = None
        self.watch_parent: WatchParentLease | None = None

    @property
    def endpoint_leases(self) -> tuple[ProcessLease, ...]:
        return self.runtime.endpoint_leases

    @property
    def holder_handle(self) -> _HolderHandle | None:
        return self.runtime.holder_handle

    def resolve(self, plan: EvalOrchestratorPlan) -> EvalAttempt:
        attempt = _initialize_orchestration(plan)
        command = [
            sys.executable,
            str(plan.evaluator_script),
            "verify-plan",
            *evaluator_cli_arguments(plan.evaluation),
        ]
        completed = subprocess.run(
            command,
            cwd=plan.outer_root,
            env=_child_environment(plan, attempt),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output_path = attempt.directory / "evaluator-verify-plan.log"
        output_path.write_bytes(completed.stdout)
        if completed.returncode != 0:
            raise OrchestratorError(
                f"held-out evaluator verify-plan failed with {completed.returncode}"
            )
        try:
            observed_contract = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise OrchestratorError("evaluator verify-plan output is not JSON") from exc
        expected_contract = _orchestrator_contract(plan)["evaluation_contract"]
        if observed_contract != expected_contract:
            raise OrchestratorError("evaluator verify-plan contract drift")
        atomic_write_json(
            attempt.directory / "preflight-receipt.json",
            {
                "schema": ATTEMPT_SCHEMA,
                "status": "resolved",
                "attempt_index": attempt.index,
                "attempt_owner_id": attempt.owner_id,
                "already_complete": attempt.already_complete,
                "evaluator_verify_plan_sha256": sha256_file(output_path),
                "endpoints_spawned": 0,
                "evaluator_spawned": False,
            },
        )
        return attempt

    def prepare_runtime(
        self, plan: EvalOrchestratorPlan, attempt: EvalAttempt
    ) -> None:
        findings = foreign_inference_processes()
        owner_ids = {plan.evaluation.run_id, attempt.owner_id}
        attempts_root = plan.orchestration_dir / "attempts"
        if attempts_root.is_dir():
            owner_ids.update(path.name for path in attempts_root.iterdir() if path.is_dir())
        owned = {
            owner_id: run_owned_processes(owner_id)
            for owner_id in sorted(owner_ids)
        }
        roots = [
            root
            for owner_id in sorted(owner_ids - {plan.evaluation.run_id})
            for root in _attempt_runtime_roots(owner_id)
        ]
        mounts = {
            str(root): mounts_below(root)
            for root in [plan.orchestration_dir, *roots]
        }
        existing_runtime_roots = [str(root) for root in roots if root.exists()]
        if findings or any(owned.values()) or any(mounts.values()) or existing_runtime_roots:
            atomic_write_json(
                attempt.directory / "foreign-inference-processes.json",
                {
                    "status": "fail",
                    "inference_findings": findings,
                    "run_owned_findings": owned,
                    "run_scoped_mounts": mounts,
                    "existing_runtime_roots": existing_runtime_roots,
                },
            )
            raise OrchestratorError(
                "pre-existing inference/run-owned process or mount residue detected"
            )
        assert_ports_available(plan.endpoints)
        atomic_write_json(
            attempt.directory / "runtime-clean-preflight.json",
            {
                "status": "pass",
                "foreign_inference_processes": [],
                "run_owned_processes": {},
                "run_scoped_mounts": {},
                "existing_runtime_roots": [],
                "endpoint_ports_available": True,
            },
        )

    def acquire_holders(
        self, plan: EvalOrchestratorPlan, attempt: EvalAttempt
    ) -> _HolderHandle:
        return self.runtime.acquire_holders(attempt.runtime_plan)

    def start_watch_parent(
        self, plan: EvalOrchestratorPlan, attempt: EvalAttempt
    ) -> WatchParentLease:
        guard = _absolute_regular(
            Path(__file__).with_name("heldout_watch_parent.py"),
            field="held-out watch-parent module",
        )
        ready = attempt.directory / "watch-parent-start.json"
        receipt = attempt.directory / "watch-parent-exit.json"
        log_handle = (attempt.directory / "watch-parent.log").open("ab", buffering=0)
        parent_pid = os.getpid()
        parent_ticks = process_start_ticks(parent_pid)
        if not parent_ticks:
            log_handle.close()
            raise OrchestratorError("cannot capture evaluator orchestrator start ticks")
        command = [
            sys.executable,
            str(guard),
            "--parent-pid",
            str(parent_pid),
            "--parent-start-ticks",
            parent_ticks,
            "--owner-id",
            attempt.owner_id,
            "--ready",
            str(ready),
            "--receipt",
            str(receipt),
        ]
        for root in _attempt_runtime_roots(attempt.owner_id):
            command.extend(("--cleanup-root", str(root)))
        process = subprocess.Popen(
            command,
            cwd=plan.outer_root,
            env=_child_environment(plan, attempt),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ticks = ""
        for _ in range(100):
            ticks = process_start_ticks(process.pid) or ""
            if ticks:
                break
            if process.poll() is not None:
                break
            time.sleep(0.02)
        watcher = WatchParentLease(
            pid=process.pid,
            start_ticks=ticks,
            process=process,
            log_handle=log_handle,
            ready_path=ready,
            receipt_path=receipt,
        )
        self.watch_parent = watcher
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if ready.is_file():
                payload = json.loads(ready.read_text(encoding="utf-8"))
                if (
                    payload.get("status") == "ready"
                    and payload.get("pid") == watcher.pid
                    and str(payload.get("start_ticks")) == watcher.start_ticks
                    and payload.get("owner_id") == attempt.owner_id
                ):
                    return watcher
                break
            if process.poll() is not None:
                break
            time.sleep(0.05)
        try:
            self.stop_watch_parent(plan, attempt, watcher)
        except Exception:
            pass
        raise OrchestratorError("held-out watch-parent did not publish exact start evidence")

    def start_endpoints(
        self, plan: EvalOrchestratorPlan, attempt: EvalAttempt
    ) -> tuple[ProcessLease, ...]:
        leases = self.runtime.start_endpoints(attempt.runtime_plan)
        rows = inspect_heldout_schedule(
            plan.evaluation.schedule_path,
            expected_sha256=plan.evaluation.schedule_sha256,
            expected_count=plan.evaluation.episode_count,
        )
        first_by_route: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            first_by_route.setdefault(str(row["route_id"]), row)
        receipts: list[dict[str, Any]] = []
        for spec in plan.endpoints:
            row = first_by_route.get(spec.route_id)
            if row is None:
                raise OrchestratorError(
                    f"held-out schedule has no reset probe row for {spec.route_id}"
                )
            receipts.append(probe_heldout_reset_identity(spec, row))
        atomic_write_json(
            attempt.directory / "heldout-reset-identity-probes.json",
            {
                "schema": "camg_heldout_reset_identity_bundle_v1",
                "status": "pass",
                "attempt_owner_id": attempt.owner_id,
                "route_order": [spec.route_id for spec in plan.endpoints],
                "receipts": receipts,
            },
        )
        return leases

    def start_evaluator(
        self, plan: EvalOrchestratorPlan, attempt: EvalAttempt
    ) -> ProcessLease:
        log_handle = (attempt.directory / "evaluator.log").open("ab", buffering=0)
        command = [
            sys.executable,
            str(plan.evaluator_script),
            "run",
            *evaluator_cli_arguments(plan.evaluation),
        ]
        evaluator = self.runtime.supervisor.start_command(
            name="heldout-evaluator",
            command=command,
            working_directory=plan.outer_root,
            environment=_child_environment(plan, attempt),
            log_handle=log_handle,
            identity_path=attempt.directory / "evaluator-process-identity.json",
            cleanup_timeout_seconds=120,
        )
        self.evaluator = evaluator
        return evaluator

    def wait_evaluator(
        self,
        plan: EvalOrchestratorPlan,
        attempt: EvalAttempt,
        evaluator: ProcessLease,
        endpoints: Sequence[ProcessLease],
        holder: _HolderHandle,
    ) -> int:
        del plan, attempt
        while True:
            return_code = self.runtime.supervisor.poll(evaluator)
            if return_code is not None:
                return return_code
            for lease in endpoints:
                if not self.runtime.supervisor.alive(lease):
                    raise OrchestratorError(
                        f"endpoint {lease.name} exited while evaluator was active"
                    )
            if not self.runtime.supervisor.alive(holder.watcher):
                raise OrchestratorError(
                    "holder transaction watcher exited while evaluator was active"
                )
            watcher = self.watch_parent
            if (
                watcher is None
                or process_start_ticks(watcher.pid) != watcher.start_ticks
                or watcher.process.poll() is not None
            ):
                raise OrchestratorError(
                    "held-out watch-parent exited while evaluator was active"
                )
            time.sleep(0.5)

    def stop_evaluator(
        self, plan: EvalOrchestratorPlan, evaluator: ProcessLease
    ) -> None:
        del plan
        self.runtime.supervisor.stop(evaluator, timeout_seconds=120)
        self.evaluator = None

    def stop_watch_parent(
        self,
        plan: EvalOrchestratorPlan,
        attempt: EvalAttempt,
        watcher: WatchParentLease,
    ) -> None:
        del plan, attempt
        if process_start_ticks(watcher.pid) == watcher.start_ticks:
            _signal_process_identity(watcher.pid, watcher.start_ticks, signal.SIGTERM)
        try:
            return_code = watcher.process.wait(timeout=15)
        except subprocess.TimeoutExpired as exc:
            if process_start_ticks(watcher.pid) == watcher.start_ticks:
                _signal_process_identity(watcher.pid, watcher.start_ticks, signal.SIGKILL)
            watcher.process.wait(timeout=5)
            raise OrchestratorError("held-out watch-parent did not stop cleanly") from exc
        finally:
            if not watcher.log_handle.closed:
                watcher.log_handle.close()
        if return_code != 0:
            raise OrchestratorError(f"held-out watch-parent exited {return_code}")
        if not watcher.receipt_path.is_file():
            raise OrchestratorError("held-out watch-parent exit receipt is missing")
        payload = json.loads(watcher.receipt_path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass" or payload.get("mode") != "signal":
            raise OrchestratorError("held-out watch-parent exit receipt is invalid")
        self.watch_parent = None

    def stop_endpoints(
        self,
        plan: EvalOrchestratorPlan,
        attempt: EvalAttempt,
        endpoints: Sequence[ProcessLease],
    ) -> None:
        del plan
        self.runtime.stop_endpoints(attempt.runtime_plan, endpoints)

    def restore_holders(
        self,
        plan: EvalOrchestratorPlan,
        attempt: EvalAttempt,
        holder: _HolderHandle,
    ) -> None:
        del plan
        self.runtime.restore_holders(attempt.runtime_plan, holder)

    def cleanup_audit(
        self, plan: EvalOrchestratorPlan, attempt: EvalAttempt
    ) -> Mapping[str, Any]:
        assert_ports_available(plan.endpoints)
        inference = foreign_inference_processes()
        owner_ids = {plan.evaluation.run_id}
        attempts_root = plan.orchestration_dir / "attempts"
        if attempts_root.is_dir():
            owner_ids.update(path.name for path in attempts_root.iterdir() if path.is_dir())
        run_owned = {
            owner_id: run_owned_processes(owner_id)
            for owner_id in sorted(owner_ids)
        }
        roots = [
            root
            for owner_id in sorted(owner_ids - {plan.evaluation.run_id})
            for root in _attempt_runtime_roots(owner_id)
        ]
        mounts = {
            str(root): mounts_below(root)
            for root in [plan.orchestration_dir, *roots]
        }
        owned = self.runtime.supervisor.owned_leases
        removed_empty_runtime_roots: list[str] = []
        if (
            not inference
            and not any(run_owned.values())
            and not any(mounts.values())
            and not owned
            and self.watch_parent is None
        ):
            removed_empty_runtime_roots = [
                str(root) for root in roots if _remove_empty_runtime_root(root)
            ]
        existing_runtime_roots = [str(root) for root in roots if root.exists()]
        if (
            inference
            or any(run_owned.values())
            or any(mounts.values())
            or existing_runtime_roots
            or owned
            or self.watch_parent is not None
        ):
            raise OrchestratorError(
                "held-out runtime residue remains: "
                f"inference={len(inference)} "
                f"run_owned={sum(len(value) for value in run_owned.values())} "
                f"mounts={sum(len(value) for value in mounts.values())} "
                f"roots={len(existing_runtime_roots)} leases={len(owned)}"
            )
        return {
            "status": "pass",
            "endpoint_listeners": 0,
            "foreign_inference_processes": 0,
            "run_owned_processes": 0,
            "run_scoped_mounts": 0,
            "run_scoped_runtime_roots": 0,
            "removed_empty_runtime_roots": removed_empty_runtime_roots,
            "owned_process_leases": 0,
            "holder_restored": True,
        }


def execute_eval_orchestrator(
    plan: EvalOrchestratorPlan, *, backend: EvalBackend
) -> int:
    """Execute one resumable attempt and always unwind evaluator→endpoints→holder."""

    attempt = backend.resolve(plan)
    if plan.resolve_only or attempt.already_complete:
        return 0
    holder: Any = None
    watcher: Any = None
    endpoints: Any = None
    evaluator: Any = None
    evaluator_rc = 125
    cleanup_errors: list[str] = []
    cleanup_audit: Mapping[str, Any] | None = None
    try:
        backend.prepare_runtime(plan, attempt)
        holder = backend.acquire_holders(plan, attempt)
        watcher = backend.start_watch_parent(plan, attempt)
        endpoints = backend.start_endpoints(plan, attempt)
        evaluator = backend.start_evaluator(plan, attempt)
        evaluator_rc = int(
            backend.wait_evaluator(
                plan, attempt, evaluator, endpoints, holder
            )
        )
    finally:
        owned_evaluator = backend.evaluator if backend.evaluator is not None else evaluator
        if owned_evaluator is not None:
            try:
                backend.stop_evaluator(plan, owned_evaluator)
            except Exception as exc:
                cleanup_errors.append(f"evaluator: {exc}")
        owned_endpoints = backend.endpoint_leases or endpoints
        if owned_endpoints:
            try:
                backend.stop_endpoints(plan, attempt, owned_endpoints)
            except Exception as exc:
                cleanup_errors.append(f"endpoints: {exc}")
        owned_watcher = backend.watch_parent if backend.watch_parent is not None else watcher
        if owned_watcher is not None:
            try:
                backend.stop_watch_parent(plan, attempt, owned_watcher)
            except Exception as exc:
                cleanup_errors.append(f"watch-parent: {exc}")
        owned_holder = backend.holder_handle if backend.holder_handle is not None else holder
        if owned_holder is not None:
            try:
                backend.restore_holders(plan, attempt, owned_holder)
            except Exception as exc:
                cleanup_errors.append(f"holder: {exc}")
        try:
            cleanup_audit = backend.cleanup_audit(plan, attempt)
        except Exception as exc:
            cleanup_errors.append(f"cleanup audit: {exc}")
        atomic_write_json(
            attempt.directory / "orchestrator-receipt.json",
            {
                "schema": ATTEMPT_SCHEMA,
                "status": (
                    "pass" if evaluator_rc == 0 and not cleanup_errors else "fail"
                ),
                "attempt_index": attempt.index,
                "attempt_owner_id": attempt.owner_id,
                "evaluator_exit_code": evaluator_rc,
                "cleanup_errors": cleanup_errors,
                "cleanup_audit": cleanup_audit,
            },
        )
    if cleanup_errors:
        raise OrchestratorError("; ".join(cleanup_errors))
    if evaluator_rc != 0:
        raise OrchestratorError(f"held-out evaluator exited {evaluator_rc}")
    finalize_run_metrics(
        plan.evaluation.run_dir,
        expected_episode_count=plan.evaluation.episode_count,
    )
    return 0


__all__ = [
    "ATTEMPT_SCHEMA",
    "EvalAttempt",
    "EvalOrchestratorPlan",
    "HeldoutEvalLocalBackend",
    "ORCHESTRATOR_SCHEMA",
    "evaluator_cli_arguments",
    "execute_eval_orchestrator",
    "foreign_inference_processes",
    "load_eval_orchestrator_plan",
    "mounts_below",
    "run_owned_processes",
]
