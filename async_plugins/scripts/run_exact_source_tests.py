#!/usr/bin/env python3
"""Run a test command only against pre/post-verified exact source bytes."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "exact_source_test_receipt_v1"
MANIFEST_SCHEMA = "exact_source_selected_files_v1"
DEFAULT_INHERITED_ENV = ("HOME", "PATH", "TMPDIR", "TMP", "TEMP")
FIXED_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
}
SECRET_ENV_PATTERN = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|(?:^|_)API_KEY(?:_|$))",
    re.IGNORECASE,
)


class ExactSourceError(RuntimeError):
    """Fail-closed source or invocation contract violation."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ExactSourceError(
            f"output parent is not a resolved directory: {path.parent}"
        )
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _bound_regular_file(
    path: Path, *, maximum_bytes: int = 32 << 20
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ExactSourceError(f"cannot open regular file {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExactSourceError(f"not a regular file: {path}")
        if before.st_size > maximum_bytes:
            raise ExactSourceError(f"file exceeds {maximum_bytes} bytes: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ExactSourceError(f"file exceeds {maximum_bytes} bytes: {path}")
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_ctime_ns", "st_size")
        ) or len(raw) != after.st_size:
            raise ExactSourceError(f"file changed while reading: {path}")
        try:
            named = path.lstat()
        except OSError as error:
            raise ExactSourceError(f"file disappeared while reading: {path}") from error
        if path.is_symlink() or any(
            getattr(named, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_ctime_ns", "st_size")
        ):
            raise ExactSourceError(f"file path was replaced while reading: {path}")
        return raw, {
            "path": str(path),
            "real_path": str(path.resolve(strict=True)),
            "device": after.st_dev,
            "inode": after.st_ino,
            "ctime_ns": after.st_ctime_ns,
            "size_bytes": after.st_size,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    finally:
        os.close(descriptor)


def _parse_named(values: Sequence[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ExactSourceError(f"{option} requires NAME=VALUE: {raw!r}")
        name, value = raw.split("=", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name) or not value:
            raise ExactSourceError(f"invalid {option} entry: {raw!r}")
        if name in parsed:
            raise ExactSourceError(f"duplicate {option} name: {name}")
        parsed[name] = value
    return parsed


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ExactSourceError(
            f"git {' '.join(arguments)} failed for {root}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def _load_selected_manifest(path: Path, *, repo_name: str) -> dict[str, Any]:
    raw, identity = _bound_regular_file(path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExactSourceError(f"invalid selected-file manifest {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ExactSourceError(f"selected-file manifest schema mismatch: {path}")
    if payload.get("repo") != repo_name:
        raise ExactSourceError(
            f"selected-file manifest repo mismatch for {repo_name}: {path}"
        )
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ExactSourceError(f"selected-file manifest has no files: {path}")
    normalized: dict[str, str] = {}
    for relative, expected_sha256 in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise ExactSourceError(f"invalid selected-file manifest entry: {path}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative == ".":
            raise ExactSourceError(f"unsafe selected-file path {relative!r}: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ExactSourceError(
                f"invalid selected-file sha256 for {relative!r}: {path}"
            )
        normalized[relative] = expected_sha256
    return {"identity": identity, "files": normalized}


def _repo_snapshot(
    *,
    name: str,
    root: Path,
    expected_head: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    top_level = Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve(
        strict=True
    )
    if top_level != root:
        raise ExactSourceError(
            f"repository root is not the git top level for {name}: "
            f"expected={root} observed={top_level}"
        )
    head = _git(root, "rev-parse", "HEAD").strip()
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    selected: dict[str, Any] = {}
    for relative, expected_sha256 in manifest["files"].items():
        _git(root, "ls-files", "--error-unmatch", "--", relative)
        raw, identity = _bound_regular_file(root / relative)
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        selected[relative] = {
            **identity,
            "expected_sha256": expected_sha256,
            "matches_expected": observed_sha256 == expected_sha256,
        }
    return {
        "name": name,
        "root": str(root),
        "git_top_level": str(top_level),
        "expected_head": expected_head,
        "head": head,
        "status_porcelain": status,
        "selected_manifest": manifest["identity"],
        "selected_files": selected,
    }


def _snapshot_violations(snapshot: Mapping[str, Any], *, phase: str) -> list[str]:
    name = str(snapshot["name"])
    violations: list[str] = []
    if snapshot["head"] != snapshot["expected_head"]:
        violations.append(
            f"{phase} HEAD mismatch for {name}: expected={snapshot['expected_head']} "
            f"observed={snapshot['head']}"
        )
    if snapshot["status_porcelain"]:
        violations.append(f"{phase} git status is dirty for {name}")
    for relative, identity in snapshot["selected_files"].items():
        if identity["matches_expected"] is not True:
            violations.append(
                f"{phase} selected-file hash mismatch for {name}:{relative}"
            )
    return violations


def _parse_environment(
    explicit_values: Sequence[str], inherited_names: Sequence[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    inherited = tuple(dict.fromkeys((*DEFAULT_INHERITED_ENV, *inherited_names)))
    environment = {
        name: os.environ[name] for name in inherited if name in os.environ
    }
    explicit = _parse_named(explicit_values, option="--env")
    for name in (*inherited, *explicit):
        if SECRET_ENV_PATTERN.search(name):
            raise ExactSourceError(
                f"refusing to record secret-bearing environment key: {name}"
            )
    environment.update(FIXED_ENV)
    environment.update(explicit)
    return environment, {
        "default_inherited_allowlist": list(DEFAULT_INHERITED_ENV),
        "additional_inherited_allowlist": list(inherited_names),
        "explicit_keys": sorted(explicit),
        "resolved": dict(sorted(environment.items())),
    }


def _resolved_output_path(raw: str, *, label: str) -> Path:
    requested = Path(raw).absolute()
    if requested.name in {"", ".", ".."}:
        raise ExactSourceError(f"invalid {label} output path: {raw!r}")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise ExactSourceError(
            f"{label} output parent must already exist: {requested.parent}"
        ) from error
    if not parent.is_dir():
        raise ExactSourceError(f"{label} output parent is not a directory: {parent}")
    resolved = parent / requested.name
    if resolved.exists() or resolved.is_symlink():
        try:
            observed = resolved.lstat()
        except OSError as error:
            raise ExactSourceError(
                f"cannot inspect {label} output: {resolved}"
            ) from error
        if resolved.is_symlink() or not stat.S_ISREG(observed.st_mode):
            raise ExactSourceError(
                f"existing {label} output is not a regular non-symlink file: {resolved}"
            )
    return resolved


def _existing_file_identity(path: Path) -> tuple[int, int] | None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ExactSourceError(
            f"cannot inspect file identity {path}: {error}"
        ) from error
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise ExactSourceError(f"file identity path is not a regular file: {path}")
    return observed.st_dev, observed.st_ino


def _outside_repositories(path: Path, repositories: Mapping[str, Path]) -> None:
    for name, root in repositories.items():
        try:
            path.relative_to(root)
        except ValueError:
            continue
        raise ExactSourceError(f"output path is inside repository {name}: {path}")


def _require_distinct_authority_paths(
    entries: Sequence[tuple[str, Path, tuple[int, int] | None]],
) -> None:
    for index, (left_label, left_path, left_identity) in enumerate(entries):
        for right_label, right_path, right_identity in entries[index + 1 :]:
            if left_path == right_path:
                raise ExactSourceError(
                    f"{left_label} and {right_label} paths overlap: {left_path}"
                )
            if (
                left_identity is not None
                and right_identity is not None
                and left_identity == right_identity
            ):
                raise ExactSourceError(
                    f"{left_label} and {right_label} resolve to the same existing file"
                )


def _prepare_output_boundaries(arguments: argparse.Namespace) -> tuple[Path, Path]:
    repo_values = _parse_named(arguments.repo, option="--repo")
    head_values = _parse_named(arguments.expected_head, option="--expected-head")
    manifest_values = _parse_named(
        arguments.selected_manifest, option="--selected-manifest"
    )
    if not repo_values or set(repo_values) != set(head_values) or set(
        repo_values
    ) != set(manifest_values):
        raise ExactSourceError(
            "--repo, --expected-head, and --selected-manifest names must match"
        )
    roots = {
        name: Path(raw).resolve(strict=True) for name, raw in repo_values.items()
    }
    for name, root in roots.items():
        if not root.is_dir():
            raise ExactSourceError(f"repository root is not a directory: {root}")
        if not re.fullmatch(r"[0-9a-f]{40}", head_values[name]):
            raise ExactSourceError(f"invalid expected HEAD for {name}")

    log_path = _resolved_output_path(arguments.log, label="log")
    receipt_path = _resolved_output_path(arguments.receipt, label="receipt")
    _outside_repositories(log_path, roots)
    _outside_repositories(receipt_path, roots)

    authority_entries: list[tuple[str, Path, tuple[int, int] | None]] = [
        ("log output", log_path, _existing_file_identity(log_path)),
        ("receipt output", receipt_path, _existing_file_identity(receipt_path)),
    ]
    for name, raw in manifest_values.items():
        manifest_path = Path(raw).resolve(strict=True)
        _raw, binding = _bound_regular_file(manifest_path)
        authority_entries.append(
            (
                f"selected manifest {name}",
                manifest_path,
                (int(binding["device"]), int(binding["inode"])),
            )
        )
    interpreter = Path(arguments.interpreter).absolute().resolve(strict=True)
    _raw, interpreter_binding = _bound_regular_file(interpreter)
    authority_entries.append(
        (
            "interpreter",
            interpreter,
            (
                int(interpreter_binding["device"]),
                int(interpreter_binding["inode"]),
            ),
        )
    )
    _require_distinct_authority_paths(authority_entries)
    return log_path, receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--expected-head", action="append", default=[], metavar="NAME=SHA"
    )
    parser.add_argument(
        "--selected-manifest", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--interpreter", required=True)
    parser.add_argument("--interpreter-sha256", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--inherit-env", action="append", default=[], metavar="NAME")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def run(arguments: argparse.Namespace) -> tuple[int, dict[str, Any], bytes]:
    started = _utc_now()
    violations: list[str] = []
    repositories: dict[str, dict[str, Any]] = {}
    command_exit: int | None = None
    log_chunks: list[bytes] = []
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "fail",
        "started_at": started,
        "finished_at": None,
        "repositories": repositories,
        "interpreter": None,
        "cwd": None,
        "command": None,
        "environment": None,
        "exit_code": None,
        "log": None,
        "final_post_output_audit": None,
        "violations": violations,
    }
    try:
        repo_values = _parse_named(arguments.repo, option="--repo")
        head_values = _parse_named(arguments.expected_head, option="--expected-head")
        manifest_values = _parse_named(
            arguments.selected_manifest, option="--selected-manifest"
        )
        if not repo_values or set(repo_values) != set(head_values) or set(
            repo_values
        ) != set(manifest_values):
            raise ExactSourceError(
                "--repo, --expected-head, and --selected-manifest names must match"
            )
        roots = {
            name: Path(raw).resolve(strict=True) for name, raw in repo_values.items()
        }
        for name, root in roots.items():
            if not root.is_dir():
                raise ExactSourceError(f"repository root is not a directory: {root}")
            if not re.fullmatch(r"[0-9a-f]{40}", head_values[name]):
                raise ExactSourceError(f"invalid expected HEAD for {name}")
        interpreter_requested = Path(arguments.interpreter).absolute()
        interpreter = interpreter_requested.resolve(strict=True)
        _raw_interpreter, interpreter_identity = _bound_regular_file(interpreter)
        if interpreter_identity["sha256"] != arguments.interpreter_sha256:
            raise ExactSourceError(
                "interpreter sha256 mismatch: "
                f"expected={arguments.interpreter_sha256} "
                f"observed={interpreter_identity['sha256']}"
            )
        receipt["interpreter"] = {
            "requested_path": str(interpreter_requested),
            "expected_sha256": arguments.interpreter_sha256,
            "pre": interpreter_identity,
            "post": None,
        }
        cwd = Path(arguments.cwd).resolve(strict=True)
        if not cwd.is_dir():
            raise ExactSourceError(f"command cwd is not a directory: {cwd}")
        receipt["cwd"] = str(cwd)
        environment, environment_receipt = _parse_environment(
            arguments.env, arguments.inherit_env
        )
        receipt["environment"] = environment_receipt
        if not command:
            raise ExactSourceError("test command is empty")
        full_command = [str(interpreter), *command]
        receipt["command"] = full_command

        manifests = {
            name: _load_selected_manifest(
                Path(manifest_values[name]).resolve(strict=True), repo_name=name
            )
            for name in roots
        }
        for name, root in roots.items():
            pre = _repo_snapshot(
                name=name,
                root=root,
                expected_head=head_values[name],
                manifest=manifests[name],
            )
            repositories[name] = {"pre": pre, "post": None}
            violations.extend(_snapshot_violations(pre, phase="pre-run"))
        if violations:
            raise ExactSourceError("pre-run source contract failed")

        preamble = {
            "schema": SCHEMA,
            "phase": "pre",
            "repositories": {name: item["pre"] for name, item in repositories.items()},
            "interpreter": receipt["interpreter"],
            "cwd": receipt["cwd"],
            "command": receipt["command"],
            "environment": receipt["environment"],
        }
        log_chunks.append(
            ("EXACT_SOURCE_TEST_PRE " + json.dumps(preamble, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
        completed = subprocess.run(
            full_command,
            cwd=cwd,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        command_exit = completed.returncode
        receipt["exit_code"] = command_exit
        log_chunks.append(completed.stdout)

        for name, root in roots.items():
            try:
                post_manifest = _load_selected_manifest(
                    Path(manifest_values[name]).resolve(strict=True), repo_name=name
                )
                post = _repo_snapshot(
                    name=name,
                    root=root,
                    expected_head=head_values[name],
                    manifest={
                        "identity": post_manifest["identity"],
                        "files": manifests[name]["files"],
                    },
                )
                repositories[name]["post"] = post
                violations.extend(_snapshot_violations(post, phase="post-run"))
                if post_manifest["files"] != manifests[name]["files"]:
                    violations.append(
                        f"post-run selected-file manifest changed for {name}"
                    )
                if post != repositories[name]["pre"]:
                    violations.append(f"post-run source snapshot changed for {name}")
            except ExactSourceError as error:
                repositories[name]["post"] = {"error": str(error)}
                violations.append(f"post-run source audit failed for {name}: {error}")
        if command_exit != 0:
            violations.append(f"test command exited {command_exit}")
        try:
            _post_interpreter_raw, post_interpreter_identity = _bound_regular_file(
                interpreter
            )
            receipt["interpreter"]["post"] = post_interpreter_identity
            if post_interpreter_identity != interpreter_identity:
                violations.append("post-run interpreter identity changed")
        except ExactSourceError as error:
            receipt["interpreter"]["post"] = {"error": str(error)}
            violations.append(f"post-run interpreter audit failed: {error}")
    except (ExactSourceError, OSError) as error:
        message = str(error)
        if message and message not in violations:
            violations.append(message)

    receipt["status"] = "pass" if not violations and command_exit == 0 else "fail"
    receipt["finished_at"] = _utc_now()
    postamble = {
        "schema": SCHEMA,
        "phase": "post",
        "status": receipt["status"],
        "exit_code": receipt["exit_code"],
        "repositories": {
            name: item.get("post") for name, item in repositories.items()
        },
        "violations": violations,
    }
    log_chunks.append(
        ("EXACT_SOURCE_TEST_POST " + json.dumps(postamble, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    log_payload = b"".join(log_chunks)
    return (0 if receipt["status"] == "pass" else 2), receipt, log_payload


def _final_post_output_audit(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Rebind all source authorities after the harness writes its outputs."""

    violations: list[str] = []
    repository_results: dict[str, Any] = {}
    repositories = receipt.get("repositories")
    if not isinstance(repositories, Mapping) or not repositories:
        violations.append("final post-output audit has no pre-run repository snapshot")
    else:
        for name, item in repositories.items():
            pre = item.get("pre") if isinstance(item, Mapping) else None
            if not isinstance(pre, Mapping):
                message = f"final post-output audit has no pre-run snapshot for {name}"
                repository_results[str(name)] = {"error": message}
                violations.append(message)
                continue
            try:
                manifest_identity = pre["selected_manifest"]
                selected_files = pre["selected_files"]
                if not isinstance(manifest_identity, Mapping) or not isinstance(
                    selected_files, Mapping
                ):
                    raise ExactSourceError("pre-run selected-file binding is invalid")
                expected_files = {
                    str(relative): str(identity["expected_sha256"])
                    for relative, identity in selected_files.items()
                }
                manifest_path = Path(str(manifest_identity["real_path"]))
                manifest = _load_selected_manifest(
                    manifest_path, repo_name=str(name)
                )
                post = _repo_snapshot(
                    name=str(name),
                    root=Path(str(pre["root"])),
                    expected_head=str(pre["expected_head"]),
                    manifest={
                        "identity": manifest["identity"],
                        "files": expected_files,
                    },
                )
                repository_results[str(name)] = post
                violations.extend(
                    _snapshot_violations(post, phase="final post-output")
                )
                if manifest["identity"] != dict(manifest_identity):
                    violations.append(
                        f"final post-output selected-file manifest changed for {name}"
                    )
                if manifest["files"] != expected_files:
                    violations.append(
                        "final post-output selected-file manifest entries changed "
                        f"for {name}"
                    )
                if post != dict(pre):
                    violations.append(
                        f"final post-output source snapshot changed for {name}"
                    )
            except (
                ExactSourceError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ) as error:
                repository_results[str(name)] = {"error": str(error)}
                violations.append(
                    f"final post-output source audit failed for {name}: {error}"
                )

    interpreter_result: dict[str, Any]
    interpreter = receipt.get("interpreter")
    pre_interpreter = (
        interpreter.get("pre") if isinstance(interpreter, Mapping) else None
    )
    if not isinstance(pre_interpreter, Mapping):
        interpreter_result = {"error": "pre-run interpreter binding is unavailable"}
        violations.append("final post-output interpreter binding is unavailable")
    else:
        try:
            _raw, observed = _bound_regular_file(
                Path(str(pre_interpreter["real_path"]))
            )
            interpreter_result = observed
            if observed != dict(pre_interpreter):
                violations.append("final post-output interpreter identity changed")
        except (ExactSourceError, KeyError, OSError) as error:
            interpreter_result = {"error": str(error)}
            violations.append(f"final post-output interpreter audit failed: {error}")

    return {
        "status": "pass" if not violations else "fail",
        "finished_at": _utc_now(),
        "repositories": repository_results,
        "interpreter": interpreter_result,
        "violations": violations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        log_path, receipt_path = _prepare_output_boundaries(arguments)
    except (ExactSourceError, OSError) as error:
        print(f"exact-source output preflight failed: {error}", file=sys.stderr)
        return 2
    arguments.log = str(log_path)
    arguments.receipt = str(receipt_path)
    _initial_status, receipt, log_payload = run(arguments)
    receipt["outputs"] = {
        "log": str(log_path),
        "receipt": str(receipt_path),
    }
    candidate_status = receipt["status"]
    receipt["publication"] = {
        "state": "pending_final_post_output_audit",
        "candidate_status": candidate_status,
    }
    # A crash after provisional publication must never leave a PASS receipt.
    # The final receipt is the only state that downstream release tooling may
    # accept, and it is published only after the post-output source audit and
    # final log are durable.
    receipt["status"] = "fail"
    # Publish provisional outputs first.  Only then can the final audit prove
    # that these writes did not mutate a repository, selected manifest, or the
    # interpreter.  The output-boundary preflight makes the subsequent output
    # rewrites disjoint from every audited authority.
    _atomic_write(log_path, log_payload)
    receipt["log"] = {
        "path": str(log_path),
        "size_bytes": len(log_payload),
        "sha256": hashlib.sha256(log_payload).hexdigest(),
    }
    _atomic_json(receipt_path, receipt)

    final_audit = _final_post_output_audit(receipt)
    receipt["final_post_output_audit"] = final_audit
    for violation in final_audit["violations"]:
        if violation not in receipt["violations"]:
            receipt["violations"].append(violation)
    receipt["status"] = (
        "pass"
        if not receipt["violations"] and receipt.get("exit_code") == 0
        else "fail"
    )
    receipt["publication"] = {
        "state": "final",
        "candidate_status": candidate_status,
    }
    receipt["finished_at"] = _utc_now()
    final_postamble = {
        "schema": SCHEMA,
        "phase": "final-post-output",
        "status": receipt["status"],
        "exit_code": receipt["exit_code"],
        "audit": final_audit,
        "violations": receipt["violations"],
    }
    final_log_payload = log_payload + (
        "EXACT_SOURCE_TEST_FINAL "
        + json.dumps(final_postamble, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _atomic_write(log_path, final_log_payload)
    receipt["log"] = {
        "path": str(log_path),
        "size_bytes": len(final_log_payload),
        "sha256": hashlib.sha256(final_log_payload).hexdigest(),
    }
    _atomic_json(receipt_path, receipt)
    status = 0 if receipt["status"] == "pass" else 2
    if status:
        print(
            "exact-source test failed: " + "; ".join(receipt["violations"]),
            file=sys.stderr,
        )
    else:
        print(f"exact-source test passed: {receipt_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
