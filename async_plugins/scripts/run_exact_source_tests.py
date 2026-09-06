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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _outside_repositories(path: Path, repositories: Mapping[str, Path]) -> None:
    absolute = path.absolute()
    for name, root in repositories.items():
        try:
            absolute.relative_to(root)
        except ValueError:
            continue
        raise ExactSourceError(f"output path is inside repository {name}: {path}")


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
        log_path = Path(arguments.log).absolute()
        receipt_path = Path(arguments.receipt).absolute()
        if log_path == receipt_path:
            raise ExactSourceError("log and receipt paths must differ")
        _outside_repositories(log_path, roots)
        _outside_repositories(receipt_path, roots)

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


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    log_path = Path(arguments.log).absolute()
    receipt_path = Path(arguments.receipt).absolute()
    status, receipt, log_payload = run(arguments)
    _atomic_write(log_path, log_payload)
    receipt["log"] = {
        "path": str(log_path),
        "size_bytes": len(log_payload),
        "sha256": hashlib.sha256(log_payload).hexdigest(),
    }
    _atomic_json(receipt_path, receipt)
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
