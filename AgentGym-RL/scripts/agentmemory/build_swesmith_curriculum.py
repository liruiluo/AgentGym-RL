#!/usr/bin/env python3
"""Build a deterministic, split-safe SWE-smith curriculum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


MANIFEST_SCHEMA = "swesmith_jsonl_manifest_v1"


@dataclass(frozen=True)
class Candidate:
    instance_id: str
    repo: str
    image_name: str
    shard_index: int
    shard_line: int
    changed_path: str
    changed_lines: int
    additions: int
    deletions: int
    f2p_count: int
    p2p_count: int
    problem_chars: int
    patch_chars: int

    @property
    def scan_order(self) -> tuple[int, int]:
        return self.shard_index, self.shard_line

    @property
    def difficulty_key(self) -> tuple[int, int, int, int, str]:
        return (
            self.changed_lines,
            self.f2p_count,
            self.patch_chars,
            self.problem_chars,
            hashlib.sha256(self.instance_id.encode("utf-8")).hexdigest(),
        )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label}: {path}") from exc


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field {key!r} must be nonempty text")
    return value.strip()


def _resolve_child(parent: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a real file: {path}")
    return path


def _load_split_ids(manifest_path: Path, manifest: Mapping[str, Any]) -> set[str]:
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("mode") != "instance_ids":
        raise ValueError("base manifest must use an instance_ids selection")
    selection_path = _resolve_child(
        manifest_path.parent,
        _required_text(selection, "path"),
    )
    raw = selection_path.read_bytes()
    expected = _required_text(selection, "sha256")
    actual = _sha256_bytes(raw)
    if actual != expected:
        raise ValueError(
            f"base selection SHA256 mismatch: expected {expected}, got {actual}"
        )
    try:
        values = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("base selection is not valid UTF-8 JSON") from exc
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise ValueError("base selection must be a list of instance IDs")
    if len(values) != len(set(values)):
        raise ValueError("base selection contains duplicate instance IDs")
    if len(values) != int(selection.get("count", -1)):
        raise ValueError("base selection count does not match its manifest")
    return set(values)


def _diff_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            raise ValueError("malformed diff --git header") from exc
        if len(parts) != 4:
            raise ValueError("malformed diff --git header")
        old_path, new_path = parts[2], parts[3]
        if old_path == "/dev/null" or new_path == "/dev/null":
            raise ValueError("curriculum excludes file creation and deletion")
        if old_path.startswith("a/"):
            old_path = old_path[2:]
        if new_path.startswith("b/"):
            new_path = new_path[2:]
        if old_path != new_path:
            raise ValueError("curriculum excludes file renames")
        paths.append(new_path)
    if not paths:
        raise ValueError("patch has no diff --git path")
    return paths


def _changed_line_counts(patch: str) -> tuple[int, int]:
    additions = 0
    deletions = 0
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        additions += int(line.startswith("+"))
        deletions += int(line.startswith("-"))
    return additions, deletions


def _is_source_path(path: str, allowed_suffixes: tuple[str, ...]) -> bool:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return False
    lowered_parts = tuple(part.lower() for part in candidate.parts)
    basename = candidate.name.lower()
    if any(part in {"test", "tests", "testing"} for part in lowered_parts):
        return False
    if basename.startswith("test_") or basename.endswith("_test.py"):
        return False
    return not allowed_suffixes or candidate.suffix.lower() in allowed_suffixes


def _candidate_from_row(
    row: Mapping[str, Any],
    *,
    shard_index: int,
    shard_line: int,
    max_changed_lines: int,
    max_f2p: int,
    min_p2p: int,
    max_problem_chars: int,
    allowed_suffixes: tuple[str, ...],
) -> Candidate | None:
    try:
        instance_id = _required_text(row, "instance_id")
        repo = _required_text(row, "repo")
        image_name = _required_text(row, "image_name")
        problem = _required_text(row, "problem_statement")
        patch = _required_text(row, "patch")
    except ValueError:
        return None
    f2p = row.get("FAIL_TO_PASS")
    p2p = row.get("PASS_TO_PASS")
    if not isinstance(f2p, list) or not isinstance(p2p, list):
        return None
    if not (1 <= len(f2p) <= max_f2p and len(p2p) >= min_p2p):
        return None
    if len(problem) > max_problem_chars:
        return None
    try:
        paths = _diff_paths(patch)
    except ValueError:
        return None
    if len(paths) != 1 or not _is_source_path(paths[0], allowed_suffixes):
        return None
    additions, deletions = _changed_line_counts(patch)
    changed_lines = additions + deletions
    if not (1 <= changed_lines <= max_changed_lines):
        return None
    return Candidate(
        instance_id=instance_id,
        repo=repo,
        image_name=image_name,
        shard_index=shard_index,
        shard_line=shard_line,
        changed_path=paths[0],
        changed_lines=changed_lines,
        additions=additions,
        deletions=deletions,
        f2p_count=len(f2p),
        p2p_count=len(p2p),
        problem_chars=len(problem),
        patch_chars=len(patch),
    )


def _atomic_write(path: Path, raw: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _iter_shard_rows(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("base manifest has no shards")
    for shard_index, spec in enumerate(shards):
        if not isinstance(spec, dict):
            raise ValueError(f"shard {shard_index} is not an object")
        path = _resolve_child(manifest_path.parent, _required_text(spec, "path"))
        digest = hashlib.sha256()
        physical_rows = 0
        usable_rows = 0
        with path.open("rb") as handle:
            for shard_line, raw in enumerate(handle, start=1):
                digest.update(raw)
                physical_rows += 1
                try:
                    row = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid JSON in {path}:{shard_line}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row in {path}:{shard_line}")
                usable_rows += int(bool(str(row.get("problem_statement", "")).strip()))
                yield shard_index, shard_line, row
        actual_sha = digest.hexdigest()
        if actual_sha != _required_text(spec, "sha256"):
            raise ValueError(f"shard SHA256 mismatch: {path}")
        if physical_rows != int(spec.get("physical_rows", -1)):
            raise ValueError(f"shard physical row mismatch: {path}")
        if usable_rows != int(spec.get("usable_rows", -1)):
            raise ValueError(f"shard usable row mismatch: {path}")


def build_curriculum(
    *,
    base_manifest_path: Path,
    output_dir: Path,
    name: str,
    expected_role: str,
    repositories: tuple[str, ...],
    per_repo: int,
    repository_quotas: Mapping[str, int] | None,
    max_changed_lines: int,
    max_f2p: int,
    min_p2p: int,
    max_problem_chars: int,
    allowed_suffixes: tuple[str, ...],
    exclude_ids: set[str],
) -> dict[str, Any]:
    base_manifest_path = base_manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest = _load_json(base_manifest_path, "base manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported base manifest")
    role = _required_text(manifest, "role")
    if role != expected_role:
        raise ValueError(f"expected role {expected_role!r}, got {role!r}")
    if not repositories or len(repositories) != len(set(repositories)):
        raise ValueError("repositories must be nonempty and unique")
    if per_repo <= 0:
        raise ValueError("per_repo must be positive")
    quotas = {repo: per_repo for repo in repositories}
    if repository_quotas is not None:
        if set(repository_quotas) != set(repositories):
            raise ValueError("repository quotas must cover every requested repository exactly")
        if any(type(value) is not int or value <= 0 for value in repository_quotas.values()):
            raise ValueError("repository quotas must be positive integers")
        quotas = dict(repository_quotas)

    split_ids = _load_split_ids(base_manifest_path, manifest)
    requested = set(repositories)
    seen_requested_repositories: set[str] = set()
    candidates: dict[str, list[Candidate]] = {repo: [] for repo in repositories}
    found_split_ids: set[str] = set()

    for shard_index, shard_line, row in _iter_shard_rows(base_manifest_path, manifest):
        instance_id = str(row.get("instance_id", ""))
        if instance_id in split_ids:
            found_split_ids.add(instance_id)
        repo = str(row.get("repo", ""))
        if repo not in requested or instance_id not in split_ids or instance_id in exclude_ids:
            continue
        seen_requested_repositories.add(repo)
        candidate = _candidate_from_row(
            row,
            shard_index=shard_index,
            shard_line=shard_line,
            max_changed_lines=max_changed_lines,
            max_f2p=max_f2p,
            min_p2p=min_p2p,
            max_problem_chars=max_problem_chars,
            allowed_suffixes=allowed_suffixes,
        )
        if candidate is not None:
            candidates[repo].append(candidate)

    missing_split_ids = split_ids - found_split_ids
    if missing_split_ids:
        raise ValueError(f"base split contains {len(missing_split_ids)} missing IDs")
    missing_repositories = requested - seen_requested_repositories
    if missing_repositories:
        raise ValueError(
            "requested repositories are absent from the frozen split: "
            + ", ".join(sorted(missing_repositories))
        )

    selected: list[Candidate] = []
    eligible_counts: dict[str, int] = {}
    for repo in repositories:
        ranked = sorted(candidates[repo], key=lambda candidate: candidate.difficulty_key)
        eligible_counts[repo] = len(ranked)
        quota = quotas[repo]
        if len(ranked) < quota:
            raise ValueError(
                f"repository {repo!r} has {len(ranked)} eligible rows, needs {quota}"
            )
        selected.extend(ranked[:quota])
    selected.sort(key=lambda candidate: candidate.scan_order)

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / f"{name}.instance_ids.json"
    selection_raw = _json_bytes([candidate.instance_id for candidate in selected])
    _atomic_write(selection_path, selection_raw)

    output_manifest = dict(manifest)
    output_manifest["dataset_id"] = f"swesmith_{name}_{manifest['upstream']['revision'][:12]}"
    output_manifest["selection"] = {
        "count": len(selected),
        "mode": "instance_ids",
        "path": selection_path.name,
        "sha256": _sha256_bytes(selection_raw),
        "split_contract": (
            f"{manifest['selection'].get('split_contract', 'frozen_split')}+"
            f"single_source_curriculum_v1"
        ),
    }
    output_manifest["shards"] = [
        {
            **spec,
            "path": os.path.relpath(
                _resolve_child(base_manifest_path.parent, _required_text(spec, "path")),
                output_dir,
            ),
        }
        for spec in manifest["shards"]
    ]
    manifest_path = output_dir / f"{name}.manifest.json"
    manifest_raw = _json_bytes(output_manifest)
    _atomic_write(manifest_path, manifest_raw)

    routing_path = output_dir / f"{name}.routing.jsonl"
    routing_raw = b"".join(
        (
            json.dumps(
                {
                    "item_id": f"swesmith_{index}",
                    "data_idx": index,
                    "extra_info": {"index": index},
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for index in range(len(selected))
    )
    _atomic_write(routing_path, routing_raw)

    report = {
        "schema": "swesmith_deterministic_curriculum_report_v1",
        "name": name,
        "role": role,
        "base_manifest": str(base_manifest_path),
        "base_manifest_sha256": _sha256_file(base_manifest_path),
        "selection_path": str(selection_path),
        "selection_sha256": _sha256_bytes(selection_raw),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "routing_path": str(routing_path),
        "routing_sha256": _sha256_bytes(routing_raw),
        "count": len(selected),
        "repositories": list(repositories),
        "repository_quotas": quotas,
        "eligible_counts": eligible_counts,
        "filters": {
            "single_existing_source_file": True,
            "max_changed_lines": max_changed_lines,
            "max_f2p": max_f2p,
            "min_p2p": min_p2p,
            "max_problem_chars": max_problem_chars,
            "allowed_suffixes": list(allowed_suffixes),
            "excluded_instance_count": len(exclude_ids),
        },
        "images": sorted({candidate.image_name for candidate in selected}),
        "records": [
            {
                "data_idx": index,
                "instance_id": candidate.instance_id,
                "repo": candidate.repo,
                "image_name": candidate.image_name,
                "shard_index": candidate.shard_index,
                "shard_line": candidate.shard_line,
                "changed_path": candidate.changed_path,
                "changed_lines": candidate.changed_lines,
                "additions": candidate.additions,
                "deletions": candidate.deletions,
                "f2p_count": candidate.f2p_count,
                "p2p_count": candidate.p2p_count,
                "problem_chars": candidate.problem_chars,
                "patch_chars": candidate.patch_chars,
            }
            for index, candidate in enumerate(selected)
        ],
    }
    report_path = output_dir / f"{name}.report.json"
    _atomic_write(report_path, _json_bytes(report))
    return report


def _load_excluded_ids(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        value = _load_json(path.expanduser().resolve(), "excluded IDs")
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(f"excluded ID file is invalid: {path}")
        excluded.update(value)
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--expected-role", choices=("train", "heldout"), required=True)
    parser.add_argument("--repository", action="append", dest="repositories", required=True)
    parser.add_argument("--per-repo", type=int, default=1)
    parser.add_argument(
        "--repository-quota",
        action="append",
        default=[],
        metavar="REPOSITORY=COUNT",
    )
    parser.add_argument("--max-changed-lines", type=int, default=12)
    parser.add_argument("--max-f2p", type=int, default=2)
    parser.add_argument("--min-p2p", type=int, default=1)
    parser.add_argument("--max-problem-chars", type=int, default=3000)
    parser.add_argument("--allowed-suffix", action="append", default=[".py"])
    parser.add_argument("--exclude-instance-id-file", action="append", type=Path, default=[])
    args = parser.parse_args()

    suffixes = tuple(sorted({suffix.lower() for suffix in args.allowed_suffix}))
    if any(not suffix.startswith(".") for suffix in suffixes):
        parser.error("--allowed-suffix values must start with '.'")
    repository_quotas: dict[str, int] | None = None
    if args.repository_quota:
        repository_quotas = {}
        for raw in args.repository_quota:
            try:
                repository, count_text = raw.rsplit("=", 1)
                count = int(count_text)
            except (ValueError, TypeError):
                parser.error(f"invalid --repository-quota value: {raw!r}")
            if not repository or count <= 0 or repository in repository_quotas:
                parser.error(f"invalid --repository-quota value: {raw!r}")
            repository_quotas[repository] = count
    report = build_curriculum(
        base_manifest_path=args.base_manifest,
        output_dir=args.output_dir,
        name=args.name,
        expected_role=args.expected_role,
        repositories=tuple(args.repositories),
        per_repo=args.per_repo,
        repository_quotas=repository_quotas,
        max_changed_lines=args.max_changed_lines,
        max_f2p=args.max_f2p,
        min_p2p=args.min_p2p,
        max_problem_chars=args.max_problem_chars,
        allowed_suffixes=suffixes,
        exclude_ids=_load_excluded_ids(args.exclude_instance_id_file),
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "name": report["name"],
                "role": report["role"],
                "count": report["count"],
                "repositories": report["repositories"],
                "eligible_counts": report["eligible_counts"],
                "images": report["images"],
                "manifest_sha256": report["manifest_sha256"],
                "routing_sha256": report["routing_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
