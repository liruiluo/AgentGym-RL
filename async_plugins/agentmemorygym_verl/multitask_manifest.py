"""Build one deterministic interleaved schedule for AMG multitask training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

_SPEC_SCHEMA = "amg_multitask_manifest_spec_v1"
_CERTIFICATE_SCHEMA = "amg_multitask_schedule_certificate_v1"
_AGENT_NAME = "amg_task_neutral_async"
_ROUTE_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VALID_ROLES = {"gate_only", "train_pool"}


def _sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _route_id(value: Any, *, field: str = "route_id") -> str:
    route_id = str(value or "").strip().lower()
    if not _ROUTE_ID.fullmatch(route_id):
        raise ValueError(f"{field} is invalid: {value!r}")
    return route_id


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a positive integer, not bool")
    try:
        normalized = int(value)
        exact = float(value) == float(normalized)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer, got {value!r}") from exc
    if not exact or normalized <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return normalized


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a non-negative integer, not bool")
    try:
        normalized = int(value)
        exact = float(value) == float(normalized)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} must be a non-negative integer, got {value!r}"
        ) from exc
    if not exact or normalized < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return normalized


def _regular_file(path: Path, *, field: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must be a regular non-symlink file: {path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceSchedule:
    route_id: str
    path: Path
    sha256: str
    route_attestation_sha256: str
    role: str
    allow_repetition: bool
    row_count: int


@dataclass(frozen=True)
class MultitaskManifestSpec:
    sha256: str
    agent_name: str
    panel_id: str
    role: str
    route_registry_sha256: str
    optimizer_updates: int
    samples_per_update: int
    sources: tuple[SourceSchedule, ...]

    @property
    def row_count(self) -> int:
        return self.optimizer_updates * self.samples_per_update

    @property
    def rows_per_route(self) -> int:
        return self.row_count // len(self.sources)


class _SourceCursor:
    def __init__(self, source: SourceSchedule) -> None:
        self.source = source
        self._handle: TextIO | None = None
        self.source_position = 0
        self.repetition = 0

    def __enter__(self) -> "_SourceCursor":
        self._open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _open(self) -> None:
        self.close()
        self._handle = self.source.path.open("r", encoding="utf-8")
        self.source_position = 0

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def next(self) -> tuple[dict[str, Any], int, int]:
        assert self._handle is not None
        while True:
            raw_line = self._handle.readline()
            if raw_line:
                position = self.source_position
                self.source_position += 1
                return (
                    _parse_source_row(
                        raw_line,
                        source=self.source,
                        source_position=position,
                    ),
                    position,
                    self.repetition,
                )
            if not self.source.allow_repetition:
                raise ValueError(
                    f"route {self.source.route_id!r} source schedule exhausted"
                )
            self.repetition += 1
            self._open()


def _parse_source_row(
    raw_line: str,
    *,
    source: SourceSchedule,
    source_position: int,
) -> dict[str, Any]:
    if not raw_line.strip():
        raise ValueError(
            f"route {source.route_id!r} source contains a blank line at "
            f"{source_position}"
        )
    try:
        row = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"route {source.route_id!r} source has invalid JSON at line "
            f"{source_position + 1}"
        ) from exc
    if not isinstance(row, Mapping):
        raise TypeError(
            f"route {source.route_id!r} source row {source_position} is not an object"
        )
    row = dict(row)
    item_id = str(row.get("item_id", ""))
    if not item_id:
        raise ValueError(
            f"route {source.route_id!r} source row {source_position} has no item_id"
        )
    data_idx = _nonnegative_int(
        row.get("data_idx"),
        field=f"route {source.route_id!r} source row {source_position} data_idx",
    )
    extra = row.get("extra_info")
    if not isinstance(extra, Mapping):
        raise TypeError(
            f"route {source.route_id!r} source row {source_position} "
            "has no extra_info object"
        )
    extra = dict(extra)
    source_index = _nonnegative_int(
        extra.get("index"),
        field=f"route {source.route_id!r} source row {source_position} index",
    )
    if source_index != source_position:
        raise ValueError(
            f"route {source.route_id!r} source global index drift at row "
            f"{source_position}: index={source_index}"
        )
    if "index" in row and _nonnegative_int(
        row["index"],
        field=f"route {source.route_id!r} source row {source_position} top-level index",
    ) != source_index:
        raise ValueError(
            f"route {source.route_id!r} source global index drift at row "
            f"{source_position}"
        )
    if extra.get("schedule_position") != source_position:
        raise ValueError(
            f"route {source.route_id!r} source schedule_position drift at row "
            f"{source_position}"
        )
    if extra.get("role") != source.role:
        raise ValueError(
            f"route {source.route_id!r} source role drift at row {source_position}"
        )
    row_route = row.get("route_id")
    extra_route = extra.get("route_id")
    for field, value in (("row.route_id", row_route), ("extra_info.route_id", extra_route)):
        if value is not None and _route_id(value, field=field) != source.route_id:
            raise ValueError(
                f"route {source.route_id!r} source route_id drift at row "
                f"{source_position}"
            )
    row["item_id"] = item_id
    row["data_idx"] = data_idx
    row["extra_info"] = extra
    return row


def _inspect_source(
    *,
    route_id: str,
    path: Path,
    expected_sha256: str,
    route_attestation_sha256: str,
    role: str,
    allow_repetition: bool,
) -> SourceSchedule:
    _regular_file(path, field=f"route {route_id!r} source schedule")
    observed_sha256 = _file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"route {route_id!r} schedule sha256 mismatch: expected "
            f"{expected_sha256}, got {observed_sha256}"
        )
    provisional = SourceSchedule(
        route_id=route_id,
        path=path.resolve(),
        sha256=observed_sha256,
        route_attestation_sha256=route_attestation_sha256,
        role=role,
        allow_repetition=allow_repetition,
        row_count=0,
    )
    item_ids: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for position, raw_line in enumerate(handle):
            row = _parse_source_row(
                raw_line,
                source=provisional,
                source_position=position,
            )
            item_id = row["item_id"]
            if item_id in item_ids:
                raise ValueError(
                    f"route {route_id!r} source has duplicate item_id {item_id!r}"
                )
            item_ids.add(item_id)
            count += 1
    if count == 0:
        raise ValueError(f"route {route_id!r} source schedule is empty")
    return SourceSchedule(
        route_id=route_id,
        path=path.resolve(),
        sha256=observed_sha256,
        route_attestation_sha256=route_attestation_sha256,
        role=role,
        allow_repetition=allow_repetition,
        row_count=count,
    )


def load_multitask_manifest_spec(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> MultitaskManifestSpec:
    spec_path = _regular_file(Path(path), field="AMG multitask manifest spec")
    expected_digest = _sha256(expected_sha256, field="multitask spec expected sha256")
    observed_digest = _file_sha256(spec_path)
    if observed_digest != expected_digest:
        raise ValueError(
            "AMG multitask spec sha256 mismatch: "
            f"expected {expected_digest}, got {observed_digest}"
        )
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid AMG multitask spec JSON: {spec_path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != _SPEC_SCHEMA:
        raise ValueError(f"AMG multitask spec schema must be {_SPEC_SCHEMA!r}")
    agent_name = str(payload.get("agent_name", ""))
    if agent_name != _AGENT_NAME:
        raise ValueError(f"AMG multitask agent_name must be {_AGENT_NAME!r}")
    panel_id = str(payload.get("panel_id", "")).strip()
    if not panel_id:
        raise ValueError("AMG multitask panel_id must not be empty")
    role = str(payload.get("role", ""))
    if role not in _VALID_ROLES:
        raise ValueError(f"AMG multitask role is invalid: {role!r}")
    route_registry_sha256 = _sha256(
        payload.get("route_registry_sha256"),
        field="multitask route_registry_sha256",
    )
    optimizer_updates = _positive_int(
        payload.get("optimizer_updates"), field="multitask optimizer_updates"
    )
    samples_per_update = _positive_int(
        payload.get("samples_per_update"), field="multitask samples_per_update"
    )
    raw_routes = payload.get("routes")
    if isinstance(raw_routes, (str, bytes)) or not isinstance(raw_routes, Sequence):
        raise TypeError("AMG multitask routes must be a sequence")
    if len(raw_routes) != 4:
        raise ValueError(
            "AMG multitask spec must contain exactly four routes, "
            f"got {len(raw_routes)}"
        )
    if samples_per_update % len(raw_routes) != 0:
        raise ValueError("samples_per_update must be divisible by the route count")

    sources: list[SourceSchedule] = []
    seen_routes: set[str] = set()
    for position, raw_route in enumerate(raw_routes):
        if not isinstance(raw_route, Mapping):
            raise TypeError(f"AMG multitask route {position} must be an object")
        route_id = _route_id(raw_route.get("route_id"))
        if route_id in seen_routes:
            raise ValueError(f"duplicate multitask route_id {route_id!r}")
        seen_routes.add(route_id)
        route_role = str(raw_route.get("role", ""))
        if route_role != role:
            raise ValueError(
                f"route {route_id!r} role {route_role!r} differs from {role!r}"
            )
        schedule_value = str(raw_route.get("schedule", "")).strip()
        if not schedule_value:
            raise ValueError(f"route {route_id!r} schedule is missing")
        schedule_path = Path(schedule_value)
        if not schedule_path.is_absolute():
            schedule_path = spec_path.parent / schedule_path
        allow_repetition = raw_route.get("allow_repetition", False)
        if not isinstance(allow_repetition, bool):
            raise TypeError(
                f"route {route_id!r} allow_repetition must be boolean"
            )
        sources.append(
            _inspect_source(
                route_id=route_id,
                path=schedule_path,
                expected_sha256=_sha256(
                    raw_route.get("schedule_sha256"),
                    field=f"route {route_id!r} schedule_sha256",
                ),
                route_attestation_sha256=_sha256(
                    raw_route.get("route_attestation_sha256"),
                    field=f"route {route_id!r} route_attestation_sha256",
                ),
                role=route_role,
                allow_repetition=allow_repetition,
            )
        )

    spec = MultitaskManifestSpec(
        sha256=observed_digest,
        agent_name=agent_name,
        panel_id=panel_id,
        role=role,
        route_registry_sha256=route_registry_sha256,
        optimizer_updates=optimizer_updates,
        samples_per_update=samples_per_update,
        sources=tuple(sources),
    )
    for source in spec.sources:
        if source.row_count < spec.rows_per_route and not source.allow_repetition:
            raise ValueError(
                f"route {source.route_id!r} source schedule would exhaust at "
                f"{source.row_count}/{spec.rows_per_route} rows without explicit repetition"
            )
    return spec


def _output_row(
    source_row: Mapping[str, Any],
    *,
    spec: MultitaskManifestSpec,
    source: SourceSchedule,
    global_index: int,
    route_ordinal: int,
    source_position: int,
    source_repetition: int,
) -> dict[str, Any]:
    row = deepcopy(dict(source_row))
    source_extra = dict(row.get("extra_info") or {})
    source_item_id = str(row["item_id"])
    source_index = int(source_extra["index"])
    source_schedule_position = int(source_extra["schedule_position"])
    source_manifest_digest = source_extra.get("manifest_digest")
    source_panel_id = source_extra.get("panel_id")

    row.update(
        {
            "index": global_index,
            "item_id": (
                f"{source.route_id}:{source_item_id}:"
                f"multitask-{route_ordinal:06d}"
            ),
            "route_id": source.route_id,
            "data_source": source.route_id,
            "agent_name": spec.agent_name,
        }
    )
    source_extra.update(
        {
            "index": global_index,
            "schedule_position": global_index,
            "schedule_repetition": route_ordinal // source.row_count,
            "role": spec.role,
            "manifest_digest": spec.sha256,
            "panel_id": spec.panel_id,
            "route_id": source.route_id,
            "route_attestation_sha256": source.route_attestation_sha256,
            "route_registry_sha256": spec.route_registry_sha256,
            "source_schedule_sha256": source.sha256,
            "source_schedule_position": source_position,
            "source_schedule_repetition": source_repetition,
            "source_index": source_index,
            "source_item_id": source_item_id,
            "source_manifest_digest": source_manifest_digest,
            "source_panel_id": source_panel_id,
            "source_schedule_position_declared": source_schedule_position,
        }
    )
    row["extra_info"] = source_extra
    return row


def compose_multitask_manifest(
    spec_path: str | os.PathLike[str],
    *,
    expected_spec_sha256: str,
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Atomically write the deterministic route-interleaved schedule."""

    spec = load_multitask_manifest_spec(
        spec_path,
        expected_sha256=expected_spec_sha256,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    cursors = [_SourceCursor(source) for source in spec.sources]
    try:
        for cursor in cursors:
            cursor.__enter__()
        global_item_ids: set[str] = set()
        with temporary_handle as handle:
            for route_ordinal in range(spec.rows_per_route):
                for route_offset, cursor in enumerate(cursors):
                    source_row, source_position, source_repetition = cursor.next()
                    global_index = route_ordinal * len(cursors) + route_offset
                    row = _output_row(
                        source_row,
                        spec=spec,
                        source=cursor.source,
                        global_index=global_index,
                        route_ordinal=route_ordinal,
                        source_position=source_position,
                        source_repetition=source_repetition,
                    )
                    item_id = row["item_id"]
                    if item_id in global_item_ids:
                        raise ValueError(f"duplicate global item_id {item_id!r}")
                    global_item_ids.add(item_id)
                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        for cursor in cursors:
            cursor.close()

    schedule_sha256 = _file_sha256(output)
    return {
        "schema": _CERTIFICATE_SCHEMA,
        "spec_sha256": spec.sha256,
        "schedule_sha256": schedule_sha256,
        "route_registry_sha256": spec.route_registry_sha256,
        "role": spec.role,
        "panel_id": spec.panel_id,
        "agent_name": spec.agent_name,
        "optimizer_updates": spec.optimizer_updates,
        "samples_per_update": spec.samples_per_update,
        "row_count": spec.row_count,
        "route_order": [source.route_id for source in spec.sources],
        "per_route_rows": {
            source.route_id: spec.rows_per_route for source in spec.sources
        },
        "sources": {
            source.route_id: {
                "schedule_sha256": source.sha256,
                "route_attestation_sha256": source.route_attestation_sha256,
                "source_row_count": source.row_count,
                "allow_repetition": source.allow_repetition,
            }
            for source in spec.sources
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--certificate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = compose_multitask_manifest(
        args.spec,
        expected_spec_sha256=args.expected_spec_sha256,
        output_path=args.output,
    )
    if args.certificate:
        _atomic_json(Path(args.certificate), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
