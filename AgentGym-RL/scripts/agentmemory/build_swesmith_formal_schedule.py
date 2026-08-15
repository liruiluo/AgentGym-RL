#!/usr/bin/env python3
"""Build a frozen SWE-smith formal schedule without losing sampling order."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from certify_swesmith_schedule_order import (
    FILTER_SOURCE_ALGORITHM,
    GLOBAL_SHUFFLE_ALGORITHM,
    certify_schedule,
    dump_json,
    load_json,
    load_jsonl,
    seeded_permutation,
    sha256_bytes,
    sha256_file,
    write_evidence,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _load_exclusions(paths: list[Path], values: list[str]) -> set[str]:
    exclusions = set(values)
    for path in paths:
        payload = load_json(path)
        if isinstance(payload, dict):
            payload = payload.get("excluded_instance_ids")
        if not isinstance(payload, list) or any(not isinstance(x, str) for x in payload):
            raise ValueError(f"malformed exclusion file: {path}")
        exclusions.update(payload)
    return exclusions


def _source_files(source_dir: Path) -> dict[str, Path]:
    files = {
        "manifest": source_dir / "formal100.manifest.json",
        "selection": source_dir / "formal100.instance_ids.json",
        "endpoint_map": source_dir / "formal100.endpoint-index-map.jsonl",
        "routing": source_dir / "formal100.routing.jsonl",
        "image_manifest": source_dir / "swesmith-image-manifest.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise ValueError(f"source schedule is incomplete: {missing}")
    return files


def build_schedule(
    *,
    source_dir: Path,
    output_dir: Path,
    mode: str,
    seed: int | None,
    target_rows: int,
    batch_size: int,
    composition_block_updates: int,
    exclude_instance_ids: set[str],
    minimum_repositories_per_block: int,
    maximum_repository_share: float,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    files = _source_files(source_dir)
    source_manifest = load_json(files["manifest"])
    source_ids = load_json(files["selection"])
    source_map = load_jsonl(files["endpoint_map"])
    source_routing = load_jsonl(files["routing"])
    if not isinstance(source_manifest, dict) or not isinstance(source_ids, list):
        raise ValueError("source manifest or selection is malformed")
    if len(source_ids) != len(source_map):
        raise ValueError("source selection and endpoint map lengths differ")
    for index, (instance_id, row) in enumerate(zip(source_ids, source_map)):
        if row.get("endpoint_data_idx") != index or row.get("instance_id") != instance_id:
            raise ValueError("source endpoint map is not dense physical order")
    unknown = exclude_instance_ids - set(source_ids)
    if unknown:
        raise ValueError(f"exclusions are absent from the source: {sorted(unknown)}")

    final_ids = [value for value in source_ids if value not in exclude_instance_ids]
    if not final_ids:
        raise ValueError("exclusions removed the complete selection")
    final_index = {instance_id: index for index, instance_id in enumerate(final_ids)}
    source_index_to_id = {
        int(row["endpoint_data_idx"]): str(row["instance_id"]) for row in source_map
    }
    final_endpoint_rows = []
    for row in source_map:
        instance_id = str(row["instance_id"])
        if instance_id not in final_index:
            continue
        copied = dict(row)
        copied["endpoint_data_idx"] = final_index[instance_id]
        final_endpoint_rows.append(copied)

    if mode == "global-shuffle":
        if seed is None:
            raise ValueError("global-shuffle mode requires --seed")
        algorithm = GLOBAL_SHUFFLE_ALGORITHM
        permutation = seeded_permutation(len(final_ids), seed)
        if target_rows < len(permutation) or target_rows > 2 * len(permutation):
            raise ValueError("target rows must contain one permutation plus at most one prefix")
        schedule_indices = permutation + permutation[: target_rows - len(permutation)]
        replay: dict[str, Any] | None = None
        unique_prefix_rows = len(permutation)
        repeated_tail_rows = target_rows - len(permutation)
    elif mode == "filter-source":
        if seed is not None:
            raise ValueError("filter-source mode does not accept --seed")
        algorithm = FILTER_SOURCE_ALGORITHM
        filtered = [
            final_index[source_index_to_id[int(row["data_idx"])]]
            for row in source_routing
            if source_index_to_id[int(row["data_idx"])] in final_index
        ]
        if set(filtered) != set(range(len(final_ids))):
            raise ValueError("filtered source routing does not cover the final selection")
        if len(filtered) > target_rows:
            raise ValueError("filtered source routing exceeds target rows")
        fill = []
        while len(filtered) + len(fill) < target_rows:
            need = target_rows - len(filtered) - len(fill)
            fill.extend(filtered[:need])
        schedule_indices = filtered + fill
        replay = {
            "source_endpoint_index_map": str(files["endpoint_map"]),
            "source_endpoint_index_map_sha256": sha256_file(files["endpoint_map"]),
            "source_routing": str(files["routing"]),
            "source_routing_sha256": sha256_file(files["routing"]),
        }
        unique_prefix_rows = 0
        repeated_tail_rows = 0
    else:
        raise ValueError(f"unsupported schedule mode: {mode}")

    if set(schedule_indices) != set(range(len(final_ids))):
        raise ValueError("generated routing does not cover the final selection")
    first_occurrence = list(dict.fromkeys(schedule_indices))
    if len(first_occurrence) > 2 and first_occurrence in (
        sorted(first_occurrence),
        sorted(first_occurrence, reverse=True),
    ):
        raise ValueError("generated routing retained monotonic physical order")

    routing_rows = []
    for position, data_idx in enumerate(schedule_indices):
        instance_id = final_ids[data_idx]
        routing_rows.append(
            {
                "data_idx": data_idx,
                "extra_info": {
                    "index": data_idx,
                    "instance_id_sha256": sha256_bytes(instance_id.encode("utf-8")),
                    "schedule_position": position,
                },
                "item_id": f"swesmith_{position}",
            }
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    try:
        selection_path = temporary / "formal100.instance_ids.json"
        endpoint_path = temporary / "formal100.endpoint-index-map.jsonl"
        routing_path = temporary / "formal100.routing.jsonl"
        image_path = temporary / "swesmith-image-manifest.json"
        dump_json(selection_path, final_ids)
        _write_jsonl(endpoint_path, final_endpoint_rows)
        _write_jsonl(routing_path, routing_rows)
        shutil.copy2(files["image_manifest"], image_path)

        manifest = copy.deepcopy(source_manifest)
        manifest["dataset_id"] = (
            f"{source_manifest.get('dataset_id', 'swesmith_formal')}"
            f"_{mode.replace('-', '_')}_{seed if seed is not None else 'preserve'}"
        )
        manifest["selection"] = dict(manifest.get("selection", {}))
        manifest["selection"].update(
            {
                "count": len(final_ids),
                "mode": "instance_ids",
                "path": selection_path.name,
                "sha256": sha256_file(selection_path),
                "split_contract": (
                    "same_frozen_swesmith_train_selection_after_explicit_exclusions+"
                    f"{algorithm}+target_rows_{target_rows}"
                ),
            }
        )
        manifest["schedule"] = {
            "algorithm": algorithm,
            "batch_size": batch_size,
            "composition_block_updates": composition_block_updates,
            "endpoint_index_map": endpoint_path.name,
            "endpoint_index_map_sha256": sha256_file(endpoint_path),
            "excluded_instance_ids": sorted(exclude_instance_ids),
            "image_manifest": image_path.name,
            "image_manifest_sha256": sha256_file(image_path),
            "maximum_repository_share": maximum_repository_share,
            "minimum_repositories_per_block": minimum_repositories_per_block,
            "repeated_tail_rows": repeated_tail_rows,
            "replay": replay,
            "routing": routing_path.name,
            "routing_sha256": sha256_file(routing_path),
            "seed": seed,
            "source_manifest": str(files["manifest"]),
            "source_manifest_sha256": sha256_file(files["manifest"]),
            "total_rows": target_rows,
            "unique_prefix_rows": unique_prefix_rows,
        }
        dump_json(temporary / "formal100.manifest.json", manifest)
        certificate = certify_schedule(temporary)
        evidence_hashes = write_evidence(temporary, certificate)
        manifest["schedule"]["evidence_sha256"] = evidence_hashes
        dump_json(temporary / "formal100.manifest.json", manifest)
        certificate = certify_schedule(temporary)
        if certificate["status"] != "pass":
            raise ValueError("schedule certificate did not pass")
        os.rename(str(temporary), str(output_dir))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return certify_schedule(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("global-shuffle", "filter-source"), required=True
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-rows", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--composition-block-updates", type=int, default=10)
    parser.add_argument("--exclude-instance-id", action="append", default=[])
    parser.add_argument("--exclude-file", action="append", type=Path, default=[])
    parser.add_argument("--minimum-repositories-per-block", type=int, default=1)
    parser.add_argument("--maximum-repository-share", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exclusions = _load_exclusions(args.exclude_file, args.exclude_instance_id)
    certificate = build_schedule(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        seed=args.seed,
        target_rows=args.target_rows,
        batch_size=args.batch_size,
        composition_block_updates=args.composition_block_updates,
        exclude_instance_ids=exclusions,
        minimum_repositories_per_block=args.minimum_repositories_per_block,
        maximum_repository_share=args.maximum_repository_share,
    )
    print(json.dumps(certificate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
