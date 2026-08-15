#!/usr/bin/env python3
"""Certify that a frozen SWE-smith routing file matches its order contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


CERTIFICATE_SCHEMA = "swesmith_schedule_order_certificate_v1"
GLOBAL_SHUFFLE_ALGORITHM = "python_random_mt19937_shuffle_v1"
FILTER_SOURCE_ALGORITHM = "filter_source_routing_preserve_order_v1"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON: {path}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"row {line_number} is not an object")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSONL: {path}") from exc
    return rows


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def repo_for_instance(instance_id: str) -> str:
    parts = instance_id.split(".", 1)
    if len(parts) != 2 or "__" not in parts[0]:
        raise ValueError(f"malformed SWE-smith instance ID: {instance_id}")
    return parts[0]


def image_map_for_instances(
    instance_ids: Iterable[str], image_manifest: Mapping[str, Any]
) -> dict[str, str]:
    images = image_manifest.get("images")
    if not isinstance(images, list):
        raise ValueError("image manifest must contain an images list")
    names = []
    for row in images:
        if not isinstance(row, dict) or not isinstance(row.get("image"), str):
            raise ValueError("image manifest contains a malformed image row")
        names.append(row["image"])

    by_instance: dict[str, str] = {}
    for instance_id in instance_ids:
        parts = instance_id.split(".", 2)
        if len(parts) != 3 or "__" not in parts[0]:
            raise ValueError(f"malformed SWE-smith instance ID: {instance_id}")
        owner, repository = parts[0].split("__", 1)
        suffix = f".{owner}_1776_{repository}.{parts[1]}".lower()
        matches = [name for name in names if name.lower().endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"expected one image for {instance_id}, found {len(matches)}"
            )
        by_instance[instance_id] = matches[0]
    return by_instance


def seeded_permutation(count: int, seed: int) -> list[int]:
    values = list(range(count))
    random.Random(seed).shuffle(values)
    return values


def _resolve(directory: Path, value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = directory / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return path


def _load_contract(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = directory / "formal100.manifest.json"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    schedule = manifest.get("schedule")
    if not isinstance(schedule, dict):
        raise ValueError("manifest is missing the structured schedule contract")
    return manifest, schedule


def _filter_source_replay(
    *,
    directory: Path,
    schedule: Mapping[str, Any],
    final_ids: list[str],
) -> list[int]:
    replay = schedule.get("replay")
    if not isinstance(replay, dict):
        raise ValueError("filter-source contract is missing replay metadata")
    source_map_path = _resolve(
        directory, str(replay.get("source_endpoint_index_map", "")), "source map"
    )
    source_routing_path = _resolve(
        directory, str(replay.get("source_routing", "")), "source routing"
    )
    if sha256_file(source_map_path) != replay.get("source_endpoint_index_map_sha256"):
        raise ValueError("source endpoint map SHA256 mismatch")
    if sha256_file(source_routing_path) != replay.get("source_routing_sha256"):
        raise ValueError("source routing SHA256 mismatch")

    source_map = load_jsonl(source_map_path)
    source_routing = load_jsonl(source_routing_path)
    index_to_id = {
        int(row["endpoint_data_idx"]): str(row["instance_id"]) for row in source_map
    }
    final_index = {instance_id: index for index, instance_id in enumerate(final_ids)}
    filtered = [
        final_index[index_to_id[int(row["data_idx"])]]
        for row in source_routing
        if index_to_id[int(row["data_idx"])] in final_index
    ]
    if set(filtered) != set(range(len(final_ids))):
        raise ValueError("filtered source routing does not cover the final selection")
    target_rows = int(schedule["total_rows"])
    if len(filtered) > target_rows:
        raise ValueError("filtered source routing exceeds the target row count")
    fill = []
    while len(filtered) + len(fill) < target_rows:
        need = target_rows - len(filtered) - len(fill)
        fill.extend(filtered[:need])
    return filtered + fill


def _composition(
    schedule_indices: list[int],
    instance_ids: list[str],
    image_by_instance: Mapping[str, str],
    block_rows: int,
) -> list[dict[str, Any]]:
    blocks = []
    for start in range(0, len(schedule_indices), block_rows):
        indices = schedule_indices[start : start + block_rows]
        repositories = Counter(repo_for_instance(instance_ids[index]) for index in indices)
        images = Counter(image_by_instance[instance_ids[index]] for index in indices)
        rows = len(indices)
        blocks.append(
            {
                "block": len(blocks) + 1,
                "end_position_exclusive": start + rows,
                "image_counts": dict(sorted(images.items())),
                "max_image_share": max(images.values()) / rows,
                "max_repository_share": max(repositories.values()) / rows,
                "repository_counts": dict(sorted(repositories.items())),
                "rows": rows,
                "start_position": start,
                "unique_images": len(images),
                "unique_repositories": len(repositories),
            }
        )
    return blocks


def certify_schedule(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest, schedule = _load_contract(directory)
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("mode") != "instance_ids":
        raise ValueError("manifest selection must use instance_ids")
    selection_path = _resolve(directory, str(selection.get("path", "")), "selection")
    if sha256_file(selection_path) != selection.get("sha256"):
        raise ValueError("selection SHA256 mismatch")
    instance_ids = load_json(selection_path)
    if not isinstance(instance_ids, list) or any(
        not isinstance(value, str) or not value for value in instance_ids
    ):
        raise ValueError("selection must be a list of nonempty instance IDs")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("selection contains duplicate instance IDs")
    if len(instance_ids) != int(selection.get("count", -1)):
        raise ValueError("selection count does not match the manifest")

    endpoint_path = _resolve(
        directory, str(schedule.get("endpoint_index_map", "")), "endpoint map"
    )
    routing_path = _resolve(directory, str(schedule.get("routing", "")), "routing")
    image_path = _resolve(
        directory, str(schedule.get("image_manifest", "")), "image manifest"
    )
    if sha256_file(endpoint_path) != schedule.get("endpoint_index_map_sha256"):
        raise ValueError("endpoint map SHA256 mismatch")
    if sha256_file(routing_path) != schedule.get("routing_sha256"):
        raise ValueError("routing SHA256 mismatch")
    if sha256_file(image_path) != schedule.get("image_manifest_sha256"):
        raise ValueError("image manifest SHA256 mismatch")

    endpoint_rows = load_jsonl(endpoint_path)
    if len(endpoint_rows) != len(instance_ids):
        raise ValueError("endpoint map length does not match the selection")
    for index, row in enumerate(endpoint_rows):
        if row.get("endpoint_data_idx") != index:
            raise ValueError("endpoint map indices are not dense and ordered")
        if row.get("instance_id") != instance_ids[index]:
            raise ValueError("endpoint map order does not match the selection")

    routing_rows = load_jsonl(routing_path)
    if len(routing_rows) != int(schedule.get("total_rows", -1)):
        raise ValueError("routing row count does not match the schedule contract")
    schedule_indices = []
    for position, row in enumerate(routing_rows):
        data_idx = row.get("data_idx")
        if not isinstance(data_idx, int) or not 0 <= data_idx < len(instance_ids):
            raise ValueError(f"invalid data_idx at schedule position {position}")
        extra = row.get("extra_info")
        if not isinstance(extra, dict):
            raise ValueError(f"missing extra_info at schedule position {position}")
        expected_digest = sha256_bytes(instance_ids[data_idx].encode("utf-8"))
        if extra.get("index") != data_idx:
            raise ValueError(f"extra_info.index mismatch at position {position}")
        if extra.get("schedule_position") != position:
            raise ValueError(f"schedule_position mismatch at position {position}")
        if extra.get("instance_id_sha256") != expected_digest:
            raise ValueError(f"instance digest mismatch at position {position}")
        if row.get("item_id") != f"swesmith_{position}":
            raise ValueError(f"item_id mismatch at position {position}")
        schedule_indices.append(data_idx)

    if set(schedule_indices) != set(range(len(instance_ids))):
        raise ValueError("routing does not cover the complete frozen selection")
    unique_order = list(dict.fromkeys(schedule_indices))
    if len(unique_order) != len(instance_ids):
        raise ValueError("routing first-occurrence order is not a permutation")
    monotonic = unique_order in (sorted(unique_order), sorted(unique_order, reverse=True))
    if len(unique_order) > 2 and monotonic:
        raise ValueError("routing first-occurrence order is monotonic physical order")

    algorithm = schedule.get("algorithm")
    if algorithm == GLOBAL_SHUFFLE_ALGORITHM:
        seed = schedule.get("seed")
        if not isinstance(seed, int):
            raise ValueError("global-shuffle contract requires an integer seed")
        permutation = seeded_permutation(len(instance_ids), seed)
        repeated_tail_rows = int(schedule.get("repeated_tail_rows", -1))
        expected = permutation + permutation[:repeated_tail_rows]
        if schedule_indices != expected:
            raise ValueError("routing does not replay from the declared shuffle seed")
        if int(schedule.get("unique_prefix_rows", -1)) != len(instance_ids):
            raise ValueError("unique_prefix_rows does not match the selection")
    elif algorithm == FILTER_SOURCE_ALGORITHM:
        expected = _filter_source_replay(
            directory=directory,
            schedule=schedule,
            final_ids=instance_ids,
        )
        if schedule_indices != expected:
            raise ValueError("routing does not preserve the filtered source order")
    else:
        raise ValueError(f"unsupported schedule algorithm: {algorithm!r}")

    batch_size = int(schedule.get("batch_size", 0))
    block_updates = int(schedule.get("composition_block_updates", 0))
    if batch_size <= 0 or block_updates <= 0:
        raise ValueError("batch and composition block sizes must be positive")
    block_rows = batch_size * block_updates
    if len(schedule_indices) % block_rows:
        raise ValueError("schedule is not divisible by its composition block size")
    image_manifest = load_json(image_path)
    if not isinstance(image_manifest, dict):
        raise ValueError("image manifest must be an object")
    image_by_instance = image_map_for_instances(instance_ids, image_manifest)
    blocks = _composition(
        schedule_indices, instance_ids, image_by_instance, block_rows
    )
    minimum_repositories = int(schedule.get("minimum_repositories_per_block", 1))
    maximum_repository_share = float(schedule.get("maximum_repository_share", 1.0))
    for block in blocks:
        if block["unique_repositories"] < minimum_repositories:
            raise ValueError(
                f"block {block['block']} has only {block['unique_repositories']} repositories"
            )
        if block["max_repository_share"] > maximum_repository_share:
            raise ValueError(
                f"block {block['block']} repository share exceeds the contract"
            )

    return {
        "algorithm": algorithm,
        "batch_size": batch_size,
        "block_composition": blocks,
        "composition_block_updates": block_updates,
        "endpoint_index_map_sha256": sha256_file(endpoint_path),
        "image_manifest_sha256": sha256_file(image_path),
        "instance_count": len(instance_ids),
        "manifest_dataset_id": manifest.get("dataset_id"),
        "nonmonotonic_first_occurrence_order": not monotonic,
        "routing_row_count": len(routing_rows),
        "routing_sha256": sha256_file(routing_path),
        "schema": CERTIFICATE_SCHEMA,
        "seed": schedule.get("seed"),
        "selection_sha256": sha256_file(selection_path),
        "status": "pass",
        "unique_routing_indices": len(set(schedule_indices)),
    }


def write_evidence(directory: Path, certificate: Mapping[str, Any]) -> dict[str, str]:
    certificate_path = directory / "schedule-order-certificate.json"
    composition_path = directory / "schedule-block-composition.json"
    repo_csv_path = directory / "schedule-block-repository-counts.csv"
    image_csv_path = directory / "schedule-block-image-counts.csv"
    pair_jsonl_path = directory / "schedule-exact-repeat-pairs.jsonl"
    pair_csv_path = directory / "schedule-exact-repeat-pairs.csv"

    dump_json(certificate_path, dict(certificate))
    dump_json(composition_path, certificate["block_composition"])

    with repo_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block", "repository", "count"])
        for block in certificate["block_composition"]:
            for repository, count in block["repository_counts"].items():
                writer.writerow([block["block"], repository, count])
    with image_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block", "image", "count"])
        for block in certificate["block_composition"]:
            for image, count in block["image_counts"].items():
                writer.writerow([block["block"], image, count])

    manifest, schedule = _load_contract(directory)
    selection = manifest["selection"]
    instance_ids = load_json(_resolve(directory, selection["path"], "selection"))
    routing = load_jsonl(_resolve(directory, schedule["routing"], "routing"))
    image_manifest = load_json(_resolve(directory, schedule["image_manifest"], "image manifest"))
    image_by_instance = image_map_for_instances(instance_ids, image_manifest)
    pair_rows = []
    prefix_rows = int(schedule.get("unique_prefix_rows", 0))
    repeated_tail_rows = int(schedule.get("repeated_tail_rows", 0))
    if repeated_tail_rows:
        for pair_id in range(repeated_tail_rows):
            early = pair_id
            late = prefix_rows + pair_id
            if routing[early]["data_idx"] != routing[late]["data_idx"]:
                raise ValueError("declared repeat tail does not match the schedule prefix")
            data_idx = routing[early]["data_idx"]
            instance_id = instance_ids[data_idx]
            pair_rows.append(
                {
                    "data_idx": data_idx,
                    "early_position": early,
                    "early_update": early // int(schedule["batch_size"]) + 1,
                    "early_within_update": early % int(schedule["batch_size"]),
                    "image": image_by_instance[instance_id],
                    "instance_id": instance_id,
                    "instance_id_sha256": sha256_bytes(instance_id.encode("utf-8")),
                    "late_position": late,
                    "late_update": late // int(schedule["batch_size"]) + 1,
                    "late_within_update": late % int(schedule["batch_size"]),
                    "pair_id": pair_id,
                    "repository": repo_for_instance(instance_id),
                }
            )
    with pair_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    fieldnames = sorted(pair_rows[0]) if pair_rows else ["pair_id"]
    with pair_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    paths = [
        certificate_path,
        composition_path,
        repo_csv_path,
        image_csv_path,
        pair_jsonl_path,
        pair_csv_path,
    ]
    return {path.name: sha256_file(path) for path in paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-dir", required=True, type=Path)
    parser.add_argument("--write-evidence", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    certificate = certify_schedule(args.schedule_dir)
    if args.write_evidence:
        hashes = write_evidence(args.schedule_dir, certificate)
        certificate = certify_schedule(args.schedule_dir)
        print(json.dumps({"certificate": certificate, "evidence_sha256": hashes}, sort_keys=True))
    else:
        print(json.dumps(certificate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
