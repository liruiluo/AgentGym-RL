from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from paired_eval.manifest import expand_manifest
from swebench_triad_eval.identity import (
    DatasetPins,
    ImageIndexPins,
    ModelFilePin,
    build_manifest,
    verify_dataset,
    verify_image_index,
    verify_model_files,
    verify_source_identity,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def canonical_json_line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def common_payload() -> dict[str, object]:
    return {
        "model": {
            "model_id": "Qwen3.5-4B",
            "revision": "model-receipt-v1",
            "tokenizer_sha256": SHA_A,
        },
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 2048,
            "stop": [],
        },
        "budgets": {
            "max_policy_turns": 250,
            "max_total_tokens": 8388608,
            "max_tool_calls": 250,
            "max_wall_seconds": 1800.0,
            "max_prompt_tokens": 30720,
            "max_model_tokens": 32768,
            "max_observation_tokens": 8192,
            "action_observation_envelope_tokens": 0,
        },
        "compaction": {
            "policy": "policy_authored_task_neutral_v1",
            "trigger": "wrapper_token_pressure_v1",
            "summary_max_tokens": 2048,
            "summary_instruction_sha256": SHA_B,
            "context_pressure_policy_sha256": SHA_C,
            "context_transition_schema": (
                "agentmemory_task_neutral_context_transition_v1"
            ),
            "action_accounting": "global_policy_action_budget_v1",
            "config_sha256": SHA_D,
        },
        "source": {
            "outer_commit": "aa2e9c80d572b513b5849c6d9b37a8dc4698bbc3",
            "inner_commit": "a0cc3ecf989ee89ba19a8e979617b4ec38909331",
            "adapter_sha256": SHA_B,
            "runner_sha256": SHA_C,
        },
        "runtime": {
            "image_digest": "sha256:" + SHA_D,
            "runtime_sha256": SHA_A,
            "compute_class": "1xB200",
        },
        "grader": {
            "name": "swebench-v4.1.0",
            "revision": "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
            "config_sha256": SHA_B,
        },
    }


class IdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_source_identity_is_exact(self) -> None:
        receipt = verify_source_identity(
            "aa2e9c80d572b513b5849c6d9b37a8dc4698bbc3",
            "a0cc3ecf989ee89ba19a8e979617b4ec38909331",
        )
        self.assertEqual(receipt["status"], "pass")
        with self.assertRaises(ValueError):
            verify_source_identity("0" * 40, receipt["inner_commit"])
        with self.assertRaises(ValueError):
            verify_source_identity(receipt["outer_commit"], "0" * 40)

    def write_dataset(self, ids: list[str]) -> tuple[Path, DatasetPins]:
        path = self.root / "dataset.jsonl"
        payload = b"".join(
            canonical_json_line(
                {
                    "instance_id": instance_id,
                    "repo": "owner/repo",
                    "base_commit": "1" * 40,
                    "problem_statement": f"problem {instance_id}",
                }
            )
            for instance_id in ids
        )
        path.write_bytes(payload)
        id_ledger = "".join(instance_id + "\n" for instance_id in ids).encode()
        return path, DatasetPins(
            row_count=500,
            jsonl_sha256=hashlib.sha256(payload).hexdigest(),
            id_ledger_sha256=hashlib.sha256(id_ledger).hexdigest(),
        )

    def test_dataset_requires_500_sorted_unique_rows_and_hashes(self) -> None:
        ids = [f"owner__repo-{index:04d}" for index in range(500)]
        path, pins = self.write_dataset(ids)
        receipt = verify_dataset(path, pins=pins)
        self.assertEqual(receipt["instance_ids"], ids)
        self.assertEqual(receipt["rows"], 500)

        duplicate_path, duplicate_pins = self.write_dataset(ids[:-1] + [ids[-2]])
        with self.assertRaises(ValueError):
            verify_dataset(duplicate_path, pins=duplicate_pins)

        unsorted = list(ids)
        unsorted[0], unsorted[1] = unsorted[1], unsorted[0]
        unsorted_path, unsorted_pins = self.write_dataset(unsorted)
        with self.assertRaises(ValueError):
            verify_dataset(unsorted_path, pins=unsorted_pins)

        with self.assertRaises(ValueError):
            verify_dataset(
                path,
                pins=DatasetPins(500, SHA_A, pins.id_ledger_sha256),
            )

    def write_image_index(self) -> tuple[Path, ImageIndexPins]:
        path = self.root / "images.jsonl"
        rows = [
            {
                "image": (
                    "swebench/sweb.eval.x86_64.owner_1776_repo-"
                    f"{index:04d}:latest"
                ),
                "digest": "sha256:" + f"{index:064x}",
                "platform": "linux/amd64",
            }
            for index in range(500)
        ]
        payload = b"".join(canonical_json_line(row) for row in rows)
        path.write_bytes(payload)
        tag_ledger = "".join(row["image"] + "\n" for row in rows).encode()
        digest_tsv = "".join(
            row["image"] + "\t" + row["digest"] + "\n" for row in rows
        ).encode()
        return path, ImageIndexPins(
            row_count=500,
            index_sha256=hashlib.sha256(payload).hexdigest(),
            tag_ledger_sha256=hashlib.sha256(tag_ledger).hexdigest(),
            digest_tsv_sha256=hashlib.sha256(digest_tsv).hexdigest(),
        )

    def test_image_index_requires_exact_sorted_digest_ledger(self) -> None:
        path, pins = self.write_image_index()
        receipt = verify_image_index(path, pins=pins)
        self.assertEqual(receipt["rows"], 500)
        self.assertEqual(receipt["digest_tsv_sha256"], pins.digest_tsv_sha256)

        rows = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join([rows[1], rows[0], *rows[2:]]) + "\n")
        drifted = path.read_bytes()
        with self.assertRaises(ValueError):
            verify_image_index(
                path,
                pins=ImageIndexPins(
                    500,
                    hashlib.sha256(drifted).hexdigest(),
                    pins.tag_ledger_sha256,
                    pins.digest_tsv_sha256,
                ),
            )

    def test_model_files_reject_content_size_and_extra_file_drift(self) -> None:
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_bytes(b"config")
        (model / "tokenizer.json").write_bytes(b"tokenizer")
        pins = {
            name: ModelFilePin(
                size=(model / name).stat().st_size,
                sha256=hashlib.sha256((model / name).read_bytes()).hexdigest(),
            )
            for name in ("config.json", "tokenizer.json")
        }
        receipt = verify_model_files(model, pins=pins)
        self.assertEqual(receipt["file_count"], 2)

        (model / "config.json").write_bytes(b"drift")
        with self.assertRaises(ValueError):
            verify_model_files(model, pins=pins)
        (model / "config.json").write_bytes(b"config")
        (model / "unexpected.txt").write_text("unexpected")
        with self.assertRaises(ValueError):
            verify_model_files(model, pins=pins)

    def test_manifest_is_exact_task_major_500_by_three(self) -> None:
        task_ids = [f"owner__repo-{index:04d}" for index in range(500)]
        manifest = build_manifest(
            task_ids,
            common=common_payload(),
            run_id="swebench-verified-triad-20260816",
        )
        configs = expand_manifest(manifest)
        self.assertEqual(len(configs), 1500)
        self.assertEqual(
            [config.capability.arm.value for config in configs[:3]],
            ["native", "amg_compaction_only", "amg_memory"],
        )
        self.assertEqual({config.task.seed for config in configs}, {0})
        self.assertEqual(
            {config.treatment_excluded_config_sha256 for config in configs[:3]},
            {configs[0].treatment_excluded_config_sha256},
        )
        self.assertEqual(manifest["common"]["budgets"]["max_policy_turns"], 250)
        self.assertEqual(manifest["common"]["decoding"]["max_output_tokens"], 2048)

        with self.assertRaises(ValueError):
            build_manifest(
                [*task_ids[:-2], task_ids[-1], task_ids[-2]],
                common=common_payload(),
                run_id="swebench-verified-triad-20260816",
            )
        with self.assertRaises(ValueError):
            build_manifest(
                task_ids[:-1],
                common=common_payload(),
                run_id="swebench-verified-triad-20260816",
            )


if __name__ == "__main__":
    unittest.main()
