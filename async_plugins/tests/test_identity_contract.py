from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import agentmemorygym_verl.identity as identity
from agentmemorygym_verl.identity import (
    EXPECTED_VERL_COMMIT,
    LOCKED_MODEL_FILE_SHA256,
    reject_ambient_identity,
    validate_outer_change_paths,
    validate_training_runtime_lock,
    verify_hash_manifest,
)

FIXTURES = Path("/tmp/openmle-v8-launch-fixtures-20260818")


class TestAMGFullyAsyncIdentity(unittest.TestCase):
    def test_verl_pin_contains_reviewed_fully_async_critic_and_fused_forward_fixes(
        self,
    ):
        self.assertEqual(
            EXPECTED_VERL_COMMIT,
            "0e9b07ff4117ff61f3594f797b0f708e8a6290fa",
        )

    def test_only_reviewed_verl_and_model_bytes_are_module_constants(self):
        self.assertRegex(EXPECTED_VERL_COMMIT, r"^[0-9a-f]{40}$")
        self.assertGreaterEqual(len(LOCKED_MODEL_FILE_SHA256), 7)
        for relative, digest in LOCKED_MODEL_FILE_SHA256.items():
            self.assertFalse(Path(relative).is_absolute())
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        for stale_name in (
            "LOCKED_OUTER_COMMIT",
            "LOCKED_INNER_COMMIT",
            "LOCKED_TRAINING_RUNTIME",
            "RICH_V8_ARTIFACT_SHA256",
        ):
            self.assertFalse(
                hasattr(identity, stale_name),
                f"dated publication constant leaked back into identity.py: {stale_name}",
            )

    def test_outer_changes_are_confined_to_the_plugin_and_bound_to_publication(self):
        publication_commit = "1" * 40
        report = validate_outer_change_paths(
            locked_outer_commit=publication_commit,
            ancestor_is_locked=True,
            committed_paths=(
                "async_plugins/agentmemorygym_verl/launch.py",
                "async_plugins/tests/test_launcher_contract.py",
            ),
            dirty_paths=(),
            require_clean=True,
        )
        self.assertEqual(report["publication_outer_commit"], publication_commit)
        self.assertEqual(report["committed_path_count"], 2)

        with self.assertRaisesRegex(RuntimeError, "not an ancestor"):
            validate_outer_change_paths(
                locked_outer_commit=publication_commit,
                ancestor_is_locked=False,
                committed_paths=(),
                dirty_paths=(),
                require_clean=True,
            )
        with self.assertRaisesRegex(RuntimeError, "outside async_plugins"):
            validate_outer_change_paths(
                locked_outer_commit=publication_commit,
                ancestor_is_locked=True,
                committed_paths=("AgentGym/agentenv/agentenv/envs/openmle_fast.py",),
                dirty_paths=(),
                require_clean=True,
            )
        with self.assertRaisesRegex(RuntimeError, "must be clean"):
            validate_outer_change_paths(
                locked_outer_commit=publication_commit,
                ancestor_is_locked=True,
                committed_paths=("async_plugins/agentmemorygym_verl/launch.py",),
                dirty_paths=("async_plugins/agentmemorygym_verl/launch.py",),
                require_clean=True,
            )

    def test_selected_file_hashes_are_verified_without_following_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inner").mkdir()
            (root / "outer").mkdir()
            (root / "inner" / "actions.py").write_text("actions\n", encoding="utf-8")
            (root / "outer" / "client.py").write_text("client\n", encoding="utf-8")
            expected = {
                "inner/actions.py": hashlib.sha256(b"actions\n").hexdigest(),
                "outer/client.py": hashlib.sha256(b"client\n").hexdigest(),
            }
            self.assertEqual(verify_hash_manifest(root, expected), expected)
            (root / "inner" / "actions.py").write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                verify_hash_manifest(root, expected)

    def test_training_runtime_is_publication_derived_but_shape_locked(self):
        runtime = {
            "base_model": "/models/qwen35",
            "python": "/runtime/bin/python3.12",
            "site_packages": "/runtime/lib/python3.12/site-packages",
            "bundle_sha256": "a" * 64,
            "bundle_sha256_file": "/runtime/bundle.sha256",
            "gpu_count": 8,
            "gpu_type": "B200",
        }
        self.assertEqual(validate_training_runtime_lock(runtime), runtime)
        b300_runtime = dict(runtime, gpu_type="B300")
        self.assertEqual(validate_training_runtime_lock(b300_runtime), b300_runtime)
        for field, value in (
            ("python", "relative/python"),
            ("bundle_sha256", "A" * 64),
            ("gpu_count", 4),
            ("gpu_type", "H200"),
        ):
            mutated = dict(runtime)
            mutated[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, field):
                validate_training_runtime_lock(mutated)

    def test_ambient_identity_conflicts_are_rejected_not_inherited(self):
        reject_ambient_identity({"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0,1"})
        for name in (
            "PYTHONPATH",
            "VERL_USE_EXTERNAL_MODULES",
            "VERL_FILE_LOGGER_PATH",
            "AMG_ENDPOINT_CLIENT_CONFIG_JSON",
            "OPENMLE_FAST_RUNTIME_OUTER_COMMIT",
            "AGENTMEMORY_MODEL_PATH",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, name):
                reject_ambient_identity({name: "caller-value"})

    def test_current_publication_fixture_is_self_describing_not_code_baked(self):
        source_path = FIXTURES / "source-lock.json"
        publication_path = FIXTURES / "publication-receipt.json"
        certificate_path = FIXTURES / "formal100-schedule-certificate.json"
        for path in (source_path, publication_path, certificate_path):
            self.assertTrue(path.is_file(), f"publication fixture missing: {path}")

        source = json.loads(source_path.read_text(encoding="utf-8"))
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        runtime = validate_training_runtime_lock(source["training_runtime"])
        self.assertEqual(runtime["gpu_count"], 8)
        for field in ("outer_commit", "inner_commit", "openmle_tasks_revision"):
            self.assertRegex(source["runtime_source"][field], r"^[0-9a-f]{40}$")
        for role in ("gate_only", "train_pool", "heldout"):
            manifest = source["integration"]["manifests"][role]
            routing = source["integration"]["routing"][role]
            self.assertGreater(manifest["task_count"], 0)
            self.assertGreater(manifest["source_family_count"], 0)
            self.assertRegex(manifest["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(routing["sha256"], r"^[0-9a-f]{64}$")
        # These documents may evolve; the code consumes their declared schemas
        # and cross-links rather than asserting this fixture's dated counts.
        self.assertIsInstance(publication, dict)
        self.assertIsInstance(certificate, dict)


if __name__ == "__main__":
    unittest.main()
