import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "agentmemory"
SCRIPT = SCRIPT_DIR / "prefetch_swesmith_oci_layers.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("prefetch_swesmith_oci_layers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def manifest_raw(*layers):
    return json.dumps(
        {
            "schemaVersion": 2,
            "config": {"digest": "sha256:" + "f" * 64, "size": 1},
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": digest,
                    "size": size,
                }
                for digest, size in layers
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")


def binding_for(raw, name):
    digest = hashlib.sha256(raw).hexdigest()
    return MODULE.parse_binding(
        f"jyangballin/{name}=swebench/{name}@sha256:{digest}"
    )


class PrefetchSwesmithOciLayersTest(unittest.TestCase):
    def test_builds_deduplicated_layer_plan(self):
        shared = ("sha256:" + "a" * 64, 7)
        first_raw = manifest_raw(shared, ("sha256:" + "b" * 64, 11))
        second_raw = manifest_raw(shared, ("sha256:" + "c" * 64, 13))
        first = binding_for(first_raw, "image-a")
        second = binding_for(second_raw, "image-b")

        plan, image_layers = MODULE.build_layer_plan(
            (first, second),
            {first.digest: first_raw, second.digest: second_raw},
        )

        self.assertEqual(len(plan), 3)
        self.assertEqual(
            plan[shared[0]].source_images,
            ("jyangballin/image-a", "jyangballin/image-b"),
        )
        self.assertEqual(len(image_layers[first.digest]), 2)
        self.assertEqual(len(image_layers[second.digest]), 2)

    def test_rejects_manifest_digest_mismatch(self):
        raw = manifest_raw(("sha256:" + "a" * 64, 7))
        binding = MODULE.parse_binding(
            "jyangballin/image=swebench/image@sha256:" + "0" * 64
        )
        with self.assertRaisesRegex(RuntimeError, "manifest bytes do not match"):
            MODULE.parse_manifest_layers(raw, binding)

    def test_resumes_partial_blob_without_exposing_token_in_argv(self):
        raw = b"verified compressed layer"
        descriptor = MODULE.LayerDescriptor(
            digest="sha256:" + hashlib.sha256(raw).hexdigest(),
            size=len(raw),
        )
        plan = MODULE.LayerPlan(
            descriptor=descriptor,
            source_images=("jyangballin/image",),
        )
        calls = []

        def fake_run(argv, *, environment, timeout):
            calls.append(tuple(argv))
            if argv[1:4] == ["auth", "token", "-H"]:
                if argv[-1].startswith("bad.registry/"):
                    return subprocess.CompletedProcess(
                        argv, 1, stdout=b"", stderr=b"unsupported auth"
                    )
                return subprocess.CompletedProcess(
                    argv, 0, stdout=b"Authorization: Bearer secret-token\n", stderr=b""
                )
            output = Path(argv[argv.index("--output") + 1])
            config = Path(argv[argv.index("--config") + 1])
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertIn("secret-token", config.read_text(encoding="utf-8"))
            self.assertNotIn("secret-token", " ".join(argv))
            existing = output.read_bytes() if output.exists() else b""
            if len(existing) < 5:
                output.write_bytes(raw[:5])
                return subprocess.CompletedProcess(argv, 18, stdout=b"", stderr=b"EOF")
            output.write_bytes(raw)
            return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(MODULE, "_run_command", side_effect=fake_run), mock.patch.object(
                MODULE.time, "sleep", return_value=None
            ):
                result = MODULE._download_layer(
                    plan,
                    shared_cache_root=root,
                    crane=Path("/fake/crane"),
                    curl=Path("/fake/curl"),
                    prefixes=("bad.registry", "docker.1ms.run"),
                    environment={},
                    attempts=3,
                    timeout_seconds=30,
                )
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["selection_attempts"], 4)
            self.assertEqual((root / descriptor.digest).read_bytes(), raw)
            self.assertFalse((root / ".partials" / f"{descriptor.digest}.partial").exists())

        curl_calls = [call for call in calls if "--continue-at" in call]
        self.assertEqual(len(curl_calls), 2)
        for call in curl_calls:
            index = call.index("--continue-at")
            self.assertEqual(call[index + 1], "-")

    def test_seeds_shard_with_hardlink_on_same_filesystem(self):
        raw = b"one immutable layer"
        descriptor = MODULE.LayerDescriptor(
            digest="sha256:" + hashlib.sha256(raw).hexdigest(),
            size=len(raw),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared" / descriptor.digest
            shard = root / "shard" / descriptor.digest
            shared.parent.mkdir()
            shared.write_bytes(raw)
            method = MODULE._install_verified_blob(shared, shard, descriptor)
            self.assertEqual(method, "hardlink")
            self.assertEqual(shard.read_bytes(), raw)
            self.assertEqual(os.stat(shared).st_ino, os.stat(shard).st_ino)

    def test_reuses_digest_check_for_unchanged_blob_identity(self):
        raw = b"immutable compressed layer"
        descriptor = MODULE.LayerDescriptor(
            digest="sha256:" + hashlib.sha256(raw).hexdigest(),
            size=len(raw),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / descriptor.digest
            path.write_bytes(raw)
            MODULE._VERIFIED_BLOB_FILES.clear()
            with mock.patch.object(
                MODULE, "_sha256_file", wraps=MODULE._sha256_file
            ) as sha256_file:
                self.assertTrue(MODULE._is_valid_blob(path, descriptor))
                self.assertTrue(MODULE._is_valid_blob(path, descriptor))
            self.assertEqual(sha256_file.call_count, 1)

    def test_relinks_existing_valid_shard_blob_to_local_shared_cache(self):
        raw = b"deduplicated immutable layer"
        descriptor = MODULE.LayerDescriptor(
            digest="sha256:" + hashlib.sha256(raw).hexdigest(),
            size=len(raw),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared" / descriptor.digest
            shard = root / "shard" / descriptor.digest
            shared.parent.mkdir()
            shard.parent.mkdir()
            shared.write_bytes(raw)
            shard.write_bytes(raw)
            self.assertNotEqual(os.stat(shared).st_ino, os.stat(shard).st_ino)
            MODULE._VERIFIED_BLOB_FILES.clear()
            method = MODULE._install_verified_blob(shared, shard, descriptor)
            self.assertEqual(method, "relinked")
            self.assertEqual(os.stat(shared).st_ino, os.stat(shard).st_ino)


if __name__ == "__main__":
    unittest.main()
