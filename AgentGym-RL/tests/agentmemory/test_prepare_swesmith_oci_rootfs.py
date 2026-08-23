import argparse
import hashlib
import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agentmemory" / "prepare_swesmith_oci_rootfs.py"
SPEC = importlib.util.spec_from_file_location("prepare_swesmith_oci_rootfs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareSwesmithOciRootfsTest(unittest.TestCase):
    def test_rootfs_contract_does_not_require_one_python_layout(self):
        with tempfile.TemporaryDirectory() as raw:
            rootfs = Path(raw)
            for relative in ("testbed", "tmp", "var/tmp", "dev", "proc", "run"):
                (rootfs / relative).mkdir(parents=True, exist_ok=True)
            for relative in (
                "bin/bash",
                "usr/bin/setpriv",
                "usr/bin/prlimit",
                "usr/bin/env",
                "bin/sleep",
                "usr/bin/cut",
                "usr/bin/python3.11",
            ):
                path = rootfs / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"executable")
            self.assertEqual(MODULE._require_rootfs_contract(rootfs), rootfs / "bin/bash")

    def test_parses_binding_and_builds_profile_manifest(self):
        first = MODULE.parse_binding(
            "jyangballin/image-a=swebench/image-a@sha256:" + "a" * 64
        )
        second = MODULE.parse_binding(
            "jyangballin/image-b=swebench/image-b@sha256:" + "b" * 64
        )
        manifest = MODULE.build_image_manifest(
            (second, first),
            dataset_revision="c" * 40,
            source_revision="d" * 40,
        )
        self.assertEqual(
            manifest["images"],
            [
                {"image": "swebench/image-a", "digest": "sha256:" + "a" * 64},
                {"image": "swebench/image-b", "digest": "sha256:" + "b" * 64},
            ],
        )
        self.assertEqual(manifest["upstream"]["dataset_revision"], "c" * 40)

    def test_rejects_invalid_binding(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.parse_binding("image-without-profile")
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.parse_binding("source=profile@sha256:not-a-digest")

    def test_rejects_duplicate_manifest_identity(self):
        first = MODULE.parse_binding(
            "jyangballin/image-a=swebench/image@sha256:" + "a" * 64
        )
        second = MODULE.parse_binding(
            "jyangballin/image-b=swebench/image@sha256:" + "b" * 64
        )
        with self.assertRaisesRegex(ValueError, "unique profile"):
            MODULE.build_image_manifest(
                (first, second),
                dataset_revision="c" * 40,
                source_revision="d" * 40,
            )

    def test_normalizes_and_deduplicates_transport_prefixes(self):
        self.assertEqual(
            MODULE._transport_prefixes(
                "docker.1ms.run/",
                ("dockerproxy.net", "docker.1ms.run", "docker.1panel.live/"),
            ),
            ("docker.1ms.run", "dockerproxy.net", "docker.1panel.live"),
        )

    def test_rejects_empty_transport_prefix(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            MODULE._transport_prefixes("docker.1ms.run", ("  ",))

    def test_purges_corrupt_crane_layer_cache_entries(self):
        with tempfile.TemporaryDirectory() as raw:
            cache = Path(raw)
            valid_raw = b"verified compressed layer"
            valid_digest = hashlib.sha256(valid_raw).hexdigest()
            valid = cache / f"sha256:{valid_digest}"
            corrupt = cache / f"sha256:{'0' * 64}"
            ignored = cache / ".pull.lock"
            valid.write_bytes(valid_raw)
            corrupt.write_bytes(b"")
            ignored.write_bytes(b"")

            self.assertEqual(
                MODULE._purge_invalid_cached_layers(cache),
                (corrupt.name,),
            )
            self.assertEqual(valid.read_bytes(), valid_raw)
            self.assertFalse(corrupt.exists())
            self.assertTrue(ignored.exists())
            self.assertEqual(MODULE._purge_invalid_cached_layers(cache), ())

    def test_builds_digest_verified_offline_tarball_from_crane_cache(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache = root / "cache"
            partial = root / "partial"
            (cache / "blobs").mkdir(parents=True)
            (cache / "manifests").mkdir()
            partial.mkdir()

            config_raw = json.dumps(
                {
                    "architecture": "amd64",
                    "os": "linux",
                    "config": {"WorkingDir": "/testbed"},
                },
                separators=(",", ":"),
            ).encode()
            layer_raw = b"cached compressed layer"
            config_digest = "sha256:" + hashlib.sha256(config_raw).hexdigest()
            layer_digest = "sha256:" + hashlib.sha256(layer_raw).hexdigest()
            manifest = {
                "schemaVersion": 2,
                "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "config": {
                    "mediaType": "application/vnd.docker.container.image.v1+json",
                    "size": len(config_raw),
                    "digest": config_digest,
                },
                "layers": [
                    {
                        "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                        "size": len(layer_raw),
                        "digest": layer_digest,
                    }
                ],
            }
            manifest_raw = json.dumps(manifest, separators=(",", ":")).encode()
            binding = MODULE.ImageBinding(
                "source/image",
                "profile/image",
                "sha256:" + hashlib.sha256(manifest_raw).hexdigest(),
            )
            (cache / "blobs" / config_digest).write_bytes(config_raw)
            (cache / "blobs" / layer_digest).write_bytes(layer_raw)
            (cache / "manifests" / f"{binding.cache_name}.json").write_bytes(
                manifest_raw
            )

            actual_manifest, actual_config = MODULE._build_offline_cached_tarball(
                binding,
                partial=partial,
                layer_cache_root=cache,
            )

            self.assertEqual(actual_manifest, manifest_raw)
            self.assertEqual(actual_config, config_raw)
            with tarfile.open(partial / "image.tar") as archive:
                self.assertEqual(
                    set(archive.getnames()),
                    {
                        "manifest.json",
                        f"{config_digest.removeprefix('sha256:')}.json",
                        f"{layer_digest.removeprefix('sha256:')}.tar.gz",
                    },
                )

    def test_offline_tarball_rejects_corrupt_cached_blob(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache = root / "cache"
            partial = root / "partial"
            (cache / "blobs").mkdir(parents=True)
            (cache / "manifests").mkdir()
            partial.mkdir()
            config_raw = b'{"architecture":"amd64","os":"linux"}'
            config_digest = "sha256:" + hashlib.sha256(config_raw).hexdigest()
            manifest = {
                "schemaVersion": 2,
                "config": {
                    "size": len(config_raw),
                    "digest": config_digest,
                },
                "layers": [],
            }
            manifest_raw = json.dumps(manifest, separators=(",", ":")).encode()
            binding = MODULE.ImageBinding(
                "source/image",
                "profile/image",
                "sha256:" + hashlib.sha256(manifest_raw).hexdigest(),
            )
            (cache / "manifests" / f"{binding.cache_name}.json").write_bytes(
                manifest_raw
            )
            (cache / "blobs" / config_digest).write_bytes(b"corrupt")

            with self.assertRaisesRegex(RuntimeError, "cached blob size mismatch"):
                MODULE._build_offline_cached_tarball(
                    binding,
                    partial=partial,
                    layer_cache_root=cache,
                )


if __name__ == "__main__":
    unittest.main()
