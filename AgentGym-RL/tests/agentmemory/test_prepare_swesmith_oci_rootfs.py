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
    @staticmethod
    def _write_tar_member(archive, name, raw):
        import io

        info = tarfile.TarInfo(name)
        info.size = len(raw)
        info.mode = 0o644
        info.mtime = 0
        archive.addfile(info, io.BytesIO(raw))

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

    def test_materializes_a_subset_without_narrowing_the_frozen_manifest(self):
        first = MODULE.parse_binding(
            "jyangballin/image-a=swebench/image-a@sha256:" + "a" * 64
        )
        second = MODULE.parse_binding(
            "jyangballin/image-b=swebench/image-b@sha256:" + "b" * 64
        )
        bindings = (first, second)
        selected = MODULE.select_materialization_bindings(
            bindings, (second.profile_image,)
        )
        self.assertEqual(selected, (second,))
        self.assertEqual(
            len(
                MODULE.build_image_manifest(
                    bindings,
                    dataset_revision="c" * 40,
                    source_revision="d" * 40,
                )["images"]
            ),
            2,
        )

    def test_rejects_unknown_or_duplicate_materialization_profiles(self):
        binding = MODULE.parse_binding(
            "jyangballin/image-a=swebench/image-a@sha256:" + "a" * 64
        )
        with self.assertRaisesRegex(ValueError, "absent from the frozen bindings"):
            MODULE.select_materialization_bindings((binding,), ("swebench/missing",))
        with self.assertRaisesRegex(ValueError, "must be unique"):
            MODULE.select_materialization_bindings(
                (binding,), (binding.profile_image, binding.profile_image)
            )

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

    def test_loads_hash_bound_offline_image_asset(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metadata = root / ".metadata-staging"
            images = root / "images"
            metadata.mkdir()
            images.mkdir()
            source_image = "jyangballin/image-a"
            profile_image = "swebench/image-a"
            config_raw = json.dumps(
                {
                    "architecture": "amd64",
                    "os": "linux",
                    "config": {"WorkingDir": "/testbed"},
                },
                sort_keys=True,
            ).encode()
            config_sha = hashlib.sha256(config_raw).hexdigest()
            layer_raw = b"compressed layer fixture"
            layer_sha = hashlib.sha256(layer_raw).hexdigest()
            oci_manifest_raw = json.dumps(
                {
                    "schemaVersion": 2,
                    "config": {"digest": f"sha256:{config_sha}"},
                    "layers": [
                        {
                            "digest": f"sha256:{layer_sha}",
                            "size": len(layer_raw),
                        }
                    ],
                },
                sort_keys=True,
            ).encode()
            image_sha = hashlib.sha256(oci_manifest_raw).hexdigest()
            binding = MODULE.parse_binding(
                f"{source_image}={profile_image}@sha256:{image_sha}"
            )
            manifest_path = metadata / "image.manifest.json"
            config_path = metadata / "image.config.json"
            manifest_path.write_bytes(oci_manifest_raw)
            config_path.write_bytes(config_raw)
            image_tar = images / f"sha256-{image_sha}.tar"
            docker_manifest = json.dumps(
                [
                    {
                        "Config": f"sha256:{config_sha}",
                        "RepoTags": [f"{source_image}:i-was-a-digest"],
                        "Layers": [f"{layer_sha}.tar.gz"],
                    }
                ],
                sort_keys=True,
            ).encode()
            with tarfile.open(image_tar, "w", format=tarfile.GNU_FORMAT) as archive:
                self._write_tar_member(archive, f"sha256:{config_sha}", config_raw)
                self._write_tar_member(archive, f"{layer_sha}.tar.gz", layer_raw)
                self._write_tar_member(archive, "manifest.json", docker_manifest)
            prestage_path = root / "metadata-prestage.json"
            prestage = {
                "schema": MODULE._OFFLINE_PRESTAGE_SCHEMA,
                "status": "pass",
                "missing_layer_count": 0,
                "bad_layers": [],
                "images": [
                    {
                        "source_image": source_image,
                        "profile_image": profile_image,
                        "digest": f"sha256:{image_sha}",
                        "manifest": manifest_path.relative_to(root).as_posix(),
                        "config": config_path.relative_to(root).as_posix(),
                        "layers": [
                            {"digest": f"sha256:{layer_sha}", "size": len(layer_raw)}
                        ],
                    }
                ],
            }
            prestage_path.write_text(json.dumps(prestage), encoding="utf-8")
            asset_manifest_path = root / "offline-image-assets.json"
            asset_manifest = {
                "schema": MODULE._OFFLINE_ASSET_SCHEMA,
                "status": "pass",
                "network_required_at_launch": False,
                "image_count": 1,
                "source_metadata_prestage": {
                    "path": prestage_path.name,
                    "sha256": hashlib.sha256(prestage_path.read_bytes()).hexdigest(),
                },
                "images": [
                    {
                        "source_image": source_image,
                        "profile_image": profile_image,
                        "digest": f"sha256:{image_sha}",
                        "manifest_sha256": image_sha,
                        "config_sha256": config_sha,
                        "image_tar": {
                            "path": image_tar.relative_to(root).as_posix(),
                            "bytes": image_tar.stat().st_size,
                            "sha256": hashlib.sha256(image_tar.read_bytes()).hexdigest(),
                        },
                    }
                ],
            }
            asset_manifest_path.write_text(json.dumps(asset_manifest), encoding="utf-8")

            assets = MODULE.load_offline_image_assets(asset_manifest_path, (binding,))
            self.assertEqual(assets[profile_image].image_tarball, image_tar.resolve())
            self.assertEqual(assets[profile_image].manifest_raw, oci_manifest_raw)
            self.assertEqual(assets[profile_image].config_raw, config_raw)

            image_tar.write_bytes(image_tar.read_bytes() + b"drift")
            with self.assertRaisesRegex(RuntimeError, "(byte count|digest) drifted"):
                MODULE.load_offline_image_assets(asset_manifest_path, (binding,))


if __name__ == "__main__":
    unittest.main()
