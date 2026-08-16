from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

from swebench_triad_eval.oci import (
    CachedOciStore,
    DockerCli,
    OciCacheError,
    attest_rootfs,
    build_docker_archive,
    ensure_repository_mirror,
    materialize_rootfs,
    recover_stale_partials,
    require_task_eviction_ready,
)


IMAGE = "swebench/sweb.eval.x86_64.owner_1776_repo-0001:latest"
ARMS = ("native", "amg_compaction_only", "amg_memory")


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_entry(name: str, payload: bytes = b"x", mode: int = 0o644) -> dict:
    return {"kind": "file", "name": name, "payload": payload, "mode": mode}


def directory_entry(name: str, mode: int = 0o755) -> dict:
    return {"kind": "directory", "name": name, "mode": mode}


def symlink_entry(name: str, target: str) -> dict:
    return {"kind": "symlink", "name": name, "target": target, "mode": 0o777}


def device_entry(name: str) -> dict:
    return {"kind": "device", "name": name, "mode": 0o600}


def layer_bytes(entries: list[dict]) -> tuple[bytes, str]:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for entry in entries:
            member = tarfile.TarInfo(entry["name"])
            member.mode = entry.get("mode", 0o644)
            member.uid = os.getuid()
            member.gid = os.getgid()
            member.mtime = 1
            kind = entry["kind"]
            if kind == "file":
                payload = entry.get("payload", b"")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            elif kind == "directory":
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            elif kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = entry["target"]
                archive.addfile(member)
            elif kind == "device":
                member.type = tarfile.CHRTYPE
                member.devmajor = 1
                member.devminor = 3
                archive.addfile(member)
            else:
                raise AssertionError(f"unsupported fixture kind: {kind}")
    uncompressed = raw.getvalue()
    return gzip.compress(uncompressed, mtime=0), "sha256:" + sha256(uncompressed)


def required_rootfs_entries() -> list[dict]:
    directories = [
        "testbed",
        "tmp",
        "var",
        "var/tmp",
        "dev",
        "proc",
        "run",
        "bin",
        "usr",
        "usr/bin",
    ]
    files = [
        "bin/bash",
        "usr/bin/setpriv",
        "usr/bin/prlimit",
        "usr/bin/env",
        "bin/sleep",
        "usr/bin/cut",
    ]
    return [directory_entry(name) for name in directories] + [
        file_entry(name, b"fixture executable\n", 0o755) for name in files
    ]


class OciFixture:
    def __init__(self, root: Path, layers: list[list[dict]]) -> None:
        self.root = root
        self.index_path = root / "instance-manifest-index.jsonl"
        self.manifest_root = root / "manifests"
        self.blob_root = root / "blob-cache"
        self.cache_root = root / "rootfs-cache"
        self.manifest_root.mkdir(parents=True)
        self.blob_root.mkdir()

        descriptors = []
        diff_ids = []
        for entries in layers:
            payload, diff_id = layer_bytes(entries)
            digest = "sha256:" + sha256(payload)
            (self.blob_root / digest).write_bytes(payload)
            descriptors.append(
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": digest,
                    "size": len(payload),
                }
            )
            diff_ids.append(diff_id)

        config = {
            "architecture": "amd64",
            "os": "linux",
            "config": {"WorkingDir": "/testbed/"},
            "rootfs": {"type": "layers", "diff_ids": diff_ids},
        }
        config_raw = json_bytes(config)
        config_digest = "sha256:" + sha256(config_raw)
        (self.blob_root / config_digest).write_bytes(config_raw)
        config_descriptor = {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": config_digest,
            "size": len(config_raw),
        }
        manifest = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": config_descriptor,
            "layers": descriptors,
        }
        manifest_raw = json_bytes(manifest)
        manifest_digest = "sha256:" + sha256(manifest_raw)
        (self.manifest_root / f"sha256-{manifest_digest[7:]}.json").write_bytes(
            manifest_raw
        )
        self.row = {
            "compressed_layer_bytes": sum(item["size"] for item in descriptors),
            "config": config_descriptor,
            "digest": manifest_digest,
            "image": IMAGE,
            "layers": descriptors,
            "manifest_sha256": manifest_digest[7:],
            "media_type": "application/vnd.oci.image.manifest.v1+json",
            "platform": "linux/amd64",
            "transport": "fixture.invalid",
        }
        self.write_rows([self.row])

    def write_rows(self, rows: list[dict]) -> None:
        self.index_path.write_bytes(b"".join(json_bytes(row) + b"\n" for row in rows))

    def store(self) -> CachedOciStore:
        return CachedOciStore(
            index_path=self.index_path,
            manifest_root=self.manifest_root,
            blob_root=self.blob_root,
        )


class CachedOciTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def fixture(self, extra_layers: list[list[dict]] | None = None) -> OciFixture:
        layers = [required_rootfs_entries()]
        if extra_layers:
            layers.extend(extra_layers)
        fixture_number = len(list(self.root.iterdir()))
        return OciFixture(self.root / f"fixture-{fixture_number}", layers)

    def test_descriptors_and_every_cached_blob_are_exact(self) -> None:
        fixture = self.fixture()
        binding = fixture.store().resolve(IMAGE)
        self.assertEqual(binding.image, IMAGE)
        self.assertEqual(binding.working_dir, "/testbed")
        self.assertEqual(binding.manifest_digest, fixture.row["digest"])

        drifted = dict(fixture.row)
        drifted["layers"] = [dict(item) for item in fixture.row["layers"]]
        drifted["layers"][0]["size"] += 1
        fixture.write_rows([drifted])
        with self.assertRaises(OciCacheError):
            fixture.store().resolve(IMAGE)

        fixture.write_rows([fixture.row])
        config_path = fixture.blob_root / fixture.row["config"]["digest"]
        config_path.write_bytes(config_path.read_bytes() + b"drift")
        with self.assertRaises(OciCacheError):
            fixture.store().resolve(IMAGE)

    def test_duplicate_image_aliases_are_rejected(self) -> None:
        fixture = self.fixture()
        fixture.write_rows([fixture.row, fixture.row])
        with self.assertRaises(OciCacheError):
            fixture.store().resolve(IMAGE)

    def test_tar_traversal_unsafe_links_devices_and_aliases_are_rejected(self) -> None:
        attacks = (
            [file_entry("../../escape", b"escape")],
            [symlink_entry("testbed/link", "../../escape")],
            [device_entry("testbed/device")],
            [file_entry("testbed/alias"), file_entry("./testbed/alias")],
        )
        for attack in attacks:
            with self.subTest(attack=attack):
                fixture = self.fixture([attack])
                binding = fixture.store().resolve(IMAGE)
                with self.assertRaises(OciCacheError):
                    materialize_rootfs(binding, fixture.cache_root)

    def test_whiteouts_are_applied_before_same_layer_entries(self) -> None:
        first = [
            file_entry("testbed/remove.txt", b"remove"),
            directory_entry("testbed/opaque"),
            file_entry("testbed/opaque/old.txt", b"old"),
            file_entry("testbed/keep.txt", b"keep"),
        ]
        second = [
            file_entry("testbed/.wh.remove.txt", b""),
            file_entry("testbed/opaque/.wh..wh..opq", b""),
            file_entry("testbed/opaque/new.txt", b"new"),
        ]
        fixture = self.fixture([first, second])
        binding = fixture.store().resolve(IMAGE)
        cache_dir = materialize_rootfs(binding, fixture.cache_root)
        rootfs = cache_dir / "rootfs"
        self.assertFalse((rootfs / "testbed/remove.txt").exists())
        self.assertFalse((rootfs / "testbed/opaque/old.txt").exists())
        self.assertEqual((rootfs / "testbed/opaque/new.txt").read_bytes(), b"new")
        self.assertEqual((rootfs / "testbed/keep.txt").read_bytes(), b"keep")
        self.assertFalse(
            any(path.name.startswith(".wh.") for path in rootfs.rglob("*"))
        )

    def test_full_tree_attestation_detects_any_mutation(self) -> None:
        fixture = self.fixture([[file_entry("testbed/keep.txt", b"keep")]])
        binding = fixture.store().resolve(IMAGE)
        cache_dir = materialize_rootfs(binding, fixture.cache_root)
        receipt = attest_rootfs(cache_dir)
        self.assertEqual(receipt["status"], "pass")
        (cache_dir / "rootfs/testbed/keep.txt").write_bytes(b"changed")
        with self.assertRaises(OciCacheError):
            attest_rootfs(cache_dir)

    def test_stale_partial_recovery_is_scoped_and_rejects_symlinks(self) -> None:
        fixture = self.fixture()
        digest = fixture.row["digest"]
        name = f"sha256-{digest[7:]}"
        fixture.cache_root.mkdir()
        stale = fixture.cache_root / f".{name}.partial-123-dead"
        stale.mkdir()
        unrelated = fixture.cache_root / ".unrelated.partial-123-dead"
        unrelated.mkdir()
        removed = recover_stale_partials(fixture.cache_root, digest)
        self.assertEqual(removed, (stale.name,))
        self.assertTrue(unrelated.is_dir())

        outside = self.root / "outside"
        outside.mkdir()
        attack = fixture.cache_root / f".{name}.partial-456-link"
        attack.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OciCacheError):
            recover_stale_partials(fixture.cache_root, digest)
        self.assertTrue(outside.is_dir())

    def test_incomplete_rootfs_cache_is_never_reused(self) -> None:
        fixture = self.fixture()
        binding = fixture.store().resolve(IMAGE)
        incomplete = fixture.cache_root / binding.cache_name
        incomplete.mkdir(parents=True)
        (incomplete / "rootfs").mkdir()

        with self.assertRaises(OciCacheError):
            materialize_rootfs(binding, fixture.cache_root)
        self.assertTrue(incomplete.is_dir())

    def test_build_docker_archive_uses_verified_config_and_layers(self) -> None:
        fixture = self.fixture([[file_entry("testbed/extra.txt", b"extra")]])
        binding = fixture.store().resolve(IMAGE)
        archive_path = build_docker_archive(binding, self.root / "image.tar")

        with tarfile.open(archive_path, mode="r:") as archive:
            manifest_file = archive.extractfile("manifest.json")
            self.assertIsNotNone(manifest_file)
            manifest = json.loads(manifest_file.read())
            self.assertEqual(manifest[0]["RepoTags"], [IMAGE])
            self.assertEqual(
                manifest[0]["Config"],
                f"{binding.config_digest[7:]}.json",
            )
            self.assertEqual(len(manifest[0]["Layers"]), len(binding.layers))
            for layer, name in zip(binding.layers, manifest[0]["Layers"]):
                layer_file = archive.extractfile(name)
                self.assertIsNotNone(layer_file)
                self.assertEqual(sha256(layer_file.read()), layer.diff_id[7:])

        receipt = json.loads(
            archive_path.with_name(f"{archive_path.name}.receipt.json").read_text()
        )
        self.assertEqual(receipt["config_digest"], binding.config_digest)
        self.assertEqual(receipt["archive_sha256"], sha256(archive_path.read_bytes()))

    def test_docker_alias_must_resolve_to_the_config_digest(self) -> None:
        fixture = self.fixture()
        binding = fixture.store().resolve(IMAGE)

        def mismatched_inspect(argv: list[str]) -> subprocess.CompletedProcess:
            self.assertIn("image", argv)
            self.assertIn("inspect", argv)
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="sha256:" + "f" * 64 + "\n",
            )

        docker = DockerCli(
            socket_path=self.root / "docker.sock",
            executor=mismatched_inspect,
            verify_socket=False,
        )
        with self.assertRaises(OciCacheError):
            docker.ensure_loaded(binding, self.root / "unused.tar")

    def test_repository_mirror_is_local_and_base_commit_bound(self) -> None:
        source_root = self.root / "rootfs"
        testbed = source_root / "testbed"
        testbed.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(testbed)], check=True)
        (testbed / "file.txt").write_text("one\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(testbed), "add", "file.txt"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(testbed),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(testbed), "rev-parse", "HEAD"], text=True
        ).strip()
        mirror_root = self.root / "mirrors"
        mirror_root.mkdir()
        stale = mirror_root / ".owner__repo.git.partial-123-dead"
        stale.mkdir()
        mirror = ensure_repository_mirror(
            source_root,
            mirror_root,
            repo="owner/repo",
            base_commit=commit,
        )
        resolved = subprocess.check_output(
            ["git", "--git-dir", str(mirror), "rev-parse", f"{commit}^{{commit}}"],
            text=True,
        ).strip()
        self.assertEqual(resolved, commit)
        self.assertFalse(stale.exists())

        outside = self.root / "outside-mirror"
        outside.mkdir()
        attack = mirror_root / ".owner__other.git.partial-456-link"
        attack.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OciCacheError):
            ensure_repository_mirror(
                source_root,
                mirror_root,
                repo="owner/other",
                base_commit=commit,
            )
        self.assertTrue(outside.is_dir())

        with self.assertRaises(OciCacheError):
            ensure_repository_mirror(
                source_root,
                mirror_root,
                repo="owner/repo",
                base_commit="0" * 40,
            )

    def test_eviction_requires_exact_triad_and_three_boolean_outcomes(self) -> None:
        accepted = [
            {"instance_id": "owner__repo-0001", "arm": arm, "status": "accepted"}
            for arm in ARMS
        ]
        outcomes = [
            {"instance_id": "owner__repo-0001", "arm": arm, "resolved": False}
            for arm in ARMS
        ]
        receipt = require_task_eviction_ready(
            "owner__repo-0001", accepted, outcomes
        )
        self.assertEqual(receipt["arms"], list(ARMS))
        with self.assertRaises(OciCacheError):
            require_task_eviction_ready(
                "owner__repo-0001", accepted[:-1], outcomes
            )
        invalid = [dict(row) for row in outcomes]
        invalid[-1]["resolved"] = 0
        with self.assertRaises(OciCacheError):
            require_task_eviction_ready("owner__repo-0001", accepted, invalid)


if __name__ == "__main__":
    unittest.main()
