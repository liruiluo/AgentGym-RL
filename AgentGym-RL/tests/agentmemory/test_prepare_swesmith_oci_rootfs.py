import argparse
import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
