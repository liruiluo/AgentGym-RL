from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_PATH = _REPO_ROOT / "scripts/agentmemory/triton_cache_tool.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "agentmemory_triton_cache_tool_under_test", _TOOL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_autotune(
    root: Path, bucket: str, kernel: str, key: list[object]
) -> Path:
    path = root / bucket / f"{kernel}.autotune.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "key": key,
                "configs_timings": [[{"kwargs": {"BT": 32}}, [0.1]]],
            }
        ),
        encoding="utf-8",
    )
    return path


class TritonCacheToolTests(unittest.TestCase):
    def test_inventory_and_prewarmer_accept_required_kernels(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, kernel in enumerate(tool.DEFAULT_REQUIRED_KERNELS):
                _write_autotune(root, f"bucket-{index}", kernel, [128, index])

            inventory = tool.inventory_cache(root)
            self.assertEqual(inventory["autotune_files"], 4)
            self.assertEqual(inventory["invalid_autotune_files"], [])
            self.assertEqual(inventory["duplicate_function_keys"], [])
            self.assertEqual(
                set(inventory["kernel_counts"]),
                set(tool.DEFAULT_REQUIRED_KERNELS),
            )
            tool.assert_prewarmer_ready(inventory, min_autotune_files=4)

    def test_prewarmer_rejects_missing_kernel(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_autotune(root, "bucket", "l2norm_fwd_kernel", [128, 1])
            inventory = tool.inventory_cache(root)
            with self.assertRaisesRegex(tool.CacheToolError, "missing kernels"):
                tool.assert_prewarmer_ready(inventory)

    def test_inventory_accepts_same_function_key_for_distinct_variants(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_autotune(root, "bucket-a", "l2norm_fwd_kernel", [128, 1])
            _write_autotune(root, "bucket-b", "l2norm_fwd_kernel", [128, 1])
            inventory = tool.inventory_cache(root)
            self.assertEqual(inventory["unique_function_keys"], 1)
            self.assertEqual(inventory["unique_variant_keys"], 2)
            self.assertEqual(inventory["duplicate_function_keys"], [])
            self.assertEqual(len(inventory["cross_variant_function_keys"]), 1)
            tool.assert_prewarmer_ready(
                inventory,
                required_kernels=("l2norm_fwd_kernel",),
            )

    def test_reference_coverage_requires_every_compiled_variant(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            cache = root / "cache"
            _write_autotune(
                reference, "bucket-a", "l2norm_fwd_kernel", [128, 1]
            )
            _write_autotune(
                reference, "bucket-b", "l2norm_fwd_kernel", [128, 1]
            )
            _write_autotune(cache, "bucket-a", "l2norm_fwd_kernel", [128, 1])

            coverage = tool.verify_reference_coverage(reference, cache)
            self.assertEqual(len(coverage["missing_function_keys"]), 1)
            self.assertEqual(
                coverage["missing_function_keys"][0]["variant"], "bucket-b"
            )

    def test_seed_is_idempotent_and_covers_reference(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            _write_autotune(source, "bucket", "l2norm_fwd_kernel", [128, 1])
            binary = source / "bucket/kernel.cubin"
            binary.write_bytes(b"compiled-kernel")

            first = tool.seed_cache(source, destination)
            second = tool.seed_cache(source, destination)
            self.assertEqual(first["copied_files"], 2)
            self.assertEqual(first["reused_files"], 0)
            self.assertEqual(second["copied_files"], 0)
            self.assertEqual(second["reused_files"], 2)
            coverage = tool.verify_reference_coverage(source, destination)
            self.assertEqual(coverage["missing_function_keys"], [])

    def test_seed_fails_closed_on_content_conflict(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "kernel.json").write_text("source", encoding="utf-8")
            (destination / "kernel.json").write_text(
                "destination", encoding="utf-8"
            )
            with self.assertRaisesRegex(tool.CacheToolError, "conflict"):
                tool.seed_cache(source, destination)

    def test_seed_command_validates_source_before_creating_destination(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            (source / "broken.autotune.json").write_text(
                "not-json", encoding="utf-8"
            )

            with self.assertRaisesRegex(tool.CacheToolError, "invalid autotune"):
                tool.main(
                    [
                        "seed",
                        "--source",
                        str(source),
                        "--destination",
                        str(destination),
                        "--require-kernel",
                        "l2norm_fwd_kernel",
                        "--min-autotune-files",
                        "1",
                    ]
                )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
