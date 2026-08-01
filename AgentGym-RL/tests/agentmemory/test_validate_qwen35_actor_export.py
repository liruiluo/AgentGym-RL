from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


DEPENDENCIES_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("safetensors") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "torch, safetensors, and transformers are required",
)
class ValidateQwen35ActorExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "agentmemory"
            / "validate_qwen35_actor_export.py"
        )
        spec = importlib.util.spec_from_file_location(
            "validate_qwen35_actor_export",
            source,
        )
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls.module)
        except ImportError as exc:
            raise unittest.SkipTest(
                f"export validator dependencies are unavailable: {exc}"
            ) from exc

    def test_weight_shapes_normalize_language_model_prefix(self) -> None:
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temp_dir:
            actor_dir = Path(temp_dir)
            save_file(
                {
                    "model.language_model.embed_tokens.weight": torch.zeros(2, 3),
                    "lm_head.weight": torch.zeros(2, 3),
                },
                actor_dir / "model.safetensors",
            )
            shapes, raw_key_count = self.module.load_weight_shapes(actor_dir)
            self.assertEqual(raw_key_count, 2)
            self.assertEqual(shapes["model.embed_tokens.weight"], (2, 3))
            self.assertEqual(shapes["lm_head.weight"], (2, 3))

    def test_weight_shape_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(SystemExit, "shape_mismatches"):
            self.module.verify_weight_shapes(
                {"model.weight": (2, 4)},
                {"model.weight": (2, 3)},
            )

    def test_normalized_key_collision_fails_closed(self) -> None:
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temp_dir:
            actor_dir = Path(temp_dir)
            save_file(
                {"model.weight": torch.zeros(2, 3)},
                actor_dir / "model-00001-of-00002.safetensors",
            )
            save_file(
                {"model.language_model.weight": torch.zeros(2, 3)},
                actor_dir / "model-00002-of-00002.safetensors",
            )
            with self.assertRaisesRegex(SystemExit, "duplicate normalized key"):
                self.module.load_weight_shapes(actor_dir)


if __name__ == "__main__":
    unittest.main()
