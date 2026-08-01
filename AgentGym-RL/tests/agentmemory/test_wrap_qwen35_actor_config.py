from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class WrapQwen35ActorConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "agentmemory"
            / "wrap_qwen35_actor_config.py"
        )
        spec = importlib.util.spec_from_file_location("wrap_qwen35_actor_config", source)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_compatible_text_and_outer_configs_pass(self) -> None:
        actor = {
            "model_type": "qwen3_5_text",
            "vocab_size": 32,
            "num_hidden_layers": 2,
            "hidden_size": 16,
        }
        base = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": dict(actor),
        }
        self.module.validate_compatible_configs(actor, base)

    def test_architecture_drift_fails_closed(self) -> None:
        actor = {
            "model_type": "qwen3_5_text",
            "vocab_size": 32,
            "num_hidden_layers": 2,
            "hidden_size": 16,
        }
        base = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {**actor, "num_attention_heads": 8},
        }
        with self.assertRaisesRegex(SystemExit, "num_attention_heads"):
            self.module.validate_compatible_configs(actor, base)

    def test_runtime_and_token_fields_may_differ(self) -> None:
        actor = {
            "model_type": "qwen3_5_text",
            "vocab_size": 32,
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "dtype": "float32",
            "eos_token_id": 7,
            "pad_token_id": 6,
            "transformers_version": "test-actor",
        }
        base = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                **actor,
                "dtype": "bfloat16",
                "eos_token_id": 6,
                "transformers_version": "test-base",
            },
        }
        self.module.validate_compatible_configs(actor, base)

    def test_json_write_is_atomic_and_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            payload = {"model_type": "qwen3_5_text", "vocab_size": 32}
            self.module.write_json(path, payload)
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
