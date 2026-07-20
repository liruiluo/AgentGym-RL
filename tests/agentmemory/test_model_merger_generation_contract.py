from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


DEPENDENCIES_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "torch and transformers are required")
class ModelMergerGenerationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (
            Path(__file__).resolve().parents[2]
            / "AgentGym-RL"
            / "scripts"
            / "model_merger.py"
        )
        spec = importlib.util.spec_from_file_location("agentmemory_model_merger", source)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls.module)
        except ImportError as exc:
            raise unittest.SkipTest(f"model merger dependencies are unavailable: {exc}") from exc

    def write_generation_config(self, path: Path, *, eos, pad) -> None:
        config = self.module.GenerationConfig(eos_token_id=eos, pad_token_id=pad)
        config.save_pretrained(path)

    def test_dual_eos_contract_survives_attach_and_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            self.write_generation_config(path, eos=[151645, 151643], pad=151643)
            model = type("DummyModel", (), {})()
            contract = self.module.attach_generation_config(
                str(path),
                model,
                expected_eos_token_ids=[151645, 151643],
                expected_pad_token_id=151643,
            )
            model.generation_config.save_pretrained(path)
            self.module.verify_generation_config_contract(str(path), contract)
            self.assertEqual(contract["eos_token_id"], [151645, 151643])
            self.assertEqual(contract["pad_token_id"], 151643)

    def test_scalar_eos_contract_remains_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            self.write_generation_config(path, eos=2, pad=0)
            model = type("DummyModel", (), {})()
            contract = self.module.attach_generation_config(str(path), model)
            model.generation_config.save_pretrained(path)
            self.module.verify_generation_config_contract(str(path), contract)
            self.assertIs(type(contract["eos_token_id"]), int)

    def test_dual_eos_collapse_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            self.write_generation_config(path, eos=[151645, 151643], pad=151643)
            model = type("DummyModel", (), {})()
            contract = self.module.attach_generation_config(str(path), model)
            self.write_generation_config(path, eos=151645, pad=151643)
            with self.assertRaisesRegex(RuntimeError, "eos_token_id"):
                self.module.verify_generation_config_contract(str(path), contract)

    def test_formal_expected_dual_eos_rejects_scalar_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            self.write_generation_config(path, eos=151645, pad=151643)
            model = type("DummyModel", (), {})()
            with self.assertRaisesRegex(RuntimeError, "frozen EOS contract"):
                self.module.attach_generation_config(
                    str(path),
                    model,
                    expected_eos_token_ids=[151645, 151643],
                    expected_pad_token_id=151643,
                )

    def test_alternate_eos_pad_collision_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            self.write_generation_config(path, eos=[151645, 151643], pad=151643)
            model = type("DummyModel", (), {})()
            contract = self.module.attach_generation_config(str(path), model)
            sampled = [42, 151643]
            self.assertIn(sampled[-1], contract["eos_token_id"])
            self.assertEqual(sampled[-1], contract["pad_token_id"])

    def test_real_tiny_model_save_roundtrip_preserves_contract_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            config = self.module.AutoConfig.for_model(
                "gpt2",
                vocab_size=32,
                n_positions=32,
                n_embd=8,
                n_layer=1,
                n_head=1,
                eos_token_id=2,
                pad_token_id=0,
            )
            model = self.module.AutoModelForCausalLM.from_config(config)
            model.save_pretrained(path)
            self.write_generation_config(path, eos=[2, 0], pad=0)
            tokenizer_path = path / "tokenizer_config.json"
            tokenizer_payload = {"sentinel": "must-not-drift", "eos_token": "</s>"}
            tokenizer_path.write_text(
                json.dumps(tokenizer_payload, sort_keys=True),
                encoding="utf-8",
            )
            config_before = self.module.AutoConfig.from_pretrained(path).to_dict()
            tokenizer_before = tokenizer_path.read_bytes()

            self.module.save_pretrained_with_generation_contract(
                model=model,
                hf_path=str(path),
                state_dict=model.state_dict(),
                expected_eos_token_ids=[2, 0],
                expected_pad_token_id=0,
            )

            self.module.verify_generation_config_contract(
                str(path),
                {"eos_token_id": [2, 0], "pad_token_id": 0},
            )
            self.assertEqual(
                self.module.AutoConfig.from_pretrained(path).to_dict(),
                config_before,
            )
            self.assertEqual(tokenizer_path.read_bytes(), tokenizer_before)

    def test_plain_and_sharded_causal_branches_use_guarded_save_helper(self) -> None:
        source = (
            Path(__file__).resolve().parents[2]
            / "AgentGym-RL"
            / "scripts"
            / "model_merger.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(source.count("save_pretrained_with_generation_contract("), 3)
        self.assertEqual(source.count("model.save_pretrained(hf_path, state_dict=state_dict)"), 1)


if __name__ == "__main__":
    unittest.main()
