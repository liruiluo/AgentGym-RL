from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_QWEN35_MODEL = _REPO_ROOT / "verl/models/transformers/qwen3_5.py"
_QWEN35_RUNTIME = _REPO_ROOT / "verl/workers/qwen35_runtime.py"
_QWEN35_SYNC = _REPO_ROOT / "verl/workers/qwen35_weight_sync.py"
_DP_ACTOR = _REPO_ROOT / "verl/workers/agent_actor/dp_actor.py"
_DP_CRITIC = _REPO_ROOT / "verl/workers/agent_critic/dp_critic.py"
_FSDP_WORKER = _REPO_ROOT / "verl/workers/agent_fsdp_workers.py"
_FSDP_VLLM = _REPO_ROOT / "verl/workers/sharding_manager/fsdp_vllm.py"
_MAIN_PPO = _REPO_ROOT / "verl/agent_trainer/main_ppo.py"
_RAY_BASE = _REPO_ROOT / "verl/single_controller/ray/base.py"
_VLLM_RUNTIME_CONFIG = (
    _REPO_ROOT
    / "verl/workers/rollout/agent_vllm_rollout/vllm_runtime_config.py"
)
_VLLM_ROLLOUT = (
    _REPO_ROOT
    / "verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self, values, *, dtype=None, device="cuda"):
        self.values = list(values)
        self.dtype = dtype
        self.device = device

    def to(self, *, dtype=None):
        return _FakeTensor(self.values, dtype=dtype, device=self.device)

    def detach(self):
        return self

    def cpu(self):
        return _FakeTensor(self.values, dtype=self.dtype, device="cpu")


class Qwen35PackedWorkerContractTests(unittest.TestCase):
    def test_actor_and_critic_forward_real_cu_seqlens(self):
        for path in (_DP_ACTOR, _DP_CRITIC):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertIn(
                    "input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input",
                    source,
                )
                self.assertIn("qwen3_5_packed_forward_kwargs(", source)
                self.assertIn("cu_seqlens,", source)

    def test_qwen35_patch_is_idempotent(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.LongTensor = object
        fake_torch.Tensor = object
        fake_torch.autograd = types.SimpleNamespace(Function=object)
        fake_torch.distributed = types.ModuleType("torch.distributed")
        fake_torch.distributed.ProcessGroup = object
        fake_nn = types.ModuleType("torch.nn")
        fake_functional = types.ModuleType("torch.nn.functional")
        fake_nn.functional = fake_functional
        fake_qwen35 = types.ModuleType(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )

        class Decoder:
            def forward(self):
                return None

        class GatedDeltaNet:
            def forward(self):
                return None

        fake_qwen35.Qwen3_5DecoderLayer = Decoder
        fake_qwen35.Qwen3_5GatedDeltaNet = GatedDeltaNet
        fake_ulysses = types.ModuleType("verl.utils.ulysses")
        fake_ulysses.get_ulysses_sequence_parallel_group = lambda: None
        fake_ulysses.get_ulysses_sequence_parallel_world_size = lambda: 1
        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "torch.distributed": fake_torch.distributed,
                "torch.nn": fake_nn,
                "torch.nn.functional": fake_functional,
                "transformers.models.qwen3_5.modeling_qwen3_5": fake_qwen35,
                "verl.utils.ulysses": fake_ulysses,
            },
        ):
            module = _load_module(_QWEN35_MODEL, "qwen35_patch_under_test")
            self.assertTrue(module.apply_qwen3_5_packed_forward_patch())
            self.assertFalse(module.apply_qwen3_5_packed_forward_patch())
            self.assertIs(Decoder.forward, module.qwen3_5_decoder_layer_forward)
            self.assertIs(
                GatedDeltaNet.forward,
                module.qwen3_5_gated_delta_net_forward,
            )


class OfficialVllmRuntimeConfigTests(unittest.TestCase):
    def _load_runtime_config(self):
        return _load_module(
            _VLLM_RUNTIME_CONFIG, "official_vllm_runtime_config_under_test"
        )

    def test_current_verl_cudagraph_default(self):
        module = self._load_runtime_config()
        self.assertEqual(
            module.resolve_official_vllm_compilation_config(
                enforce_eager=False
            ),
            {"cudagraph_mode": "FULL_AND_PIECEWISE"},
        )
        self.assertIsNone(
            module.resolve_official_vllm_compilation_config(
                enforce_eager=True
            )
        )

    def test_explicit_config_and_capture_sizes_are_forwarded(self):
        module = self._load_runtime_config()
        self.assertEqual(
            module.resolve_official_vllm_compilation_config(
                enforce_eager=False,
                configured='{"backend": "inductor"}',
                cudagraph_capture_sizes=[1, 8, 48],
            ),
            {
                "backend": "inductor",
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": [1, 8, 48],
            },
        )

    def test_rollout_gates_old_assertion_to_legacy_vllm(self):
        source = _VLLM_ROLLOUT.read_text(encoding="utf-8")
        self.assertIn("if not self._official_vllm:", source)
        self.assertIn(
            "resolve_official_vllm_compilation_config(", source
        )
        self.assertIn(
            "official_vllm_kwargs['compilation_config']", source
        )

    @staticmethod
    def _fake_dynamic_triton():
        class CacheKnob:
            @property
            def dir(self):
                return os.environ.get("TRITON_CACHE_DIR", "/default/triton")

        return types.SimpleNamespace(
            __version__="3.6.0",
            knobs=types.SimpleNamespace(cache=CacheKnob()),
        )

    def test_training_triton_cache_is_opt_in(self):
        module = self._load_runtime_config()
        with patch.dict(os.environ, {}, clear=True):
            result = module.restore_training_triton_cache_after_vllm(
                triton_module=self._fake_dynamic_triton()
            )
            self.assertIsNone(result)
            self.assertNotIn("TRITON_CACHE_DIR", os.environ)
            self.assertNotIn("FLA_CACHE_RESULTS", os.environ)

    def test_training_triton_cache_requires_absolute_path(self):
        module = self._load_runtime_config()
        with patch.dict(
            os.environ,
            {"VERL_TRAINING_TRITON_CACHE_DIR": "relative/cache"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "absolute"):
                module.restore_training_triton_cache_after_vllm(
                    triton_module=self._fake_dynamic_triton()
                )

    def test_training_triton_cache_restores_runtime_knob(self):
        module = self._load_runtime_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory) / "stable-triton"
            with patch.dict(
                os.environ,
                {
                    "VERL_TRAINING_TRITON_CACHE_DIR": str(cache_dir),
                    "FLA_CACHE_RESULTS": "0",
                },
                clear=True,
            ):
                result = module.restore_training_triton_cache_after_vllm(
                    triton_module=self._fake_dynamic_triton()
                )
                self.assertTrue(cache_dir.is_dir())
                stable_dir = str(cache_dir.resolve())
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], stable_dir)
                self.assertEqual(os.environ["FLA_CACHE_RESULTS"], "1")
                self.assertEqual(result["requested_dir"], stable_dir)
                self.assertEqual(result["runtime_dir"], stable_dir)
                self.assertEqual(result["triton_version"], "3.6.0")

    def test_training_triton_cache_fails_on_runtime_mismatch(self):
        module = self._load_runtime_config()
        fake_triton = types.SimpleNamespace(
            __version__="3.6.0",
            knobs=types.SimpleNamespace(
                cache=types.SimpleNamespace(dir="/unexpected/triton")
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory) / "stable-triton"
            with patch.dict(
                os.environ,
                {"VERL_TRAINING_TRITON_CACHE_DIR": str(cache_dir)},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "did not take effect"):
                    module.restore_training_triton_cache_after_vllm(
                        triton_module=fake_triton
                    )

    def test_training_triton_cache_restore_order_and_env_propagation(self):
        rollout_source = _VLLM_ROLLOUT.read_text(encoding="utf-8")
        engine_init = rollout_source.index(
            "self.inference_engine = LLM(**official_vllm_kwargs)"
        )
        cache_restore = rollout_source.index(
            "restore_training_triton_cache_after_vllm(", engine_init
        )
        engine_client = rollout_source.index(
            "llm_engine = getattr(self.inference_engine", engine_init
        )
        self.assertLess(engine_init, cache_restore)
        self.assertLess(cache_restore, engine_client)

        for path in (_MAIN_PPO, _RAY_BASE):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertIn("'VERL_TRAINING_TRITON_CACHE_DIR'", source)
                self.assertIn("'FLA_CACHE_RESULTS'", source)


class Qwen35RuntimeContractTests(unittest.TestCase):
    def _load_runtime(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "test"
        fake_torch.long = "long"
        fake_model = types.ModuleType("verl.models.transformers.qwen3_5")
        fake_model.is_qwen3_5_model_type = lambda value: value in {
            "qwen3_5",
            "qwen3_5_text",
        }
        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "verl.models.transformers.qwen3_5": fake_model,
            },
        ):
            return _load_module(_QWEN35_RUNTIME, "qwen35_runtime_under_test")

    def test_packed_kwargs_preserve_boundaries(self):
        module = self._load_runtime()
        wrapped = types.SimpleNamespace(
            module=types.SimpleNamespace(
                config=types.SimpleNamespace(model_type="qwen3_5_text")
            )
        )
        boundaries = _FakeTensor([0, 3, 7], dtype="int32")
        result = module.qwen3_5_packed_forward_kwargs(wrapped, boundaries, 1)
        self.assertEqual(result["cu_seqlens"].values, [0, 3, 7])
        self.assertEqual(result["cu_seqlens"].dtype, "long")
        self.assertEqual(result["cu_seqlens_cpu"].device, "cpu")

    def test_sp_greater_than_one_remains_fail_closed(self):
        module = self._load_runtime()
        wrapped = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="qwen3_5_text")
        )
        with self.assertRaisesRegex(NotImplementedError, "SP>1"):
            module.qwen3_5_packed_forward_kwargs(
                wrapped,
                _FakeTensor([0, 4]),
                2,
            )

    def test_runtime_gate_names_all_three_packed_kernel_interfaces(self):
        source = _QWEN35_RUNTIME.read_text(encoding="utf-8")
        self.assertIn('chunk_gated_delta_rule, "cu_seqlens"', source)
        self.assertIn('causal_conv1d_fn, "seq_idx"', source)
        self.assertIn('_flash_attention_forward,', source)
        self.assertIn('"position_ids"', source)


class Qwen35WeightSyncContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_model = types.ModuleType("verl.models.transformers.qwen3_5")
        fake_model.is_qwen3_5_model_type = lambda value: value in {
            "qwen3_5",
            "qwen3_5_text",
        }
        with patch.dict(
            sys.modules,
            {"verl.models.transformers.qwen3_5": fake_model},
        ):
            cls.module = _load_module(
                _QWEN35_SYNC,
                "qwen35_weight_sync_under_test",
            )

    def test_actor_namespace_mapping(self):
        self.assertEqual(
            self.module.map_actor_weight_name_for_vllm(
                "_fsdp_wrapped_module.model.layers.0.self_attn.q_proj.weight",
                model_type="qwen3_5_text",
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(
            self.module.map_actor_weight_name_for_vllm(
                "module.lm_head.weight",
                model_type="qwen3_5_text",
            ),
            "lm_head.weight",
        )

    def test_weight_sync_requires_complete_source_and_target_manifests(self):
        source_names = [
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
        ]
        result = self.module.validate_qwen35_mapped_source_names(
            source_names,
            expected_names=source_names,
        )
        self.assertEqual(result["mapped_source_parameter_count"], 2)
        with self.assertRaisesRegex(RuntimeError, "source mapping mismatch"):
            self.module.validate_qwen35_mapped_source_names(
                source_names[:1],
                expected_names=source_names,
            )

        target_names = {
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            "visual.patch_embed.weight",
        }
        result = self.module.validate_qwen35_vllm_load_coverage(
            loaded_names={
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
            },
            target_parameter_names=target_names,
        )
        self.assertEqual(result["loaded_text_parameter_count"], 2)
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            self.module.validate_qwen35_vllm_load_coverage(
                loaded_names={"language_model.model.embed_tokens.weight"},
                target_parameter_names=target_names,
            )

    def test_formal_path_has_no_skip_weight_sync_escape_hatch(self):
        for path in (_MAIN_PPO, _RAY_BASE, _FSDP_VLLM):
            with self.subTest(path=path):
                self.assertNotIn(
                    "VERL_AGENTMEMORY_SKIP_VLLM_WEIGHT_SYNC",
                    path.read_text(encoding="utf-8"),
                )
        worker_source = _FSDP_WORKER.read_text(encoding="utf-8")
        self.assertIn("sync_weight_format=hf", worker_source)


if __name__ == "__main__":
    unittest.main()
