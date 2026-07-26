from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_QWEN35_MODEL = _REPO_ROOT / "verl/models/transformers/qwen3_5.py"
_QWEN35_RUNTIME = _REPO_ROOT / "verl/workers/qwen35_runtime.py"
_QWEN35_SYNC = _REPO_ROOT / "verl/workers/qwen35_weight_sync.py"
_FUSED_PPO = _REPO_ROOT / "verl/utils/experimental/torch_functional.py"
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

        class CausalLM:
            def forward(self, *args, **kwargs):
                return (args, kwargs)

        fake_qwen35.Qwen3_5DecoderLayer = Decoder
        fake_qwen35.Qwen3_5GatedDeltaNet = GatedDeltaNet
        fake_qwen35.Qwen3_5ForCausalLM = CausalLM
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
            self.assertIs(
                CausalLM.forward,
                module.qwen3_5_causal_lm_forward_dispatch,
            )
            self.assertEqual(CausalLM().forward("normal", flag=True), (("normal",), {"flag": True}))

    def test_fused_ppo_backport_uses_upstream_gradient_fix(self):
        source = _FUSED_PPO.read_text(encoding="utf-8")
        self.assertIn(
            "hidden_states_requires_grad = hidden_states.requires_grad",
            source,
        )
        self.assertIn(
            "hidden_states.requires_grad_(hidden_states_requires_grad)",
            source,
        )

    def test_actor_fused_path_is_explicit_and_critic_remains_separate(self):
        actor_source = _DP_ACTOR.read_text(encoding="utf-8")
        critic_source = _DP_CRITIC.read_text(encoding="utf-8")
        self.assertIn("_verl_fused_ppo=True", actor_source)
        self.assertIn("shift_labels=input_ids_rmpad_rolled.unsqueeze(0)", actor_source)
        self.assertNotIn("_verl_fused_ppo=True", critic_source)


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
        self.assertIn("enable_prefix_caching=rollout_config.get(", source)


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
