import importlib.util
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRANSPORT = (
    _REPO_ROOT
    / "verl/workers/sharding_manager/vllm_hf_sync_transport.py"
)
_FSDP_VLLM = _REPO_ROOT / "verl/workers/sharding_manager/fsdp_vllm.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _typed_object(module, name):
    cls = type(name, (), {})
    cls.__module__ = module
    return cls()


class DirectInprocTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module(_TRANSPORT, "vllm_hf_sync_transport_under_test")

    def test_file_is_default_and_direct_transports_are_explicit(self):
        self.assertEqual(self.module.resolve_hf_sync_transport({}), "file")
        self.assertEqual(
            self.module.resolve_hf_sync_transport(
                {self.module.TRANSPORT_ENV: "direct_inproc"}
            ),
            "direct_inproc",
        )
        self.assertEqual(
            self.module.resolve_hf_sync_transport(
                {self.module.TRANSPORT_ENV: "direct_inproc_streaming"}
            ),
            "direct_inproc_streaming",
        )
        self.assertTrue(
            self.module.uses_streaming_sharded_state_dict(
                "direct_inproc_streaming"
            )
        )
        self.assertFalse(
            self.module.uses_streaming_sharded_state_dict("direct_inproc")
        )
        with self.assertRaisesRegex(
            ValueError, "direct_inproc, direct_inproc_streaming, file"
        ):
            self.module.resolve_hf_sync_transport(
                {self.module.TRANSPORT_ENV: "automatic"}
            )

    def test_streaming_requires_exact_dtensor_contract(self):
        dtensor = _typed_object("torch.distributed.tensor", "DTensor")
        dtensor.full_tensor = lambda: "full"
        self.assertEqual(
            self.module.require_streaming_sharded_tensor("weight", dtensor),
            "torch.distributed.tensor.DTensor",
        )

        regular_tensor = _typed_object("torch", "Tensor")
        regular_tensor.full_tensor = lambda: "full"
        with self.assertRaisesRegex(RuntimeError, "every sharded state-dict"):
            self.module.require_streaming_sharded_tensor(
                "weight", regular_tensor
            )

        missing_method = _typed_object("torch.distributed.tensor", "DTensor")
        with self.assertRaisesRegex(RuntimeError, "missing full_tensor"):
            self.module.require_streaming_sharded_tensor(
                "weight", missing_method
            )

    def make_runtime(self):
        executor = _typed_object(
            "vllm.v1.executor.uniproc_executor", "UniProcExecutor"
        )
        core = _typed_object("vllm.v1.engine.core", "EngineCore")
        core.model_executor = executor
        client = _typed_object("vllm.v1.engine.core_client", "InprocClient")
        client.engine_core = core
        engine = object.__new__(type("FakeLLMEngine", (), {}))
        engine.engine_core = client
        return engine

    def test_exact_inproc_uniproc_chain_passes(self):
        actual = self.module.require_direct_inproc_runtime(
            self.make_runtime(), infer_tp_size=1
        )
        self.assertEqual(
            actual["model_executor"],
            "vllm.v1.executor.uniproc_executor.UniProcExecutor",
        )

    def test_multiprocess_or_tp_runtime_fails_closed(self):
        engine = self.make_runtime()
        engine.engine_core = _typed_object(
            "vllm.v1.engine.core_client", "SyncMPClient"
        )
        with self.assertRaisesRegex(RuntimeError, "non-serializing"):
            self.module.require_direct_inproc_runtime(engine, infer_tp_size=1)
        with self.assertRaisesRegex(RuntimeError, "tensor_model_parallel_size=1"):
            self.module.require_direct_inproc_runtime(
                self.make_runtime(), infer_tp_size=2
            )


class FSDPTransportWiringTests(unittest.TestCase):
    def test_direct_inproc_is_opt_in_and_file_fallback_remains(self):
        source = _FSDP_VLLM.read_text(encoding="utf-8")
        self.assertIn("self._hf_sync_transport = (", source)
        self.assertIn("resolve_hf_sync_transport()", source)
        self.assertIn("effective_transport = self._hf_sync_transport", source)
        self.assertIn("if effective_transport == 'direct_inproc':", source)
        self.assertIn("require_direct_inproc_runtime(", source)
        self.assertIn("agentmemory_load_hf_direct)", source)
        self.assertIn("torch.save(hf_weights, sync_file)", source)
        self.assertIn("transport=effective_transport", source)

    def test_streaming_selects_sharded_state_dict_and_never_materializes_batch(self):
        source = _FSDP_VLLM.read_text(encoding="utf-8")
        self.assertIn("self._streaming_hf_sync", source)
        self.assertIn("ShardedStateDictConfig(offload_to_cpu=False)", source)
        self.assertIn("sharded_tensor.full_tensor()", source)
        self.assertIn("agentmemory_load_hf_streaming", source)
        self.assertIn("extract_streamed_source_fingerprint", source)
        self.assertIn("vLLM did not consume every streamed actor tensor", source)


if __name__ == "__main__":
    unittest.main()
