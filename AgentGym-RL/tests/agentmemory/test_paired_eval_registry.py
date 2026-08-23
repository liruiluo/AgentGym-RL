from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


OUTER_ROOT = Path(__file__).resolve().parents[2]
INNER_ROOT = OUTER_ROOT.parent / "AgentGym"
for package_root in (
    INNER_ROOT / "agentenv",
    INNER_ROOT / "agentenv-agentmemory",
    INNER_ROOT / "agentenv-gaia-text",
    INNER_ROOT / "agentenv-swesmith",
    INNER_ROOT / "agentenv-swebench-verified",
    INNER_ROOT / "agentenv-mlebench-lite",
):
    path = str(package_root)
    if path not in sys.path:
        sys.path.insert(0, path)

# Importing agentenv.controller normally pulls the model/training stack and
# torch. The frozen adapter clients need only the dependency-light env/types
# modules, matching the adapters' own client-test boundary.
controller = ModuleType("agentenv.controller")
controller.__path__ = [
    str(INNER_ROOT / "agentenv" / "agentenv" / "controller")
]
sys.modules.setdefault("agentenv.controller", controller)
envs = ModuleType("agentenv.envs")
envs.__path__ = [str(INNER_ROOT / "agentenv" / "agentenv" / "envs")]
sys.modules.setdefault("agentenv.envs", envs)

from agentenv.controller.env import BaseEnvClient  # noqa: E402
from agentenv.controller.types import (  # noqa: E402
    PolicyContextPressure,
    StepOutput as AgentStepOutput,
)


class BaseTask:
    def __init__(self, client_args, n_clients=1, *args, **kwargs) -> None:
        del args, kwargs
        self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]


controller.BaseEnvClient = BaseEnvClient
controller.BaseTask = BaseTask
controller.StepOutput = AgentStepOutput

from test_paired_eval_support import (  # noqa: E402
    Arm,
    ManualClock,
    make_config,
    make_fake_runtime,
)

from paired_eval.controller import DependencyLightPolicyTurnController  # noqa: E402
from paired_eval.contracts import (  # noqa: E402
    CONTEXT_OPERATION_APPEND,
    CONTEXT_OPERATION_REPLACE,
    CONTEXT_TRANSITION_SCHEMA,
    EXTERNAL_MEMORY_ROUTE,
    POLICY_COMPACTION_ROUTE,
    TaskNeutralStepReceipt,
    capability_for_arm,
)
from paired_eval.evidence import (  # noqa: E402
    AppendSafeJsonlWriter,
    PrivateEvidenceStore,
)
from paired_eval.manifest import (  # noqa: E402
    RuntimeBindings,
    execute_manifest,
)
from paired_eval.registry import (  # noqa: E402
    AdapterHooks,
    ClientEnvironmentAdapter,
    ClientStepProxy,
    DEFAULT_ADAPTER_SPECS,
    PairedEvalRegistry,
    lifecycle_roots,
    make_runtime_factory,
    treatment_excluded_messages,
)
from paired_eval.runner import PairedRunner  # noqa: E402


EXPECTED_CLIENTS = {
    "gaia_text": "agentenv_gaia_text.client:GaiaTextEnvClient",
    "swebench_verified": (
        "agentenv.envs.swebench_verified:SwebenchVerifiedEnvClient"
    ),
    "mlebench_lite": "agentenv.envs.mlebench_lite:MLEBenchLiteEnvClient",
}


class BoundModel:
    def __init__(self, model_config) -> None:
        self.model_config = model_config


def config_for(spec, arm: Arm):
    config = make_config(
        benchmark=spec.benchmark,
        arm=arm,
        artifact_type=spec.artifact_type,
    )
    return replace(
        config,
        task=replace(config.task, native_tools=spec.native_tools),
        capability=capability_for_arm(arm),
    )


def structured_raw_step_info(
    spec,
    kind: str,
    *,
    raw_native_before: int = 0,
    policy_before: int = 0,
    context_before: int = 0,
):
    if kind not in {"ordinary", "compaction", "memory"}:
        raise ValueError("unsupported raw fixture step kind")
    operation = (
        CONTEXT_OPERATION_REPLACE
        if kind == "compaction"
        else CONTEXT_OPERATION_APPEND
    )
    raw_native_after = raw_native_before + (kind != "compaction")
    context_after = context_before + (kind == "compaction")
    info = {
        "schema": "agentmemory_task_neutral_transition_v1",
        "env_info": {},
        "action_submission": {"kind": "ordinary"},
        "native_step_before": raw_native_before,
        "native_step_after": raw_native_after,
        "native_call_count_before": raw_native_before,
        "native_call_count_after": raw_native_after,
        "context_epoch_before": context_before,
        "context_epoch_after": context_after,
        "session_epoch_before": 0,
        "session_epoch_after": 0,
        "policy_step_before": policy_before,
        "policy_step_after": policy_before + 1,
        "context_transition": {
            "schema": CONTEXT_TRANSITION_SCHEMA,
            "operation": operation,
            "messages": (
                [{"role": "system", "content": "retained"}]
                if operation == CONTEXT_OPERATION_REPLACE
                else []
            ),
        },
        "wrapper_evidence": {"event": kind},
    }
    if kind == "memory":
        if spec.benchmark == "gaia_text":
            info["env_info"] = {"domain_action": "workspace"}
            info["action_submission"] = {"kind": "workspace"}
        else:
            info["env_info"] = {
                "external_memory_operation": "read_write"
            }
    return info


def raw_step_info(spec, arm: Arm):
    kind = {
        Arm.NATIVE: "ordinary",
        Arm.AMG_COMPACTION_ONLY: "compaction",
        Arm.AMG_MEMORY: "memory",
    }[arm]
    return structured_raw_step_info(spec, kind)


def registry_manifest_payload():
    base = config_for(DEFAULT_ADAPTER_SPECS[0], Arm.NATIVE)
    return {
        "schema": "amg.paired_eval.manifest",
        "schema_version": "2.0.0",
        "run_id": base.run_id,
        "arms": [arm.value for arm in Arm],
        "common": {
            "model": base.model.to_payload(),
            "decoding": base.decoding.to_payload(),
            "budgets": base.budgets.to_payload(),
            "compaction": base.compaction.to_payload(),
            "source": base.source.to_payload(),
            "runtime": base.runtime.to_payload(),
            "grader": base.grader.to_payload(),
        },
        "tasks": [
            replace(
                config_for(spec, Arm.NATIVE).task,
                protocol=f"{spec.benchmark}@integration-fixture-v1",
                task_id=f"{spec.benchmark}-fixture-001",
                task_index=index,
            ).to_payload()
            for index, spec in enumerate(DEFAULT_ADAPTER_SPECS)
        ],
    }


class _FixtureResponse:
    status_code = 200
    text = ""

    def __init__(self, value) -> None:
        self.value = value

    def json(self):
        return self.value


class _FixtureTransport:
    def __init__(self, spec, config, client_type) -> None:
        self.spec = spec
        self.config = config
        self.client_module = sys.modules[client_type.__module__]
        self.native_steps = 0
        self.reset_response = self._canonical_reset_response()
        self.close_response = (
            True
            if spec.benchmark == "gaia_text"
            else {"closed": True, "id": 1}
            if spec.benchmark == "swebench_verified"
            else {"closed": True}
        )

    def _canonical_reset_response(self):
        info = {}
        if self.spec.benchmark == "mlebench_lite":
            info = {"counters": self.client_module._zero_counters()}
        return {
            "observation": "fixture observation",
            "reward": 0.0,
            "done": False,
            "info": info,
        }

    def _mle_metadata(self):
        module = self.client_module
        contract = module._resource_contract(
            max_actions=30,
            max_submission_bytes=100_000_000,
            max_shell_timeout_ms=3_600_000,
        )
        return {
            "schema": module.METADATA_SCHEMA,
            "upstream_commit": module.UPSTREAM_COMMIT,
            "split_sha256": module.SPLIT_SHA256,
            "competition_ids": list(module.LITE_COMPETITION_IDS),
            "task_count": 22,
            "public_manifest_sha256": "1" * 64,
            "runner_sha256": "2" * 64,
            "runtime_digest": "3" * 64,
            "submission_path": module.SUBMISSION_PATH,
            "modes": list(module.MODES),
            "resource_contract": contract,
            "resource_contract_sha256": module._resource_contract_sha256(contract),
        }

    def request_value(self, method, path, **kwargs):
        if path == "metadata":
            if self.spec.benchmark == "gaia_text":
                return {"task_count": 127, "max_policy_steps": 12}
            if self.spec.benchmark == "mlebench_lite":
                return self._mle_metadata()
            return {}
        if path == "create":
            if self.spec.benchmark == "swebench_verified":
                return {"id": 1, "capability": "fixture-slot-capability"}
            if self.spec.benchmark == "mlebench_lite":
                return {"id": 1, "capability_token": "a" * 64}
            return {"id": 1}
        if path == "reset":
            self.native_steps = 0
            return self.reset_response
        if path == "step" and self.spec.benchmark == "swebench_verified":
            self.native_steps += 1
            action = kwargs["json"]["action"]
            info = {"step": self.native_steps}
            if (
                self.config.capability.external_read_write_memory
                and "/run/amg_memory/" in action
            ):
                info["external_memory_operation"] = "write"
            return {
                "observation": "fixture step",
                "reward": 0.0,
                "done": False,
                "info": info,
            }
        if path in {"close", "abort"}:
            return self.close_response
        raise AssertionError((method, path))

    def request_http(self, method, url, **kwargs):
        kwargs.pop("timeout", None)
        path = url.rsplit("/", 1)[-1]
        return _FixtureResponse(self.request_value(method, path, **kwargs))


def fixture_client(client_type, spec, config, *, synthetic_step=True):
    """Construct the frozen client normally while replacing only its transport."""

    transport = _FixtureTransport(spec, config, client_type)
    client_kwargs = {}

    def construct():
        arguments = dict(client_kwargs)
        arguments[spec.arm_parameter] = config.capability.arm.value
        return client_type(
            env_server_base="http://fixture.invalid",
            **arguments,
        )

    if spec.benchmark == "gaia_text":
        with (
            patch.object(
                client_type,
                "_request",
                new=lambda _self, method, path, **kwargs: transport.request_value(
                    method, path, **kwargs
                ),
            ),
            patch.object(
                client_type,
                "_validate_metadata",
                new=lambda _self, *_args, **_kwargs: ({}, "0" * 64),
            ),
        ):
            client = construct()
        client._request = transport.request_value
        client._request_json = transport.request_value
    elif spec.benchmark == "swebench_verified":
        client_kwargs = {
            "run_id": config.run_id,
            "run_capability": "r" * 43,
            "image_manifest_sha256": "b" * 64,
        }
        with (
            patch.object(
                client_type,
                "_request",
                new=lambda _self, method, path, **kwargs: transport.request_value(
                    method, path, **kwargs
                ),
            ),
            patch.object(
                client_type,
                "_validate_metadata",
                new=lambda _self, _metadata: None,
            ),
        ):
            client = construct()
        client._request = transport.request_value
    else:
        client_kwargs = {
            "expected_public_manifest_sha256": "1" * 64,
            "expected_runner_sha256": "2" * 64,
            "expected_runtime_digest": "3" * 64,
            "requester": transport.request_http,
        }
        client = construct()

    client._fixture_transport = transport
    if synthetic_step:
        client.step = lambda policy_output: SimpleNamespace(
            state="fixture step",
            reward=0.0,
            done=True,
            info=raw_step_info(spec, config.capability.arm),
        )
    return client


class PairedEvalRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = PrivateEvidenceStore(
            Path(self.temporary.name) / "evidence"
        )

    def builders(self, seen):
        def build(*, config, spec, client_type, evidence_store):
            seen.append(
                (
                    config.task.benchmark,
                    config.capability.arm.value,
                    spec.client_specification,
                    client_type,
                )
            )
            adapter = ClientEnvironmentAdapter(
                config=config,
                spec=spec,
                raw_client=fixture_client(client_type, spec, config),
                hooks=AdapterHooks(
                    finalize_artifact=lambda *_: None,
                    handoff_to_grader=lambda *_: None,
                ),
                evidence_store=evidence_store,
            )
            return RuntimeBindings(
                adapter=adapter,
                model=BoundModel(config.model),
            )

        return {spec.benchmark: build for spec in DEFAULT_ADAPTER_SPECS}

    def test_default_registry_is_lazy_and_pins_exact_clients(self) -> None:
        self.assertEqual(
            {
                spec.benchmark: spec.client_specification
                for spec in DEFAULT_ADAPTER_SPECS
            },
            EXPECTED_CLIENTS,
        )
        script = """
import sys
from paired_eval.registry import DEFAULT_ADAPTER_SPECS
assert len(DEFAULT_ADAPTER_SPECS) == 3
prefixes = ('agentenv_gaia_text', 'agentenv_swebench_verified',
            'agentenv_mlebench_lite', 'agentenv.envs.swebench_verified',
            'agentenv.envs.mlebench_lite')
assert not [name for name in sys.modules if name.startswith(prefixes)]
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(
            OUTER_ROOT / "scripts" / "agentmemory"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_all_nine_cells_traverse_one_factory_and_frozen_client_types(
        self,
    ) -> None:
        seen = []
        registry = PairedEvalRegistry(builders=self.builders(seen))
        expected_routes = {
            Arm.NATIVE: ("benchmark_task", "benchmark_task"),
            Arm.AMG_COMPACTION_ONLY: POLICY_COMPACTION_ROUTE,
            Arm.AMG_MEMORY: EXTERNAL_MEMORY_ROUTE,
        }

        for spec in DEFAULT_ADAPTER_SPECS:
            for arm in Arm:
                with self.subTest(benchmark=spec.benchmark, arm=arm.value):
                    config = config_for(spec, arm)
                    bindings = registry.build_runtime(
                        config,
                        evidence_store=self.store,
                    )
                    reset = bindings.adapter.reset(config)
                    base_prompt = reset.treatment_excluded_messages[0]["content"]
                    full_prompt = reset.initial_messages[0]["content"]
                    if arm is Arm.AMG_MEMORY:
                        self.assertEqual(
                            full_prompt,
                            base_prompt + spec.memory_prompt_suffix(),
                        )
                    else:
                        self.assertEqual(full_prompt, base_prompt)
                    step = bindings.adapter.client.step("fixture policy action")
                    receipt = TaskNeutralStepReceipt.from_info(step.info)
                    self.assertEqual(receipt.route.route, expected_routes[arm])
                    bindings.adapter.close()

        self.assertEqual(len(seen), 9)
        self.assertEqual(
            {(benchmark, arm) for benchmark, arm, _, _ in seen},
            {
                (spec.benchmark, arm.value)
                for spec in DEFAULT_ADAPTER_SPECS
                for arm in Arm
            },
        )
        for benchmark, _, specification, client_type in seen:
            self.assertEqual(specification, EXPECTED_CLIENTS[benchmark])
            self.assertEqual(
                f"{client_type.__module__}:{client_type.__name__}",
                EXPECTED_CLIENTS[benchmark],
            )

    def test_captured_factory_executes_all_nine_cells_through_manifest(
        self,
    ) -> None:
        seen = []

        def build(*, config, spec, client_type, evidence_store):
            seen.append((config.task.benchmark, config.capability.arm.value))
            fake = make_fake_runtime(config, evidence_store)
            adapter = ClientEnvironmentAdapter(
                config=config,
                spec=spec,
                raw_client=fixture_client(client_type, spec, config),
                hooks=AdapterHooks(
                    finalize_artifact=(
                        lambda _client, context, _config, _store: (
                            fake.adapter.finalize_artifact(context)
                        )
                    ),
                    handoff_to_grader=(
                        lambda _client, artifact, _config, _store: (
                            fake.adapter.handoff_to_grader(artifact)
                        )
                    ),
                ),
                evidence_store=evidence_store,
            )
            return RuntimeBindings(adapter=adapter, model=fake.model)

        builders = {
            spec.benchmark: build for spec in DEFAULT_ADAPTER_SPECS
        }
        factory = make_runtime_factory(
            builders,
            evidence_store=self.store,
        )
        writer = AppendSafeJsonlWriter(
            Path(self.temporary.name) / "results.jsonl"
        )
        rows = execute_manifest(
            registry_manifest_payload(),
            runner=PairedRunner(
                controller=DependencyLightPolicyTurnController(),
                evidence_store=self.store,
                clock=ManualClock(),
            ),
            runtime_factory=factory,
            writer=writer,
        )

        self.assertEqual(len(rows), 9)
        self.assertEqual(
            set(seen),
            {
                (spec.benchmark, arm.value)
                for spec in DEFAULT_ADAPTER_SPECS
                for arm in Arm
            },
        )

    def test_proxy_reconciles_raw_server_calls_with_route_accounting(
        self,
    ) -> None:
        for spec in DEFAULT_ADAPTER_SPECS:
            with self.subTest(benchmark=spec.benchmark):
                config = config_for(spec, Arm.AMG_MEMORY)
                raw_client = fixture_client(
                    spec.resolve_client_type(), spec, config
                )
                raw_steps = [
                    structured_raw_step_info(
                        spec,
                        "memory",
                        raw_native_before=0,
                        policy_before=0,
                        context_before=0,
                    ),
                    structured_raw_step_info(
                        spec,
                        "compaction",
                        raw_native_before=1,
                        policy_before=1,
                        context_before=0,
                    ),
                    structured_raw_step_info(
                        spec,
                        "ordinary",
                        raw_native_before=1,
                        policy_before=2,
                        context_before=1,
                    ),
                ]
                raw_client.step = lambda _output: SimpleNamespace(
                    state="fixture step",
                    reward=0.0,
                    done=False,
                    info=raw_steps.pop(0),
                )
                proxy = ClientStepProxy(
                    raw_client,
                    config=config,
                    spec=spec,
                    roots=lifecycle_roots(config),
                )

                memory = proxy.step("memory action").task_neutral_receipt
                compact = proxy.step("retained summary").task_neutral_receipt
                ordinary = proxy.step("benchmark action").task_neutral_receipt

                self.assertEqual(memory.route.route, EXTERNAL_MEMORY_ROUTE)
                self.assertEqual(
                    (memory.native_call_count_before, memory.native_call_count_after),
                    (0, 0),
                )
                self.assertEqual(compact.route.route, POLICY_COMPACTION_ROUTE)
                self.assertEqual(
                    (compact.native_call_count_before, compact.native_call_count_after),
                    (0, 0),
                )
                self.assertEqual(
                    (compact.context_epoch_before, compact.context_epoch_after),
                    (0, 1),
                )
                self.assertEqual(
                    ordinary.route.route,
                    ("benchmark_task", "benchmark_task"),
                )
                self.assertEqual(
                    (
                        ordinary.native_call_count_before,
                        ordinary.native_call_count_after,
                    ),
                    (0, 1),
                )

                drifted = structured_raw_step_info(spec, "memory")
                drifted["native_call_count_after"] = 0
                drifted["native_step_after"] = 0
                raw_client = fixture_client(
                    spec.resolve_client_type(), spec, config
                )
                raw_client.step = lambda _output: SimpleNamespace(
                    state="fixture step",
                    reward=0.0,
                    done=False,
                    info=drifted,
                )
                invalid_proxy = ClientStepProxy(
                    raw_client,
                    config=config,
                    spec=spec,
                    roots=lifecycle_roots(config),
                )
                with self.assertRaises(ValueError):
                    invalid_proxy.step("memory action")

    def test_proxy_rejects_untyped_step_payloads(self) -> None:
        spec = DEFAULT_ADAPTER_SPECS[0]
        config = config_for(spec, Arm.NATIVE)
        invalid_fields = (
            {"state": {"not": "text"}},
            {"reward": "0.0"},
            {"reward": True},
            {"reward": float("inf")},
            {"done": 0},
            {"info": []},
        )
        for drift in invalid_fields:
            with self.subTest(drift=drift):
                raw_client = fixture_client(
                    spec.resolve_client_type(), spec, config
                )
                payload = {
                    "state": "fixture step",
                    "reward": 0.0,
                    "done": False,
                    "info": raw_step_info(spec, Arm.NATIVE),
                    **drift,
                }
                raw_client.step = lambda _output, payload=payload: SimpleNamespace(
                    **payload
                )
                proxy = ClientStepProxy(
                    raw_client,
                    config=config,
                    spec=spec,
                    roots=lifecycle_roots(config),
                )
                with self.assertRaises((TypeError, ValueError)):
                    proxy.step("benchmark action")

    def test_real_swe_client_memory_then_compaction_routes_do_not_alias(self) -> None:
        spec = DEFAULT_ADAPTER_SPECS[1]
        config = config_for(spec, Arm.AMG_MEMORY)
        raw_client = fixture_client(
            spec.resolve_client_type(),
            spec,
            config,
            synthetic_step=False,
        )
        raw_client.reset(config.task.task_index)
        initial = raw_client.policy_framing() + [
            {"role": "user", "content": raw_client.observe()}
        ]
        raw_client.bind_policy_context(initial, initial=True)
        proxy = ClientStepProxy(
            raw_client,
            config=config,
            spec=spec,
            roots=lifecycle_roots(config),
        )

        memory = proxy.step(
            'shell_command {"command":"printf clue > '
            '/run/amg_memory/notes.md"}'
        ).task_neutral_receipt
        self.assertEqual(memory.route.route, EXTERNAL_MEMORY_ROUTE)
        selected = raw_client.prepare_policy_turn(
            PolicyContextPressure(
                action_prompt_tokens=800,
                candidate_prompt_tokens=850,
                max_prompt_tokens=1000,
                max_model_tokens=1200,
                max_response_tokens=100,
                max_observation_tokens=100,
            )
        )
        self.assertIsNotNone(selected)
        compacted = proxy.step("retain the note path").task_neutral_receipt
        self.assertEqual(compacted.route.route, POLICY_COMPACTION_ROUTE)

    def test_registry_and_prompt_suffix_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            PairedEvalRegistry(builders={})
        with self.assertRaises(ValueError):
            PairedEvalRegistry(
                specs=(*DEFAULT_ADAPTER_SPECS, DEFAULT_ADAPTER_SPECS[0]),
                builders=self.builders([]),
            )
        altered = replace(
            DEFAULT_ADAPTER_SPECS[0],
            artifact_type="altered-answer",
        )
        with self.assertRaises(ValueError):
            PairedEvalRegistry(
                specs=(altered, *DEFAULT_ADAPTER_SPECS[1:]),
                builders=self.builders([]),
            )

        registry = PairedEvalRegistry(builders=self.builders([]))
        unknown = make_config(benchmark="unknown")
        with self.assertRaises(KeyError):
            registry.build_runtime(unknown, evidence_store=self.store)

        spec = DEFAULT_ADAPTER_SPECS[0]
        wrong_tools = make_config(
            benchmark=spec.benchmark,
            artifact_type=spec.artifact_type,
        )
        with self.assertRaises(ValueError):
            registry.build_runtime(wrong_tools, evidence_store=self.store)

        nonbinding = {
            item.benchmark: (lambda **_: object())
            for item in DEFAULT_ADAPTER_SPECS
        }
        with self.assertRaises(TypeError):
            PairedEvalRegistry(builders=nonbinding).build_runtime(
                config_for(spec, Arm.NATIVE),
                evidence_store=self.store,
            )

        native_config = config_for(spec, Arm.NATIVE)
        missing_arm_client = fixture_client(
            spec.resolve_client_type(), spec, native_config
        )
        delattr(missing_arm_client, spec.arm_parameter)
        fake = make_fake_runtime(native_config, self.store)
        with self.assertRaises(ValueError):
            ClientEnvironmentAdapter(
                config=native_config,
                spec=spec,
                raw_client=missing_arm_client,
                hooks=AdapterHooks(
                    finalize_artifact=lambda *_: fake.adapter.finalize_artifact,
                    handoff_to_grader=lambda *_: fake.adapter.handoff_to_grader,
                ),
                evidence_store=self.store,
            )

        correct_builders = self.builders([])
        correct_builder = correct_builders[spec.benchmark]

        def wrong_arm_builder(**kwargs):
            bindings = correct_builder(**kwargs)
            setattr(
                bindings.adapter.raw_client,
                spec.arm_parameter,
                Arm.AMG_MEMORY.value,
            )
            return bindings

        correct_builders[spec.benchmark] = wrong_arm_builder
        with self.assertRaises(ValueError):
            PairedEvalRegistry(builders=correct_builders).build_runtime(
                config_for(spec, Arm.NATIVE),
                evidence_store=self.store,
            )

        memory_config = config_for(spec, Arm.AMG_MEMORY)
        suffix = spec.memory_prompt_suffix()
        with self.assertRaises(ValueError):
            treatment_excluded_messages(
                spec,
                memory_config,
                (
                    {
                        "role": "system",
                        "content": "frozen task framing" + suffix + " drift",
                    },
                    {"role": "user", "content": "fixture observation"},
                ),
            )
        with self.assertRaises(ValueError):
            treatment_excluded_messages(
                spec,
                memory_config,
                (
                    {
                        "role": "system",
                        "content": "frozen task framing" + suffix + suffix,
                    },
                    {"role": "user", "content": "fixture observation"},
                ),
            )

    def test_bridge_rejects_malformed_reset_and_close_receipts(self) -> None:
        spec = DEFAULT_ADAPTER_SPECS[1]
        config = config_for(spec, Arm.NATIVE)
        fake = make_fake_runtime(config, self.store)

        def adapter_for(client):
            return ClientEnvironmentAdapter(
                config=config,
                spec=spec,
                raw_client=client,
                hooks=AdapterHooks(
                    finalize_artifact=lambda *_: fake.adapter.finalize_artifact,
                    handoff_to_grader=lambda *_: fake.adapter.handoff_to_grader,
                ),
                evidence_store=self.store,
            )

        malformed_resets = (
            "not-a-mapping",
            {"observation": "fixture observation"},
            {
                "observation": 1,
                "reward": 0.0,
                "done": False,
                "info": {},
            },
            {
                "observation": "fixture observation",
                "reward": True,
                "done": False,
                "info": {},
            },
            {
                "observation": "fixture observation",
                "reward": float("nan"),
                "done": False,
                "info": {},
            },
            {
                "observation": "fixture observation",
                "reward": 0.1,
                "done": False,
                "info": {},
            },
            {
                "observation": "fixture observation",
                "reward": 0.0,
                "done": True,
                "info": {},
            },
            {
                "observation": "fixture observation",
                "reward": 0.0,
                "done": False,
                "info": {},
                "state": "different observation",
            },
            {
                "observation": "fixture observation",
                "reward": 0.0,
                "done": False,
                "info": [],
            },
            {
                "observation": "fixture observation",
                "reward": 0.0,
                "done": False,
                "info": {},
                "unexpected": "field",
            },
        )
        for response in malformed_resets:
            with self.subTest(reset=response):
                client = fixture_client(spec.resolve_client_type(), spec, config)

                def malformed_reset(_index, response=response):
                    client.info = response
                    return response

                client.reset = malformed_reset
                with self.assertRaises((TypeError, ValueError)):
                    adapter_for(client).reset(config)

        mismatched_observation = fixture_client(
            spec.resolve_client_type(), spec, config
        )
        mismatched_observation.observe = lambda: "different observation"
        with self.assertRaises(ValueError):
            adapter_for(mismatched_observation).reset(config)

        changing_observation = fixture_client(
            spec.resolve_client_type(), spec, config
        )
        observations = iter(("fixture observation", "unvalidated observation"))
        changing_observation.observe = lambda: next(observations)
        reset_result = adapter_for(changing_observation).reset(config)
        self.assertEqual(
            reset_result.initial_messages[-1],
            {"role": "user", "content": "fixture observation"},
        )
        self.assertEqual(next(observations), "unvalidated observation")
        with self.assertRaises(StopIteration):
            next(observations)

        mismatched_info = fixture_client(
            spec.resolve_client_type(), spec, config
        )
        reset = mismatched_info.reset

        def reset_with_drift(index):
            response = reset(index)
            mismatched_info.info = {
                **response,
                "observation": "different observation",
            }
            return response

        mismatched_info.reset = reset_with_drift
        mismatched_info.observe = lambda: "fixture observation"
        with self.assertRaises(ValueError):
            adapter_for(mismatched_info).reset(config)

        for response in (
            None,
            False,
            "not-a-mapping",
            {},
            {"closed": False},
            {"closed": 1},
        ):
            with self.subTest(close=response):
                client = fixture_client(spec.resolve_client_type(), spec, config)
                client._fixture_transport.close_response = response
                with self.assertRaises((TypeError, ValueError)):
                    adapter_for(client).close()

        for response in (True, {"closed": True}, {"closed": True, "id": 1}):
            with self.subTest(valid_close=response):
                client = fixture_client(spec.resolve_client_type(), spec, config)
                client._fixture_transport.close_response = response
                adapter_for(client).close()

    def test_registry_requires_exact_evidence_store_identity(self) -> None:
        seen = []
        builders = self.builders(seen)
        spec = DEFAULT_ADAPTER_SPECS[0]
        original = builders[spec.benchmark]
        other_store = PrivateEvidenceStore(
            Path(self.temporary.name) / "other-evidence"
        )

        def mismatched_store_builder(**kwargs):
            kwargs["evidence_store"] = other_store
            return original(**kwargs)

        builders[spec.benchmark] = mismatched_store_builder
        with self.assertRaises(ValueError):
            PairedEvalRegistry(builders=builders).build_runtime(
                config_for(spec, Arm.NATIVE),
                evidence_store=self.store,
            )

    def test_registry_rejects_builder_bound_to_wrong_client(self) -> None:
        spec = DEFAULT_ADAPTER_SPECS[0]
        other_spec = DEFAULT_ADAPTER_SPECS[1]
        builders = self.builders([])
        original = builders[spec.benchmark]

        def wrong_client_builder(**kwargs):
            bindings = original(**kwargs)
            other_config = config_for(other_spec, kwargs["config"].capability.arm)
            bindings.adapter.raw_client = fixture_client(
                other_spec.resolve_client_type(),
                other_spec,
                other_config,
            )
            return bindings

        builders[spec.benchmark] = wrong_client_builder
        with self.assertRaises(TypeError):
            PairedEvalRegistry(builders=builders).build_runtime(
                config_for(spec, Arm.NATIVE),
                evidence_store=self.store,
            )

    def test_bridge_requires_the_exact_frozen_client_type(self) -> None:
        spec = DEFAULT_ADAPTER_SPECS[0]
        config = config_for(spec, Arm.NATIVE)
        client_type = spec.resolve_client_type()
        subclass_type = type("BehaviorOverridingClient", (client_type,), {})
        subclass_client = fixture_client(subclass_type, spec, config)
        fake = make_fake_runtime(config, self.store)

        with self.assertRaises(TypeError):
            ClientEnvironmentAdapter(
                config=config,
                spec=spec,
                raw_client=subclass_client,
                hooks=AdapterHooks(
                    finalize_artifact=lambda *_: fake.adapter.finalize_artifact,
                    handoff_to_grader=lambda *_: fake.adapter.handoff_to_grader,
                ),
                evidence_store=self.store,
            )


if __name__ == "__main__":
    unittest.main()
