"""Published SWE client binding and deployment-owned artifact hooks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping

from paired_eval.contracts import (
    ArtifactResult,
    FinalizationContext,
    RunConfig,
    ScorerResult,
)
from paired_eval.evidence import PrivateEvidenceStore
from paired_eval.manifest import RuntimeBindings
from paired_eval.model_client import (
    JsonTransport,
    OpenAICompatibleModelClient,
    UrllibJsonTransport,
)
from paired_eval.registry import (
    AdapterHooks,
    AdapterSpec,
    ClientEnvironmentAdapter,
    DEFAULT_ADAPTER_SPECS,
    make_runtime_factory,
)

from .model_transport import ExactTokenVllmTransport


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EndpointResolver = Callable[[RunConfig], "SwebenchRuntimeEndpoint"]
TransportFactory = Callable[[], JsonTransport]


@dataclass(frozen=True)
class SwebenchRuntimeEndpoint:
    env_server_base: str
    private_run_id: str
    run_capability: str
    image_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in ("env_server_base", "private_run_id", "run_capability"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be nonempty normalized text")
        if SHA256_PATTERN.fullmatch(self.image_manifest_sha256) is None:
            raise ValueError("image manifest identity must be lowercase SHA-256")


def client_done(client: Any) -> bool:
    info = getattr(client, "info", None)
    if not isinstance(info, Mapping) or type(info.get("done")) is not bool:
        raise RuntimeError("SWE client terminal state is unavailable")
    return info["done"]


def finalize_swebench_artifact(
    client: Any,
    context: FinalizationContext,
    config: RunConfig,
    evidence_store: PrivateEvidenceStore,
) -> ArtifactResult:
    if not isinstance(context, FinalizationContext):
        raise TypeError("SWE artifact context must be FinalizationContext")
    if not client_done(client):
        client.finalize_policy_horizon()
        if not client_done(client):
            raise RuntimeError("SWE horizon did not close the endpoint")
    prediction = client.prediction()
    if not isinstance(prediction, Mapping):
        raise RuntimeError("SWE prediction must be an object")
    expected = {"instance_id", "model_name_or_path", "model_patch"}
    if set(prediction) != expected:
        raise RuntimeError("SWE prediction fields drifted")
    if prediction.get("instance_id") != config.task.task_id:
        raise RuntimeError("SWE prediction belongs to another task")
    if any(not isinstance(prediction.get(key), str) for key in expected):
        raise RuntimeError("SWE prediction values must be text")
    prediction_ref = evidence_store.put_json(
        "swebench_predictions",
        dict(prediction),
    )
    patch_bytes = len(prediction["model_patch"].encode("utf-8"))
    return ArtifactResult(
        artifact_type="patch",
        protected_ref=prediction_ref.protected_ref,
        sha256=prediction_ref.sha256,
        receipt={
            "schema": "swebench_verified_prediction_artifact_v1",
            "status": "closed",
            "instance_id": prediction["instance_id"],
            "model_name_or_path": prediction["model_name_or_path"],
            "model_patch_bytes": patch_bytes,
            "prediction_sha256": prediction_ref.sha256,
            "termination_reason": context.termination_reason,
            "horizon_cause": context.horizon_cause,
            "failure_class": context.failure_class,
            "timed_out": context.timed_out,
        },
    )


def handoff_swebench_grader(
    client: Any,
    artifact: ArtifactResult,
    config: RunConfig,
    evidence_store: PrivateEvidenceStore,
) -> ScorerResult:
    del client
    if artifact.artifact_type != "patch":
        raise ValueError("SWE grader handoff requires a patch artifact")
    if artifact.receipt.get("instance_id") != config.task.task_id:
        raise ValueError("SWE grader handoff task identity drifted")
    queue_payload = {
        "schema": "swebench_verified_official_grader_queue_v1",
        "status": "queued",
        "instance_id": config.task.task_id,
        "task_index": config.task.task_index,
        "arm": config.capability.arm.value,
        "artifact_type": artifact.artifact_type,
        "artifact_ref": artifact.protected_ref,
        "artifact_sha256": artifact.sha256,
        "grader": config.grader.to_payload(),
        "official_resolved": None,
    }
    queue_ref = evidence_store.put_json(
        "swebench_grader_queue",
        queue_payload,
    )
    return ScorerResult(
        name=config.grader.name,
        revision=config.grader.revision,
        config_sha256=config.grader.config_sha256,
        public_metrics={"official_resolved": None},
        receipt={
            **queue_payload,
            "queue_ref": queue_ref.protected_ref,
            "queue_sha256": queue_ref.sha256,
        },
    )


def make_swebench_runtime_factory(
    *,
    evidence_store: PrivateEvidenceStore,
    endpoint_resolver: EndpointResolver,
    model_base_url: str,
    transport_factory: TransportFactory = UrllibJsonTransport,
    model_timeout_seconds: float,
    environment_timeout_seconds: int,
):
    if not callable(endpoint_resolver):
        raise TypeError("SWE endpoint resolver must be callable")
    if not callable(transport_factory):
        raise TypeError("model transport factory must be callable")
    if isinstance(model_timeout_seconds, bool) or model_timeout_seconds <= 0:
        raise ValueError("model timeout must be positive")
    if (
        isinstance(environment_timeout_seconds, bool)
        or not isinstance(environment_timeout_seconds, int)
        or environment_timeout_seconds <= 0
    ):
        raise ValueError("environment timeout must be a positive integer")

    def build_swebench(
        *,
        config: RunConfig,
        spec: AdapterSpec,
        client_type: type,
        evidence_store: PrivateEvidenceStore,
    ) -> RuntimeBindings:
        del client_type
        endpoint = endpoint_resolver(config)
        if not isinstance(endpoint, SwebenchRuntimeEndpoint):
            raise TypeError("SWE endpoint resolver returned the wrong type")
        raw_client = spec.instantiate_client(
            config,
            env_server_base=endpoint.env_server_base,
            client_kwargs={
                "run_id": endpoint.private_run_id,
                "run_capability": endpoint.run_capability,
                "image_manifest_sha256": endpoint.image_manifest_sha256,
                "data_len": 500,
                "timeout": environment_timeout_seconds,
            },
        )
        adapter = ClientEnvironmentAdapter(
            config=config,
            spec=spec,
            raw_client=raw_client,
            hooks=AdapterHooks(
                finalize_artifact=finalize_swebench_artifact,
                handoff_to_grader=handoff_swebench_grader,
            ),
            evidence_store=evidence_store,
        )
        model = OpenAICompatibleModelClient(
            base_url=model_base_url,
            model_config=config.model,
            transport=ExactTokenVllmTransport(transport_factory()),
            evidence_store=evidence_store,
            timeout_seconds=model_timeout_seconds,
            enable_thinking=False,
        )
        return RuntimeBindings(adapter=adapter, model=model)

    def reject_other_benchmark(**kwargs: Any) -> RuntimeBindings:
        config = kwargs.get("config")
        benchmark = getattr(getattr(config, "task", None), "benchmark", None)
        raise RuntimeError(
            f"SWE deployment factory cannot build benchmark {benchmark!r}"
        )

    builders = {
        spec.benchmark: (
            build_swebench
            if spec.benchmark == "swebench_verified"
            else reject_other_benchmark
        )
        for spec in DEFAULT_ADAPTER_SPECS
    }
    return make_runtime_factory(builders, evidence_store=evidence_store)


__all__ = [
    "SwebenchRuntimeEndpoint",
    "finalize_swebench_artifact",
    "handoff_swebench_grader",
    "make_swebench_runtime_factory",
]
