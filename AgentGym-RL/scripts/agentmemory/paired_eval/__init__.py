"""Task-agnostic paired external-evaluation orchestration."""

from .contracts import (
    AMG_COMPACTION_ONLY_CAPABILITY,
    AMG_MEMORY_CAPABILITY,
    CAPABILITY_LATTICE,
    EXTERNAL_MEMORY_CAPABILITY_SURFACES,
    NATIVE_CAPABILITY,
    Arm,
    RunConfig,
)
from .manifest import RuntimeBindings, execute_manifest, expand_manifest
from .runner import PairedRunner
from .registry import (
    AdapterHooks,
    AdapterSpec,
    ClientEnvironmentAdapter,
    DEFAULT_ADAPTER_SPECS,
    PairedEvalRegistry,
    make_runtime_factory,
)
from .verifier import verify_pair_completeness


__all__ = [
    "AMG_COMPACTION_ONLY_CAPABILITY",
    "AMG_MEMORY_CAPABILITY",
    "CAPABILITY_LATTICE",
    "EXTERNAL_MEMORY_CAPABILITY_SURFACES",
    "NATIVE_CAPABILITY",
    "Arm",
    "AdapterHooks",
    "AdapterSpec",
    "ClientEnvironmentAdapter",
    "DEFAULT_ADAPTER_SPECS",
    "PairedRunner",
    "PairedEvalRegistry",
    "RunConfig",
    "RuntimeBindings",
    "execute_manifest",
    "expand_manifest",
    "make_runtime_factory",
    "verify_pair_completeness",
]
