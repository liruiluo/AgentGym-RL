"""Task-agnostic paired external-evaluation orchestration."""

from .contracts import (
    AMG_MEMORY_CAPABILITY,
    NATIVE_CAPABILITY,
    Arm,
    RunConfig,
)
from .manifest import RuntimeBindings, execute_manifest, expand_manifest
from .runner import PairedRunner
from .verifier import verify_pair_completeness


__all__ = [
    "AMG_MEMORY_CAPABILITY",
    "NATIVE_CAPABILITY",
    "Arm",
    "PairedRunner",
    "RunConfig",
    "RuntimeBindings",
    "execute_manifest",
    "expand_manifest",
    "verify_pair_completeness",
]
