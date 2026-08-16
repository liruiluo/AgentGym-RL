"""Guarded process entrypoint for one production SWE-bench cell server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import agentenv_swebench_verified.launch as published_launch
from agentenv_swebench_verified.sandbox import (
    VerifiedLinuxNamespaceEpisodeSandbox,
)

from .resource_guard import (
    GuardedEpisodeSandboxMixin,
    LinuxTmpfsMountBackend,
    QuotaMountSpec,
    RootfsMutationGuard,
    TmpfsQuotaMounts,
)


ENV_PREFIX = "SWEBENCH_TRIAD_"


def required_path(name: str) -> Path:
    full_name = ENV_PREFIX + name
    value = os.environ.get(full_name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {full_name}")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"{full_name} must be absolute")
    return path.resolve(strict=True)


def required_integer(name: str) -> int:
    full_name = ENV_PREFIX + name
    raw = os.environ.get(full_name, "")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{full_name} must be an integer") from error
    if value <= 0:
        raise RuntimeError(f"{full_name} must be positive")
    return value


class GuardedVerifiedLinuxNamespaceEpisodeSandbox(
    GuardedEpisodeSandboxMixin,
    VerifiedLinuxNamespaceEpisodeSandbox,
):
    """Published sandbox plus deployment-owned quotas and rootfs attestation."""

    @classmethod
    def from_environment(cls, **kwargs: Any):
        kwargs["run_preflight"] = False
        sandbox = super().from_environment(**kwargs)
        configured = False
        try:
            sandbox.configure_deployment_guards(
                quota_mounts=TmpfsQuotaMounts(LinuxTmpfsMountBackend()),
                workspace_quota=QuotaMountSpec(
                    byte_limit=required_integer("WORKSPACE_BYTES"),
                    inode_limit=required_integer("WORKSPACE_INODES"),
                    purpose="workspace",
                ),
                external_memory_quota=QuotaMountSpec(
                    byte_limit=required_integer("EXTERNAL_MEMORY_BYTES"),
                    inode_limit=required_integer("EXTERNAL_MEMORY_INODES"),
                    purpose="external-memory",
                ),
                rootfs_guard=RootfsMutationGuard(required_path("ROOTFS_CACHE")),
            )
            configured = True
            sandbox.preflight()
            return sandbox
        except BaseException:
            if configured:
                sandbox.close()
            else:
                VerifiedLinuxNamespaceEpisodeSandbox.close(sandbox)
            raise


def main() -> int:
    if (
        published_launch.VerifiedLinuxNamespaceEpisodeSandbox
        is not VerifiedLinuxNamespaceEpisodeSandbox
    ):
        raise RuntimeError("published SWE sandbox binding was already replaced")
    published_launch.VerifiedLinuxNamespaceEpisodeSandbox = (
        GuardedVerifiedLinuxNamespaceEpisodeSandbox
    )
    published_launch.launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
