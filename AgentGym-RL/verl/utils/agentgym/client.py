from __future__ import annotations

from contextlib import contextmanager
import os
import time
from urllib.parse import urlparse


def _client_timeout_seconds(default: int = 2400) -> int:
    raw = os.environ.get("AGENTGYM_CLIENT_TIMEOUT_SECONDS")
    if raw is None or raw == "":
        return default
    try:
        value = int(float(raw))
    except ValueError:
        print(f"Invalid AGENTGYM_CLIENT_TIMEOUT_SECONDS={raw!r}; using {default}")
        return default
    return max(1, value)
from agentenv.envs import (
    AgentMemoryEnvClient,
    AcademiaEnvClient,
    AlfWorldEnvClient,
    BabyAIEnvClient,
    MazeEnvClient,
    MovieEnvClient,
    SciworldEnvClient,
    SheetEnvClient,
    SqlGymEnvClient,
    TextCraftEnvClient,
    TodoEnvClient,
    WeatherEnvClient,
    WebarenaEnvClient,
    WebshopEnvClient,
    WordleEnvClient,
    SearchQAEnvClient,
    SwesmithEnvClient,
)


ENVCLIENT_CLASSES = {
    "agentmemory": AgentMemoryEnvClient,
    "webshop": WebshopEnvClient,
    "alfworld": AlfWorldEnvClient,
    "babyai": BabyAIEnvClient,
    "sciworld": SciworldEnvClient,
    "textcraft": TextCraftEnvClient,
    "webarena": WebarenaEnvClient,
    "sqlgym": SqlGymEnvClient,
    "maze": MazeEnvClient,
    "wordle": WordleEnvClient,
    "weather": WeatherEnvClient,
    "todo": TodoEnvClient,
    "movie": MovieEnvClient,
    "sheet": SheetEnvClient,
    "academia": AcademiaEnvClient,
    "searchqa": SearchQAEnvClient,
    "swesmith": SwesmithEnvClient,
}

def configured_multitask_env_addrs(args) -> tuple[str, ...]:
    raw = getattr(args, "multitask_env_addrs", None)
    if raw is None:
        return ()
    if isinstance(raw, str):
        raise ValueError("multitask_env_addrs must be a sequence, not a string")
    try:
        values = tuple(str(value).rstrip("/") for value in raw)
    except TypeError as exc:
        raise ValueError("multitask_env_addrs must be a sequence") from exc
    if not values:
        raise ValueError("multitask_env_addrs must not be empty")
    for index, value in enumerate(values):
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"multitask_env_addrs[{index}] is not an HTTP endpoint: {value!r}"
            )
    if len(set(values)) != len(values):
        raise ValueError("multitask_env_addrs must contain distinct endpoints")
    return values


def configured_multitask_task_names(args) -> tuple[str, ...]:
    raw = getattr(args, "multitask_task_names", None)
    if raw is None:
        return ()
    if isinstance(raw, str):
        raise ValueError("multitask_task_names must be a sequence, not a string")
    try:
        values = tuple(str(value).strip().lower() for value in raw)
    except TypeError as exc:
        raise ValueError("multitask_task_names must be a sequence") from exc
    route_addrs = configured_multitask_env_addrs(args)
    if len(values) != len(route_addrs):
        raise ValueError(
            "multitask_task_names must align one-to-one with "
            "multitask_env_addrs"
        )
    for index, value in enumerate(values):
        if value not in ENVCLIENT_CLASSES:
            raise ValueError(
                f"multitask_task_names[{index}] is unsupported: {value!r}"
            )
    return values


def env_addr_for_surface_slot(args, surface_slot: int | None = None) -> str:
    route_addrs = configured_multitask_env_addrs(args)
    if not route_addrs:
        if surface_slot not in {None, 0}:
            raise ValueError(
                "a nonzero surface slot requires configured multitask endpoints"
            )
        return str(args.env_addr).rstrip("/")
    if surface_slot is None:
        surface_slot = 0
    if isinstance(surface_slot, bool) or not isinstance(surface_slot, int):
        raise ValueError("surface_slot must be an integer")
    if surface_slot < 0 or surface_slot >= len(route_addrs):
        raise ValueError(
            f"surface_slot {surface_slot} is outside [0, {len(route_addrs)})"
        )
    return route_addrs[surface_slot]


def task_name_for_env_addr(args, env_addr: str | None = None) -> str:
    default_task_name = str(args.task_name).strip().lower()
    route_task_names = configured_multitask_task_names(args)
    if not route_task_names:
        return default_task_name
    route_addrs = configured_multitask_env_addrs(args)
    resolved_env_addr = (
        env_addr_for_surface_slot(args)
        if env_addr is None
        else str(env_addr).rstrip("/")
    )
    try:
        route_index = route_addrs.index(resolved_env_addr)
    except ValueError as exc:
        raise ValueError(
            "env_addr is not present in configured multitask_env_addrs: "
            f"{resolved_env_addr!r}"
        ) from exc
    return route_task_names[route_index]


def init_env_client(args, *, env_addr: str | None = None):
    resolved_env_addr = (
        env_addr_for_surface_slot(args)
        if env_addr is None
        else str(env_addr).rstrip("/")
    )
    resolved_task_name = task_name_for_env_addr(args, resolved_env_addr)
    envclient_class = ENVCLIENT_CLASSES.get(resolved_task_name)
    if envclient_class is None:
        raise ValueError(f"Unsupported task name: {resolved_task_name}")
    retry = 0
    while True:
        try:
            data_len = getattr(args, "data_len", 1)
            if resolved_task_name in {"agentmemory", "swesmith"} and not hasattr(args, "data_len"):
                data_len = None
            env_client = envclient_class(env_server_base=resolved_env_addr, data_len=data_len, timeout=_client_timeout_seconds())
            if resolved_task_name == "agentmemory":
                _configure_agentmemory_policy_prompt(env_client)
            break
        except Exception as e:
            retry += 1
            print(f"Failed to connect to env server, retrying...({retry}/{args.max_retries})")
            if retry > args.max_retries:
                raise e
            time.sleep(5)
    return env_client


def _configure_agentmemory_policy_prompt(env_client) -> None:
    """Resolve the formal AgentMemory prompt outside the shared rollout loop."""

    from verl.utils.agentgym.formal_domain_v3 import (
        FormalDomainV3Error,
        resolve_formal_runtime_contract,
        validate_webshop_action_listing_mode,
        validate_webshop_filesystem_surface,
        validate_webshop_ltm_inventory_mode,
        validate_webshop_memory_prompt_mode,
    )
    from verl.workers.rollout.schemas import (
        agentmemory_action_listing_mode,
        agentmemory_action_system_prompt,
        agentmemory_ltm_inventory_mode,
        agentmemory_memory_prompt_mode,
    )

    metadata = getattr(env_client, "metadata", None)
    if not isinstance(metadata, dict):
        raise RuntimeError("AgentMemory client is missing formal runtime metadata")
    try:
        validate_webshop_ltm_inventory_mode(
            metadata,
            expected_mode=agentmemory_ltm_inventory_mode(),
        )
        validate_webshop_memory_prompt_mode(
            metadata,
            expected_mode=agentmemory_memory_prompt_mode(),
        )
        validate_webshop_filesystem_surface(
            metadata,
            expected_prompt_mode=agentmemory_memory_prompt_mode(),
        )
        validate_webshop_action_listing_mode(
            metadata,
            expected_mode=agentmemory_action_listing_mode(),
        )
        _, system_prompt, _ = resolve_formal_runtime_contract(
            metadata,
            webshop_v2_system_prompt=agentmemory_action_system_prompt(
                surface=metadata.get("surface")
            ),
        )
    except FormalDomainV3Error as exc:
        raise RuntimeError(
            f"Invalid formal AgentMemory runtime contract: {exc}"
        ) from exc
    env_client.configure_policy_system_prompt(system_prompt)
