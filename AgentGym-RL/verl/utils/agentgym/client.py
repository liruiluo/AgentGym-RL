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


def init_env_client(args, *, env_addr: str | None = None):
    # task_name - task dict
    envclient_classes = {
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
    # select task according to the name
    envclient_class = envclient_classes.get(args.task_name.lower(), None)
    if envclient_class is None:
        raise ValueError(f"Unsupported task name: {args.task_name}")
    retry = 0
    while True:
        try:
            data_len = getattr(args, "data_len", 1)
            if args.task_name.lower() in {"agentmemory", "swesmith"} and not hasattr(args, "data_len"):
                data_len = None
            resolved_env_addr = (
                env_addr_for_surface_slot(args)
                if env_addr is None
                else str(env_addr).rstrip("/")
            )
            env_client = envclient_class(env_server_base=resolved_env_addr, data_len=data_len, timeout=_client_timeout_seconds())
            break
        except Exception as e:
            retry += 1
            print(f"Failed to connect to env server, retrying...({retry}/{args.max_retries})")
            if retry > args.max_retries:
                raise e
            time.sleep(5)
    return env_client
