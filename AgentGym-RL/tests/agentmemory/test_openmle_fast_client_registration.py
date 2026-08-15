#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "verl" / "utils" / "agentgym" / "client.py"
CLIENT_NAMES = (
    "AgentMemoryEnvClient",
    "AcademiaEnvClient",
    "AlfWorldEnvClient",
    "BabyAIEnvClient",
    "MazeEnvClient",
    "MovieEnvClient",
    "SciworldEnvClient",
    "SheetEnvClient",
    "SqlGymEnvClient",
    "TextCraftEnvClient",
    "TodoEnvClient",
    "WeatherEnvClient",
    "WebarenaEnvClient",
    "WebshopEnvClient",
    "WordleEnvClient",
    "SearchQAEnvClient",
    "SwesmithEnvClient",
)


class RecordingOpenMLEFastEnvClient:
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.calls.append(dict(kwargs))


def load_client_module(
    *,
    export_openmle: bool,
    openmle_import_error: ImportError | None = None,
):
    fake_agentenv = types.ModuleType("agentenv")
    fake_envs = types.ModuleType("agentenv.envs")
    for class_name in CLIENT_NAMES:
        setattr(fake_envs, class_name, type(class_name, (), {}))
    if export_openmle:
        fake_envs.OpenMLEFastEnvClient = RecordingOpenMLEFastEnvClient
    elif openmle_import_error is not None:

        def fail_openmle_import(name: str):
            if name == "OpenMLEFastEnvClient":
                raise openmle_import_error
            raise AttributeError(name)

        fake_envs.__getattr__ = fail_openmle_import

    original_agentenv = sys.modules.get("agentenv")
    original_envs = sys.modules.get("agentenv.envs")
    sys.modules["agentenv"] = fake_agentenv
    sys.modules["agentenv.envs"] = fake_envs
    try:
        spec = importlib.util.spec_from_file_location(
            f"openmle_fast_client_registration_{export_openmle}",
            MODULE_PATH,
        )
        if spec is None or spec.loader is None:
            raise AssertionError("could not load AgentGym client registry")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if original_agentenv is None:
            sys.modules.pop("agentenv", None)
        else:
            sys.modules["agentenv"] = original_agentenv
        if original_envs is None:
            sys.modules.pop("agentenv.envs", None)
        else:
            sys.modules["agentenv.envs"] = original_envs


class OpenMLEFastClientRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        RecordingOpenMLEFastEnvClient.calls.clear()

    def test_registers_exported_client_and_derives_server_length(self) -> None:
        module = load_client_module(export_openmle=True)
        self.assertIs(
            module.ENVCLIENT_CLASSES["openmle_fast"],
            RecordingOpenMLEFastEnvClient,
        )
        args = SimpleNamespace(
            task_name="openmle_fast",
            env_addr="http://127.0.0.1:65432/",
            max_retries=0,
        )
        with mock.patch.object(module.time, "sleep"):
            module.init_env_client(args)
        self.assertEqual(
            RecordingOpenMLEFastEnvClient.calls,
            [
                {
                    "env_server_base": "http://127.0.0.1:65432",
                    "data_len": None,
                    "timeout": 2400,
                }
            ],
        )

    def test_preserves_an_explicit_data_length_cap(self) -> None:
        module = load_client_module(export_openmle=True)
        args = SimpleNamespace(
            task_name="openmle_fast",
            env_addr="http://127.0.0.1:65432",
            max_retries=0,
            data_len=7,
        )
        module.init_env_client(args)
        self.assertEqual(RecordingOpenMLEFastEnvClient.calls[0]["data_len"], 7)

    def test_missing_inner_export_does_not_break_existing_client_imports(self) -> None:
        module = load_client_module(export_openmle=False)
        self.assertNotIn("openmle_fast", module.ENVCLIENT_CLASSES)
        args = SimpleNamespace(
            task_name="openmle_fast",
            env_addr="http://127.0.0.1:65432",
            max_retries=0,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported task name"):
            module.init_env_client(args)

    def test_does_not_hide_an_inner_dependency_import_failure(self) -> None:
        dependency_error = ImportError(
            "missing inner dependency",
            name="openmle_runtime_dependency",
        )
        with self.assertRaisesRegex(ImportError, "missing inner dependency"):
            load_client_module(
                export_openmle=False,
                openmle_import_error=dependency_error,
            )


if __name__ == "__main__":
    unittest.main()
