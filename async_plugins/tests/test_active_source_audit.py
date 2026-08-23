from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from agentmemorygym_verl.active_source_audit import (
    SourceModuleContract,
    audit_resolved_active_sources,
    audit_source_paths,
)

ROOT = Path(__file__).resolve().parents[2]
VERL_ROOT = Path(
    "/Users/luolirui.1/Projects/agentmemorygym-rl-workspace/"
    "worktrees/verl-main-exact-eos-20260823"
)


class TestActiveSourceAudit(unittest.TestCase):
    def test_actual_accepted_verl_and_shared_plugin_sources_have_no_domain_dispatch(
        self,
    ):
        if not VERL_ROOT.is_dir():
            self.skipTest("accepted veRL source worktree is unavailable")
        files = {
            "fully_async_main": VERL_ROOT
            / "verl/experimental/fully_async_policy/fully_async_main.py",
            "fully_async_rollouter": VERL_ROOT
            / "verl/experimental/fully_async_policy/fully_async_rollouter.py",
            "fully_async_trainer": VERL_ROOT
            / "verl/experimental/fully_async_policy/fully_async_trainer.py",
            "message_queue": VERL_ROOT
            / "verl/experimental/fully_async_policy/message_queue.py",
            "upstream_agent_loop": VERL_ROOT
            / "verl/experimental/agent_loop/agent_loop.py",
            "amg_agent_loop": ROOT / "async_plugins/agentmemorygym_verl/agent_loop.py",
            "amg_dataset": ROOT / "async_plugins/agentmemorygym_verl/dataset.py",
            "amg_action_gae": ROOT / "async_plugins/agentmemorygym_verl/action_gae.py",
        }
        report = audit_source_paths(files)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(set(report["files"]), set(files))

    def test_ast_audit_rejects_concrete_environment_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.py"
            source.write_text(
                "def dispatch(route_id):\n"
                "    if route_id == 'webshop':\n"
                "        return 1\n"
                "    return 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "domain-specific"):
                audit_source_paths({"bad": source})

    def test_ast_audit_rejects_domain_constant_alias_in_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad_alias.py"
            source.write_text(
                "WEBSHOP = 'webshop'\n"
                "def dispatch(route_id):\n"
                "    if route_id == WEBSHOP:\n"
                "        return 1\n"
                "    return 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "domain-specific"):
                audit_source_paths({"bad_alias": source})

    def test_ast_audit_rejects_domain_keyed_dispatch_table(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad_table.py"
            source.write_text(
                "def handler():\n"
                "    return 1\n"
                "HANDLERS = {'webshop': handler}\n"
                "def dispatch(route_id):\n"
                "    return HANDLERS[route_id]\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "domain-specific"):
                audit_source_paths({"bad_table": source})

    def test_ast_audit_rejects_domain_named_call(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad_call.py"
            source.write_text(
                "def generate_webshop_rollout():\n"
                "    return 1\n"
                "def run():\n"
                "    return generate_webshop_rollout()\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "domain-specific"):
                audit_source_paths({"bad_call": source})

    def test_ast_audit_rejects_unused_domain_named_definitions(self):
        definitions = {
            "function": "def generate_webshop_rollout():\n    return 1\n",
            "async_function": ("async def run_swesmith_episode():\n    return 1\n"),
            "class": "class LiteResearcherSampler:\n    pass\n",
            "camel_case_class": "class OpenMLEFastEndpoint:\n    pass\n",
        }
        for kind, body in definitions.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / f"bad_{kind}.py"
                source.write_text(body, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "domain-specific definition"):
                    audit_source_paths({kind: source})

    def test_runtime_resolved_audit_rejects_inactive_legacy_verl_and_records_paths(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verl_root = root / "verl-source"
            plugin_root = root / "outer/async_plugins"
            controller_root = root / "outer/AgentGym/agentenv"
            legacy_root = root / "outer/AgentGym-RL/verl"
            for path in (verl_root, plugin_root, controller_root, legacy_root):
                path.mkdir(parents=True)
            active = {
                "upstream": verl_root / "upstream.py",
                "plugin": plugin_root / "plugin.py",
                "controller": controller_root / "controller.py",
            }
            for path in active.values():
                path.write_text("VALUE = 1\n", encoding="utf-8")
            modules = {}
            for label, path in active.items():
                module = types.ModuleType(f"fixture_{label}")
                module.__file__ = str(path)
                modules[module.__name__] = module
            contracts = (
                SourceModuleContract("upstream", "fixture_upstream", "verl"),
                SourceModuleContract("plugin", "fixture_plugin", "plugin"),
                SourceModuleContract("controller", "fixture_controller", "agentgym"),
            )
            with mock.patch(
                "agentmemorygym_verl.active_source_audit.importlib.import_module",
                side_effect=lambda name: modules[name],
            ):
                report = audit_resolved_active_sources(
                    verl_root=verl_root,
                    outer_root=root / "outer",
                    module_contracts=contracts,
                )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                Path(report["files"]["upstream"]["path"]),
                active["upstream"].resolve(),
            )

            legacy = legacy_root / "shadow.py"
            legacy.write_text("VALUE = 1\n", encoding="utf-8")
            modules["fixture_upstream"].__file__ = str(legacy)
            with (
                mock.patch(
                    "agentmemorygym_verl.active_source_audit.importlib.import_module",
                    side_effect=lambda name: modules[name],
                ),
                self.assertRaisesRegex(RuntimeError, "inactive nested legacy"),
            ):
                audit_resolved_active_sources(
                    verl_root=verl_root,
                    outer_root=root / "outer",
                    module_contracts=contracts,
                )


if __name__ == "__main__":
    unittest.main()
