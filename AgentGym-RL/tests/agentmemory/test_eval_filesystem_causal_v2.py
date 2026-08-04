from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "scripts/agentmemory/eval_v3_openai.py"
CAUSAL_PATH = ROOT / "scripts/agentmemory/eval_filesystem_causal_v2.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("eval_v3_openai", CORE_PATH)
CAUSAL = load_module("eval_filesystem_causal_v2_test", CAUSAL_PATH)


def workspace_state(value: str | None) -> dict:
    files = []
    total_bytes = 0
    if value is not None:
        data = value.encode("utf-8")
        total_bytes = len(data)
        files.append(
            {
                "path": "MEMORY.md",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "content_base64": base64.b64encode(data).decode("ascii"),
            }
        )
    manifest_files = [
        {key: item[key] for key in ("path", "sha256", "bytes")}
        for item in files
    ]
    manifest = json.dumps(
        {"directories": [], "files": manifest_files},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "agentmemory_workspace_transfer_state_v1",
        "file_count": len(files),
        "directory_count": 0,
        "total_bytes": total_bytes,
        "directories": [],
        "files": files,
        "tree_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def snapshot(value: str | None) -> dict:
    state = workspace_state(value)
    return {
        "schema": "agentmemory_workspace_snapshot_v2",
        "file_count": state["file_count"],
        "directory_count": state["directory_count"],
        "total_bytes": state["total_bytes"],
        "directories": state["directories"],
        "files": [
            {key: item[key] for key in ("path", "sha256", "bytes")}
            for item in state["files"]
        ],
        "tree_sha256": state["tree_sha256"],
    }


def filesystem_metadata(token: str) -> dict:
    limits = {
        "max_path_chars": 240,
        "max_files": 64,
        "max_directories": 64,
        "max_file_bytes": 65_536,
        "max_total_bytes": 524_288,
        "max_command_chars": 32_768,
        "max_patch_bytes": 262_144,
        "default_timeout_ms": 10_000,
        "max_timeout_ms": 30_000,
        "cpu_seconds": 10,
        "address_space_bytes": 1_073_741_824,
        "max_processes": 32,
        "max_open_files": 64,
        "stdout_bytes": 16_384,
        "stderr_bytes": 16_384,
        "tmp_bytes": 67_108_864,
        "tmp_inodes": 512,
    }
    resources = {
        name: limits[name] for name in CORE.FILESYSTEM_SANDBOX_SHARED_LIMIT_FIELDS
    }
    resources.update(
        {
            "workspace_bytes": limits["max_total_bytes"],
            "workspace_inodes": limits["max_files"] + limits["max_directories"] + 1,
        }
    )
    prompt = CORE.FILESYSTEM_WEBSHOP_SYSTEM_PROMPT
    return {
        "surface": CORE.FILESYSTEM_WEBSHOP_V2_SURFACE,
        "paper_eligible": False,
        "memory_prompt_mode": "natural_filesystem",
        "memory_management": "policy_managed_persistent_workspace",
        "workspace_surface": "codex_workspace_v2",
        "workspace_tool_contract": "codex_shell_command_apply_patch_v1",
        "workspace_tool_ops": list(CORE.FILESYSTEM_TOOL_OPS),
        "workspace_persistence": "episode_across_sessions",
        "workspace_episode_isolation": True,
        "workspace_shell_enabled": True,
        "workspace_apply_patch_enabled": True,
        "workspace_host_path_exposed": False,
        "workspace_limits": limits,
        "workspace_sandbox": {
            **CORE.FILESYSTEM_SANDBOX_FIELDS,
            "ripgrep_sha256": "c" * 64,
            "ripgrep_expected_sha256": "c" * 64,
            "ripgrep_version": "ripgrep 15.1.0",
            "ripgrep_startup_fingerprint": {
                "device": 1,
                "inode": 2,
                "mode": 33_237,
                "size": 5_000_000,
                "mtime_ns": 1,
                "ctime_ns": 1,
            },
            "resource_limits": resources,
        },
        "reward_contract": {
            "workspace_action_reward": 0.0,
            "shell_command_reward": 0.0,
            "apply_patch_reward": 0.0,
            "memory_specific_shaping": "none",
        },
        "system_prompt": prompt,
        "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "service": {
            "role": "intervention_eval",
            "runtime_source_id": "a" * 40,
            "fingerprint_sha256": "b" * 64,
        },
        "workspace_intervention_control": {
            "enabled": True,
            "contract": "authenticated_first_boundary_counterfactual_copy_v1",
            "allowed_arms": list(CORE.FILESYSTEM_CAUSAL_ARMS),
            "boundary_session_index": 1,
            "source_state": "policy_authored_workspace_only",
            "authenticated_export": True,
            "hidden_answer_injection": False,
            "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        },
    }


class FakeModel:
    model_url = "http://model.test/v1"
    model = "fake-model"
    temperature = 0.0
    max_tokens = 128
    enable_thinking = False

    def __init__(self) -> None:
        self.system_prompts = []

    def _chat_template_kwargs(self):
        return {"enable_thinking": False}

    def tokenize(self, messages):
        self.system_prompts.append(messages[0]["content"])
        return [1, 2, 3], {"tokens": [1, 2, 3]}, "http://model.test/tokenize"

    def complete(self, messages):
        system = messages[0]["content"]
        observation = messages[-1]["content"]
        if observation.startswith("source:") and "written=0" in observation:
            value = "black" if "data=0" in observation else "gray"
            action = (
                "apply_patch\n*** Begin Patch\n*** Add File: MEMORY.md\n"
                f"+{value}\n*** End Patch"
            )
        elif observation.startswith("source:"):
            action = "click[Buy Now]"
        elif "tool_output=" in observation:
            action = "click[Buy Now]"
        elif "without a persistent workspace" in system:
            action = "click[Buy Now]"
        else:
            action = 'shell_command {"command":"cat MEMORY.md","workdir":"."}'
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": action},
                    "finish_reason": "stop",
                }
            ]
        }


class FakeEnv:
    def __init__(self, registry, env_id: int, metadata: dict):
        self.registry = registry
        self.env_id = env_id
        self.base_url = "http://env.test"
        self.metadata = json.loads(json.dumps(metadata))
        self.surface = self.metadata["surface"]
        self.system_prompt = self.metadata["system_prompt"]
        self.system_prompt_source = "server_metadata"
        self.info = {}
        self.last_submitted_action = None
        self.data_idx = 0
        self.session = 0
        self.value = None
        self.enabled = True
        self.audit_count = 0
        self.control_event = None
        self.done = False
        self.closed = False

    def _observation(self):
        if self.session == 0:
            return f"source:data={self.data_idx};written={int(self.value is not None)}"
        if hasattr(self, "tool_output"):
            return f"dependent:tool_output={self.tool_output}"
        contract = (
            "Persistent workspace: available."
            if self.enabled
            else "Persistent workspace: unavailable in this intervention."
        )
        return f"dependent:choose finish.\n{contract}"

    def _info(self, *, success=False, reward_components=()):
        return {
            "current_subtask_index": self.session,
            "phase_count": 2,
            "episode_success": success,
            "reward_components": list(reward_components),
            "workspace_surface": "codex_workspace_v2",
            "workspace_tool_contract": "codex_shell_command_apply_patch_v1",
            "workspace_tool_ops": ["SHELL_COMMAND", "APPLY_PATCH"],
            "workspace_intervention": "enabled" if self.enabled else "no_workspace",
            "workspace_causal_arm": (
                None if self.control_event is None else self.control_event["arm"]
            ),
            "workspace_control_event": self.control_event,
            "workspace_shell_enabled": self.enabled,
            "workspace_apply_patch_enabled": self.enabled,
            "workspace_snapshot": snapshot(self.value if self.enabled else None),
            "workspace_audit_event_count": self.audit_count,
            "workspace_ops": [],
            "workspace_latest_event": None,
            "tool_ops": [],
            "memory_ops": [],
        }

    def _response(self, reward=0.0, *, success=False, reward_components=()):
        response = {
            "id": self.env_id,
            "observation": self._observation(),
            "reward": reward,
            "done": self.done,
            "info": self._info(
                success=success,
                reward_components=reward_components,
            ),
        }
        self.info = {
            "observation": response["observation"],
            "reward": reward,
            "done": self.done,
            "env_info": response["info"],
            "metadata": self.metadata,
        }
        return response

    def reset(self, data_idx):
        self.data_idx = data_idx
        self.session = 0
        self.value = None
        self.enabled = True
        self.audit_count = 0
        self.control_event = None
        self.done = False
        if hasattr(self, "tool_output"):
            del self.tool_output
        return self._response()

    def step(self, action):
        self.last_submitted_action = action
        if action.startswith("apply_patch"):
            self.value = "black" if "+black" in action else "gray"
            self.audit_count += 1
            return self._response()
        if action.startswith("shell_command"):
            self.tool_output = self.value or "<empty>"
            self.audit_count += 1
            return self._response()
        if action == "click[Buy Now]" and self.session == 0:
            self.session = 1
            return self._response(
                1.0,
                reward_components=(
                    {"name": "correct_buy", "value": 1.0, "op": "BUY"},
                ),
            )
        if action == "click[Buy Now]" and self.session == 1:
            self.done = True
            success = self.value == "black" and getattr(self, "tool_output", None) == "black"
            reward = 2.0 if success else -1.0
            return self._response(
                reward,
                success=success,
                reward_components=(
                    {"name": "final_buy", "value": reward, "op": "BUY"},
                ),
            )
        raise AssertionError(f"unexpected action: {action}")

    def workspace_export(self, *, token):
        del token
        return {
            "schema": "agentmemory_workspace_authenticated_export_v1",
            "id": self.env_id,
            "data_idx": self.data_idx,
            "workspace_state": workspace_state(self.value),
            "policy_authored": True,
            "hidden_answer_injection": False,
        }

    def workspace_intervention(self, arm, *, token, source_env_id=None):
        del token
        before = snapshot(self.value)["tree_sha256"]
        source_tree = None
        if arm == "blank":
            self.value = None
        elif arm == "swapped":
            source = self.registry[source_env_id]
            self.value = source.value
            source_tree = snapshot(source.value)["tree_sha256"]
        elif arm == "no_workspace":
            self.value = None
            self.enabled = False
        elif arm == "correct":
            source_tree = snapshot(self.value)["tree_sha256"]
        else:
            raise AssertionError(arm)
        self.audit_count = 0
        self.control_event = {
            "arm": arm,
            "policy_action": False,
            "task_reward": 0.0,
            "source_tree_sha256": source_tree,
            "workspace_tree_sha256_before": before,
            "workspace_tree_sha256_after": snapshot(
                self.value if self.enabled else None
            )["tree_sha256"],
        }
        return self._response()

    def close(self):
        self.closed = True
        return True


class FilesystemCausalEvalTest(unittest.TestCase):
    def test_contract_hash_ignores_only_dynamic_environment_counts(self):
        token = "t" * 48
        metadata = filesystem_metadata(token)
        metadata.update(
            {
                "source": "agentmemory_programmatic_generator",
                "active_environment_count": 0,
                "backend": {
                    "active_session_count": 0,
                    "price_seed": 233,
                },
            }
        )
        reference = CAUSAL._causal_metadata_sha256(metadata)

        dynamic = json.loads(json.dumps(metadata))
        dynamic["active_environment_count"] = 17
        dynamic["backend"]["active_session_count"] = 19
        self.assertEqual(reference, CAUSAL._causal_metadata_sha256(dynamic))

        static_mutations = {
            "source": lambda value: value.__setitem__("source", "other_source"),
            "backend": lambda value: value["backend"].__setitem__(
                "price_seed", 234
            ),
            "sandbox": lambda value: value["workspace_sandbox"].__setitem__(
                "network", "host"
            ),
        }
        for label, mutate in static_mutations.items():
            with self.subTest(label=label):
                changed = json.loads(json.dumps(metadata))
                mutate(changed)
                self.assertNotEqual(
                    reference,
                    CAUSAL._causal_metadata_sha256(changed),
                )

    def test_contract_mismatch_closes_every_created_environment(self):
        token = "t" * 48
        registry = {}
        next_id = 0

        def factory():
            nonlocal next_id
            metadata = filesystem_metadata(token)
            metadata["backend"] = {
                "active_session_count": next_id,
                "price_seed": 233 if next_id == 0 else 234,
            }
            metadata["active_environment_count"] = next_id
            env = FakeEnv(registry, next_id, metadata)
            registry[next_id] = env
            next_id += 1
            return env

        with tempfile.TemporaryDirectory() as temporary:
            runner = CAUSAL.FilesystemCausalEvalRunner(
                factory,
                FakeModel(),
                indices=[0],
                max_policy_turns=8,
                output_dir=Path(temporary),
                intervention_token=token,
            )
            with self.assertRaisesRegex(
                CORE.EvalError,
                "causal arms resolved to different environment contracts",
            ):
                runner.run_orbit(0)

        self.assertEqual(set(registry), {0, 1})
        self.assertTrue(all(env.closed for env in registry.values()))

    def test_real_policy_source_then_four_exact_replays_and_interventions(self):
        token = "t" * 48
        metadata = filesystem_metadata(token)
        registry = {}
        next_id = 0

        def factory():
            nonlocal next_id
            env = FakeEnv(registry, next_id, metadata)
            registry[next_id] = env
            next_id += 1
            return env

        model = FakeModel()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = CAUSAL.FilesystemCausalEvalRunner(
                factory,
                model,
                indices=[0],
                max_policy_turns=8,
                output_dir=output,
                intervention_token=token,
            ).run()
            persisted = json.loads((output / "manifest.json").read_text())

        orbit = manifest["orbits"][0]
        self.assertTrue(orbit["eligible"])
        self.assertEqual(orbit["sources"]["target"]["actions"], [
            "apply_patch\n*** Begin Patch\n*** Add File: MEMORY.md\n+black\n*** End Patch",
            "click[Buy Now]",
        ])
        self.assertEqual(
            orbit["sources"]["paired"]["workspace_export"]["workspace_state"][
                "files"
            ][0]["content_base64"],
            base64.b64encode(b"gray").decode("ascii"),
        )
        self.assertTrue(all(item["matches_source"] for item in orbit["replays"].values()))
        self.assertTrue(orbit["arms"]["correct"]["episode_success"])
        for arm in ("blank", "swapped", "no_workspace"):
            self.assertFalse(orbit["arms"][arm]["episode_success"])
        self.assertEqual(
            manifest["summary"]["strict_four_arm_separation_count"],
            1,
        )
        no_workspace_prompt = orbit["arms"]["no_workspace"]["system_prompt"]
        self.assertIn("without a persistent workspace", no_workspace_prompt)
        self.assertNotIn("shell_command JSON action", no_workspace_prompt)
        self.assertEqual(persisted["summary"], manifest["summary"])
        self.assertNotIn(token, json.dumps(persisted))


if __name__ == "__main__":
    unittest.main()
