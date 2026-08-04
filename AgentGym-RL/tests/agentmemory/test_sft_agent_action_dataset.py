from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from verl.utils.agent_dataset.agent_action_schema import (
    AGENT_ACTION_SCHEMA_V1,
    AgentActionSchemaError,
    canonical_json_sha256,
    finalize_agent_action_record,
    text_sha256,
    validate_agent_action_record,
)
from verl.utils.agent_dataset.sft_dataset import SFTDataset


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, *, prompt_ids=None, action_ids=None):
        self.prompt_ids = list(prompt_ids or [10, 11, 12])
        self.action_ids = list(action_ids or [20, 21])
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), dict(kwargs)))
        return list(self.prompt_ids)

    def encode(self, text, *, add_special_tokens):
        self.calls.append((text, {"add_special_tokens": add_special_tokens}))
        if text == "<|im_end|>":
            return [99]
        return list(self.action_ids)


SURFACE = "agentmemory_webshop_procedural_natural_chain_filesystem_v2"
TASK_FAMILY = "procedural_natural_attribute_chain_shopping"
SCENARIO_ID = "finish"
TARGET_ASIN = "ABCDE12345"


def _seal(record):
    execution = record["execution"]
    receipt = {
        "submitted_action": execution["submitted_action"],
        "observation_after": execution["observation_after"],
        "reward": execution["reward"],
        "terminated": execution["terminated"],
        "truncated": execution["truncated"],
        "info_after": execution["info_after"],
    }
    execution["receipt_sha256"] = canonical_json_sha256(receipt)
    provenance = record["provenance"]
    provenance.update(
        {
            "system_prompt_sha256": text_sha256(record["system_prompt"]),
            "observation_sha256": text_sha256(record["observation"]),
            "assistant_action_sha256": text_sha256(record["assistant_action"]),
            "observation_after_sha256": text_sha256(execution["observation_after"]),
            "env_info_before_sha256": canonical_json_sha256(execution["info_before"]),
            "env_info_after_sha256": canonical_json_sha256(execution["info_after"]),
        }
    )
    return finalize_agent_action_record(record)


def _environment_info(*, phase_index, tree_sha256, event_count, event=None):
    workspace_event = copy.deepcopy(event) if event and event["op"] in {
        "SHELL_COMMAND",
        "APPLY_PATCH",
    } else None
    tool_ops = [] if event is None else [copy.deepcopy(event)]
    return {
        "surface": SURFACE,
        "task_family": TASK_FAMILY,
        "split": "train",
        "scenario_id": SCENARIO_ID,
        "current_subtask_index": phase_index,
        "phase_count": 6,
        "workspace_snapshot": {"tree_sha256": tree_sha256},
        "workspace_audit_event_count": event_count,
        "tool_ops": tool_ops,
        "workspace_ops": [] if workspace_event is None else [workspace_event],
        "workspace_latest_event": workspace_event,
    }


def _bind_workspace_event(event, *, action, event_id, phase_index):
    event["event_id"] = event_id
    event["phase_index"] = phase_index
    event["episode_id"] = "fixture:episode:1"
    if action.startswith("shell_command "):
        payload = json.loads(action.removeprefix("shell_command "))
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        command = payload["command"]
        event["request_sha256"] = text_sha256(canonical)
        event["command_sha256"] = text_sha256(command)
        event["command_bytes"] = len(command.encode("utf-8"))
    else:
        patch_text = action.removeprefix("apply_patch\n")
        event["request_sha256"] = text_sha256(patch_text)
        event["patch_sha256"] = text_sha256(patch_text)
        event["patch_bytes"] = len(patch_text.encode("utf-8"))
    return event


def _record(
    action="shell_command {\"command\":\"cat .agent_memory/MEMORY.md\",\"workdir\":\".\"}",
    *,
    final_buy=False,
):
    system_prompt = "Exact filesystem system prompt."
    observation = "Latest observation only."
    observation_after = "Chunk ID: 123\nProcess exited with code 0\nFinal output:\nnote contents"
    phase_index = 5 if final_buy else 1
    before_tree = "1" * 64
    after_tree = before_tree
    before_event_count = 3
    after_event_count = before_event_count
    reward = 0.0
    terminated = False
    native_event = None
    purchase_receipt = None
    workspace_event = None

    if action.startswith("shell_command "):
        action_kind = "workspace_shell_command"
        workspace_event = {
            "op": "SHELL_COMMAND",
            "status": "executed",
            "exit_code": 0,
            "timed_out": False,
            "workspace_tree_sha256_before": before_tree,
            "workspace_tree_sha256_after": after_tree,
        }
        _bind_workspace_event(
            workspace_event,
            action=action,
            event_id=before_event_count,
            phase_index=phase_index,
        )
        after_event_count += 1
    elif action.startswith("apply_patch\n"):
        action_kind = "workspace_apply_patch"
        after_tree = "8" * 64
        workspace_event = {
            "op": "APPLY_PATCH",
            "status": "executed",
            "transactional": True,
            "changed_paths": [".agent_memory/MEMORY.md"],
            "workspace_tree_sha256_before": before_tree,
            "workspace_tree_sha256_after": after_tree,
        }
        _bind_workspace_event(
            workspace_event,
            action=action,
            event_id=before_event_count,
            phase_index=phase_index,
        )
        after_event_count += 1
    elif action.startswith("search["):
        action_kind = "native_search"
        native_event = {
            "op": "SEARCH",
            "raw_action": action,
            "result_count": 2,
        }
    elif action == "click[Buy Now]":
        action_kind = "native_click"
        reward = 2.0 if final_buy else 1.0
        terminated = final_buy
        native_event = {
            "op": "BUY",
            "raw_action": action,
            "committed": True,
            "purchase_correct": True,
            "session_advanced": True,
            "terminal": final_buy,
            "step": 8,
            "session_index": phase_index,
        }
        purchase_receipt = {
            **native_event,
            "actual_asin": TARGET_ASIN,
            "actual_price_cents": 1299,
            "selected_options": {},
            "budget_ok": True,
        }
    elif action.startswith("click["):
        action_kind = "native_click"
        native_event = {"op": "CLICK", "raw_action": action}
    else:  # pragma: no cover - fixture callers own the closed action set.
        raise AssertionError(f"unsupported fixture action: {action}")

    info_before = _environment_info(
        phase_index=phase_index,
        tree_sha256=before_tree,
        event_count=before_event_count,
    )
    info_after = _environment_info(
        phase_index=phase_index + (1 if native_event and native_event["op"] == "BUY" else 0),
        tree_sha256=after_tree,
        event_count=after_event_count,
        event=workspace_event or native_event,
    )
    record = {
        "schema": AGENT_ACTION_SCHEMA_V1,
        "system_prompt": system_prompt,
        "observation": observation,
        "assistant_action": action,
        "action_kind": action_kind,
        "chat_template": {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "assistant_terminator": "<|im_end|>",
        },
        "execution": {
            "accepted": True,
            "action_effect_verified": True,
            "submitted_action": action,
            "observation_after": observation_after,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info_before": info_before,
            "info_after": info_after,
            "receipt_sha256": "",
        },
        "task": {
            "surface": SURFACE,
            "task_family": TASK_FAMILY,
            "task_id": "task_0_a",
            "orbit_id": "orbit_0",
            "scenario_id": SCENARIO_ID,
            "split": "train",
            "data_index": 0,
            "orbit_index": 0,
            "branch_index": 0,
            "phase_index": phase_index,
            "turn_index": 4,
        },
        "workspace_audit": {
            "applicable": workspace_event is not None,
            "committed": workspace_event is not None,
            "tree_sha256_before": before_tree,
            "tree_sha256_after": after_tree,
            "event": copy.deepcopy(workspace_event),
        },
        "native_audit": {
            "applicable": native_event is not None,
            "target_asin": TARGET_ASIN if native_event is not None else None,
            "event": copy.deepcopy(native_event),
            "purchase_receipt": (
                copy.deepcopy(purchase_receipt)
                if purchase_receipt is not None
                else None
            ),
        },
        "provenance": {
            "outer_source_commit": "2" * 40,
            "agentgym_source_commit": "3" * 40,
            "provider_proof_sha256": "4" * 64,
            "product_pool_sha256": "5" * 64,
            "system_prompt_sha256": "",
            "observation_sha256": "",
            "assistant_action_sha256": "",
            "observation_after_sha256": "",
            "env_info_before_sha256": "",
            "env_info_after_sha256": "",
            "task_semantic_sha256": "6" * 64,
            "target_product_record_sha256": "7" * 64,
        },
    }
    record["execution"]["reward"] = reward
    record["execution"]["terminated"] = terminated
    return _seal(record)


class AgentActionSchemaTests(unittest.TestCase):
    def test_valid_record_is_content_addressed(self):
        record = _record()
        fields = validate_agent_action_record(record)
        self.assertEqual(fields["assistant_action"], record["assistant_action"])
        unhashed = dict(record)
        observed = unhashed.pop("record_sha256")
        self.assertEqual(observed, canonical_json_sha256(unhashed))

    def test_validates_each_executed_action_kind(self):
        patch = "\n".join(
            [
                "apply_patch",
                "*** Begin Patch",
                "*** Add File: .agent_memory/MEMORY.md",
                "+preferred finish: black",
                "*** End Patch",
            ]
        )
        for action in (
            patch,
            "search[black insulated bottle]",
            f"click[{TARGET_ASIN}]",
            "click[Buy Now]",
        ):
            with self.subTest(action=action):
                fields = validate_agent_action_record(_record(action))
                self.assertEqual(fields["assistant_action"], action)
        final = _record("click[Buy Now]", final_buy=True)
        self.assertEqual(
            validate_agent_action_record(final)["assistant_action"],
            "click[Buy Now]",
        )

    def test_rejects_unexecuted_or_tampered_record(self):
        unexecuted = _record()
        unexecuted["execution"]["action_effect_verified"] = False
        unexecuted = finalize_agent_action_record(unexecuted)
        with self.assertRaisesRegex(AgentActionSchemaError, "effect_verified"):
            validate_agent_action_record(unexecuted)

        tampered = _record()
        tampered["observation"] += " changed"
        with self.assertRaisesRegex(AgentActionSchemaError, "observation_sha256"):
            validate_agent_action_record(tampered)

    def test_rejects_tampered_execution_receipt(self):
        record = _record()
        record["execution"]["observation_after"] += " changed"
        record = finalize_agent_action_record(record)
        with self.assertRaisesRegex(AgentActionSchemaError, "receipt_sha256"):
            validate_agent_action_record(record)

    def test_rejects_workspace_event_not_bound_to_info_after(self):
        record = _record()
        event = copy.deepcopy(record["workspace_audit"]["event"])
        event["exit_code"] = 17
        record["workspace_audit"]["event"] = event
        record = _seal(record)
        with self.assertRaisesRegex(AgentActionSchemaError, "exact workspace event"):
            validate_agent_action_record(record)

    def test_rejects_workspace_request_or_command_hash_tampering(self):
        cases = (
            ("shell request", _record(), "request_sha256", "f" * 64),
            ("shell command", _record(), "command_sha256", "e" * 64),
            ("shell byte count", _record(), "command_bytes", 1),
            (
                "patch request",
                _record(
                    "\n".join(
                        [
                            "apply_patch",
                            "*** Begin Patch",
                            "*** Add File: .agent_memory/MEMORY.md",
                            "+preferred finish: black",
                            "*** End Patch",
                        ]
                    )
                ),
                "request_sha256",
                "d" * 64,
            ),
            (
                "patch digest",
                _record(
                    "\n".join(
                        [
                            "apply_patch",
                            "*** Begin Patch",
                            "*** Add File: .agent_memory/MEMORY.md",
                            "+preferred finish: black",
                            "*** End Patch",
                        ]
                    )
                ),
                "patch_sha256",
                "c" * 64,
            ),
            (
                "patch byte count",
                _record(
                    "\n".join(
                        [
                            "apply_patch",
                            "*** Begin Patch",
                            "*** Add File: .agent_memory/MEMORY.md",
                            "+preferred finish: black",
                            "*** End Patch",
                        ]
                    )
                ),
                "patch_bytes",
                1,
            ),
        )
        for name, record, field, value in cases:
            with self.subTest(name=name):
                event = record["workspace_audit"]["event"]
                event[field] = value
                record["execution"]["info_after"]["tool_ops"][0][field] = value
                record["execution"]["info_after"]["workspace_ops"][0][field] = value
                record["execution"]["info_after"]["workspace_latest_event"][field] = value
                record = _seal(record)
                with self.assertRaisesRegex(
                    AgentActionSchemaError,
                    "bound to assistant_action|bound to the submitted command|wrong request size",
                ):
                    validate_agent_action_record(record)

    def test_rejects_tampered_native_execution_evidence(self):
        search = _record("search[black insulated bottle]")
        search["native_audit"]["event"]["result_count"] = 0
        search["execution"]["info_after"]["tool_ops"][0]["result_count"] = 0
        search = _seal(search)
        with self.assertRaisesRegex(AgentActionSchemaError, "at least one"):
            validate_agent_action_record(search)

        click = _record(f"click[{TARGET_ASIN}]")
        click["native_audit"]["target_asin"] = "ZZZZZ99999"
        click = _seal(click)
        with self.assertRaisesRegex(AgentActionSchemaError, "target-ASIN"):
            validate_agent_action_record(click)

        buy = _record("click[Buy Now]")
        buy["native_audit"]["purchase_receipt"]["actual_asin"] = "ZZZZZ99999"
        buy = _seal(buy)
        with self.assertRaisesRegex(AgentActionSchemaError, "target ASIN"):
            validate_agent_action_record(buy)

    def test_rejects_non_finite_execution_reward(self):
        record = _record()
        record["execution"]["reward"] = float("nan")
        with self.assertRaisesRegex(AgentActionSchemaError, "finite numeric"):
            validate_agent_action_record(record)


class AgentActionDatasetTests(unittest.TestCase):
    def _dataset(self, record, tokenizer, *, max_length=8, truncation="error"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(
                json.dumps([record], ensure_ascii=False), encoding="utf-8"
            )
            dataset = SFTDataset(
                json_file=str(path),
                tokenizer=tokenizer,
                max_length=max_length,
                truncation=truncation,
                data_mode="agent_action_v1",
            )
            return dataset[0]

    def test_exact_prompt_action_terminator_and_loss_mask(self):
        tokenizer = FakeTokenizer()
        item = self._dataset(_record(), tokenizer)
        self.assertEqual(item["input_ids"].tolist(), [10, 11, 12, 20, 21, 99, 0, 0])
        self.assertEqual(item["attention_mask"].tolist(), [1, 1, 1, 1, 1, 1, 0, 0])
        self.assertEqual(item["loss_mask"].tolist(), [0, 0, 0, 1, 1, 1, 0, 0])
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": _record()["system_prompt"]},
                {"role": "user", "content": _record()["observation"]},
            ],
        )
        self.assertEqual(
            kwargs,
            {
                "add_generation_prompt": True,
                "tokenize": True,
                "enable_thinking": False,
            },
        )

    def test_left_truncation_preserves_complete_target(self):
        tokenizer = FakeTokenizer(prompt_ids=[10, 11, 12, 13, 14])
        item = self._dataset(
            _record(), tokenizer, max_length=5, truncation="left"
        )
        self.assertEqual(item["input_ids"].tolist(), [13, 14, 20, 21, 99])
        self.assertEqual(item["loss_mask"].tolist(), [0, 0, 1, 1, 1])

    def test_right_truncation_cannot_cut_supervised_target(self):
        tokenizer = FakeTokenizer(prompt_ids=[10, 11, 12, 13, 14])
        with self.assertRaisesRegex(ValueError, "removed supervised action"):
            self._dataset(_record(), tokenizer, max_length=6, truncation="right")


if __name__ == "__main__":
    unittest.main()
