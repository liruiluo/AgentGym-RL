from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from test_paired_eval_support import ManualClock, make_config, make_fake_runtime

from paired_eval.controller import AgentGymPolicyTurnController
from paired_eval.evidence import PrivateEvidenceStore
from paired_eval.runner import PairedRunner


class ExactControllerAdapterTest(unittest.TestCase):
    def test_integration_adapter_calls_exact_controller_functions(self) -> None:
        calls = []

        def bind_initial(client, messages):
            calls.append("bind_initial_policy_context")
            client.bind_policy_context(messages, initial=True)
            return [dict(message) for message in messages]

        def prepare(client, messages, **kwargs):
            calls.append("prepare_policy_turn")
            count = kwargs["count_prompt_tokens"](messages)
            return SimpleNamespace(
                messages=tuple(dict(message) for message in messages),
                prompt_token_count=count,
                control_request=None,
            )

        def complete(client, prepared, policy_output):
            calls.append("complete_policy_turn")
            output = client.step(policy_output)
            messages = [dict(message) for message in prepared.messages]
            messages.extend(
                (
                    {"role": "assistant", "content": policy_output},
                    {"role": "user", "content": output.state},
                )
            )
            return output, messages

        with tempfile.TemporaryDirectory() as temp_dir:
            store = PrivateEvidenceStore(Path(temp_dir) / "evidence")
            config = make_config()
            bindings = make_fake_runtime(config, store)
            runner = PairedRunner(
                controller=AgentGymPolicyTurnController(
                    bind_initial_policy_context=bind_initial,
                    prepare_policy_turn=prepare,
                    complete_policy_turn=complete,
                ),
                evidence_store=store,
                clock=ManualClock(),
            )
            row = runner.run_task(config, bindings.adapter, bindings.model)

        self.assertEqual(
            calls,
            [
                "bind_initial_policy_context",
                "prepare_policy_turn",
                "complete_policy_turn",
            ],
        )
        self.assertEqual(row["termination"]["reason"], "terminal")


if __name__ == "__main__":
    unittest.main()
