from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from test_paired_eval_support import (
    Arm,
    ManualClock,
    make_config,
    make_fake_runtime,
)

from paired_eval.controller import DependencyLightPolicyTurnController
from paired_eval.evidence import PrivateEvidenceStore
from paired_eval.runner import PairedRunner
from paired_eval.verifier import validate_result_row


class PairedRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store = PrivateEvidenceStore(Path(self.temp_dir.name) / "evidence")
        self.clock = ManualClock()
        self.runner = PairedRunner(
            controller=DependencyLightPolicyTurnController(),
            evidence_store=self.store,
            clock=self.clock,
        )

    def run_case(self, config, **runtime_kwargs):
        bindings = make_fake_runtime(config, self.store, **runtime_kwargs)
        row = self.runner.run_task(config, bindings.adapter, bindings.model)
        validate_result_row(row)
        return row, bindings

    def test_three_adapters_traverse_identical_ordered_loop(self) -> None:
        receipt_shapes = []
        for benchmark, artifact_type in (
            ("gaia_text", "answer"),
            ("swebench_verified", "patch"),
            ("mlebench_lite", "submission"),
        ):
            config = make_config(
                benchmark=benchmark,
                artifact_type=artifact_type,
            )
            row, bindings = self.run_case(config)
            self.assertEqual(bindings.adapter.client.reset_calls, [0])
            self.assertEqual(bindings.adapter.close_calls, 1)
            self.assertEqual(
                bindings.adapter.client.step_calls,
                ["ordinary-policy-output"],
            )
            self.assertEqual(len(bindings.model.calls), 1)
            self.assertEqual(row["termination"]["reason"], "terminal")
            self.assertEqual(row["usage"]["policy_turns"], 1)
            self.assertEqual(row["usage"]["tool_calls"], 1)
            self.assertEqual(row["usage"]["policy_outputs"], 1)
            self.assertEqual(row["usage"]["parsed_actions"], 1)
            self.assertEqual(row["usage"]["domain_tool_attempts"], 1)
            self.assertEqual(row["usage"]["successful_backend_calls"], 1)
            self.assertEqual(row["usage"]["answers"], 0)
            self.assertEqual(row["final_artifact"]["type"], artifact_type)
            shape = [receipt["kind"] for receipt in row["receipts"]]
            receipt_shapes.append(shape)
            self.assertEqual(
                shape,
                [
                    "namespace_reset",
                    "model",
                    "environment",
                    "context_transition",
                    "artifact",
                    "scorer",
                    "close",
                ],
            )
        self.assertEqual(receipt_shapes[0], receipt_shapes[1])
        self.assertEqual(receipt_shapes[1], receipt_shapes[2])

    def test_memory_and_compaction_rows_use_ordinary_budgets(self) -> None:
        config = make_config(
            arm=Arm.AMG_MEMORY,
            max_policy_turns=3,
            max_tool_calls=3,
        )
        plan = [
            {
                "state": "memory write receipt",
                "done": False,
                "wrapper_evidence": {"row_class": "memory"},
            },
            {
                "state": "not appended",
                "done": False,
                "control_request": "Author a compact continuation state.",
                "operation": "replace_messages",
                "wrapper_evidence": {"row_class": "compaction"},
            },
            {
                "state": "terminal",
                "reward": 1.0,
                "done": True,
                "wrapper_evidence": {"row_class": "ordinary"},
            },
        ]
        outputs = (
            "MEMORY_WRITE(policy-note,private-value)",
            "Policy-authored compact continuation.",
            "ordinary final output",
        )
        row, bindings = self.run_case(config, plan=plan, outputs=outputs)

        self.assertEqual(bindings.adapter.client.step_calls, list(outputs))
        self.assertEqual(row["usage"]["policy_turns"], 3)
        self.assertEqual(row["usage"]["tool_calls"], 2)
        self.assertEqual(row["usage"]["workspace_actions"], 1)
        self.assertEqual(row["usage"]["compactions"], 1)
        self.assertEqual(len(row["turns"]), 3)
        self.assertEqual(
            [turn["root_kind"] for turn in row["turns"]],
            ["external_memory", "policy_context", "benchmark_task"],
        )
        self.assertEqual(
            [turn["context_operation"] for turn in row["turns"]],
            ["append_observation", "replace_messages", "append_observation"],
        )

    def test_compaction_only_uses_compaction_but_memory_fails_closed(self) -> None:
        config = make_config(
            arm=Arm.AMG_COMPACTION_ONLY,
            max_policy_turns=2,
            max_tool_calls=2,
        )
        row, bindings = self.run_case(
            config,
            plan=(
                {
                    "state": "compacted",
                    "done": False,
                    "control_request": "Author a compact continuation state.",
                    "operation": "replace_messages",
                },
                {"state": "terminal", "reward": 1.0, "done": True},
            ),
            outputs=("Policy-authored compact state.", "ordinary final output"),
        )

        self.assertIsNone(bindings.adapter.memory_service)
        self.assertEqual(row["usage"]["policy_turns"], 2)
        self.assertEqual(row["usage"]["tool_calls"], 1)
        self.assertEqual(row["compaction"]["receipt_count"], 1)
        self.assertEqual(
            [turn["root_kind"] for turn in row["turns"]],
            ["policy_context", "benchmark_task"],
        )

        leaked, leaked_bindings = self.run_case(
            replace(config, task=replace(config.task, task_id="memory-leak")),
            plan=({"state": "must not execute", "done": True},),
            outputs=("MEMORY_WRITE(note,forbidden)",),
        )
        self.assertIsNone(leaked_bindings.adapter.memory_service)
        self.assertEqual(leaked["failure"]["class"], "environment_failure")
        self.assertFalse(leaked["comparable"])

    def test_enabled_arms_share_compaction_and_action_accounting(self) -> None:
        plan = (
            {
                "state": "compacted",
                "done": False,
                "control_request": "Author a compact continuation state.",
                "operation": "replace_messages",
            },
            {"state": "terminal", "reward": 1.0, "done": True},
        )
        outputs = ("Policy-authored compact state.", "ordinary final output")
        rows = []
        for arm in (Arm.AMG_COMPACTION_ONLY, Arm.AMG_MEMORY):
            row, _ = self.run_case(
                make_config(
                    arm=arm,
                    max_policy_turns=2,
                    max_tool_calls=2,
                ),
                plan=plan,
                outputs=outputs,
            )
            rows.append(row)

        self.assertEqual(rows[0]["compaction"], rows[1]["compaction"])
        self.assertEqual(rows[0]["budgets"], rows[1]["budgets"])
        self.assertEqual(
            {
                name: rows[0]["usage"][name]
                for name in ("policy_turns", "tool_calls")
            },
            {
                name: rows[1]["usage"][name]
                for name in ("policy_turns", "tool_calls")
            },
        )
        self.assertLessEqual(
            rows[0]["turns"][0]["response_token_count"],
            rows[0]["compaction"]["summary_max_tokens"],
        )
        self.assertEqual(
            [turn["execution_kind"] for turn in rows[0]["turns"]],
            ["policy_compaction", "benchmark_action"],
        )
        self.assertEqual(
            [turn["execution_kind"] for turn in rows[1]["turns"]],
            ["policy_compaction", "benchmark_action"],
        )

    def test_wrapper_owned_memory_routing_and_namespace_isolation(self) -> None:
        memory_config = make_config(
            task_id="memory-routing",
            arm=Arm.AMG_MEMORY,
            max_policy_turns=2,
            max_tool_calls=2,
        )
        memory_row, memory_bindings = self.run_case(
            memory_config,
            plan=(
                {"state": "write receipt", "done": False},
                {"state": "read receipt", "done": True},
            ),
            outputs=("MEMORY_WRITE(note,private-value)", "MEMORY_READ(note)"),
        )
        memory_service = memory_bindings.adapter.memory_service
        self.assertIsNotNone(memory_service)
        self.assertEqual(
            memory_service.events,
            [
                ("write", "note", "private-value"),
                ("read", "note", "private-value"),
            ],
        )
        self.assertTrue(memory_service.closed)
        self.assertNotIn("private-value", str(memory_row))
        self.assertEqual(memory_row["usage"]["policy_turns"], 2)
        self.assertEqual(memory_row["usage"]["tool_calls"], 2)
        self.assertEqual(
            [
                turn["execution_receipt"]["operation"]
                for turn in memory_row["turns"]
            ],
            ["read_write", "read_write"],
        )
        self.assertEqual(
            {
                route["root_kind"]
                for route in memory_bindings.adapter.client.route_calls
            },
            {"external_memory"},
        )

        stable_fields = {
            "schema",
            "env_info",
            "action_submission",
            "native_step_before",
            "native_step_after",
            "native_call_count_before",
            "native_call_count_after",
            "context_epoch_before",
            "context_epoch_after",
            "session_epoch_before",
            "session_epoch_after",
            "policy_step_before",
            "policy_step_after",
            "context_transition",
            "wrapper_evidence",
        }
        self.assertEqual(
            set(memory_bindings.adapter.client.step_infos[0]), stable_fields
        )

        isolated_config = replace(
            memory_config,
            run_id="isolated-run",
            task=replace(memory_config.task, task_id="isolated-task", seed=8),
        )
        isolated_bindings = make_fake_runtime(isolated_config, self.store)
        self.assertNotEqual(
            memory_service.root_id,
            isolated_bindings.adapter.memory_service.root_id,
        )
        self.assertNotIn("note", isolated_bindings.adapter.memory_service.values)

        native_config = make_config(task_id="native-memory-attempt")
        native_row, native_bindings = self.run_case(
            native_config,
            plan=({"state": "must not execute", "done": True},),
            outputs=("MEMORY_WRITE(note,forbidden)",),
        )
        self.assertIsNone(native_bindings.adapter.memory_service)
        self.assertEqual(native_row["failure"]["class"], "environment_failure")
        self.assertFalse(native_row["comparable"])

    def test_wrong_capability_or_root_receipts_fail_closed(self) -> None:
        wrong_capability, _ = self.run_case(
            make_config(task_id="wrong-capability"),
            plan=(
                {
                    "state": "must not be accepted",
                    "done": True,
                    "route_override": {
                        "capability_id": "external_memory",
                        "root_kind": "external_memory",
                        "root_id": "f" * 64,
                    },
                },
            ),
            outputs=("ordinary output",),
        )
        self.assertEqual(
            wrong_capability["failure"]["class"],
            "environment_failure",
        )
        self.assertIsNone(wrong_capability["turns"][0]["environment_ref"])
        self.assertFalse(wrong_capability["comparable"])

        wrong_root, _ = self.run_case(
            make_config(task_id="wrong-root", arm=Arm.AMG_MEMORY),
            plan=(
                {
                    "state": "must not be accepted",
                    "done": True,
                    "route_override": {"root_id": "f" * 64},
                },
            ),
            outputs=("MEMORY_WRITE(note,private-value)",),
        )
        self.assertEqual(wrong_root["failure"]["class"], "environment_failure")
        self.assertIsNone(wrong_root["turns"][0]["environment_ref"])
        self.assertFalse(wrong_root["comparable"])

        cross_spoof, _ = self.run_case(
            make_config(task_id="cross-spoof", arm=Arm.AMG_MEMORY),
            plan=(
                {
                    "state": "must not be accepted",
                    "done": True,
                    "execution_kind_override": "benchmark_action",
                },
            ),
            outputs=("MEMORY_WRITE(note,private-value)",),
        )
        self.assertEqual(cross_spoof["failure"]["class"], "environment_failure")
        self.assertIsNone(cross_spoof["turns"][0]["environment_ref"])

        wrong_output_binding, _ = self.run_case(
            make_config(task_id="wrong-output-binding"),
            plan=(
                {
                    "state": "must not be accepted",
                    "done": True,
                    "policy_output_sha256_override": "f" * 64,
                },
            ),
            outputs=("ordinary output",),
        )
        self.assertEqual(
            wrong_output_binding["failure"]["class"],
            "environment_failure",
        )
        self.assertIsNone(
            wrong_output_binding["turns"][0]["environment_ref"]
        )

    def test_native_compaction_and_prompt_drift_fail_closed(self) -> None:
        compaction_row, compaction_bindings = self.run_case(
            make_config(task_id="native-compaction"),
            plan=(
                {
                    "state": "must not execute",
                    "done": True,
                    "control_request": "compact now",
                    "operation": "replace_messages",
                },
            ),
            outputs=("forbidden compact state",),
        )
        self.assertEqual(
            compaction_row["failure"]["class"], "environment_failure"
        )
        self.assertEqual(compaction_bindings.model.calls, [])

        prompt_row, prompt_bindings = self.run_case(
            make_config(task_id="prompt-drift", arm=Arm.AMG_MEMORY),
            prompt_declaration_override="",
        )
        self.assertEqual(prompt_row["failure"]["class"], "environment_failure")
        self.assertIsNone(prompt_row["prompt"]["full_sha256"])
        self.assertEqual(prompt_bindings.model.calls, [])

    def test_horizon_stops_before_an_unbudgeted_third_row(self) -> None:
        config = make_config(max_policy_turns=2, max_tool_calls=2)
        plan = [
            {"state": "one", "done": False},
            {"state": "two", "done": False},
            {"state": "must not execute", "done": True},
        ]
        row, bindings = self.run_case(
            config,
            plan=plan,
            outputs=("one", "two", "three"),
        )

        self.assertEqual(bindings.adapter.client.step_calls, ["one", "two"])
        self.assertEqual(row["termination"]["reason"], "horizon")
        self.assertEqual(row["termination"]["horizon_cause"], "policy_turn_limit")
        self.assertTrue(row["comparable"])

    def test_tool_and_token_horizons_stop_before_extra_sampling(self) -> None:
        tool_config = make_config(max_policy_turns=3, max_tool_calls=1)
        tool_row, tool_bindings = self.run_case(
            tool_config,
            plan=(
                {"state": "one", "done": False},
                {"state": "must not execute", "done": True},
            ),
            outputs=("one", "two"),
        )
        self.assertEqual(tool_bindings.adapter.client.step_calls, ["one"])
        self.assertEqual(tool_row["termination"]["horizon_cause"], "tool_call_limit")

        token_config = make_config(
            task_id="token-horizon",
            max_policy_turns=3,
            max_total_tokens=15,
        )
        token_row, token_bindings = self.run_case(
            token_config,
            plan=(
                {"state": "one", "done": False},
                {"state": "must not execute", "done": True},
            ),
            outputs=("abcde", "unused"),
        )
        self.assertEqual(token_bindings.adapter.client.step_calls, ["abcde"])
        self.assertEqual(token_row["usage"]["total_tokens"], 15)
        self.assertEqual(token_row["termination"]["horizon_cause"], "token_limit")

    def test_response_capacity_respects_model_context_limit(self) -> None:
        config = make_config(task_id="model-context-limit")
        config = replace(
            config,
            budgets=replace(
                config.budgets,
                max_prompt_tokens=15,
                max_model_tokens=16,
            ),
        )
        row, bindings = self.run_case(
            config,
            plan=({"state": "terminal", "done": True},),
            outputs=("123456",),
        )

        requested = bindings.model.calls[0]["decoding"]["max_output_tokens"]
        self.assertEqual(requested, 6)
        self.assertEqual(row["usage"]["total_tokens"], 16)

    def test_terminal_model_environment_timeout_and_finalization_taxonomy(self) -> None:
        terminal, _ = self.run_case(make_config())
        self.assertEqual(terminal["failure"]["class"], None)
        self.assertTrue(terminal["comparable"])

        model_failure, _ = self.run_case(
            make_config(task_id="model-failure"),
            model_fail_on_call=1,
        )
        self.assertEqual(model_failure["failure"]["class"], "model_failure")
        self.assertEqual(model_failure["usage"]["policy_turns"], 0)
        self.assertEqual(model_failure["usage"]["tool_calls"], 0)
        self.assertFalse(model_failure["comparable"])

        tokenization_failure, _ = self.run_case(
            make_config(task_id="tokenization-failure"),
            tokenize_fail_on_call=1,
        )
        self.assertEqual(
            tokenization_failure["failure"]["class"],
            "model_failure",
        )
        self.assertEqual(
            tokenization_failure["failure"]["stage"],
            "model_tokenization",
        )
        self.assertEqual(tokenization_failure["usage"]["policy_turns"], 0)
        self.assertEqual(tokenization_failure["usage"]["tool_calls"], 0)

        environment_failure, _ = self.run_case(
            make_config(task_id="environment-failure"),
            plan=({"raise": RuntimeError("private environment failure")},),
            outputs=("ordinary action",),
        )
        self.assertEqual(
            environment_failure["failure"]["class"],
            "environment_failure",
        )
        self.assertEqual(environment_failure["usage"]["policy_turns"], 1)
        self.assertEqual(environment_failure["usage"]["tool_calls"], 0)
        self.assertFalse(environment_failure["comparable"])

        timeout_config = make_config(
            task_id="timeout",
            max_policy_turns=2,
            max_wall_seconds=1.0,
        )
        timeout, _ = self.run_case(
            timeout_config,
            plan=({"state": "late observation", "done": False},),
            outputs=("slow action",),
            clock=self.clock,
            model_advance_seconds=2.0,
        )
        self.assertEqual(timeout["failure"]["class"], "wall_timeout")
        self.assertTrue(timeout["failure"]["timed_out"])
        self.assertEqual(timeout["usage"]["policy_turns"], 1)
        self.assertEqual(timeout["usage"]["tool_calls"], 0)
        self.assertFalse(timeout["comparable"])

        finalization_timeout, _ = self.run_case(
            make_config(task_id="finalization-timeout", max_wall_seconds=1.0),
            clock=self.clock,
            artifact_advance_seconds=2.0,
        )
        self.assertEqual(
            finalization_timeout["failure"]["class"], "wall_timeout"
        )
        self.assertEqual(
            finalization_timeout["failure"]["stage"], "after_close"
        )
        self.assertEqual(
            finalization_timeout["termination"]["reason"], "timeout"
        )
        self.assertFalse(finalization_timeout["comparable"])

        artifact_failure, _ = self.run_case(
            make_config(task_id="artifact-failure"),
            artifact_error=RuntimeError("private artifact failure"),
        )
        self.assertEqual(
            artifact_failure["failure"]["class"], "artifact_failure"
        )
        self.assertIsNone(artifact_failure["final_artifact"])
        self.assertIsNone(artifact_failure["scorer"])

        scorer_failure, _ = self.run_case(
            make_config(task_id="scorer-failure"),
            scorer_error=RuntimeError("private scorer failure"),
        )
        self.assertEqual(scorer_failure["failure"]["class"], "scorer_failure")
        self.assertIsNotNone(scorer_failure["final_artifact"])
        self.assertIsNone(scorer_failure["scorer"])

        close_failure, close_bindings = self.run_case(
            make_config(task_id="close-failure"),
            invalid_close_result=True,
        )
        self.assertEqual(close_failure["failure"]["class"], "environment_failure")
        self.assertEqual(close_failure["failure"]["stage"], "close")
        self.assertIsNone(close_failure["lifecycle"]["closed_roots"])
        self.assertIsNone(close_failure["lifecycle"]["close_receipt_ref"])
        self.assertEqual(close_bindings.adapter.close_calls, 1)
        self.assertFalse(close_failure["comparable"])

    def test_deferred_scorer_is_scorable_but_never_comparable(self) -> None:
        row, _ = self.run_case(
            make_config(task_id="deferred-scorer"),
            scorer_status="deferred",
        )

        self.assertEqual(row["scorer"]["status"], "deferred")
        self.assertEqual(row["scorer"]["public_metrics"], {})
        self.assertTrue(row["scorable"])
        self.assertFalse(row["comparable"])
        forged = {**row, "comparable": True}
        with self.assertRaisesRegex(ValueError, "comparability"):
            validate_result_row(forged)

    def test_action_accounting_uses_explicit_dispatch_boundaries(self) -> None:
        config = make_config(max_policy_turns=7, max_tool_calls=4)
        zero = {
            "parsed_actions": 0,
            "domain_tool_attempts": 0,
            "successful_backend_calls": 0,
            "workspace_actions": 0,
            "compactions": 0,
            "answers": 0,
            "invalid_actions": 0,
            "parser_corrections": 0,
        }
        deltas = [
            {**zero, "invalid_actions": 1, "parser_corrections": 1},
            {**zero, "parsed_actions": 1, "domain_tool_attempts": 1},
            {
                **zero,
                "parsed_actions": 1,
                "domain_tool_attempts": 1,
                "successful_backend_calls": 1,
            },
            {**zero, "parsed_actions": 1, "answers": 1},
        ]
        plan = tuple(
            {
                "state": f"state-{index}",
                "done": index == len(deltas) - 1,
                "wrapper_evidence": {"action_accounting_delta": delta},
            }
            for index, delta in enumerate(deltas)
        )
        row, _ = self.run_case(
            config,
            plan=plan,
            outputs=("invalid", "failed search", "search", "answer"),
        )
        self.assertEqual(
            {
                name: row["usage"][name]
                for name in (
                    "policy_outputs",
                    "parsed_actions",
                    "domain_tool_attempts",
                    "successful_backend_calls",
                    "workspace_actions",
                    "compactions",
                    "answers",
                    "invalid_actions",
                    "parser_corrections",
                    "tool_calls",
                )
            },
            {
                "policy_outputs": 4,
                "parsed_actions": 3,
                "domain_tool_attempts": 2,
                "successful_backend_calls": 1,
                "workspace_actions": 0,
                "compactions": 0,
                "answers": 1,
                "invalid_actions": 1,
                "parser_corrections": 1,
                "tool_calls": 2,
            },
        )

    def test_private_policy_and_environment_content_is_digest_addressed(self) -> None:
        secret = "GATED-GAIA-QUESTION-AND-GOLD"
        config = make_config(benchmark="gaia_text")
        plan = (
            {
                "state": secret,
                "done": True,
                "reward": 1.0,
                "wrapper_evidence": {"private": secret},
            },
        )
        row, _ = self.run_case(config, plan=plan, outputs=(secret,))
        serialized = str(row)

        self.assertNotIn(secret, serialized)
        self.assertTrue(
            all(
                receipt["protected_ref"].startswith("evidence://")
                for receipt in row["receipts"]
            )
        )

    def test_sampling_source_has_no_benchmark_or_arm_dispatch(self) -> None:
        runner_path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "agentmemory"
            / "paired_eval"
            / "runner.py"
        )
        source = runner_path.read_text(encoding="utf-8")
        lowered = source.lower()
        for literal in ("gaia", "swebench", "mlebench"):
            self.assertNotIn(literal, lowered)

        tree = ast.parse(source)
        string_literals = {
            node.value.casefold()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for command in ("write", "read", "memory_write", "memory_read"):
            self.assertNotIn(command, string_literals)
        referenced_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        self.assertNotIn("parse_policy_action", referenced_names)
        run_task = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_task"
        )
        branch_types = (ast.If, ast.IfExp)
        match_type = getattr(ast, "Match", None)
        if match_type is not None:
            branch_types += (match_type,)
        for node in ast.walk(run_task):
            if isinstance(node, branch_types):
                segment = ast.get_source_segment(source, node) or ""
                self.assertNotIn("arm", segment.lower())


if __name__ == "__main__":
    unittest.main()
