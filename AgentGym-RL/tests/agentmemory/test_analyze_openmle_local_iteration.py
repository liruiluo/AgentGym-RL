from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "agentmemory"
    / "analyze_openmle_local_iteration.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_openmle_local_iteration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CHECKPOINT_PATH = ".agent_memory/CONTINUATION.md"
CHECKPOINT_BODY = "RMSE 0.421; next add features\n"
CHECKPOINT_SHA256 = __import__("hashlib").sha256(
    CHECKPOINT_BODY.encode("utf-8")
).hexdigest()


def _checkpoint_receipt() -> dict:
    return {
        "schema": "agentmemory_filesystem_checkpoint_receipt_v1",
        "path": CHECKPOINT_PATH,
        "action_kind": "apply_patch",
        "action_completed": True,
        "changed": True,
        "exists": True,
        "regular_file": True,
        "size_bytes": len(CHECKPOINT_BODY.encode("utf-8")),
        "sha256": CHECKPOINT_SHA256,
    }


def _read_receipt() -> dict:
    return {
        "schema": "agentmemory_filesystem_checkpoint_read_receipt_v1",
        "path": CHECKPOINT_PATH,
        "observed": True,
        "size_bytes": len(CHECKPOINT_BODY.encode("utf-8")),
        "sha256": CHECKPOINT_SHA256,
    }


def _row(
    order: int,
    *,
    action: str,
    action_kind: str,
    changed_paths: list[str] | None = None,
    stdout: str = "",
    terminal: bool = False,
    compaction: bool = False,
    context_epoch: int = 0,
):
    changed_paths = changed_paths or []
    execution = None
    managed_execution = action_kind == "shell_command" and "python" in action
    if action_kind in {"shell_command", "apply_patch"}:
        execution = {
            "status": "completed",
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "changed_paths": changed_paths,
            "execution_attempt_delta": int(managed_execution),
        }
    info = {
        "action_kind": action_kind,
        "action_status": "graded" if action_kind == "submit" else "completed",
        "execution": execution,
        "counter_delta": {"execution_attempt_count": int(managed_execution)},
        "terminal_reason": "graded" if terminal else None,
        "episode_success": bool(terminal),
        "grade": (
            {
                "submission_valid": True,
                "normalized_reward": 0.4,
            }
            if terminal
            else None
        ),
        "counters": {
            "execution_attempt_count": 3,
            "execution_completed_count": 3,
            "fit_count": 2,
        },
    }
    wrapper_evidence = {}
    context_transition = None
    control_request = None
    context_after = context_epoch
    if compaction:
        receipt = _checkpoint_receipt()
        execution["filesystem_checkpoint"] = dict(receipt)
        wrapper_evidence = {
            "event": "context_compaction",
            "continuation_path": CHECKPOINT_PATH,
            "continuation_persisted": True,
            "checkpoint_receipt": dict(receipt),
            "checkpoint_failure_reason": None,
            "context_replaced": True,
            "retry_pending": False,
            "preserved_policy_output": True,
            "preserved_native_observation": True,
            "checkpoint_action_in_successor_context": False,
            "checkpoint_observation_in_successor_context": False,
            "checkpoint_content_in_successor_context": False,
        }
        context_after += 1
        marker = (
            "Earlier conversation was removed after the continuation snapshot "
            "write succeeded. The workspace persists, but "
            f"`{CHECKPOINT_PATH}` was not copied into this prompt. Use the next "
            "normal action to read that file, then continue from its evidence and "
            "next action. Other workspace files remain available and may still be "
            "read or updated normally. Verified receipt: "
            f"size_bytes={receipt['size_bytes']}, sha256={receipt['sha256']}."
        )
        context_transition = {
            "schema": "agentmemory_task_neutral_context_transition_v1",
            "operation": "replace_messages",
            "messages": [
                {"role": "system", "content": "OpenMLE task contract"},
                {"role": "user", "content": "task observation\n\n" + marker},
            ],
        }
        control_request = "Persist continuation state before compaction."
    elif action_kind == "shell_command" and f"cat {CHECKPOINT_PATH}" in action:
        receipt = _read_receipt()
        execution["filesystem_checkpoint_read"] = dict(receipt)
        wrapper_evidence = {
            "memory_event": "read",
            "document_read_observed": True,
            "filesystem_checkpoint_read": dict(receipt),
        }
    return {
        "formal_step_record": {
            "trajectory_uid": "trajectory-1",
            "trajectory_row_order": order,
            "item_id": 7,
            "action": action,
            "trajectory_terminal": terminal,
            "trajectory_return": 0.4 if terminal else None,
            "immediate_reward": 0.4 if terminal else 0.0,
            "wrapper_evidence": wrapper_evidence,
            "context_transition": context_transition,
            "control_request": control_request,
            "action_submission": {"raw_policy_output": action},
            "native_step_before": order - 1,
            "native_step_after": order,
            "native_call_count_before": order - 1,
            "native_call_count_after": order,
            "policy_step_before": order - 1,
            "policy_step_after": order,
            "context_epoch_before": context_epoch,
            "context_epoch_after": context_after,
            "env_info_after": info,
        }
    }


def _complete_document():
    note = CHECKPOINT_PATH
    return {
        "rows": [
            _row(
                1,
                action='shell_command {"command":"python train.py"}',
                action_kind="shell_command",
                stdout="validation RMSE=0.421\n",
            ),
            _row(
                2,
                action="apply_patch\n*** Begin Patch\n*** Add File: "
                + "experiments.md"
                + "\n+RMSE 0.421; next add features\n*** End Patch",
                action_kind="apply_patch",
                changed_paths=["experiments.md"],
            ),
            _row(
                3,
                action="apply_patch\n*** Begin Patch\n*** Update File: "
                + note
                + "\n@@\n-RMSE 0.421\n+RMSE 0.421; next add features\n*** End Patch",
                action_kind="apply_patch",
                changed_paths=[note],
                compaction=True,
            ),
            _row(
                4,
                action='shell_command {"command":"cat '
                + note
                + '"}',
                action_kind="shell_command",
                stdout=CHECKPOINT_BODY,
                context_epoch=1,
            ),
            _row(
                5,
                action="apply_patch\n*** Begin Patch\n*** Update File: train.py\n"
                "@@\n-old\n+new\n*** End Patch",
                action_kind="apply_patch",
                changed_paths=["train.py"],
                context_epoch=1,
            ),
            _row(
                6,
                action='shell_command {"command":"python train.py"}',
                action_kind="shell_command",
                stdout="validation RMSE=0.390\n",
                context_epoch=1,
            ),
            _row(
                7,
                action="submit",
                action_kind="submit",
                terminal=True,
                context_epoch=1,
            ),
        ]
    }


def test_complete_local_iteration_memory_chain_is_detected():
    result = MODULE.analyze_documents([(1, _complete_document())])
    assert result["trajectory_count"] == 1
    assert result["counts"] == {
        "complete_iteration_memory_chain": 1,
        "has_compaction": 1,
        "has_document_write": 1,
        "has_continuation_write": 1,
        "has_legacy_continuation_write": 0,
        "has_local_validation": 1,
        "post_compaction_continuation_read": 1,
        "post_compaction_document_read": 1,
        "post_compaction_legacy_continuation_read": 0,
        "terminal_submit": 1,
    }
    case = result["complete_chain_cases"][0]
    assert case["chain"] == {
        "validation_order": 1,
        "document_write_order": 3,
        "document_paths": [".agent_memory/CONTINUATION.md"],
        "compaction_order": 3,
        "document_read_order": 4,
        "code_edit_order": 5,
        "code_paths": ["train.py"],
        "rerun_order": 6,
        "submit_order": 7,
    }
    assert case["submission_valid"] is True
    assert case["normalized_reward"] == 0.4
    assert result["aggregate"] == {
        "trajectory_count": 1,
        "valid_submission_count": 1,
        "terminal_submit_count": 1,
        "vsr": 1.0,
        "terminal_submit_rate": 1.0,
        "ps": 1.0,
        "bbr_all": 1.0,
        "bbr_valid": 1.0,
        "mean_normalized_reward_valid": 0.4,
        "mean_trajectory_return": 0.4,
        "local_validation_rate": 1.0,
        "document_write_rate": 1.0,
        "continuation_write_rate": 1.0,
        "post_compaction_document_read_rate": 1.0,
        "post_compaction_continuation_read_rate": 1.0,
        "complete_iteration_memory_chain_rate": 1.0,
        "mean_execution_attempt_count": 3.0,
        "mean_execution_completed_count": 3.0,
        "mean_fit_count": 2.0,
    }
    assert result["step_summaries"][0]["step"] == 1
    assert result["block_summaries"][0]["observed_steps"] == [1]


def test_missing_post_compaction_read_rejects_complete_chain():
    document = _complete_document()
    document["rows"] = [row for index, row in enumerate(document["rows"], 1) if index != 4]
    result = MODULE.analyze_documents([(1, document)])
    assert result["counts"].get("complete_iteration_memory_chain", 0) == 0
    assert result["counts"].get("post_compaction_document_read", 0) == 0


def test_explicit_validation_metric_prefix_accepts_short_metric_names():
    for line in (
        "validation_rmse=0.0640",
        "validation_f1=0.6284",
        "validation_ap=0.7920",
    ):
        document = _complete_document()
        first = document["rows"][0]["formal_step_record"]["env_info_after"][
            "execution"
        ]
        rerun = document["rows"][5]["formal_step_record"]["env_info_after"][
            "execution"
        ]
        first["stdout"] = line + "\n"
        rerun["stdout"] = "training_loss=0.1\n"
        result = MODULE.analyze_documents([(1, document)])
        assert result["trajectories"][0]["local_validation_rows"] == 1
        assert result["trajectories"][0]["local_validation_evidence"][0][
            "matched_excerpts"
        ] == [line]


def test_validation_not_measured_marker_is_not_local_validation():
    document = _complete_document()
    first = document["rows"][0]["formal_step_record"]["env_info_after"]["execution"]
    rerun = document["rows"][5]["formal_step_record"]["env_info_after"]["execution"]
    first["stdout"] = "validation: not measured yet\n"
    rerun["stdout"] = "validation_rmse=nan\n"
    result = MODULE.analyze_documents([(1, document)])
    assert result["trajectories"][0]["local_validation_rows"] == 0


def test_metric_word_without_measured_number_is_not_local_validation():
    document = _complete_document()
    execution = document["rows"][0]["formal_step_record"]["env_info_after"]["execution"]
    execution["stdout"] = "validation finished but no metric was printed"
    result = MODULE.analyze_documents([(1, document)])
    assert result["counts"].get("has_local_validation", 0) == 1  # later rerun still prints one
    assert result["trajectories"][0]["local_validation_rows"] == 1


def test_post_compaction_read_is_reported_without_claiming_full_chain():
    document = _complete_document()
    document["rows"] = document["rows"][:4]
    result = MODULE.analyze_documents([(1, document)])
    assert result["counts"].get("post_compaction_document_read", 0) == 1
    assert result["counts"].get("complete_iteration_memory_chain", 0) == 0


def test_training_metric_without_validation_context_is_not_counted():
    document = _complete_document()
    first = document["rows"][0]["formal_step_record"]["env_info_after"]["execution"]
    rerun = document["rows"][5]["formal_step_record"]["env_info_after"]["execution"]
    first["stdout"] = "Train RMSE: 0.421\n"
    rerun["stdout"] = "Training MAE: 0.390\n"
    result = MODULE.analyze_documents([(1, document)])
    assert result["counts"].get("has_local_validation", 0) == 0
    assert result["trajectories"][0]["local_validation_rows"] == 0


def test_dataset_text_with_metric_substring_is_not_counted():
    document = _complete_document()
    first = document["rows"][0]["formal_step_record"]["env_info_after"]["execution"]
    rerun = document["rows"][5]["formal_step_record"]["env_info_after"]["execution"]
    first["stdout"] = "Row 0: Global Map dataset, text length 56\n"
    rerun["stdout"] = "Prediction rows: 390\n"
    result = MODULE.analyze_documents([(1, document)])
    assert result["counts"].get("has_local_validation", 0) == 0


def test_plain_shell_after_code_edit_is_not_a_model_rerun():
    document = _complete_document()
    rerun = document["rows"][5]["formal_step_record"]
    rerun["action"] = 'shell_command {"command":"cat train.py"}'
    rerun["env_info_after"]["counter_delta"]["execution_attempt_count"] = 0
    rerun["env_info_after"]["execution"]["execution_attempt_delta"] = 0
    rerun["env_info_after"]["execution"]["stdout"] = "print('candidate')\n"
    result = MODULE.analyze_documents([(1, document)])
    assert result["counts"].get("complete_iteration_memory_chain", 0) == 0


def test_require_chain_failure_still_writes_diagnostic_output(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    document = _complete_document()
    document["rows"] = document["rows"][:4]
    (diagnostics / "ppo_batch_step1_post_adv.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    output = tmp_path / "analysis.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--diagnostics-dir",
            str(diagnostics),
            "--output",
            str(output),
            "--require-chain",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "no complete OpenMLE local-iteration memory chain found" in completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["counts"].get("post_compaction_document_read", 0) == 1
    assert result["counts"].get("complete_iteration_memory_chain", 0) == 0


def test_ordinary_document_chain_does_not_satisfy_canonical_continuation_gate():
    document = _complete_document()
    canonical = ".agent_memory/CONTINUATION.md"
    for row in document["rows"]:
        record = row["formal_step_record"]
        record["action"] = record["action"].replace(canonical, "experiments.md")
        execution = record["env_info_after"].get("execution")
        if execution:
            execution["changed_paths"] = [
                "experiments.md" if path == canonical else path
                for path in execution["changed_paths"]
            ]
    result = MODULE.analyze_documents([(1, document)])
    assert result["counts"].get("post_compaction_document_read", 0) == 1
    assert result["counts"].get("post_compaction_continuation_read", 0) == 0
    assert result["counts"].get("complete_iteration_memory_chain", 0) == 0

def test_legacy_openmle_continuation_path_is_reported_but_not_canonical():
    document = _complete_document()
    canonical = ".agent_memory/CONTINUATION.md"
    legacy = ".agent_memory/OPENMLE_CONTINUATION.md"
    for row in document["rows"]:
        record = row["formal_step_record"]
        record["action"] = record["action"].replace(canonical, legacy)
        execution = record["env_info_after"].get("execution")
        if execution:
            execution["changed_paths"] = [
                legacy if path == canonical else path
                for path in execution["changed_paths"]
            ]
    result = MODULE.analyze_documents([(1, document)])
    assert result["counts"].get("has_legacy_continuation_write", 0) == 1
    assert result["counts"].get("post_compaction_legacy_continuation_read", 0) == 1
    assert result["counts"].get("has_continuation_write", 0) == 0
    assert result["counts"].get("post_compaction_continuation_read", 0) == 0
    assert result["counts"].get("complete_iteration_memory_chain", 0) == 0


def test_legacy_event_label_without_canonical_receipts_cannot_satisfy_gate():
    document = _complete_document()
    compaction = document["rows"][2]["formal_step_record"]
    compaction["wrapper_evidence"] = {"event": "policy_context_compaction"}
    compaction["context_transition"] = {"operation": "replace_messages"}
    compaction["env_info_after"]["execution"].pop("filesystem_checkpoint")
    read = document["rows"][3]["formal_step_record"]
    read["wrapper_evidence"] = {}
    read["env_info_after"]["execution"].pop("filesystem_checkpoint_read")
    result = MODULE.analyze_documents([(1, document)])
    assert result["counts"].get("complete_iteration_memory_chain", 0) == 0


def test_read_digest_must_match_the_checkpoint_write():
    document = _complete_document()
    read = document["rows"][3]["formal_step_record"]
    read["wrapper_evidence"]["filesystem_checkpoint_read"]["sha256"] = "c" * 64
    read["env_info_after"]["execution"]["filesystem_checkpoint_read"][
        "sha256"
    ] = "c" * 64
    result = MODULE.analyze_documents([(1, document)])
    assert result["counts"].get("complete_iteration_memory_chain", 0) == 0
