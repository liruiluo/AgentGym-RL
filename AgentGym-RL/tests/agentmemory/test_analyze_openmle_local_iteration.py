from __future__ import annotations

import importlib.util
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


def _row(
    order: int,
    *,
    action: str,
    action_kind: str,
    changed_paths: list[str] | None = None,
    stdout: str = "",
    terminal: bool = False,
    compaction: bool = False,
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
    return {
        "formal_step_record": {
            "trajectory_uid": "trajectory-1",
            "trajectory_row_order": order,
            "item_id": 7,
            "action": action,
            "trajectory_terminal": terminal,
            "trajectory_return": 0.4 if terminal else None,
            "immediate_reward": 0.4 if terminal else 0.0,
            "wrapper_evidence": (
                {"event": "context_compaction"} if compaction else {}
            ),
            "context_transition": (
                {"operation": "replace_messages"} if compaction else None
            ),
            "env_info_after": info,
        }
    }


def _complete_document():
    note = ".agent_memory/OPENMLE_CONTINUATION.md"
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
                + note
                + "\n+RMSE 0.421; next add features\n*** End Patch",
                action_kind="apply_patch",
                changed_paths=[note],
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
                stdout="RMSE 0.421; next add features\n",
            ),
            _row(
                5,
                action="apply_patch\n*** Begin Patch\n*** Update File: train.py\n"
                "@@\n-old\n+new\n*** End Patch",
                action_kind="apply_patch",
                changed_paths=["train.py"],
            ),
            _row(
                6,
                action='shell_command {"command":"python train.py"}',
                action_kind="shell_command",
                stdout="validation RMSE=0.390\n",
            ),
            _row(7, action="submit", action_kind="submit", terminal=True),
        ]
    }


def test_complete_local_iteration_memory_chain_is_detected():
    result = MODULE.analyze_documents([(1, _complete_document())])
    assert result["trajectory_count"] == 1
    assert result["counts"] == {
        "complete_iteration_memory_chain": 1,
        "has_compaction": 1,
        "has_document_write": 1,
        "has_local_validation": 1,
        "post_compaction_document_read": 1,
        "terminal_submit": 1,
    }
    case = result["complete_chain_cases"][0]
    assert case["chain"] == {
        "validation_order": 1,
        "document_write_order": 2,
        "document_paths": [".agent_memory/OPENMLE_CONTINUATION.md"],
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
        "post_compaction_document_read_rate": 1.0,
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
