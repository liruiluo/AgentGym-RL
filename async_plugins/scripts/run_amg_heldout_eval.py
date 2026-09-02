#!/usr/bin/env python3
"""Compose, verify, or run the CAMG native held-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentmemorygym_verl.heldout_eval import (
    EXPECTED_BATCH_SIZE,
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_NUM_GPUS,
    derive_eval_config,
    load_eval_plan,
    run_contract,
    run_evaluation,
)
from agentmemorygym_verl.heldout_eval_contract import (
    canonical_json_bytes,
    compose_heldout_schedule,
    sha256_bytes,
)


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--resolved-config-sha256", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--schedule-manifest-sha256", required=True)
    parser.add_argument("--route-registry", type=Path, required=True)
    parser.add_argument("--route-registry-sha256", required=True)
    parser.add_argument("--agent-loop-config", type=Path, required=True)
    parser.add_argument("--agent-loop-config-sha256", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--training-outer-commit", required=True)
    parser.add_argument("--training-inner-commit", required=True)
    parser.add_argument("--training-verl-commit", required=True)
    parser.add_argument("--evaluator-outer-commit", required=True)
    parser.add_argument("--evaluator-inner-commit", required=True)
    parser.add_argument("--evaluator-verl-commit", required=True)
    parser.add_argument(
        "--checkpoint-step", type=int, default=EXPECTED_CHECKPOINT_STEP
    )
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--num-gpus", type=int, default=EXPECTED_NUM_GPUS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)


def _plan(args: argparse.Namespace):
    return load_eval_plan(
        run_id=args.run_id,
        run_dir=args.run_dir,
        resolved_config_path=args.resolved_config,
        expected_resolved_config_sha256=args.resolved_config_sha256,
        schedule_path=args.schedule,
        expected_schedule_sha256=args.schedule_sha256,
        schedule_manifest_path=args.schedule_manifest,
        expected_schedule_manifest_sha256=args.schedule_manifest_sha256,
        route_registry_path=args.route_registry,
        expected_route_registry_sha256=args.route_registry_sha256,
        agent_loop_config_path=args.agent_loop_config,
        expected_agent_loop_config_sha256=args.agent_loop_config_sha256,
        model_manifest_path=args.model_manifest,
        expected_model_manifest_sha256=args.model_manifest_sha256,
        training_run_id=args.training_run_id,
        training_outer_commit=args.training_outer_commit,
        training_inner_commit=args.training_inner_commit,
        training_verl_commit=args.training_verl_commit,
        evaluator_outer_commit=args.evaluator_outer_commit,
        evaluator_inner_commit=args.evaluator_inner_commit,
        evaluator_verl_commit=args.evaluator_verl_commit,
        checkpoint_step=args.checkpoint_step,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    compose = commands.add_parser("compose-schedule")
    compose.add_argument("--spec", type=Path, required=True)
    compose.add_argument("--spec-sha256", required=True)
    compose.add_argument("--output", type=Path, required=True)
    compose.add_argument("--manifest", type=Path, required=True)

    verify = commands.add_parser("verify-plan")
    _add_plan_arguments(verify)

    run = commands.add_parser("run")
    _add_plan_arguments(run)

    orchestrate = commands.add_parser("orchestrate")
    _add_plan_arguments(orchestrate)
    orchestrate.add_argument("--orchestration-dir", type=Path, required=True)
    orchestrate.add_argument(
        "--outer-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    orchestrate.add_argument("--inner-root", type=Path, required=True)
    orchestrate.add_argument("--verl-root", type=Path, required=True)
    orchestrate.add_argument("--evaluator-script-sha256", required=True)
    orchestrate.add_argument("--endpoint-registry", type=Path, required=True)
    orchestrate.add_argument("--endpoint-registry-sha256", required=True)
    orchestrate.add_argument("--holder-lease", type=Path, required=True)
    orchestrate.add_argument("--holder-lease-sha256", required=True)
    orchestrate.add_argument(
        "--holder-lock",
        type=Path,
        default=Path("/tmp/amg-heldout-eval-holder-transaction.lock"),
    )
    orchestrate.add_argument("--resolve-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compose-schedule":
        result = compose_heldout_schedule(
            args.spec,
            expected_spec_sha256=args.spec_sha256,
            output_path=args.output,
            manifest_path=args.manifest,
        )
    elif args.command in {"verify-plan", "run"}:
        plan = _plan(args)
        if args.command == "verify-plan":
            config = derive_eval_config(plan)
            config_sha256 = sha256_bytes(canonical_json_bytes(config))
            result = run_contract(plan, config_sha256)
        else:
            result = run_evaluation(plan)
    else:
        from agentmemorygym_verl.heldout_eval_orchestrator import (
            HeldoutEvalLocalBackend,
            execute_eval_orchestrator,
            load_eval_orchestrator_plan,
        )
        from agentmemorygym_verl.multitask_orchestrator import _termination_guard

        evaluation = _plan(args)
        plan = load_eval_orchestrator_plan(
            evaluation=evaluation,
            orchestration_dir=args.orchestration_dir,
            outer_root=args.outer_root,
            inner_root=args.inner_root,
            verl_root=args.verl_root,
            evaluator_script=Path(__file__).resolve(),
            expected_evaluator_script_sha256=args.evaluator_script_sha256,
            endpoint_registry_path=args.endpoint_registry,
            expected_endpoint_registry_sha256=args.endpoint_registry_sha256,
            holder_lease_path=args.holder_lease,
            expected_holder_lease_sha256=args.holder_lease_sha256,
            holder_lock_path=args.holder_lock,
            resolve_only=args.resolve_only,
        )
        with _termination_guard():
            exit_code = execute_eval_orchestrator(
                plan, backend=HeldoutEvalLocalBackend()
            )
        result = {
            "schema": "camg_heldout_eval_orchestrator_cli_result_v1",
            "status": "pass" if exit_code == 0 else "fail",
            "exit_code": exit_code,
            "run_id": evaluation.run_id,
            "orchestration_dir": str(plan.orchestration_dir),
            "evaluation_run_dir": str(evaluation.run_dir),
            "resolve_only": plan.resolve_only,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
