#!/usr/bin/env python3
"""Generate executed Codex-workspace SFT actions on native WebShop.

The native catalog, price table, and Lucene searcher are initialized once for
the complete generation job.  Every supervised action is then executed by the
real filesystem-v2 environment and sealed together with its before/after
observations and authoritative environment evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "agentmemory_agent_action_sft_v1"
MANIFEST_SCHEMA = "agentmemory_filesystem_sft_dataset_manifest_v1"
SURFACE = "agentmemory_webshop_procedural_natural_chain_filesystem_v2"
TASK_FAMILY = "procedural_natural_attribute_chain_shopping"
MEMORY_PATH = ".agent_memory/MEMORY.md"
RECORDS_PER_TASK = 28
CHAT_TEMPLATE = {
    "add_generation_prompt": True,
    "enable_thinking": False,
    "assistant_terminator": "<|im_end|>",
}


class GenerationError(RuntimeError):
    """Raised when an executed demonstration cannot be proved exactly."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memoryarena-root", required=True, type=Path)
    parser.add_argument("--memoryarena-base-commit", required=True)
    parser.add_argument("--items-file", required=True, type=Path)
    parser.add_argument("--attributes-file", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--java-home", required=True, type=Path)
    parser.add_argument("--lucene-index-manifest", required=True, type=Path)
    parser.add_argument("--product-pool", required=True, type=Path)
    parser.add_argument("--product-pool-file-sha256", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--generator-seed", type=int, default=233)
    parser.add_argument("--start-orbit", type=int, default=0)
    parser.add_argument("--orbit-count", type=int, required=True)
    parser.add_argument("--price-seed", type=int, default=233)
    parser.add_argument("--workspace-root-parent", type=Path)
    parser.add_argument("--workspace-rg-binary", required=True, type=Path)
    parser.add_argument("--workspace-rg-sha256", required=True)
    parser.add_argument("--expected-outer-source-commit", required=True)
    parser.add_argument("--expected-agentgym-source-commit", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_cli_args(args)
    outer_root = _repository_root(Path(__file__).resolve())
    source = attest_clean_source(
        outer_root,
        expected_outer_commit=args.expected_outer_source_commit,
        expected_agentgym_commit=args.expected_agentgym_source_commit,
    )
    runtime = _load_runtime()
    record_helpers = _load_record_helpers(outer_root)
    system_prompt = _load_system_prompt()

    pool = runtime["load_certified_product_pool"](
        args.product_pool,
        expected_file_sha256=args.product_pool_file_sha256,
    )
    generator = runtime["NaturalAttributeChainGenerator"](
        pool=pool,
        seed=args.generator_seed,
    )
    provider = runtime["VerifiedProceduralBundleProvider"](
        generator=generator,
        split=args.split,
        task_count=args.orbit_count * 2,
        start_orbit=args.start_orbit,
    )
    backend = runtime["MemoryArenaNativeWebShopBackend"](
        memoryarena_root=args.memoryarena_root,
        items_file=args.items_file,
        attributes_file=args.attributes_file,
        search_root=args.search_root,
        java_home=args.java_home,
        expected_memoryarena_commit=args.memoryarena_base_commit,
        price_seed=args.price_seed,
    )
    workspace_limits = runtime["WorkspaceLimits"]()
    shell_sandbox = runtime["LinuxNamespaceShellSandbox"].from_environment(
        limits=workspace_limits.shell_limits(),
        rg_binary=args.workspace_rg_binary,
        expected_rg_sha256=args.workspace_rg_sha256,
    )
    env = runtime["ProceduralFilesystemWebShopEnv"](
        provider=provider,
        backend=backend,
        env_uid=(
            f"filesystem_sft_v1_{args.split}_{args.start_orbit}_"
            f"{source['outer_source_commit'][:10]}"
        ),
        shell_sandbox=shell_sandbox,
        workspace_root_parent=args.workspace_root_parent,
        workspace_limits=workspace_limits,
    )

    output = args.output_json.expanduser().resolve()
    manifest_path = args.manifest_json.expanduser().resolve()
    if output == manifest_path:
        raise GenerationError("dataset and manifest paths must differ")
    for path in (output, manifest_path):
        if path.exists():
            raise GenerationError(f"refusing to overwrite existing output: {path}")

    writer = JsonArrayWriter(output)
    action_counts: Counter[str] = Counter()
    phase_counts: Counter[int] = Counter()
    task_summaries: list[dict[str, Any]] = []
    runtime_metadata: dict[str, Any] | None = None
    try:
        runtime["attest_procedural_runtime_inputs"](
            pool,
            backend,
            items_file=args.items_file.resolve(),
            attributes_file=args.attributes_file.resolve(),
            search_root=args.search_root.resolve(),
            lucene_manifest=args.lucene_index_manifest.resolve(),
        )
        for local_orbit in range(args.orbit_count):
            absolute_orbit = args.start_orbit + local_orbit
            orbit = generator.generate_orbit(absolute_orbit, split=args.split)
            for branch_index, task in enumerate(orbit.tasks):
                data_index = local_orbit * 2 + branch_index
                records = generate_task_records(
                    env,
                    backend=backend,
                    provider=provider,
                    task=task,
                    data_index=data_index,
                    branch_index=branch_index,
                    system_prompt=system_prompt,
                    source=source,
                    validate_record=record_helpers["validate"],
                    finalize_record=record_helpers["finalize"],
                    canonical_sha256=record_helpers["canonical_sha256"],
                    text_sha256=record_helpers["text_sha256"],
                )
                if len(records) != RECORDS_PER_TASK:
                    raise GenerationError(
                        f"task {task.task_id} produced {len(records)} records; "
                        f"expected {RECORDS_PER_TASK}"
                    )
                for record in records:
                    writer.write(record)
                    action_counts[record["action_kind"]] += 1
                    phase_counts[record["task"]["phase_index"]] += 1
                task_summaries.append(
                    {
                        "data_index": data_index,
                        "task_id": task.task_id,
                        "orbit_id": task.orbit_id,
                        "orbit_index": task.orbit_index,
                        "branch_index": branch_index,
                        "scenario_id": task.scenario_id,
                        "task_semantic_sha256": task.semantic_sha256,
                        "provider_proof_sha256": provider.proof_for_index(
                            data_index
                        ).proof_sha256,
                        "record_count": len(records),
                        "first_record_sha256": records[0]["record_sha256"],
                        "last_record_sha256": records[-1]["record_sha256"],
                    }
                )
                if backend.active_session_count() != 0:
                    raise GenerationError(
                        f"native session leaked after task {task.task_id}"
                    )
        runtime_metadata = snapshot_runtime_metadata(
            provider=provider,
            backend=backend,
            shell_sandbox=shell_sandbox,
        )
        # Keep the sealed JSON in its temporary path until its manifest has
        # been staged.  Publishing the pair happens after all post-generation
        # count and metadata checks, so a manifest failure cannot leave a
        # dataset that looks complete but has no authentication record.
        writer.seal()
    except BaseException:
        writer.abort()
        raise
    finally:
        env.close()
        backend.close()

    try:
        expected_records = args.orbit_count * 2 * RECORDS_PER_TASK
        if writer.record_count != expected_records:
            raise GenerationError(
                f"dataset has {writer.record_count} records; expected {expected_records}"
            )
        if runtime_metadata is None:
            raise GenerationError("runtime metadata was not captured before shutdown")
        expected_action_counts = {
            "native_search": args.orbit_count * 2 * 6,
            "native_click": args.orbit_count * 2 * 12,
            "workspace_shell_command": args.orbit_count * 2 * 5,
            "workspace_apply_patch": args.orbit_count * 2 * 5,
        }
        if dict(action_counts) != expected_action_counts:
            raise GenerationError(
                f"dataset action counts diverged: {dict(action_counts)} != "
                f"{expected_action_counts}"
            )

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_schema": SCHEMA,
            "surface": SURFACE,
            "task_family": TASK_FAMILY,
            "split": args.split,
            "generator_seed": args.generator_seed,
            "start_orbit": args.start_orbit,
            "orbit_count": args.orbit_count,
            "task_count": args.orbit_count * 2,
            "records_per_task": RECORDS_PER_TASK,
            "record_count": writer.record_count,
            "action_kind_counts": dict(sorted(action_counts.items())),
            "phase_record_counts": {
                str(key): value for key, value in sorted(phase_counts.items())
            },
            "counterfactual_pair_complete": True,
            "real_environment_execution_required": True,
            "catalog_and_lucene_loaded_once_per_process": True,
            "system_prompt_sha256": record_helpers["text_sha256"](system_prompt),
            "product_pool_file": str(args.product_pool.resolve()),
            "product_pool_file_sha256": args.product_pool_file_sha256,
            "product_pool_semantic_sha256": pool.semantic_sha256,
            "provider_metadata": runtime_metadata["provider"],
            "native_backend_metadata": runtime_metadata["native_backend"],
            "workspace_sandbox": runtime_metadata["workspace_sandbox"],
            "source": source,
            "dataset_file": str(output),
            "dataset_file_sha256": file_sha256(writer.staged_path),
            "record_sha256_sequence_sha256": record_helpers["canonical_sha256"](
                writer.record_hashes
            ),
            "tasks": task_summaries,
        }
        publish_dataset_and_manifest(writer, manifest_path, manifest)
        print(
            "AGENTMEMORY_FILESYSTEM_SFT_V1_OK "
            f"orbits={args.orbit_count} tasks={args.orbit_count * 2} "
            f"records={writer.record_count} dataset_sha256={manifest['dataset_file_sha256']}"
        )
    except BaseException:
        # ``publish_dataset_and_manifest`` also rolls back a partially
        # published pair.  This covers validation and manifest-construction
        # failures before that helper is entered.
        writer.abort()
        raise


def generate_task_records(
    env,
    *,
    backend,
    provider,
    task,
    data_index: int,
    branch_index: int,
    system_prompt: str,
    source: Mapping[str, str],
    validate_record: Callable[[Any], Mapping[str, str]],
    finalize_record: Callable[[Mapping[str, Any]], dict[str, Any]],
    canonical_sha256: Callable[[Any], str],
    text_sha256: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Execute and seal the 28-action expert trajectory for one task."""

    bundle = provider.get(data_index)
    proof = provider.proof_for_index(data_index)
    if (
        bundle.task_id != task.task_id
        or bundle.orbit_id != task.orbit_id
        or bundle.scenario_id != task.scenario_id
        or bundle.proof_sha256 != proof.proof_sha256
        or bundle.product_pool_sha256 != task.product_pool_sha256
    ):
        raise GenerationError("provider bundle, proof, and generated task disagree")
    if task.orbit_index != data_index // 2 + provider.start_orbit:
        raise GenerationError("task orbit identity disagrees with provider window")

    observation, info = env.reset(data_idx=data_index)
    if info.get("surface") != SURFACE or info.get("task_family") != TASK_FAMILY:
        raise GenerationError("environment reset returned the wrong filesystem surface")
    snapshot = _mapping(info, "workspace_snapshot")
    if snapshot.get("file_count") != 0 or info.get("workspace_audit_event_count") != 0:
        raise GenerationError("task workspace did not start empty and unaudited")

    records: list[dict[str, Any]] = []
    turn_index = 0
    previous_note: str | None = None
    for phase in task.phases:
        target = _target_candidate(phase)
        if phase.phase_index > 0:
            if previous_note is None:
                raise GenerationError("later phase lacks the preceding note")
            action = shell_read_action()
            record, observation, info = _execute_and_record(
                env,
                action=action,
                action_kind="workspace_shell_command",
                observation=observation,
                info_before=info,
                task=task,
                phase=phase,
                target=target,
                proof_sha256=proof.proof_sha256,
                data_index=data_index,
                branch_index=branch_index,
                turn_index=turn_index,
                system_prompt=system_prompt,
                source=source,
                validate_record=validate_record,
                finalize_record=finalize_record,
                canonical_sha256=canonical_sha256,
                text_sha256=text_sha256,
                expected_note=previous_note,
            )
            records.append(record)
            turn_index += 1

        for action, action_kind in (
            (f"search[{_native_argument(target.search_query)}]", "native_search"),
            (f"click[{target.asin}]", "native_click"),
        ):
            record, observation, info = _execute_and_record(
                env,
                action=action,
                action_kind=action_kind,
                observation=observation,
                info_before=info,
                task=task,
                phase=phase,
                target=target,
                proof_sha256=proof.proof_sha256,
                data_index=data_index,
                branch_index=branch_index,
                turn_index=turn_index,
                system_prompt=system_prompt,
                source=source,
                validate_record=validate_record,
                finalize_record=finalize_record,
                canonical_sha256=canonical_sha256,
                text_sha256=text_sha256,
            )
            records.append(record)
            turn_index += 1

        if phase.phase_index < 5:
            current_note = note_content(phase, target)
            patch_action = (
                add_note_action(current_note)
                if previous_note is None
                else update_note_action(previous_note, current_note)
            )
            record, observation, info = _execute_and_record(
                env,
                action=patch_action,
                action_kind="workspace_apply_patch",
                observation=observation,
                info_before=info,
                task=task,
                phase=phase,
                target=target,
                proof_sha256=proof.proof_sha256,
                data_index=data_index,
                branch_index=branch_index,
                turn_index=turn_index,
                system_prompt=system_prompt,
                source=source,
                validate_record=validate_record,
                finalize_record=finalize_record,
                canonical_sha256=canonical_sha256,
                text_sha256=text_sha256,
                expected_note=current_note,
            )
            records.append(record)
            turn_index += 1
            previous_note = current_note

        record, observation, info = _execute_and_record(
            env,
            action="click[Buy Now]",
            action_kind="native_click",
            observation=observation,
            info_before=info,
            task=task,
            phase=phase,
            target=target,
            proof_sha256=proof.proof_sha256,
            data_index=data_index,
            branch_index=branch_index,
            turn_index=turn_index,
            system_prompt=system_prompt,
            source=source,
            validate_record=validate_record,
            finalize_record=finalize_record,
            canonical_sha256=canonical_sha256,
            text_sha256=text_sha256,
        )
        records.append(record)
        turn_index += 1

    if turn_index != RECORDS_PER_TASK:
        raise GenerationError(
            f"expert trajectory has {turn_index} turns; expected {RECORDS_PER_TASK}"
        )
    if not env.done or env.status != "success" or env.current_session_index != 6:
        raise GenerationError("expert trajectory did not finish all six purchases")
    if len(env.purchase_ledger) != 6 or any(
        receipt.get("purchase_correct") is not True
        for receipt in env.purchase_ledger
    ):
        raise GenerationError("expert trajectory lacks six correct private receipts")
    if backend.active_session_count() != 0:
        raise GenerationError("final BUY did not close its native WebShop session")
    return records


def _execute_and_record(
    env,
    *,
    action: str,
    action_kind: str,
    observation: str,
    info_before: Mapping[str, Any],
    task,
    phase,
    target,
    proof_sha256: str,
    data_index: int,
    branch_index: int,
    turn_index: int,
    system_prompt: str,
    source: Mapping[str, str],
    validate_record: Callable[[Any], Mapping[str, str]],
    finalize_record: Callable[[Mapping[str, Any]], dict[str, Any]],
    canonical_sha256: Callable[[Any], str],
    text_sha256: Callable[[str], str],
    expected_note: str | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    before = _json_copy(info_before)
    observation_after, reward, terminated, truncated, info_after = env.step(action)
    after = _json_copy(info_after)
    _verify_executed_effect(
        env,
        action=action,
        action_kind=action_kind,
        phase=phase,
        target=target,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info_before=before,
        info_after=after,
        expected_note=expected_note,
    )

    workspace_action = action_kind.startswith("workspace_")
    workspace_event = after["workspace_ops"][0] if workspace_action else None
    native_event = after["tool_ops"][0] if not workspace_action else None
    purchase_receipt = None
    if native_event is not None and native_event.get("op") == "BUY":
        purchase_receipt = _json_copy(env.purchase_ledger[-1])
    before_tree = before["workspace_snapshot"]["tree_sha256"]
    after_tree = after["workspace_snapshot"]["tree_sha256"]
    receipt = {
        "submitted_action": action,
        "observation_after": observation_after,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info_after": after,
    }
    record = {
        "schema": SCHEMA,
        "system_prompt": system_prompt,
        "observation": observation,
        "assistant_action": action,
        "action_kind": action_kind,
        "chat_template": dict(CHAT_TEMPLATE),
        "execution": {
            "accepted": True,
            "action_effect_verified": True,
            "submitted_action": action,
            "observation_after": observation_after,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info_before": before,
            "info_after": after,
            "receipt_sha256": canonical_sha256(receipt),
        },
        "task": {
            "surface": SURFACE,
            "task_family": TASK_FAMILY,
            "task_id": task.task_id,
            "orbit_id": task.orbit_id,
            "scenario_id": task.scenario_id,
            "split": task.split,
            "data_index": data_index,
            "orbit_index": task.orbit_index,
            "branch_index": branch_index,
            "phase_index": phase.phase_index,
            "turn_index": turn_index,
        },
        "workspace_audit": {
            "applicable": workspace_action,
            "committed": workspace_action,
            "tree_sha256_before": before_tree,
            "tree_sha256_after": after_tree,
            "event": _json_copy(workspace_event),
        },
        "native_audit": {
            "applicable": not workspace_action,
            "target_asin": target.asin if not workspace_action else None,
            "event": _json_copy(native_event),
            "purchase_receipt": purchase_receipt,
        },
        "provenance": {
            "outer_source_commit": source["outer_source_commit"],
            "agentgym_source_commit": source["agentgym_source_commit"],
            "provider_proof_sha256": proof_sha256,
            "product_pool_sha256": task.product_pool_sha256,
            "system_prompt_sha256": text_sha256(system_prompt),
            "observation_sha256": text_sha256(observation),
            "assistant_action_sha256": text_sha256(action),
            "observation_after_sha256": text_sha256(observation_after),
            "env_info_before_sha256": canonical_sha256(before),
            "env_info_after_sha256": canonical_sha256(after),
            "task_semantic_sha256": task.semantic_sha256,
            "target_product_record_sha256": target.catalog_record_sha256,
        },
    }
    sealed = finalize_record(record)
    visible = validate_record(sealed)
    if visible != {
        "system_prompt": system_prompt,
        "observation": observation,
        "assistant_action": action,
    }:
        raise GenerationError("record validator changed the policy-visible fields")
    return sealed, observation_after, after


def _verify_executed_effect(
    env,
    *,
    action: str,
    action_kind: str,
    phase,
    target,
    reward: float,
    terminated: bool,
    truncated: bool,
    info_before: Mapping[str, Any],
    info_after: Mapping[str, Any],
    expected_note: str | None,
) -> None:
    if info_before.get("current_subtask_index") != phase.phase_index:
        raise GenerationError("action began in the wrong shopping phase")
    if info_before.get("surface") != SURFACE or info_after.get("surface") != SURFACE:
        raise GenerationError("action escaped the filesystem-v2 surface")
    workspace_action = action_kind.startswith("workspace_")
    if workspace_action:
        events = info_after.get("workspace_ops")
        if not isinstance(events, list) or len(events) != 1:
            raise GenerationError("workspace action lacks one exact audit event")
        event = events[0]
        expected_op = (
            "SHELL_COMMAND"
            if action_kind == "workspace_shell_command"
            else "APPLY_PATCH"
        )
        if event.get("op") != expected_op or event.get("status") != "executed":
            raise GenerationError("workspace action was not executed as submitted")
        if reward != 0 or terminated or truncated:
            raise GenerationError("workspace action changed reward or episode state")
        if info_after.get("current_subtask_index") != phase.phase_index:
            raise GenerationError("workspace action advanced the shopping phase")
        if expected_note is None:
            raise GenerationError("workspace demonstration lacks an expected note")
        note_path = env.workspace.host_root / MEMORY_PATH
        if not note_path.is_file() or note_path.read_text(encoding="utf-8") != expected_note:
            raise GenerationError("workspace action did not preserve the exact note bytes")
        if expected_op == "SHELL_COMMAND":
            if (
                event.get("exit_code") != 0
                or event.get("timed_out") is not False
                or event.get("stdout") != expected_note
            ):
                raise GenerationError("shell_command did not read the exact note")
        elif MEMORY_PATH not in event.get("changed_paths", []):
            raise GenerationError("apply_patch did not change the memory note")
        return

    events = info_after.get("tool_ops")
    if not isinstance(events, list) or len(events) != 1:
        raise GenerationError("native action lacks one authoritative tool event")
    event = events[0]
    if event.get("raw_action") != action:
        raise GenerationError("native event does not bind the submitted action")
    if action_kind == "native_search":
        if (
            event.get("op") != "SEARCH"
            or event.get("result_count", 0) < 1
            or reward != 0
            or terminated
            or truncated
        ):
            raise GenerationError("native search did not execute successfully")
        page = env.native_page
        if page is None or target.asin.casefold() not in {
            str(value).casefold() for value in page.clickables
        }:
            raise GenerationError("native search did not expose the target ASIN")
    elif action != "click[Buy Now]":
        if (
            event.get("op") != "CLICK"
            or action != f"click[{target.asin}]"
            or reward != 0
            or terminated
            or truncated
        ):
            raise GenerationError("native product click did not open the target")
        page = env.native_page
        if page is None or "Buy Now" not in page.clickables:
            raise GenerationError("target product page does not expose Buy Now")
    else:
        final = phase.phase_index == 5
        expected_reward = 2.0 if final else 1.0
        if (
            event.get("op") != "BUY"
            or event.get("committed") is not True
            or event.get("purchase_correct") is not True
            or event.get("session_advanced") is not True
            or event.get("terminal") is not final
            or reward != expected_reward
            or terminated is not final
            or truncated
        ):
            raise GenerationError("native BUY did not produce a correct exact receipt")
        if not env.purchase_ledger or env.purchase_ledger[-1].get(
            "actual_asin"
        ) != target.asin:
            raise GenerationError("private BUY receipt has the wrong ASIN")


def note_content(phase, target) -> str:
    value = str(target.attribute_display_name).strip()
    attribute = str(phase.attribute_name).strip()
    if not value or not attribute:
        raise GenerationError("target natural attribute is empty")
    if attribute not in phase.question or value not in phase.question:
        raise GenerationError("memory note contains a non-visible attribute")
    note = f"Latest selected product attribute: {attribute} = {value}\n"
    forbidden = [candidate.asin for candidate in phase.candidates]
    forbidden.extend(candidate.title for candidate in phase.candidates)
    if any(item and item.casefold() in note.casefold() for item in forbidden):
        raise GenerationError("memory note leaked product identity")
    return note


def shell_read_action() -> str:
    return "shell_command " + json.dumps(
        {
            "command": f"cat -- {MEMORY_PATH}",
            "workdir": ".",
            "timeout_ms": 10_000,
        },
        separators=(",", ":"),
    )


def snapshot_runtime_metadata(*, provider, backend, shell_sandbox) -> dict[str, Any]:
    """Freeze metadata while the already-loaded backend is still alive."""

    provider_metadata = _json_copy(provider.metadata())
    native_backend_metadata = _json_copy(backend.metadata())
    if native_backend_metadata.get("active_session_count") != 0:
        raise GenerationError(
            "cannot seal SFT metadata while native WebShop sessions remain active"
        )
    sandbox_metadata = _json_copy(shell_sandbox.metadata)
    if not isinstance(sandbox_metadata, Mapping):
        raise GenerationError("workspace sandbox metadata must be an object")
    return {
        "provider": provider_metadata,
        "native_backend": native_backend_metadata,
        "workspace_sandbox": sandbox_metadata,
    }


def add_note_action(content: str) -> str:
    lines = ["apply_patch", "*** Begin Patch", f"*** Add File: {MEMORY_PATH}"]
    lines.extend("+" + line for line in content.splitlines())
    lines.append("*** End Patch")
    return "\n".join(lines)


def update_note_action(before: str, after: str) -> str:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    if not before_lines or not after_lines:
        raise GenerationError("note update requires non-empty before and after text")
    lines = [
        "apply_patch",
        "*** Begin Patch",
        f"*** Update File: {MEMORY_PATH}",
        "@@",
    ]
    lines.extend("-" + line for line in before_lines)
    lines.extend("+" + line for line in after_lines)
    lines.append("*** End Patch")
    return "\n".join(lines)


class JsonArrayWriter:
    """Write a JSON array atomically without retaining all records in memory."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        )
        self._handle = handle
        self._temporary = Path(handle.name)
        self._sealed = False
        self._finished = False
        self.record_count = 0
        self.record_hashes: list[str] = []
        handle.write("[\n")

    def write(self, record: Mapping[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot append to a finished dataset")
        if self.record_count:
            self._handle.write(",\n")
        self._handle.write(
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        self.record_count += 1
        self.record_hashes.append(str(record["record_sha256"]))

    @property
    def staged_path(self) -> Path:
        if not self._sealed:
            raise RuntimeError("dataset writer has not been sealed")
        return self._temporary

    def seal(self) -> None:
        if self._finished or self._sealed:
            raise RuntimeError("dataset writer is already sealed or finished")
        self._handle.write("\n]\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._sealed = True

    def publish(self) -> None:
        if self._finished:
            raise RuntimeError("dataset writer is already finished")
        if not self._sealed:
            raise RuntimeError("dataset writer must be sealed before publishing")
        if self.output.exists():
            raise GenerationError(f"refusing to overwrite existing output: {self.output}")
        os.replace(self._temporary, self.output)
        self._finished = True

    def finish(self) -> None:
        """Seal and publish a dataset for callers that do not need a manifest."""

        self.seal()
        self.publish()

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        if not self._finished:
            self._temporary.unlink(missing_ok=True)


def publish_dataset_and_manifest(
    writer: JsonArrayWriter,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Publish a sealed dataset and manifest with rollback on any failure."""

    if not writer._sealed or writer._finished:  # pylint: disable=protected-access
        raise RuntimeError("dataset writer must be sealed and unpublished")
    if writer.output.exists() or manifest_path.exists():
        raise GenerationError("refusing to overwrite an existing dataset pair")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_temp: Path | None = None
    dataset_published = False
    manifest_published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=manifest_path.parent,
            delete=False,
        ) as handle:
            manifest_temp = Path(handle.name)
            json.dump(
                manifest,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        writer.publish()
        dataset_published = True
        os.replace(manifest_temp, manifest_path)
        manifest_published = True
        manifest_temp = None
    except BaseException:
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)
        if manifest_published:
            manifest_path.unlink(missing_ok=True)
        if dataset_published:
            writer.output.unlink(missing_ok=True)
        writer.abort()
        raise


def attest_clean_source(
    outer_root: Path,
    *,
    expected_outer_commit: str,
    expected_agentgym_commit: str,
) -> dict[str, str]:
    outer_commit = _git(outer_root, "rev-parse", "HEAD")
    agentgym_root = outer_root / "AgentGym"
    agentgym_commit = _git(agentgym_root, "rev-parse", "HEAD")
    dirty = _git(outer_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise GenerationError(
            "formal SFT generation requires a clean source worktree; dirty paths:\n"
            + dirty
        )
    if outer_commit != expected_outer_commit:
        raise GenerationError(
            f"outer source commit mismatch: {outer_commit} != {expected_outer_commit}"
        )
    if agentgym_commit != expected_agentgym_commit:
        raise GenerationError(
            "AgentGym source commit mismatch: "
            f"{agentgym_commit} != {expected_agentgym_commit}"
        )
    return {
        "outer_source_commit": outer_commit,
        "agentgym_source_commit": agentgym_commit,
    }


def _load_runtime() -> dict[str, Any]:
    procedural = importlib.import_module("agentenv_agentmemory.procedural")
    filesystem = importlib.import_module(
        "agentenv_agentmemory.filesystem_webshop_env"
    )
    backend = importlib.import_module("agentenv_agentmemory.native_webshop_backend")
    wrapper = importlib.import_module("agentenv_agentmemory.procedural_wrapper")
    workspace = importlib.import_module("agentenv_agentmemory.persistent_workspace")
    sandbox = importlib.import_module("agentenv_agentmemory.workspace_sandbox")
    return {
        name: getattr(module, name)
        for module, names in (
            (
                procedural,
                (
                    "NaturalAttributeChainGenerator",
                    "VerifiedProceduralBundleProvider",
                    "load_certified_product_pool",
                ),
            ),
            (filesystem, ("ProceduralFilesystemWebShopEnv",)),
            (backend, ("MemoryArenaNativeWebShopBackend",)),
            (wrapper, ("attest_procedural_runtime_inputs",)),
            (workspace, ("WorkspaceLimits",)),
            (sandbox, ("LinuxNamespaceShellSandbox",)),
        )
        for name in names
    }


def _load_record_helpers(outer_root: Path) -> dict[str, Callable[..., Any]]:
    path = (
        outer_root
        / "AgentGym-RL"
        / "verl"
        / "utils"
        / "agent_dataset"
        / "agent_action_schema.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_agentmemory_agent_action_schema_v1", path
    )
    if spec is None or spec.loader is None:
        raise GenerationError(f"cannot load record schema from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "validate": module.validate_agent_action_record,
        "finalize": module.finalize_agent_action_record,
        "canonical_sha256": module.canonical_json_sha256,
        "text_sha256": module.text_sha256,
    }


def _load_system_prompt() -> str:
    module = importlib.import_module("verl.workers.rollout.schemas")
    prompt = getattr(module, "AGENTMEMORY_ACTION_SYSTEM_PROMPT_NATURAL_FILESYSTEM")
    if not isinstance(prompt, str) or not prompt.strip():
        raise GenerationError("canonical natural-filesystem prompt is unavailable")
    return prompt


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.start_orbit < 0:
        raise SystemExit("--start-orbit must be non-negative")
    if args.orbit_count <= 0:
        raise SystemExit("--orbit-count must be positive")
    for name in (
        "product_pool_file_sha256",
        "workspace_rg_sha256",
    ):
        value = getattr(args, name)
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise SystemExit(f"--{name.replace('_', '-')} must be lowercase SHA-256")
    for name in ("expected_outer_source_commit", "expected_agentgym_source_commit"):
        value = getattr(args, name)
        if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise SystemExit(f"--{name.replace('_', '-')} must be a full git commit")


def _target_candidate(phase):
    matches = [item.product for item in phase.candidates if item.asin == phase.target_asin]
    if len(matches) != 1:
        raise GenerationError("phase does not have exactly one target candidate")
    return matches[0]


def _native_argument(value: str) -> str:
    text = " ".join(str(value).split())
    if not text or any(char in text for char in "[]\r\n"):
        raise GenerationError(f"unsafe native action argument: {text!r}")
    return text


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise GenerationError(f"environment info lacks object field {key!r}")
    return result


def _json_copy(value: Any) -> Any:
    if value is None:
        return None
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _repository_root(path: Path) -> Path:
    return Path(_git(path.parent, "rev-parse", "--show-toplevel")).resolve()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise GenerationError(
            f"git {' '.join(args)} failed in {root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
