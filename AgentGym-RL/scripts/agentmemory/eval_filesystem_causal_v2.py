#!/usr/bin/env python3
"""Run a real-model four-arm causal evaluation of filesystem memory.

The evaluator never preloads an answer. It first lets the policy solve the
source session for a target task and its exact counterfactual pair. Four fresh
target environments then replay the target's exact policy actions before the
server installs correct, blank, swapped, or no-workspace state out of band.
Only dependent sessions are sampled again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import eval_v3_openai as core


SCHEMA = "agentmemory_filesystem_causal_eval_v1"
ARMS = core.FILESYSTEM_CAUSAL_ARMS
_INFO_EXCLUDED_FROM_NATIVE_COMPARISON = frozenset(
    {
        "tool_ops",
        "memory_ops",
        "workspace_ops",
        "workspace_latest_event",
        "workspace_snapshot",
        "workspace_control_event",
    }
)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _session_index(info: Mapping[str, Any]) -> int:
    value = info.get("current_subtask_index", info.get("phase_index"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise core.EvalError(
            "filesystem causal evaluation requires a non-negative session index"
        )
    return value


def _native_info_projection(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _json_copy(value)
        for key, value in info.items()
        if key not in _INFO_EXCLUDED_FROM_NATIVE_COMPARISON
        and not key.startswith("workspace_")
    }


def _workspace_nonempty(state: Mapping[str, Any]) -> bool:
    return bool(state.get("file_count") or state.get("directory_count"))


def _empty_workspace_tree_sha256() -> str:
    return _sha256_json({"directories": [], "files": []})


def _causal_metadata_sha256(metadata: Mapping[str, Any]) -> str:
    payload = _json_copy(metadata)
    payload.pop("active_environment_count", None)
    service = payload.get("service")
    if isinstance(service, dict):
        service.pop("instance_run_id", None)
    return _sha256_json(payload)


def _read_private_token(path: Path) -> str:
    path = path.expanduser()
    try:
        info = path.lstat()
    except OSError as exc:
        raise core.EvalError(f"cannot stat intervention token file: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise core.EvalError("intervention token path must be a regular non-symlink file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise core.EvalError("intervention token file must not grant group/other access")
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise core.EvalError(f"cannot read intervention token file: {exc}") from exc
    if len(token) < 32 or "\n" in token or "\r" in token:
        raise core.EvalError("intervention token must be one line with at least 32 characters")
    return token


class FilesystemCausalEvalRunner:
    def __init__(
        self,
        env_factory: Callable[[], core.AgentMemoryEnvClient],
        model: core.OpenAIChatClient,
        *,
        indices: Sequence[int],
        max_policy_turns: int,
        output_dir: Path,
        intervention_token: str,
    ) -> None:
        if not indices:
            raise ValueError("causal evaluation requires at least one target index")
        if max_policy_turns < 2:
            raise ValueError("max_policy_turns must allow source and dependent actions")
        if not math.isclose(model.temperature, 0.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "four-arm causal evaluation requires temperature=0 for paired decoding"
            )
        self.env_factory = env_factory
        self.model = model
        self.indices = [int(index) for index in indices]
        self.max_policy_turns = int(max_policy_turns)
        self.output_dir = output_dir
        self.token = intervention_token
        self._reference_fingerprint: str | None = None
        self._reference_contract_sha256: str | None = None
        self._reference_metadata: dict[str, Any] | None = None

    def _new_env(self) -> core.AgentMemoryEnvClient:
        env = self.env_factory()
        core.validate_filesystem_intervention_control(
            env.metadata,
            token=self.token,
        )
        service = env.metadata.get("service")
        fingerprint = service.get("fingerprint_sha256") if isinstance(service, Mapping) else None
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            env.close()
            raise core.EvalError("intervention service lacks a valid runtime fingerprint")
        if self._reference_fingerprint is None:
            self._reference_fingerprint = fingerprint
            self._reference_contract_sha256 = _causal_metadata_sha256(env.metadata)
            self._reference_metadata = _json_copy(env.metadata)
        elif fingerprint != self._reference_fingerprint:
            env.close()
            raise core.EvalError("causal arms resolved to different environment fingerprints")
        elif _causal_metadata_sha256(env.metadata) != self._reference_contract_sha256:
            env.close()
            raise core.EvalError("causal arms resolved to different environment contracts")
        prompt_has_native_thinking = "You may first reason inside" in env.system_prompt
        if prompt_has_native_thinking is not self.model.enable_thinking:
            env.close()
            raise core.EvalError(
                "model enable_thinking does not match the attested environment prompt"
            )
        core.filesystem_no_workspace_system_prompt(env.system_prompt)
        return env

    def _source_session(
        self,
        env: core.AgentMemoryEnvClient,
        *,
        data_idx: int,
        label: str,
    ) -> dict[str, Any]:
        reset = env.reset(data_idx)
        initial_info = _json_copy(env.info.get("env_info", {}))
        if _session_index(initial_info) != 0 or reset.get("done") is not False:
            raise core.EvalError(f"{label} source did not reset at session zero")
        observation = str(reset.get("observation", ""))
        steps = []
        reached_boundary = False
        for turn in range(1, self.max_policy_turns + 1):
            step, observation, _, done, _ = core.execute_policy_turn(
                env,
                self.model,
                system_prompt=env.system_prompt,
                observation=observation,
                turn=turn,
            )
            steps.append(step)
            index = _session_index(step["env_info_after"])
            if index > 1:
                raise core.EvalError(f"{label} source skipped the frozen intervention boundary")
            if index == 1:
                if done:
                    raise core.EvalError(
                        f"{label} source terminated while advancing to session one"
                    )
                reached_boundary = True
                break
            if done:
                break
        record = {
            "label": label,
            "data_idx": data_idx,
            "paired_data_idx": data_idx ^ 1,
            "reset_response": _json_copy(reset),
            "initial_env_info": initial_info,
            "steps": steps,
            "actions": [step["action_submitted"] for step in steps],
            "actions_sha256": _sha256_json(
                [step["action_submitted"] for step in steps]
            ),
            "reached_boundary": reached_boundary,
            "terminated_before_boundary": bool(steps and steps[-1]["done"]),
            "timed_out_before_boundary": (
                not reached_boundary and not (steps and steps[-1]["done"])
            ),
        }
        if not reached_boundary:
            record["eligible"] = False
            record["ineligible_reason"] = (
                f"{label}_terminated_before_boundary"
                if record["terminated_before_boundary"]
                else f"{label}_timed_out_before_boundary"
            )
            return record
        exported = env.workspace_export(token=self.token)
        state = exported["workspace_state"]
        record.update(
            {
                "eligible": _workspace_nonempty(state),
                "ineligible_reason": (
                    None
                    if _workspace_nonempty(state)
                    else f"{label}_workspace_empty"
                ),
                "boundary_observation": observation,
                "boundary_env_info": _json_copy(env.info.get("env_info", {})),
                "workspace_export": exported,
            }
        )
        return record

    def _replay_target_source(
        self,
        env: core.AgentMemoryEnvClient,
        *,
        source: Mapping[str, Any],
    ) -> dict[str, Any]:
        reset = env.reset(int(source["data_idx"]))
        actions = list(source["actions"])
        mismatches = []
        if reset.get("observation") != source["reset_response"].get("observation"):
            mismatches.append("reset_observation")
        replays = []
        for index, action in enumerate(actions):
            before = _json_copy(env.info)
            response = env.step(action)
            replays.append(
                {
                    "index": index,
                    "action": action,
                    "env_info_before": before,
                    "response": _json_copy(response),
                }
            )
            expected = source["steps"][index]["env_response"]
            if response.get("observation") != expected.get("observation"):
                mismatches.append(f"step_{index}_observation")
            observed_projection = _native_info_projection(response.get("info", {}))
            expected_projection = _native_info_projection(expected.get("info", {}))
            if observed_projection != expected_projection:
                mismatches.append(f"step_{index}_native_info")
            if response.get("done") is not False and index != len(actions) - 1:
                mismatches.append(f"step_{index}_premature_done")
        info = env.info.get("env_info", {})
        if _session_index(info) != 1 or env.info.get("done") is not False:
            mismatches.append("final_boundary")
        exported = env.workspace_export(token=self.token) if not mismatches else None
        if exported is not None and exported["workspace_state"] != source["workspace_export"]["workspace_state"]:
            mismatches.append("workspace_state")
        return {
            "reset_response": _json_copy(reset),
            "actions": actions,
            "actions_sha256": _sha256_json(actions),
            "steps": replays,
            "boundary_observation": str(env.info.get("observation", "")),
            "boundary_env_info": _json_copy(info),
            "workspace_export": exported,
            "matches_source": not mismatches,
            "mismatches": mismatches,
        }

    def _validate_intervention_result(
        self,
        *,
        arm: str,
        response: Mapping[str, Any],
        target_state: Mapping[str, Any],
        paired_state: Mapping[str, Any],
    ) -> None:
        info = response["info"]
        event = info.get("workspace_control_event")
        if (
            not isinstance(event, Mapping)
            or event.get("arm") != arm
            or event.get("policy_action") is not False
            or event.get("task_reward") != 0.0
            or info.get("workspace_audit_event_count") != 0
            or info.get("workspace_ops") != []
        ):
            raise core.EvalError(f"{arm} intervention leaked into the policy ledger")
        snapshot = info.get("workspace_snapshot")
        if not isinstance(snapshot, Mapping):
            raise core.EvalError(f"{arm} intervention lacks a workspace snapshot")
        expected_tree = {
            "correct": target_state["tree_sha256"],
            "blank": _empty_workspace_tree_sha256(),
            "swapped": paired_state["tree_sha256"],
            "no_workspace": _empty_workspace_tree_sha256(),
        }[arm]
        if snapshot.get("tree_sha256") != expected_tree:
            raise core.EvalError(f"{arm} intervention installed the wrong workspace tree")
        expected_enabled = arm != "no_workspace"
        if (
            info.get("workspace_shell_enabled") is not expected_enabled
            or info.get("workspace_apply_patch_enabled") is not expected_enabled
        ):
            raise core.EvalError(f"{arm} intervention tool availability is inconsistent")

    def _dependent_sessions(
        self,
        env: core.AgentMemoryEnvClient,
        *,
        arm: str,
        observation: str,
        source_action_count: int,
    ) -> dict[str, Any]:
        prompt = (
            core.filesystem_no_workspace_system_prompt(env.system_prompt)
            if arm == "no_workspace"
            else env.system_prompt
        )
        steps = []
        episode_return = 0.0
        done = False
        success = False
        remaining_turns = self.max_policy_turns - source_action_count
        for turn in range(1, max(0, remaining_turns) + 1):
            step, observation, reward, done, success = core.execute_policy_turn(
                env,
                self.model,
                system_prompt=prompt,
                observation=observation,
                turn=turn,
            )
            steps.append(step)
            episode_return += reward
            if done:
                break
        final_info = _json_copy(env.info.get("env_info", {}))
        return {
            "arm": arm,
            "system_prompt": prompt,
            "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "dependent_steps": steps,
            "dependent_return": episode_return,
            "done": done,
            "episode_success": success,
            "timed_out": not done,
            "final_session_index": _session_index(final_info),
            "final_env_info": final_info,
            "buy_actions": [
                step["action_submitted"]
                for step in steps
                if isinstance(step.get("action_submitted"), str)
                and step["action_submitted"].startswith("click[Buy Now]")
            ],
        }

    def run_orbit(self, data_idx: int) -> dict[str, Any]:
        clients: list[core.AgentMemoryEnvClient] = []
        close_errors = []
        try:
            target_source_env = self._new_env()
            paired_source_env = self._new_env()
            clients.extend((target_source_env, paired_source_env))
            target_source = self._source_session(
                target_source_env,
                data_idx=data_idx,
                label="target",
            )
            paired_source = self._source_session(
                paired_source_env,
                data_idx=data_idx ^ 1,
                label="paired",
            )
            result = {
                "schema": SCHEMA,
                "target_data_idx": data_idx,
                "paired_data_idx": data_idx ^ 1,
                "sources": {
                    "target": target_source,
                    "paired": paired_source,
                },
                "eligible": False,
                "ineligible_reasons": [],
                "arms": {},
            }
            source_reasons = [
                source.get("ineligible_reason")
                for source in (target_source, paired_source)
                if not source.get("eligible")
            ]
            if source_reasons:
                result["ineligible_reasons"] = source_reasons
                return result
            target_source_state = target_source["workspace_export"][
                "workspace_state"
            ]
            paired_source_state = paired_source["workspace_export"][
                "workspace_state"
            ]
            if (
                target_source_state["tree_sha256"]
                == paired_source_state["tree_sha256"]
            ):
                result["ineligible_reasons"] = [
                    "target_and_paired_policy_workspaces_are_identical"
                ]
                return result

            arm_envs = {arm: self._new_env() for arm in ARMS}
            clients.extend(arm_envs.values())
            replays = {
                arm: self._replay_target_source(env, source=target_source)
                for arm, env in arm_envs.items()
            }
            replay_reasons = [
                f"{arm}_replay:" + ",".join(replay["mismatches"])
                for arm, replay in replays.items()
                if not replay["matches_source"]
            ]
            result["replays"] = replays
            if replay_reasons:
                result["ineligible_reasons"] = replay_reasons
                return result
            action_hashes = {replay["actions_sha256"] for replay in replays.values()}
            boundary_observations = {
                replay["boundary_observation"] for replay in replays.values()
            }
            if len(action_hashes) != 1 or len(boundary_observations) != 1:
                raise core.EvalError("target arms diverged before intervention")

            target_state = target_source_state
            paired_state = paired_source_state
            intervention_responses = {}
            for arm, env in arm_envs.items():
                response = env.workspace_intervention(
                    arm,
                    token=self.token,
                    source_env_id=(
                        paired_source_env.env_id if arm == "swapped" else None
                    ),
                )
                self._validate_intervention_result(
                    arm=arm,
                    response=response,
                    target_state=target_state,
                    paired_state=paired_state,
                )
                intervention_responses[arm] = response
            enabled_observations = {
                intervention_responses[arm]["observation"]
                for arm in ("correct", "blank", "swapped")
            }
            if len(enabled_observations) != 1:
                raise core.EvalError(
                    "enabled causal arms expose different target observations"
                )
            no_workspace_observation = intervention_responses["no_workspace"][
                "observation"
            ]
            if "unavailable in this intervention" not in no_workspace_observation:
                raise core.EvalError(
                    "no_workspace arm does not visibly attest unavailable tools"
                )

            arms = {}
            for arm, env in arm_envs.items():
                outcome = self._dependent_sessions(
                    env,
                    arm=arm,
                    observation=str(intervention_responses[arm]["observation"]),
                    source_action_count=len(target_source["actions"]),
                )
                arms[arm] = {
                    "intervention_response": _json_copy(
                        intervention_responses[arm]
                    ),
                    **outcome,
                }
            result.update(
                {
                    "eligible": True,
                    "ineligible_reasons": [],
                    "same_target_source_actions": True,
                    "same_enabled_boundary_observation": True,
                    "hidden_answer_injection": False,
                    "arms": arms,
                }
            )
            return result
        finally:
            active_exception = sys.exc_info()[0] is not None
            for env in reversed(clients):
                try:
                    env.close()
                except Exception as exc:  # pragma: no cover - network cleanup failure
                    close_errors.append(f"{type(exc).__name__}: {exc}")
            if close_errors:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                (self.output_dir / f"close_errors_{data_idx:06d}.txt").write_text(
                    "\n".join(close_errors) + "\n",
                    encoding="utf-8",
                )
                if not active_exception:
                    raise core.EvalError("causal environment cleanup failed")

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        orbits = []
        for data_idx in self.indices:
            orbit = self.run_orbit(data_idx)
            core.write_json(
                self.output_dir / f"orbit_{data_idx:06d}.json",
                orbit,
            )
            orbits.append(orbit)
        summary = summarize_causal_orbits(orbits)
        manifest = {
            "schema": SCHEMA,
            "started_unix": started,
            "finished_unix": time.time(),
            "environment": {
                "metadata": self._reference_metadata,
                "service_fingerprint_sha256": self._reference_fingerprint,
                "contract_sha256": self._reference_contract_sha256,
            },
            "model": {
                "url": self.model.model_url,
                "model": self.model.model,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "enable_thinking": self.model.enable_thinking,
            },
            "indices": self.indices,
            "max_policy_turns": self.max_policy_turns,
            "prompt_history_policy": "latest_observation_only",
            "source_actions_replayed_exactly": True,
            "hidden_answer_injection": False,
            "intervention_token_persisted": False,
            "orbits": orbits,
            "summary": summary,
        }
        core.write_json(self.output_dir / "manifest.json", manifest)
        return manifest


def summarize_causal_orbits(orbits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [orbit for orbit in orbits if orbit.get("eligible") is True]
    arms = {}
    for arm in ARMS:
        outcomes = [orbit["arms"][arm] for orbit in eligible]
        success_count = sum(outcome.get("episode_success") is True for outcome in outcomes)
        arms[arm] = {
            "episode_count": len(outcomes),
            "success_count": success_count,
            "success_rate": success_count / len(outcomes) if outcomes else 0.0,
            "mean_final_session_index": (
                sum(float(outcome["final_session_index"]) for outcome in outcomes)
                / len(outcomes)
                if outcomes
                else 0.0
            ),
            "mean_dependent_return": (
                sum(float(outcome["dependent_return"]) for outcome in outcomes)
                / len(outcomes)
                if outcomes
                else 0.0
            ),
        }
    strict = sum(
        orbit["arms"]["correct"].get("episode_success") is True
        and all(
            orbit["arms"][arm].get("episode_success") is False
            for arm in ("blank", "swapped", "no_workspace")
        )
        for orbit in eligible
    )
    return {
        "orbit_count": len(orbits),
        "eligible_orbit_count": len(eligible),
        "ineligible_orbit_count": len(orbits) - len(eligible),
        "strict_four_arm_separation_count": strict,
        "strict_four_arm_separation_rate": strict / len(eligible) if eligible else 0.0,
        "arm_metrics": arms,
        "operation_counts_prove_memory_capability": False,
        "causal_claim_requires_panel_level_intervention_effect": True,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--indices", default="0")
    parser.add_argument("--max-policy-turns", type=int, default=56)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intervention-token-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        token = _read_private_token(args.intervention_token_file)
        indices = core.parse_indices(args.indices)
        env_transport = core.JsonHttp(timeout=args.timeout)
        model_transport = core.JsonHttp(timeout=args.timeout, api_key=args.api_key)

        def env_factory() -> core.AgentMemoryEnvClient:
            return core.AgentMemoryEnvClient(args.env_url, env_transport)

        model = core.OpenAIChatClient(
            args.model_url,
            args.model,
            model_transport,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
        )
        manifest = FilesystemCausalEvalRunner(
            env_factory,
            model,
            indices=indices,
            max_policy_turns=args.max_policy_turns,
            output_dir=args.output_dir,
            intervention_token=token,
        ).run()
        print(json.dumps(manifest["summary"], ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            f"filesystem causal eval failed closed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
