from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer

from verl.utils.agentgym.continuous_agent_v1 import (
    POLICY_COMPACTION_REQUEST,
    POLICY_CONTINUATION_MARKER,
    continuous_prompt_capacity,
    should_request_policy_compaction,
)
from verl.workers.rollout.agent_vllm_rollout.vllm_rollout import vLLMRollout
from verl.workers.rollout.schemas import Message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-tokens", type=int, default=32768)
    parser.add_argument("--max-prompt-tokens", type=int, default=30720)
    parser.add_argument("--max-response-tokens", type=int, default=2048)
    parser.add_argument("--max-observation-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    rollout = object.__new__(vLLMRollout)
    rollout.tokenizer = tokenizer

    full_observation = "\n".join(
        f"line-{index:05d}: deterministic observation payload for tokenizer audit"
        for index in range(4096)
    )
    bounded = rollout._bound_continuous_observation(
        full_observation,
        max_observation_tokens=args.max_observation_tokens,
    )
    assert bounded["truncated"] is True
    assert bounded["truncation_marker"] in bounded["policy_visible_text"]
    assert len(bounded["policy_visible_token_ids"]) <= args.max_observation_tokens
    assert rollout._continuous_plain_token_ids(bounded["policy_visible_text"]) == bounded[
        "policy_visible_token_ids"
    ]
    expected_visible_text = (
        rollout._continuous_decode_plain_token_ids(
            bounded["full_token_ids"][: bounded["head_token_count"]]
        )
        + bounded["truncation_marker"]
        + rollout._continuous_decode_plain_token_ids(
            bounded["full_token_ids"][-bounded["tail_token_count"] :]
        )
    )
    assert bounded["policy_visible_text"] == expected_visible_text

    immutable_messages = [
        Message(role="system", content="You are a coding agent."),
        Message(role="user", content="Fix the repository issue and verify the patch."),
    ]
    messages = list(immutable_messages)
    capacity = continuous_prompt_capacity(
        max_prompt_tokens=args.max_prompt_tokens,
        max_model_tokens=args.max_model_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    trigger = None
    for transition_index in range(256):
        action_prompt_ids = rollout._continuous_prompt_from_messages(messages)
        compaction_prompt_ids = rollout._continuous_prompt_from_messages(
            list(messages) + [Message(role="user", content=POLICY_COMPACTION_REQUEST)]
        )
        envelope_tokens = rollout._continuous_action_observation_envelope_tokens(
            messages,
            action_prompt_ids,
        )
        request_compaction = should_request_policy_compaction(
            action_prompt_token_count=len(action_prompt_ids),
            compaction_prompt_token_count=len(compaction_prompt_ids),
            max_prompt_tokens=args.max_prompt_tokens,
            max_model_tokens=args.max_model_tokens,
            max_response_tokens=args.max_response_tokens,
            max_observation_tokens=args.max_observation_tokens,
            action_observation_envelope_tokens=envelope_tokens,
        )
        if request_compaction:
            trigger = {
                "transition_index": transition_index,
                "action_prompt_tokens": len(action_prompt_ids),
                "compaction_prompt_tokens": len(compaction_prompt_ids),
                "envelope_tokens": envelope_tokens,
            }
            break
        messages.extend(
            [
                Message(
                    role="assistant",
                    content='shell_command({"command":"true","workdir":"."})',
                ),
                Message(role="user", content=bounded["policy_visible_text"]),
            ]
        )
    assert trigger is not None
    assert trigger["compaction_prompt_tokens"] <= capacity

    summary = (
        "Repository state is unchanged. Continue by inspecting the failing test "
        "and preserve any durable notes in the workspace."
    )
    summary_ids = rollout._continuous_plain_token_ids(summary)
    assert len(summary_ids) < args.max_response_tokens
    post_compaction_messages = list(immutable_messages) + [
        Message(role="assistant", content=summary),
        Message(role="user", content=POLICY_CONTINUATION_MARKER),
    ]
    post_compaction_ids = rollout._continuous_prompt_from_messages(
        post_compaction_messages
    )
    assert len(post_compaction_ids) <= capacity

    print(
        json.dumps(
            {
                "status": "QWEN35_CONTINUOUS_TOKENIZER_PROBE_OK",
                "model": args.model,
                "capacity": {
                    "max_model_tokens": args.max_model_tokens,
                    "max_prompt_tokens": args.max_prompt_tokens,
                    "max_response_tokens": args.max_response_tokens,
                    "effective_prompt_capacity": capacity,
                    "max_observation_tokens": args.max_observation_tokens,
                },
                "observation": {
                    "full_tokens": len(bounded["full_token_ids"]),
                    "visible_tokens": len(bounded["policy_visible_token_ids"]),
                    "head_tokens": bounded["head_token_count"],
                    "tail_tokens": bounded["tail_token_count"],
                },
                "trigger": trigger,
                "summary_tokens": len(summary_ids),
                "post_compaction_prompt_tokens": len(post_compaction_ids),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
