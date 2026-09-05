from __future__ import annotations

import hashlib
import json

from agentmemorygym_verl.finalizer import _checkpoint_successor_is_safe


def _digest(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(
            messages,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _marker(receipt: dict[str, object]) -> str:
    return (
        "Earlier conversation was removed after the continuation snapshot write "
        "succeeded. The workspace persists, but `.agent_memory/CONTINUATION.md` "
        "was not copied into this prompt. Use the next normal action to read that "
        "file, then continue from its evidence and next action. Other workspace "
        "files remain available and may still be read or updated normally. "
        f"Verified receipt: size_bytes={receipt['size_bytes']}, "
        f"sha256={receipt['sha256']}."
    )


def test_finalizer_accepts_named_native_tool_result_in_checkpoint_framing() -> None:
    framing = [
        {"role": "system", "content": "task framing"},
        {"role": "user", "content": "task observation"},
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"shell_command",' 
            '"arguments":{"command":"pwd"}}</tool_call>',
        },
        {"role": "tool", "name": "shell_command", "content": "/workspace"},
    ]
    receipt = {"size_bytes": 12, "sha256": "a" * 64}
    messages = framing + [{"role": "user", "content": _marker(receipt)}]
    evidence = {"checkpoint_framing_sha256": _digest(framing)}
    assert _checkpoint_successor_is_safe({}, messages, receipt, evidence)


def test_finalizer_rejects_unnamed_native_tool_result() -> None:
    framing = [
        {"role": "system", "content": "task framing"},
        {"role": "tool", "content": "/workspace"},
    ]
    receipt = {"size_bytes": 12, "sha256": "a" * 64}
    messages = framing + [{"role": "user", "content": _marker(receipt)}]
    evidence = {"checkpoint_framing_sha256": _digest(framing)}
    assert not _checkpoint_successor_is_safe({}, messages, receipt, evidence)
