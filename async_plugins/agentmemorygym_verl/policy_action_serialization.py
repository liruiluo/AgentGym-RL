"""Strict, side-effect-free parsing of policy-authored action envelopes.

This module is for evidence readers such as the run finalizer and offline
analyzers.  It recognizes only action serializations already accepted by the
environment endpoint; it never executes the parsed command.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_SHELL_PREFIX = "shell_command "
_SHELL_BLOCK_PREFIX = "shell_command\n"
_NATIVE_TOOL_START = "<tool_call>"
_NATIVE_TOOL_END = "</tool_call>"
_NATIVE_FUNCTION_END = "</function>"
_NATIVE_PARAMETER_END = "</parameter>"
_NATIVE_FUNCTION_RE = re.compile(r"\A<function=([a-z][a-z0-9_]*)>\Z")
_NATIVE_PARAMETER_RE = re.compile(r"\A<parameter=([a-z][a-z0-9_]*)>\Z")
_ALLOWED_SHELL_ARGUMENTS = frozenset({"command", "workdir", "timeout_ms"})


def _valid_shell_payload(payload: Mapping[str, Any]) -> str | None:
    if set(payload) - _ALLOWED_SHELL_ARGUMENTS or "command" not in payload:
        return None
    command = payload.get("command")
    if not isinstance(command, str) or not command or "\x00" in command:
        return None
    workdir = payload.get("workdir", ".")
    if not isinstance(workdir, str) or not workdir or "\x00" in workdir:
        return None
    timeout_ms = payload.get("timeout_ms")
    if timeout_ms is not None and (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, int)
        or timeout_ms <= 0
    ):
        return None
    return command


def _parse_native_shell_payload(text: str) -> Mapping[str, Any] | None:
    lines = text.splitlines()
    if (
        len(lines) < 6
        or lines[0] != _NATIVE_TOOL_START
        or lines[-1] != _NATIVE_TOOL_END
        or lines[-2] != _NATIVE_FUNCTION_END
    ):
        return None
    function_match = _NATIVE_FUNCTION_RE.fullmatch(lines[1])
    if function_match is None or function_match.group(1) != "shell_command":
        return None

    parameters: dict[str, Any] = {}
    cursor = 2
    while cursor < len(lines) - 2:
        parameter_match = _NATIVE_PARAMETER_RE.fullmatch(lines[cursor])
        if parameter_match is None:
            return None
        name = parameter_match.group(1)
        if name in parameters:
            return None
        cursor += 1
        value_start = cursor
        while cursor < len(lines) - 2 and lines[cursor] != _NATIVE_PARAMETER_END:
            if _NATIVE_PARAMETER_RE.fullmatch(lines[cursor]):
                return None
            cursor += 1
        if cursor >= len(lines) - 2:
            return None
        parameters[name] = "\n".join(lines[value_start:cursor])
        cursor += 1

    if "timeout_ms" in parameters:
        raw_timeout = str(parameters["timeout_ms"]).strip()
        if not raw_timeout.isdecimal():
            return None
        parameters["timeout_ms"] = int(raw_timeout)
    return parameters


def parse_shell_command_text(raw_output: str) -> str | None:
    """Return one shell command from a delimiter-light, JSON, or native envelope.

    Leading/trailing whitespace and one generated ``</s>`` token are normalized
    exactly as the endpoint parser does.  Visible prose, multiple actions,
    unsupported functions, duplicate parameters, and incomplete blocks remain
    rejected.
    """

    if not isinstance(raw_output, str):
        return None
    text = raw_output.strip()
    if text.endswith("</s>"):
        text = text[: -len("</s>")].rstrip()

    if text.startswith(_SHELL_BLOCK_PREFIX):
        command = text[len(_SHELL_BLOCK_PREFIX) :]
        if not command.strip() or "\x00" in command:
            return None
        return command

    if text.startswith(_SHELL_PREFIX):
        try:
            payload = json.loads(text[len(_SHELL_PREFIX) :])
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        return _valid_shell_payload(payload)

    if not text.startswith(_NATIVE_TOOL_START):
        return None
    payload = _parse_native_shell_payload(text)
    if payload is None:
        return None
    return _valid_shell_payload(payload)


__all__ = ["parse_shell_command_text"]
