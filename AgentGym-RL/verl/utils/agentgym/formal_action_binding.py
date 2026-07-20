from __future__ import annotations

import json
import re
from typing import Any


FORMAL_ACTION_NAMES = (
    "ADD",
    "UPDATE",
    "DELETE",
    "RETRIEVE",
    "SUMMARY",
    "FILTER",
    "SEARCH",
    "PAGE",
    "BUY",
    "ANSWER",
)
FORMAL_ACTION_OPS = frozenset(FORMAL_ACTION_NAMES)
ACTION_ALLOWED_FIELDS = {
    "ADD": frozenset({"key", "value"}),
    "UPDATE": frozenset({"memory_id", "key", "value"}),
    "DELETE": frozenset({"memory_id", "key"}),
    "RETRIEVE": frozenset({"query", "top_k"}),
    "SUMMARY": frozenset({"text", "source_ids"}),
    "FILTER": frozenset({"keep_ids", "drop_ids", "scope"}),
    "SEARCH": frozenset({"query"}),
    "PAGE": frozenset({"cursor"}),
    "BUY": frozenset({"product_id"}),
    "ANSWER": frozenset({"text"}),
}
ACTION_REQUIRED_FIELDS = {
    "ADD": frozenset({"key", "value"}),
    "UPDATE": frozenset({"value"}),
    "DELETE": frozenset(),
    "RETRIEVE": frozenset({"query", "top_k"}),
    "SUMMARY": frozenset({"text", "source_ids"}),
    "FILTER": frozenset({"scope"}),
    "SEARCH": frozenset({"query"}),
    "PAGE": frozenset({"cursor"}),
    "BUY": frozenset({"product_id"}),
    "ANSWER": frozenset({"text"}),
}
_ACTION_MARKER_RE = re.compile(r"^\s*action\s*:\s*", re.IGNORECASE | re.MULTILINE)
_BARE_ACTION_RE = re.compile(
    r"(?P<action>\b(?:"
    + "|".join(re.escape(name) for name in FORMAL_ACTION_NAMES)
    + r")\s*\{)",
    re.IGNORECASE,
)


def normalize_action_name(action_name: str) -> str:
    return str(action_name).strip().upper()


def validate_nonempty_string_list(action_name: str, field: str, value: Any) -> None:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(f"{action_name}.{field} must be a non-empty list of strings")


def validate_action_payload(action_name: str, arguments: Any) -> None:
    if action_name not in FORMAL_ACTION_OPS:
        raise ValueError(f"Invalid action name: {action_name}")
    if not isinstance(arguments, dict):
        raise ValueError(f"{action_name} payload must be a JSON object")
    extra = sorted(set(arguments) - ACTION_ALLOWED_FIELDS[action_name])
    if extra:
        raise ValueError(
            f"{action_name} payload contains unsupported fields: {', '.join(extra)}"
        )
    missing = sorted(ACTION_REQUIRED_FIELDS[action_name] - set(arguments))
    if missing:
        raise ValueError(
            f"{action_name} payload is missing required fields: {', '.join(missing)}"
        )

    string_fields = {
        "ADD": ("key", "value"),
        "UPDATE": ("value",),
        "RETRIEVE": ("query",),
        "SUMMARY": ("text",),
        "SEARCH": ("query",),
        "PAGE": ("cursor",),
        "BUY": ("product_id",),
        "ANSWER": ("text",),
    }.get(action_name, ())
    for field in string_fields:
        value = arguments[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{action_name}.{field} must be a non-empty string")

    if action_name in {"UPDATE", "DELETE"}:
        identifiers = [
            field for field in ("memory_id", "key") if field in arguments
        ]
        if not identifiers:
            raise ValueError(f"{action_name} requires memory_id or key")
        for field in identifiers:
            value = arguments[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{action_name}.{field} must be a non-empty string")
    if action_name == "RETRIEVE":
        top_k = arguments["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != 3:
            raise ValueError("RETRIEVE.top_k must be exactly 3")
    if action_name == "SUMMARY":
        validate_nonempty_string_list(
            action_name, "source_ids", arguments["source_ids"]
        )
    if action_name == "FILTER":
        selected = [
            field for field in ("keep_ids", "drop_ids") if field in arguments
        ]
        if len(selected) != 1:
            raise ValueError("FILTER requires exactly one of keep_ids or drop_ids")
        validate_nonempty_string_list(action_name, selected[0], arguments[selected[0]])
        if arguments["scope"] not in {"active", "session", "all"}:
            raise ValueError("FILTER.scope must be active, session, or all")


def format_action(action_name: str, arguments: dict[str, Any]) -> str:
    canonical = normalize_action_name(action_name)
    validate_action_payload(canonical, arguments)
    return f"{canonical} {json.dumps(arguments, ensure_ascii=False)}"


def parse_env_action(action: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(action, str):
        raise ValueError("Formal action must be text")
    parts = action.strip().split(None, 1)
    action_name = normalize_action_name(parts[0]) if parts else ""
    if len(parts) != 2:
        raise ValueError(f"{action_name or 'Action'} payload is missing")
    arguments = json.loads(parts[1])
    validate_action_payload(action_name, arguments)
    return action_name, arguments


def extract_bare_env_action(text: str) -> str:
    """Extract and canonicalize the first balanced formal action."""

    if not isinstance(text, str):
        raise ValueError("Policy response must be text")
    cleaned = text.strip()
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-4].strip()
    markers = list(_ACTION_MARKER_RE.finditer(cleaned))
    action_region = cleaned[markers[-1].end() :] if markers else cleaned
    match = _BARE_ACTION_RE.search(action_region)
    if match is None:
        raise ValueError("Policy response contains no formal action")
    start = match.start("action")
    brace_start = action_region.find("{", start)
    depth = 0
    in_string = False
    escaped = False
    for index in range(brace_start, len(action_region)):
        char = action_region[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                raw_action = action_region[start : index + 1]
                action_name, arguments = parse_env_action(raw_action)
                return format_action(action_name, arguments)
    raise ValueError("Formal action JSON payload is not balanced")


def canonicalize_react_action(text: str) -> str:
    """Canonical action used by both the execution client and formal validator."""

    return extract_bare_env_action(text)
