"""Runtime-resolved source and AST audit for the shared multitask lineage."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DOMAIN_LITERALS = (
    "webshop",
    "swesmith",
    "literesearcher",
    "openmle_fast",
    "openmle-fast",
    "memoryarena",
)


@dataclass(frozen=True)
class SourceModuleContract:
    """One import that must resolve below a specific active source root."""

    label: str
    module_name: str
    root_kind: str


_ACTIVE_MODULES = (
    SourceModuleContract(
        "fully_async_main",
        "verl.experimental.fully_async_policy.fully_async_main",
        "verl",
    ),
    SourceModuleContract(
        "fully_async_rollouter",
        "verl.experimental.fully_async_policy.fully_async_rollouter",
        "verl",
    ),
    SourceModuleContract(
        "fully_async_trainer",
        "verl.experimental.fully_async_policy.fully_async_trainer",
        "verl",
    ),
    SourceModuleContract(
        "message_queue",
        "verl.experimental.fully_async_policy.message_queue",
        "verl",
    ),
    SourceModuleContract(
        "upstream_agent_loop",
        "verl.experimental.agent_loop.agent_loop",
        "verl",
    ),
    SourceModuleContract("amg_agent_loop", "agentmemorygym_verl.agent_loop", "plugin"),
    SourceModuleContract("amg_dataset", "agentmemorygym_verl.dataset", "plugin"),
    SourceModuleContract("amg_action_gae", "agentmemorygym_verl.action_gae", "plugin"),
    SourceModuleContract("policy_turn", "agentenv.controller.policy_turn", "agentgym"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _domain_literals(node: ast.AST) -> tuple[str, ...]:
    values = {
        value.strip().lower()
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
        for value in (child.value,)
    }
    return tuple(
        domain
        for domain in _DOMAIN_LITERALS
        if any(domain in value for value in values)
    )


def _domain_mentions(
    node: ast.AST,
    bindings: Mapping[str, frozenset[str]],
) -> tuple[str, ...]:
    domains = set(_domain_literals(node))
    domains.update(
        domain
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        for domain in bindings.get(child.id, ())
    )
    domains.update(
        domain
        for child in ast.walk(node)
        if isinstance(child, (ast.Name, ast.Attribute))
        for domain in _identifier_domains(
            child.id if isinstance(child, ast.Name) else child.attr
        )
    )
    return tuple(sorted(domains))


def _assignment_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(name for item in node.elts for name in _assignment_names(item))
    return ()


def _domain_bindings(tree: ast.AST) -> dict[str, frozenset[str]]:
    assignments: list[tuple[tuple[str, ...], ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = tuple(
                name for target in node.targets for name in _assignment_names(target)
            )
            assignments.append((names, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assignments.append((_assignment_names(node.target), node.value))

    bindings: dict[str, frozenset[str]] = {}
    changed = True
    while changed:
        changed = False
        for names, value in assignments:
            domains = frozenset(_domain_mentions(value, bindings))
            if not domains:
                continue
            for name in names:
                combined = bindings.get(name, frozenset()) | domains
                if combined != bindings.get(name):
                    bindings[name] = combined
                    changed = True
    return bindings


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _identifier_domains(value: str) -> tuple[str, ...]:
    lowered = value.lower()
    compact = lowered.replace("_", "").replace("-", "")
    return tuple(
        domain
        for domain in _DOMAIN_LITERALS
        if domain in lowered or domain.replace("_", "").replace("-", "") in compact
    )


def _audit_tree(label: str, path: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(
            f"cannot AST-audit active source {label}: {path}: {exc}"
        ) from exc
    bindings = _domain_bindings(tree)
    branch_nodes = (ast.If, ast.IfExp)
    match_node = getattr(ast, "Match", None)
    if match_node is not None:
        branch_nodes += (match_node,)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            domains = _identifier_domains(node.name)
            if domains:
                raise RuntimeError(
                    f"active source {label} has a domain-specific definition "
                    f"for {domains!r} at {path}:{getattr(node, 'lineno', '?')}"
                )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            lowered = " ".join(names).lower()
            if any(domain in lowered for domain in _DOMAIN_LITERALS):
                raise RuntimeError(
                    f"active source {label} imports a domain-specific module at "
                    f"{path}:{getattr(node, 'lineno', '?')}"
                )
        if isinstance(node, branch_nodes):
            if match_node is not None and isinstance(node, match_node):
                selector_nodes = [node.subject]
                selector_nodes.extend(case.pattern for case in node.cases)
                selector_nodes.extend(
                    case.guard for case in node.cases if case.guard is not None
                )
            else:
                selector_nodes = [node.test]
            domains = tuple(
                sorted(
                    {
                        domain
                        for selector in selector_nodes
                        for domain in _domain_mentions(selector, bindings)
                    }
                )
            )
            if domains:
                raise RuntimeError(
                    f"active source {label} has a domain-specific executable branch "
                    f"for {domains!r} at {path}:{getattr(node, 'lineno', '?')}"
                )
        if isinstance(node, ast.Subscript):
            domains = _domain_mentions(node.value, bindings)
            if domains:
                raise RuntimeError(
                    f"active source {label} has a domain-specific dispatch table "
                    f"for {domains!r} at {path}:{getattr(node, 'lineno', '?')}"
                )
        if isinstance(node, ast.Call):
            domains = tuple(
                sorted(
                    set(_identifier_domains(_call_name(node.func)))
                    | set(_domain_mentions(node.func, bindings))
                )
            )
            if domains:
                raise RuntimeError(
                    f"active source {label} calls a domain-specific function "
                    f"for {domains!r} at {path}:{getattr(node, 'lineno', '?')}"
                )


def audit_source_paths(files: Mapping[str, Path]) -> dict[str, Any]:
    """AST-audit exact resolved files without inferring inactive alternatives."""

    if not files:
        raise RuntimeError("active-source audit received no files")
    report: dict[str, dict[str, str]] = {}
    for label, raw_path in files.items():
        source_path = Path(raw_path)
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(
                f"active source {label} is missing or symlinked: {source_path}"
            )
        path = source_path.resolve()
        _audit_tree(str(label), path)
        report[str(label)] = {"path": str(path), "sha256": _sha256(path)}
    return {
        "schema": "amg_runtime_active_source_ast_audit_v1",
        "status": "pass",
        "files": report,
        "forbidden_domain_literals": list(_DOMAIN_LITERALS),
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def audit_resolved_active_sources(
    *,
    verl_root: Path,
    outer_root: Path,
    module_contracts: Sequence[SourceModuleContract] = _ACTIVE_MODULES,
) -> dict[str, Any]:
    """Import the active stack, verify its roots, then AST-audit those files."""

    roots = {
        "verl": Path(verl_root).resolve(),
        "plugin": (Path(outer_root) / "async_plugins").resolve(),
        "agentgym": (Path(outer_root) / "AgentGym" / "agentenv").resolve(),
    }
    legacy_verl = (Path(outer_root) / "AgentGym-RL" / "verl").resolve()
    resolved: dict[str, Path] = {}
    modules: dict[str, str] = {}
    for contract in module_contracts:
        if contract.root_kind not in roots:
            raise RuntimeError(
                f"active-source contract {contract.label} has unknown root kind "
                f"{contract.root_kind!r}"
            )
        module = importlib.import_module(contract.module_name)
        source = inspect.getsourcefile(module) or getattr(module, "__file__", None)
        if not source:
            raise RuntimeError(
                f"active module {contract.module_name!r} has no Python source file"
            )
        source_path = Path(source)
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(
                f"active module {contract.module_name!r} source is missing or "
                f"symlinked: {source_path}"
            )
        path = source_path.resolve()
        if _is_relative_to(path, legacy_verl):
            raise RuntimeError(
                f"active module {contract.module_name!r} resolved to inactive nested "
                f"legacy veRL: {path}"
            )
        expected_root = roots[contract.root_kind]
        if not _is_relative_to(path, expected_root):
            raise RuntimeError(
                f"active module {contract.module_name!r} resolved outside "
                f"{contract.root_kind} root {expected_root}: {path}"
            )
        resolved[contract.label] = path
        modules[contract.label] = contract.module_name
    report = audit_source_paths(resolved)
    report["modules"] = modules
    report["roots"] = {key: str(value) for key, value in roots.items()}
    report["inactive_legacy_verl_root"] = str(legacy_verl)
    return report


__all__ = [
    "SourceModuleContract",
    "audit_resolved_active_sources",
    "audit_source_paths",
]
