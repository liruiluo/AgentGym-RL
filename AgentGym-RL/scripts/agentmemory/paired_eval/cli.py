"""Dependency-light CLI for manifest execution and evidence verification."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Optional, Sequence

from .contracts import RunConfig
from .controller import AgentGymPolicyTurnController
from .evidence import AppendSafeJsonlWriter, PrivateEvidenceStore
from .manifest import RuntimeBindings, execute_manifest, expand_manifest
from .runner import PairedRunner
from .serialization import canonical_json_bytes
from .verifier import (
    build_public_summary,
    validate_result_row,
    verify_pair_completeness,
)


def load_json_object(path: Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON document must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError("refusing to read a JSONL file with a partial final line")
    rows = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid JSONL line {line_number}") from error
        if not isinstance(value, Mapping):
            raise TypeError(f"JSONL line {line_number} must contain an object")
        rows.append(value)
    return rows


def load_runtime_factory(
    specification: str,
    evidence_store: PrivateEvidenceStore,
) -> Callable[[RunConfig], RuntimeBindings]:
    if ":" not in specification:
        raise ValueError("runtime factory must use module:callable syntax")
    module_name, attribute_name = specification.rsplit(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("runtime factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError("runtime factory target is not callable")

    def bind(config: RunConfig) -> RuntimeBindings:
        result = factory(config, evidence_store=evidence_store)
        if not isinstance(result, RuntimeBindings):
            raise TypeError("integration factory must return RuntimeBindings")
        return result

    return bind


def write_stdout(value: Any) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paired-eval")
    commands = parser.add_subparsers(dest="command", required=True)

    expand = commands.add_parser("expand", help="expand a paired manifest")
    expand.add_argument("--manifest", type=Path, required=True)

    run = commands.add_parser("run", help="execute a paired manifest")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--results", type=Path, required=True)
    run.add_argument("--evidence-dir", type=Path, required=True)
    run.add_argument("--runtime-factory", required=True)

    verify = commands.add_parser("verify", help="verify private result pairs")
    verify.add_argument("--results", type=Path, required=True)

    summary = commands.add_parser(
        "public-summary", help="emit a privacy-safe scalar summary"
    )
    summary.add_argument("--results", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "expand":
        configs = expand_manifest(load_json_object(arguments.manifest))
        for config in configs:
            write_stdout(config.to_payload())
        return 0
    if arguments.command == "verify":
        write_stdout(verify_pair_completeness(load_jsonl(arguments.results)))
        return 0
    if arguments.command == "public-summary":
        write_stdout(build_public_summary(load_jsonl(arguments.results)))
        return 0

    evidence_store = PrivateEvidenceStore(arguments.evidence_dir)
    runner = PairedRunner(
        controller=AgentGymPolicyTurnController.from_agentenv(),
        evidence_store=evidence_store,
    )
    rows = execute_manifest(
        load_json_object(arguments.manifest),
        runner=runner,
        runtime_factory=load_runtime_factory(
            arguments.runtime_factory,
            evidence_store,
        ),
        writer=AppendSafeJsonlWriter(
            arguments.results,
            validator=validate_result_row,
        ),
    )
    write_stdout(verify_pair_completeness(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
