#!/usr/bin/env python3
"""Analyze frozen AgentMemory WebShop rollouts without relying on final return."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "agentmemory_webshop_rollout_analysis_v1"
PROGRESS_PATTERN = re.compile(r"(?m)^Progress:\s*(\d+)/6\s*$")
ACTION_LABEL_PATTERN = re.compile(r"(?im)^\s*Action:\s*")
MEMORY_ACTION_PATTERN = re.compile(
    r"^(ADD|UPDATE|DELETE|RETRIEVE|SUMMARY|FILTER)\s+(\{.*\})\s*$",
    re.IGNORECASE,
)
NATIVE_ACTION_PATTERN = re.compile(r"^(search|click)\[(.*)\]\s*$", re.IGNORECASE)
ASIN_PATTERN = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)
STORED_MEMORY_PATTERN = re.compile(
    r"Stored memory \[(mem_\d+)\]\s+([^:\n]+):\s*([^\n]+)", re.IGNORECASE
)
RETRIEVED_MEMORY_PATTERN = re.compile(
    r"(?m)^\[(mem_\d+)\]\s+([^:\n]+):\s*(.+?)\s*$"
)
CURRENT_PRODUCT_PATTERN = re.compile(
    r"\[SEP\]\s*< Prev\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*"
    r"Price:\s*(.*?)\s*\[SEP\]\s*Rating:",
    re.IGNORECASE | re.DOTALL,
)
SAMPLE_PATTERN = re.compile(r"steptest_batch_(\d+)_sample_(\d+)")
WORD_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "base",
    "buy",
    "for",
    "from",
    "in",
    "item",
    "of",
    "one",
    "option",
    "pack",
    "product",
    "select",
    "the",
    "to",
    "with",
}
WRONG_BUY_TERMINAL_TEXT = "The shopping episode has ended."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-replicas", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sample_sort_key(path: Path) -> tuple[int, int, str]:
    match = SAMPLE_PATTERN.search(str(path))
    if match is None:
        return (sys.maxsize, sys.maxsize, str(path))
    return (int(match.group(1)), int(match.group(2)), str(path))


def parse_progress(text: str) -> int:
    match = PROGRESS_PATTERN.search(text)
    if match is None:
        raise ValueError("observation has no authoritative Progress line")
    value = int(match.group(1))
    if not 0 <= value <= 6:
        raise ValueError(f"invalid progress value {value}")
    return value


def parse_action(text: str) -> dict[str, Any]:
    labels = list(ACTION_LABEL_PATTERN.finditer(text))
    candidate = text[labels[-1].end() :] if labels else text
    candidate = candidate.strip().splitlines()[0].strip() if candidate.strip() else ""
    memory_match = MEMORY_ACTION_PATTERN.fullmatch(candidate)
    if memory_match:
        kind = memory_match.group(1).upper()
        try:
            payload = json.loads(memory_match.group(2))
        except json.JSONDecodeError:
            return {"kind": "INVALID", "raw": candidate, "reason": "invalid_json"}
        return {"kind": kind, "raw": candidate, "payload": payload}
    native_match = NATIVE_ACTION_PATTERN.fullmatch(candidate)
    if native_match:
        return {
            "kind": native_match.group(1).upper(),
            "raw": candidate,
            "value": native_match.group(2).strip(),
        }
    return {"kind": "INVALID", "raw": candidate, "reason": "unparsed"}


def parse_candidate_options(observation: str) -> list[str]:
    marker = "**Available Options:**"
    marker_index = observation.find(marker)
    if marker_index < 0:
        return []
    options: list[str] = []
    tail = observation[marker_index + len(marker) :]
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped and not options:
            continue
        if not stripped.startswith("- "):
            if options:
                break
            continue
        option = stripped[2:].split(" [SEP]", 1)[0].strip()
        if option:
            options.append(option)
        if " [SEP]" in stripped:
            break
    return options


def current_product(observation: str) -> dict[str, str] | None:
    match = CURRENT_PRODUCT_PATTERN.search(observation)
    if match is None:
        return None
    return {"title": " ".join(match.group(1).split()), "price": match.group(2).strip()}


def terms(text: str) -> set[str]:
    return {
        token
        for token in WORD_PATTERN.findall(text.lower())
        if token not in STOPWORDS and (len(token) >= 3 or token.isdigit())
    }


def overlap_evidence(left: str, right: str) -> dict[str, Any]:
    left_terms = terms(left)
    right_terms = terms(right)
    shared = sorted(left_terms & right_terms)
    denominator = min(len(left_terms), len(right_terms)) or 1
    score = len(shared) / denominator
    return {
        "shared_terms": shared,
        "overlap": round(score, 6),
        "linked": len(shared) >= 2 and (score >= 0.3 or len(shared) >= 4),
    }


def add_matches_buy(add_event: dict[str, Any], buy_event: dict[str, Any]) -> dict[str, Any]:
    add_asin = add_event.get("page_asin")
    buy_asin = buy_event.get("page_asin")
    if add_asin and buy_asin:
        return {
            "method": "page_asin",
            "linked": add_asin.lower() == buy_asin.lower(),
            "add_asin": add_asin,
            "buy_asin": buy_asin,
        }
    buy_title = (buy_event.get("product") or {}).get("title", "")
    evidence = overlap_evidence(add_event["value"], buy_title)
    evidence["method"] = "value_to_buy_page_title"
    return evidence


def retrieved_value_matches(expected: str, observed: str) -> dict[str, Any]:
    if " ".join(expected.split()).casefold() == " ".join(observed.split()).casefold():
        return {"method": "normalized_exact", "linked": True, "overlap": 1.0}
    evidence = overlap_evidence(expected, observed)
    evidence["method"] = "lexical_overlap"
    return evidence


def parse_stored_memory(observation: str) -> dict[str, str] | None:
    matches = list(STORED_MEMORY_PATTERN.finditer(observation))
    if not matches:
        return None
    match = matches[-1]
    return {
        "memory_id": match.group(1),
        "key": match.group(2).strip(),
        "value": match.group(3).strip(),
    }


def parse_retrieved_memories(observation: str) -> dict[str, dict[str, str]]:
    if "Result: Retrieved memories:" not in observation:
        return {}
    segment = observation.rsplit("Result: Retrieved memories:", 1)[1]
    segment = segment.split("Active retrieved/summary context:", 1)[0]
    result: dict[str, dict[str, str]] = {}
    for match in RETRIEVED_MEMORY_PATTERN.finditer(segment):
        result[match.group(1)] = {
            "key": match.group(2).strip(),
            "value": match.group(3).strip(),
        }
    return result


def query_option_match(query: str, options: list[str]) -> dict[str, Any] | None:
    query_terms = terms(query)
    if len(query_terms) < 2 or not options:
        return None
    scored = []
    for position, option in enumerate(options, start=1):
        shared = sorted(query_terms & terms(option))
        scored.append((len(shared), position, shared))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if scored[0][0] < 2:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return {
        "position": scored[0][1],
        "shared_terms": scored[0][2],
        "query_terms": sorted(query_terms),
    }


def hash_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def analyze_trajectory(
    row: dict[str, Any], replica_index: int, rollout_log: str
) -> dict[str, Any]:
    conversations = row["conversations"]
    if (
        len(conversations) < 3
        or conversations[0].get("role") != "user"
        or conversations[1] != {"role": "assistant", "content": "Ok."}
    ):
        raise ValueError(f"{rollout_log} item={row.get('item_id')}: invalid seed pair")
    actual_assistants = [
        index
        for index, message in enumerate(conversations[2:], start=2)
        if message.get("role") == "assistant"
    ]
    if len(actual_assistants) != int(row["task_rounds"]):
        raise ValueError(
            f"{rollout_log} item={row.get('item_id')}: sampled turn count mismatch"
        )

    action_counts: collections.Counter[str] = collections.Counter()
    add_events: list[dict[str, Any]] = []
    retrieve_events: list[dict[str, Any]] = []
    buy_events: list[dict[str, Any]] = []
    session_searches: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    current_asin: str | None = None
    progress_values = []
    for message in conversations[2:]:
        if message.get("role") == "user":
            progress_values.append(parse_progress(message["content"]))

    for step_index, message_index in enumerate(actual_assistants):
        if message_index + 1 >= len(conversations):
            raise ValueError(f"{rollout_log}: sampled action has no resulting observation")
        before = conversations[message_index - 1]["content"]
        after = conversations[message_index + 1]["content"]
        before_progress = parse_progress(before)
        after_progress = parse_progress(after)
        action = parse_action(conversations[message_index]["content"])
        kind = action["kind"]
        action_counts[kind] += 1

        if kind == "SEARCH":
            current_asin = None
            options = parse_candidate_options(before)
            mapped = query_option_match(action["value"], options)
            session_searches[before_progress].append(
                {
                    "step_index": step_index,
                    "query": action["value"],
                    "options": options,
                    "mapped": mapped,
                }
            )
        elif kind == "CLICK":
            value = action["value"]
            if ASIN_PATTERN.fullmatch(value):
                current_asin = value.upper()
            elif value.casefold() in {"back to search", "< prev", "search"}:
                current_asin = None
            if value.casefold() == "buy now":
                correct = after_progress > before_progress
                wrong_committed = WRONG_BUY_TERMINAL_TEXT in after
                buy_events.append(
                    {
                        "step_index": step_index,
                        "from_progress": before_progress,
                        "to_progress": after_progress,
                        "correct": correct,
                        "committed": correct or wrong_committed,
                        "status": (
                            "correct_committed"
                            if correct
                            else (
                                "wrong_committed"
                                if wrong_committed
                                else "uncommitted_click"
                            )
                        ),
                        "page_asin": current_asin,
                        "product": current_product(before),
                    }
                )
                current_asin = None
        elif kind == "ADD":
            stored = parse_stored_memory(after)
            payload = action.get("payload", {})
            if stored is not None:
                add_events.append(
                    {
                        "step_index": step_index,
                        "progress": before_progress,
                        "memory_id": stored["memory_id"],
                        "key": str(payload.get("key", stored["key"])),
                        "value": str(payload.get("value", stored["value"])),
                        "stored_value": stored["value"],
                        "page_asin": current_asin,
                        "product": current_product(before),
                    }
                )
        elif kind == "RETRIEVE":
            retrieve_events.append(
                {
                    "step_index": step_index,
                    "progress": before_progress,
                    "payload": action.get("payload", {}),
                    "memories": parse_retrieved_memories(after),
                }
            )

    correct_buys = [event for event in buy_events if event["correct"]]
    strict_chains = []
    for add_event in add_events:
        candidate_buys = [
            event
            for event in correct_buys
            if event["from_progress"] == add_event["progress"]
            and event["step_index"] > add_event["step_index"]
        ]
        for buy_event in candidate_buys:
            buy_link = add_matches_buy(add_event, buy_event)
            if not buy_link["linked"]:
                continue
            candidate_retrieves = [
                event
                for event in retrieve_events
                if event["step_index"] > buy_event["step_index"]
                and event["progress"] >= buy_event["to_progress"]
                and add_event["memory_id"] in event["memories"]
            ]
            for retrieve_event in candidate_retrieves:
                retrieved = retrieve_event["memories"][add_event["memory_id"]]
                retrieve_link = retrieved_value_matches(
                    add_event["stored_value"], retrieved["value"]
                )
                if not retrieve_link["linked"]:
                    continue
                downstream = next(
                    (
                        event
                        for event in correct_buys
                        if event["step_index"] > retrieve_event["step_index"]
                        and event["from_progress"] >= retrieve_event["progress"]
                    ),
                    None,
                )
                strict_chains.append(
                    {
                        "memory_id": add_event["memory_id"],
                        "add": add_event,
                        "correct_buy": buy_event,
                        "buy_link": buy_link,
                        "later_retrieve": retrieve_event,
                        "retrieve_link": retrieve_link,
                        "downstream_correct_buy": downstream,
                    }
                )
                break
            if strict_chains and strict_chains[-1]["memory_id"] == add_event["memory_id"]:
                break

    first_searches = []
    for progress, searches in sorted(session_searches.items()):
        first = searches[0]
        first_mapped = next((item for item in searches if item["mapped"]), None)
        first_searches.append(
            {
                "progress": progress,
                "first_query": first["query"],
                "first_query_mapping": first["mapped"],
                "first_mapped_query": (
                    {
                        "query": first_mapped["query"],
                        "mapping": first_mapped["mapped"],
                    }
                    if first_mapped
                    else None
                ),
                "candidate_order": first["options"],
                "session_succeeded": any(
                    event["correct"] and event["from_progress"] == progress
                    for event in buy_events
                ),
            }
        )

    final_progress = max(progress_values) if progress_values else 0
    return {
        "replica_index": replica_index,
        "rollout_log": rollout_log,
        "item_id": row["item_id"],
        "task_rounds": row["task_rounds"],
        "sample_excluded": bool(row.get("sample_excluded", False)),
        "final_progress": final_progress,
        "action_counts": dict(sorted(action_counts.items())),
        "buy_events": buy_events,
        "add_events": add_events,
        "retrieve_events": retrieve_events,
        "strict_chains": strict_chains,
        "first_searches": first_searches,
    }


def merge_counts(counters: Iterable[dict[str, int]]) -> dict[str, int]:
    total: collections.Counter[str] = collections.Counter()
    for counter in counters:
        total.update(counter)
    return dict(sorted(total.items()))


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "webshop_rollout_analysis.json").resolve()
    log_paths = sorted(
        run_dir.glob("results/*/executer_logs/steptest_batch_*_sample_*/0.json"),
        key=sample_sort_key,
    )
    if not log_paths:
        raise SystemExit(f"no rollout 0.json files found under {run_dir}")
    if args.expected_replicas is not None and len(log_paths) != args.expected_replicas:
        raise SystemExit(
            f"expected {args.expected_replicas} replicas, found {len(log_paths)}"
        )

    trajectories = []
    for replica_index, path in enumerate(log_paths):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            trajectories.append(
                analyze_trajectory(
                    row,
                    replica_index=replica_index,
                    rollout_log=str(path.relative_to(run_dir)),
                )
            )

    included_trajectories = [
        trajectory for trajectory in trajectories if not trajectory["sample_excluded"]
    ]
    if not included_trajectories:
        raise ValueError("all trajectories were excluded as infrastructure failures")
    progress_distribution = collections.Counter(
        trajectory["final_progress"] for trajectory in included_trajectories
    )
    buy_events = [
        event
        for trajectory in included_trajectories
        for event in trajectory["buy_events"]
    ]
    retrieve_events = [
        event
        for trajectory in included_trajectories
        for event in trajectory["retrieve_events"]
    ]
    chains = [
        {
            "replica_index": trajectory["replica_index"],
            "item_id": trajectory["item_id"],
            **chain,
        }
        for trajectory in included_trajectories
        for chain in trajectory["strict_chains"]
    ]
    first_searches = [
        {
            "replica_index": trajectory["replica_index"],
            "item_id": trajectory["item_id"],
            **search,
        }
        for trajectory in included_trajectories
        for search in trajectory["first_searches"]
    ]
    first_mapped_positions = collections.Counter(
        search["first_query_mapping"]["position"]
        for search in first_searches
        if search["first_query_mapping"] is not None
    )
    first_specific_positions = collections.Counter(
        search["first_mapped_query"]["mapping"]["position"]
        for search in first_searches
        if search["first_mapped_query"] is not None
    )
    option_set_hashes: dict[str, set[str]] = collections.defaultdict(set)
    observed_orders: dict[str, set[str]] = collections.defaultdict(set)
    for search in first_searches:
        key = f"item={search['item_id']}:progress={search['progress']}"
        options = search["candidate_order"]
        if options:
            option_set_hashes[key].add(hash_json(sorted(options)))
            observed_orders[key].add(hash_json(options))
    inconsistent_option_sets = {
        key: sorted(values) for key, values in option_set_hashes.items() if len(values) > 1
    }
    if inconsistent_option_sets:
        raise ValueError(
            f"candidate content changed across replicas: {inconsistent_option_sets}"
        )

    trajectory_count = len(included_trajectories)
    total_progress = sum(
        trajectory["final_progress"] for trajectory in included_trajectories
    )
    strict_trajectory_keys = {
        (chain["replica_index"], chain["item_id"]) for chain in chains
    }
    four_stage_chains = [
        chain for chain in chains if chain["downstream_correct_buy"] is not None
    ]
    result = {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "analysis_contract": {
            "progress": "first authoritative Progress line in each observation",
            "ps": "sum(final_progress)/(6*trajectory_count)",
            "buy_status": (
                "progress increase means correct committed; terminal episode text means "
                "wrong committed; otherwise click[Buy Now] was uncommitted"
            ),
            "strict_three_stage": (
                "ADD on actual product page -> same-session correct BUY with matching "
                "page ASIN/title -> later same-memory-id nonempty RETRIEVE with matching value"
            ),
            "position_mapping": (
                "diagnostic lexical mapping of search query to a unique displayed candidate; "
                "not an answer-label proof"
            ),
        },
        "summary": {
            "replica_count": len(log_paths),
            "trajectory_count": trajectory_count,
            "excluded_trajectory_count": len(trajectories) - trajectory_count,
            "progress_distribution": {
                str(key): progress_distribution.get(key, 0) for key in range(7)
            },
            "progress_score": total_progress / (6 * trajectory_count),
            "completed_six": progress_distribution.get(6, 0),
            "action_counts": merge_counts(
                trajectory["action_counts"] for trajectory in included_trajectories
            ),
            "correct_buy": sum(event["correct"] for event in buy_events),
            "wrong_buy": sum(
                event["status"] == "wrong_committed" for event in buy_events
            ),
            "uncommitted_buy_click": sum(
                event["status"] == "uncommitted_click" for event in buy_events
            ),
            "add": sum(
                len(trajectory["add_events"])
                for trajectory in included_trajectories
            ),
            "retrieve": len(retrieve_events),
            "nonempty_retrieve": sum(bool(event["memories"]) for event in retrieve_events),
            "strict_three_stage_chain_count": len(chains),
            "strict_three_stage_trajectory_count": len(strict_trajectory_keys),
            "strict_four_stage_chain_count": len(four_stage_chains),
            "session_with_search_count": len(first_searches),
            "first_search_uniquely_mapped_count": sum(
                search["first_query_mapping"] is not None for search in first_searches
            ),
            "first_search_mapped_positions": {
                str(key): first_mapped_positions.get(key, 0) for key in range(1, 6)
            },
            "first_specific_search_mapped_positions": {
                str(key): first_specific_positions.get(key, 0) for key in range(1, 6)
            },
        },
        "presentation_diagnostics": {
            "option_set_consistency": "pass",
            "observed_unique_order_counts": {
                key: len(values) for key, values in sorted(observed_orders.items())
            },
            "first_searches": first_searches,
        },
        "strict_chains": chains,
        "trajectories": trajectories,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(json.dumps(result["summary"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
