#!/usr/bin/env python3
"""Replay frozen LiteResearcher searches while measuring latency and memory.

The released LiteResearcher formatter randomizes snippets, so response bodies
are intentionally excluded from parity checks.  Retrieval parity is reported
on four separate axes: ordered URLs, top-1 URL, top-k set overlap, and
same-URL score differences within a configurable floating-point tolerance.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib import request

GIB = 1024**3


def normalize_query(value: object) -> str:
    if isinstance(value, str):
        query = value.strip()
    elif (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) for item in value)
    ):
        query = " ".join(item.strip() for item in value).strip()
    else:
        raise ValueError("query must be a non-empty string or list of strings")
    if not query:
        raise ValueError("query must not be empty")
    return query


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def ordered_url_scores(results: list[dict[str, Any]]) -> list[list[object]]:
    pairs: list[list[object]] = []
    for result in results:
        url = result.get("url", result.get("link"))
        score = result.get("score")
        if (
            not isinstance(url, str)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            raise TypeError("search result lacks string URL or numeric score")
        pairs.append([url, float(score)])
    return pairs


def compare_url_scores(
    expected: list[list[object]],
    actual: list[list[object]],
    *,
    score_atol: float,
) -> dict[str, Any]:
    expected_urls = [str(pair[0]) for pair in expected]
    actual_urls = [str(pair[0]) for pair in actual]
    expected_scores = {str(url): float(score) for url, score in expected}
    actual_scores = {str(url): float(score) for url, score in actual}
    common_urls = sorted(set(expected_scores) & set(actual_scores))
    score_deltas = [
        abs(expected_scores[url] - actual_scores[url]) for url in common_urls
    ]
    overlap_count = len(set(expected_urls) & set(actual_urls))
    overlap_denominator = max(1, len(set(expected_urls)))
    ordered_url_exact = actual_urls == expected_urls
    scores_within_tolerance = bool(common_urls) and all(
        delta <= score_atol for delta in score_deltas
    )
    return {
        "ordered_url_exact": ordered_url_exact,
        "top1_url_exact": bool(
            expected_urls and actual_urls and expected_urls[0] == actual_urls[0]
        ),
        "topk_set_overlap_count": overlap_count,
        "topk_set_overlap_ratio": overlap_count / overlap_denominator,
        "common_url_count": len(common_urls),
        "common_url_score_max_abs_diff": max(score_deltas) if score_deltas else None,
        "common_url_scores_within_tolerance": scores_within_tolerance,
        "ordered_url_score_within_tolerance": ordered_url_exact
        and scores_within_tolerance,
    }


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def process_start_tick(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()[21])
    except (OSError, ValueError, IndexError):
        return None


def process_state(pid: int) -> str | None:
    try:
        for line in (
            Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines()
        ):
            if line.startswith("State:"):
                return line.split()[1]
    except OSError:
        return None
    return None


def discover_milvus_pid() -> int | None:
    candidates: list[tuple[int, int]] = []
    proc = Path("/proc")
    if not proc.exists():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        pid = int(entry.name)
        tick = process_start_tick(pid)
        if (
            b"/milvus/bin/milvus run standalone" in cmdline
            and process_state(pid) != "Z"
            and tick is not None
        ):
            candidates.append((tick, pid))
    return max(candidates)[1] if candidates else None


def resolve_milvus_process(
    pid: int | None, expected_start_tick: int | None
) -> tuple[int, int]:
    selected = pid if pid is not None else discover_milvus_pid()
    if selected is None:
        raise RuntimeError("no live Milvus standalone process found")
    actual_start_tick = process_start_tick(selected)
    if actual_start_tick is None or process_state(selected) == "Z":
        raise RuntimeError(f"Milvus pid {selected} is not a live process")
    if expected_start_tick is not None and actual_start_tick != expected_start_tick:
        raise RuntimeError(
            f"Milvus process identity mismatch: pid={selected} "
            f"expected_start_tick={expected_start_tick} actual_start_tick={actual_start_tick}"
        )
    return selected, actual_start_tick


def memory_snapshot(cgroup_path: str, milvus_pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checked_at": time.time(),
        "cgroup_bytes": read_int(cgroup_path),
        "milvus_pid": milvus_pid,
        "milvus_start_tick": process_start_tick(milvus_pid),
    }
    try:
        fields = {}
        for line in Path(f"/proc/{milvus_pid}/status").read_text().splitlines():
            if line.startswith(
                ("VmRSS:", "RssAnon:", "RssFile:", "RssShmem:", "Threads:")
            ):
                name, raw = line.split(":", 1)
                fields[name] = int(raw.strip().split()[0])
        result.update(
            milvus_rss_kib=fields.get("VmRSS"),
            milvus_anon_kib=fields.get("RssAnon"),
            milvus_file_kib=fields.get("RssFile"),
            milvus_shmem_kib=fields.get("RssShmem"),
            milvus_threads=fields.get("Threads"),
        )
    except (OSError, ValueError):
        result["milvus_process_missing"] = True
    return result


def fetch(endpoint: str, query: str, timeout: float) -> tuple[dict[str, Any], float]:
    body = json.dumps(
        {
            "query": query,
            "limit": 5,
            "search_type": "hybrid",
            "sparse_weight": 0.7,
            "dense_weight": 1.0,
        },
        ensure_ascii=True,
    ).encode("utf-8")
    req = request.Request(
        endpoint.rstrip("/") + "/search",
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with request.urlopen(req, timeout=timeout) as response:
        raw = response.read(32 * 1024 * 1024 + 1)
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
    elapsed = time.monotonic() - started
    if len(raw) > 32 * 1024 * 1024:
        raise RuntimeError("response exceeded 32 MiB")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise TypeError("invalid search response")
    return payload, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--waves", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--cgroup-memory-path",
        default="/sys/fs/cgroup/memory/memory.usage_in_bytes",
    )
    parser.add_argument("--cgroup-limit-gib", type=float, default=1600.0)
    parser.add_argument("--min-headroom-gib", type=float, default=220.0)
    parser.add_argument("--milvus-pid", type=int)
    parser.add_argument("--milvus-start-tick", type=int)
    parser.add_argument("--score-atol", type=float, default=1e-5)
    parser.add_argument("--min-ordered-url-exact-ratio", type=float, default=0.95)
    parser.add_argument("--min-top1-url-exact-ratio", type=float, default=0.95)
    parser.add_argument("--min-topk-overlap-ratio", type=float, default=0.8)
    parser.add_argument("--memory-sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--post-run-observe-seconds", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def ratio(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def main() -> int:
    args = parse_args()
    if args.concurrency < 1 or args.waves < 1:
        raise SystemExit("concurrency and waves must be positive")
    if args.memory_sample_interval_seconds <= 0 or args.post_run_observe_seconds < 0:
        raise SystemExit(
            "memory sample interval must be positive and observation time non-negative"
        )
    if not 0 <= args.min_ordered_url_exact_ratio <= 1:
        raise SystemExit("min ordered URL exact ratio must be in [0, 1]")
    if not 0 <= args.min_top1_url_exact_ratio <= 1:
        raise SystemExit("min top-1 URL exact ratio must be in [0, 1]")
    if not 0 <= args.min_topk_overlap_ratio <= 1:
        raise SystemExit("min top-k overlap ratio must be in [0, 1]")

    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    records = baseline.get("records")
    if not isinstance(records, list) or not records:
        raise SystemExit("baseline records must be a non-empty list")

    work = []
    for wave in range(args.waves):
        for index, record in enumerate(records):
            work.append((wave, index, normalize_query(record.get("query")), record))

    milvus_pid, milvus_start_tick = resolve_milvus_process(
        args.milvus_pid,
        args.milvus_start_tick,
    )
    threshold = int((args.cgroup_limit_gib - args.min_headroom_gib) * GIB)
    started = time.time()
    output: dict[str, Any] = {
        "schema": "amg_literesearcher_retrieval_memory_probe_v2",
        "status": "running",
        "started_at": started,
        "endpoint": args.endpoint,
        "baseline": str(baseline_path),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "concurrency": args.concurrency,
        "waves": args.waves,
        "request_count": len(work),
        "memory_abort_threshold_bytes": threshold,
        "milvus_process": {"pid": milvus_pid, "start_tick": milvus_start_tick},
        "parity_thresholds": {
            "score_atol": args.score_atol,
            "min_ordered_url_exact_ratio": args.min_ordered_url_exact_ratio,
            "min_top1_url_exact_ratio": args.min_top1_url_exact_ratio,
            "min_topk_overlap_ratio_per_request": args.min_topk_overlap_ratio,
        },
        "memory_before": memory_snapshot(args.cgroup_memory_path, milvus_pid),
        "memory_samples": [],
        "records": [],
    }
    lock = threading.Lock()
    stop = threading.Event()
    sampler_done = threading.Event()
    guard_error: list[str] = []

    def sample_memory() -> None:
        while not sampler_done.is_set():
            snapshot = memory_snapshot(args.cgroup_memory_path, milvus_pid)
            with lock:
                output["memory_samples"].append(snapshot)
            current = snapshot.get("cgroup_bytes")
            if isinstance(current, int) and current >= threshold:
                stop.set()
                if not guard_error:
                    guard_error.append(
                        f"memory headroom guard tripped: current={current} threshold={threshold}"
                    )
            sampler_done.wait(args.memory_sample_interval_seconds)

    sampler = threading.Thread(target=sample_memory, name="memory-sampler", daemon=True)
    sampler.start()

    def run_one(item: tuple[int, int, str, dict[str, Any]]) -> dict[str, Any]:
        wave, index, query, expected_record = item
        if stop.is_set():
            raise RuntimeError(guard_error[0] if guard_error else "probe stopped")
        payload, elapsed = fetch(args.endpoint, query, args.timeout_seconds)
        actual = ordered_url_scores(payload["results"])
        expected = ordered_url_scores(expected_record["results"])
        parity = compare_url_scores(expected, actual, score_atol=args.score_atol)
        return {
            "wave": wave,
            "baseline_record_index": index,
            "update": expected_record.get("update"),
            "trajectory_uid": expected_record.get("trajectory_uid"),
            "row_order": expected_record.get("row_order"),
            "query": query,
            "elapsed_s": elapsed,
            "server_search_time_s": payload.get("search_time"),
            "server_embedding_time_s": payload.get("embedding_time"),
            "server_milvus_time_s": payload.get("milvus_time"),
            "parity": parity,
            "expected": expected,
            "actual": actual,
        }

    error: str | None = None
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = [pool.submit(run_one, item) for item in work]
            for future in concurrent.futures.as_completed(futures):
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve partial evidence before failing closed
                    error = f"{type(exc).__name__}: {exc}"
                    stop.set()
                    for pending in futures:
                        pending.cancel()
                    break
                with lock:
                    output["records"].append(record)
                    completed = len(output["records"])
                if args.progress_every > 0 and (
                    completed % args.progress_every == 0 or completed == len(work)
                ):
                    snapshot = memory_snapshot(args.cgroup_memory_path, milvus_pid)
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "completed": completed,
                                "total": len(work),
                                "cgroup_bytes": snapshot.get("cgroup_bytes"),
                                "milvus_rss_kib": snapshot.get("milvus_rss_kib"),
                                "milvus_anon_kib": snapshot.get("milvus_anon_kib"),
                                "milvus_threads": snapshot.get("milvus_threads"),
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
        output["memory_at_request_completion"] = memory_snapshot(
            args.cgroup_memory_path,
            milvus_pid,
        )
        if error is None and not guard_error and args.post_run_observe_seconds:
            observation_deadline = time.monotonic() + args.post_run_observe_seconds
            while time.monotonic() < observation_deadline and not stop.is_set():
                time.sleep(min(1.0, observation_deadline - time.monotonic()))
    finally:
        sampler_done.set()
        sampler.join(timeout=args.memory_sample_interval_seconds + 1.0)
        output["ended_at"] = time.time()
        output["elapsed_s"] = output["ended_at"] - started
        output["memory_after_observation"] = memory_snapshot(
            args.cgroup_memory_path, milvus_pid
        )
        output["records"].sort(
            key=lambda row: (row["wave"], row["baseline_record_index"])
        )
        output["completed_requests"] = len(output["records"])
        latencies = [record["elapsed_s"] for record in output["records"]]
        denominator = len(output["records"])
        ordered_exact_count = sum(
            record["parity"]["ordered_url_exact"] for record in output["records"]
        )
        top1_exact_count = sum(
            record["parity"]["top1_url_exact"] for record in output["records"]
        )
        overlap_pass_count = sum(
            record["parity"]["topk_set_overlap_ratio"] >= args.min_topk_overlap_ratio
            for record in output["records"]
        )
        score_pass_count = sum(
            record["parity"]["common_url_scores_within_tolerance"]
            for record in output["records"]
        )
        ordered_exact_ratio = ratio(ordered_exact_count, denominator)
        top1_exact_ratio = ratio(top1_exact_count, denominator)
        semantic_parity_pass = (
            denominator == len(work)
            and ordered_exact_ratio >= args.min_ordered_url_exact_ratio
            and top1_exact_ratio >= args.min_top1_url_exact_ratio
            and overlap_pass_count == denominator
            and score_pass_count == denominator
        )
        output["latency_s"] = {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
            "max": max(latencies) if latencies else None,
        }
        output["parity"] = {
            "semantic_pass": semantic_parity_pass,
            "ordered_url_exact_count": ordered_exact_count,
            "ordered_url_exact_ratio": ordered_exact_ratio,
            "top1_url_exact_count": top1_exact_count,
            "top1_url_exact_ratio": top1_exact_ratio,
            "topk_overlap_pass_count": overlap_pass_count,
            "score_tolerance_pass_count": score_pass_count,
        }
        numeric_cgroup = [
            sample["cgroup_bytes"]
            for sample in output["memory_samples"]
            if isinstance(sample.get("cgroup_bytes"), int)
        ]
        numeric_rss = [
            sample["milvus_rss_kib"]
            for sample in output["memory_samples"]
            if isinstance(sample.get("milvus_rss_kib"), int)
        ]
        numeric_anon = [
            sample["milvus_anon_kib"]
            for sample in output["memory_samples"]
            if isinstance(sample.get("milvus_anon_kib"), int)
        ]
        numeric_threads = [
            sample["milvus_threads"]
            for sample in output["memory_samples"]
            if isinstance(sample.get("milvus_threads"), int)
        ]
        output["memory_peaks"] = {
            "cgroup_bytes": max(numeric_cgroup) if numeric_cgroup else None,
            "milvus_rss_kib": max(numeric_rss) if numeric_rss else None,
            "milvus_anon_kib": max(numeric_anon) if numeric_anon else None,
            "milvus_threads": max(numeric_threads) if numeric_threads else None,
        }
        if guard_error and error is None:
            error = guard_error[0]
        if error:
            output["status"] = "fail"
            output["error"] = error
        elif denominator != len(work):
            output["status"] = "fail"
            output["error"] = "incomplete request set"
        elif not semantic_parity_pass:
            output["status"] = "fail"
            output["error"] = "retrieval parity threshold failed"
        else:
            output["status"] = "pass"
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    print(
        json.dumps(
            {
                key: output[key]
                for key in (
                    "status",
                    "request_count",
                    "completed_requests",
                    "elapsed_s",
                    "latency_s",
                    "parity",
                    "memory_before",
                    "memory_at_request_completion",
                    "memory_after_observation",
                    "memory_peaks",
                )
                if key in output
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
