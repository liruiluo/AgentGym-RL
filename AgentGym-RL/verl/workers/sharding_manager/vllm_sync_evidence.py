"""Bounded, structured evidence for actor-to-vLLM weight synchronization."""

import hashlib
import json
import os
from datetime import datetime, timezone


SCHEMA_VERSION = 2
EVENT_TYPE = "official_vllm_hf_apply_model_sync"
LEGACY_EVENT_TYPE = "official_vllm_file_backed_apply_model_sync"


def _evenly_spaced_indices(length, limit):
    if length <= 0 or limit <= 0:
        return []
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [0]
    return sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)})


def _json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_tensor(name, tensor, max_values_per_tensor):
    value = tensor.detach() if hasattr(tensor, "detach") else tensor
    shape = [int(size) for size in value.shape]
    numel = int(value.numel())
    positions = _evenly_spaced_indices(numel, max_values_per_tensor)
    if positions:
        flat = value.reshape(-1)
        selected = flat[positions]
        if hasattr(selected, "detach"):
            selected = selected.detach()
        if hasattr(selected, "cpu"):
            selected = selected.cpu()
        values = selected.tolist()
        if not isinstance(values, list):
            values = [values]
    else:
        values = []
    return {
        "name": str(name),
        "shape": shape,
        "dtype": str(value.dtype),
        "numel": numel,
        "positions": positions,
        "values": values,
    }


class StreamingTensorFingerprint:
    """Build the bounded fingerprint while tensors are materialized one at a time."""

    def __init__(self, names, max_tensors=64, max_values_per_tensor=1024):
        self.names = sorted(str(name) for name in names)
        if len(self.names) != len(set(self.names)):
            raise ValueError("Streaming tensor fingerprint names must be unique")
        selected_indices = _evenly_spaced_indices(len(self.names), max_tensors)
        self.selected_names = [self.names[index] for index in selected_indices]
        self._expected = set(self.names)
        self._selected = set(self.selected_names)
        self.max_tensors = int(max_tensors)
        self.max_values_per_tensor = int(max_values_per_tensor)
        self._observed = set()
        self._samples = {}

    def observe(self, name, tensor):
        name = str(name)
        if name not in self._expected:
            raise RuntimeError(f"Unexpected streamed tensor name: {name}")
        if name in self._observed:
            raise RuntimeError(f"Streamed tensor name was observed twice: {name}")
        self._observed.add(name)
        if name in self._selected:
            self._samples[name] = _sample_tensor(
                name, tensor, self.max_values_per_tensor
            )

    def finalize(self):
        missing = sorted(set(self.names) - self._observed)
        if missing:
            raise RuntimeError(
                f"Streaming tensor fingerprint did not observe all tensors: {missing[:8]}"
            )
        samples = [self._samples[name] for name in self.selected_names]
        sampled_value_count = sum(len(sample["values"]) for sample in samples)
        digest_payload = {
            "total_tensor_count": len(self.names),
            "samples": samples,
        }
        return {
            "algorithm": "sha256-bounded-even-samples-v1",
            "sha256": _json_sha256(digest_payload),
            "total_tensor_count": len(self.names),
            "sampled_tensor_count": len(samples),
            "sampled_value_count": sampled_value_count,
            "max_tensors": self.max_tensors,
            "max_values_per_tensor": self.max_values_per_tensor,
            "sampled_tensor_names": self.selected_names,
        }


def extract_streamed_source_fingerprint(apply_model_results):
    """Extract one identical streamed-source fingerprint from executor results."""
    results = flatten_apply_model_results(apply_model_results)
    fingerprints = []
    for result in results:
        fingerprint = result.pop("_streamed_source_before", None)
        if not isinstance(fingerprint, dict) or not fingerprint.get("sha256"):
            raise RuntimeError(
                "direct_inproc_streaming result is missing its source fingerprint"
            )
        fingerprints.append(fingerprint)
    if not fingerprints:
        raise RuntimeError(
            "direct_inproc_streaming returned no source fingerprints"
        )
    first = fingerprints[0]
    if any(fingerprint != first for fingerprint in fingerprints[1:]):
        raise RuntimeError(
            "direct_inproc_streaming executor results disagree on source fingerprint"
        )
    return first


def bounded_tensor_fingerprint(named_tensors, max_tensors=64, max_values_per_tensor=1024):
    """Hash deterministic samples without copying or hashing the full model."""
    items = sorted(list(named_tensors), key=lambda item: item[0])
    fingerprint = StreamingTensorFingerprint(
        (name for name, _ in items),
        max_tensors=max_tensors,
        max_values_per_tensor=max_values_per_tensor,
    )
    for name, tensor in items:
        fingerprint.observe(name, tensor)
    return fingerprint.finalize()


def flatten_apply_model_results(value):
    """Normalize vLLM executor result containers while rejecting opaque results."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(flatten_apply_model_results(item))
        return flattened
    raise RuntimeError(f"Unsupported vLLM apply_model result type: {type(value).__name__}")


def _target_aggregate_sha256(targets):
    return _json_sha256([
        {
            "engine_result_index": target["engine_result_index"],
            "model": target["model"],
            "sha256": target["target_after"]["sha256"],
        }
        for target in targets
    ])


def build_sync_event(*, rank, pid, global_steps, sync_sequence, sync_id,
                     source_before, apply_model_results, previous_event=None,
                     transport="file"):
    targets = []
    for result_index, result in enumerate(flatten_apply_model_results(apply_model_results)):
        target = dict(result)
        target["engine_result_index"] = result_index
        targets.append(target)

    target_aggregate_sha256 = _target_aggregate_sha256(targets)
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_TYPE,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "rank": int(rank),
        "pid": int(pid),
        "global_steps": global_steps,
        "sync_sequence": int(sync_sequence),
        "sync_id": sync_id,
        "transport": str(transport),
        "source_before": source_before,
        "targets_after": targets,
        "target_aggregate_sha256": target_aggregate_sha256,
        "previous_sync_id": previous_event.get("sync_id") if previous_event else None,
        "source_changed_from_previous": (
            source_before["sha256"] != previous_event["source_before"]["sha256"]
            if previous_event else None
        ),
        "target_changed_from_previous": (
            target_aggregate_sha256 != previous_event["target_aggregate_sha256"]
            if previous_event else None
        ),
    }
    validate_sync_event(event, previous_event=previous_event, require_change=False)
    return event


def validate_sync_event(event, previous_event=None, require_change=False):
    schema = event.get("schema_version")
    event_type = event.get("event_type")
    valid_schema = (
        (schema == SCHEMA_VERSION and event_type == EVENT_TYPE)
        or (schema == 1 and event_type == LEGACY_EVENT_TYPE)
    )
    if not valid_schema:
        raise RuntimeError("Unexpected vLLM sync evidence schema")
    if schema == SCHEMA_VERSION and event.get("transport") not in {
        "file",
        "direct_inproc",
        "direct_inproc_streaming",
    }:
        raise RuntimeError("vLLM sync evidence has an invalid transport")
    if event.get("global_steps") is None:
        raise RuntimeError("vLLM sync evidence is missing prompts.meta_info global_steps")
    if not isinstance(event.get("sync_sequence"), int) or event["sync_sequence"] < 1:
        raise RuntimeError("vLLM sync evidence has an invalid sync_sequence")
    if not event.get("sync_id"):
        raise RuntimeError("vLLM sync evidence is missing sync_id")

    source_sha = event.get("source_before", {}).get("sha256")
    if not source_sha:
        raise RuntimeError("vLLM sync evidence is missing source fingerprint")
    targets = event.get("targets_after") or []
    if not targets:
        raise RuntimeError("vLLM sync evidence contains no target results")
    for target in targets:
        if target.get("sync_id") != event["sync_id"]:
            raise RuntimeError("vLLM target result is not bound to the current sync_id")
        if target.get("source_fingerprint_sha256") != source_sha:
            raise RuntimeError("vLLM target result is not bound to the current source fingerprint")
        if target.get("loaded_source_fingerprint_sha256") != source_sha:
            raise RuntimeError("vLLM engine did not read back the expected source fingerprint")
        if int(target.get("loaded_count", -1)) <= 0:
            raise RuntimeError("vLLM target reported no loaded weights")
        if not target.get("target_after", {}).get("sha256"):
            raise RuntimeError("vLLM sync evidence is missing target fingerprint")

    if previous_event is not None:
        if event["sync_sequence"] != previous_event["sync_sequence"] + 1:
            raise RuntimeError("vLLM sync_sequence is not contiguous")
        if event.get("previous_sync_id") != previous_event.get("sync_id"):
            raise RuntimeError("vLLM sync event is not linked to its predecessor")
        current_step = event["global_steps"]
        previous_step = previous_event["global_steps"]
        if isinstance(current_step, int) and isinstance(previous_step, int) and current_step <= previous_step:
            raise RuntimeError("vLLM sync global_steps did not advance")
        if require_change:
            if event.get("source_changed_from_previous") is not True:
                raise RuntimeError("Actor source fingerprint did not change before post-update generation")
            if event.get("target_changed_from_previous") is not True:
                raise RuntimeError("vLLM target fingerprint did not change before post-update generation")
    elif require_change:
        raise RuntimeError("Post-update vLLM sync change requires a previous sync event")
    return event


def append_and_readback_event(path, event):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    readback = read_last_event(path)
    if (readback.get("sync_id"), readback.get("sync_sequence")) != (
            event.get("sync_id"), event.get("sync_sequence")):
        raise RuntimeError("vLLM sync evidence readback does not match the appended event")
    return readback


def read_last_event(path):
    last_line = None
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last_line = line
    if last_line is None:
        raise RuntimeError(f"vLLM sync evidence file is empty: {path}")
    return json.loads(last_line)
