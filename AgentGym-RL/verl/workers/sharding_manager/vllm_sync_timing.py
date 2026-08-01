# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
from contextlib import contextmanager


EVENT_TYPE = "agentmemory_vllm_weight_sync_timing_v1"
LOG_PREFIX = "agentmemory_vllm_sync_timing="


class SyncPhaseTimer:
    """Accumulate wall time for disjoint phases of one weight synchronization."""

    def __init__(self, enabled=False, clock=None):
        self.enabled = bool(enabled)
        self._clock = clock or time.perf_counter
        self._started_at = self._clock() if self.enabled else None
        self._timing_s = {}

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        started_at = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - started_at
            self._timing_s[name] = self._timing_s.get(name, 0.0) + elapsed

    def build_event(self, **metadata):
        if not self.enabled:
            return None
        timing_s = {
            name: round(seconds, 6)
            for name, seconds in sorted(self._timing_s.items())
        }
        timing_s["total"] = round(self._clock() - self._started_at, 6)
        return {
            "event_type": EVENT_TYPE,
            **metadata,
            "timing_s": timing_s,
        }


def format_sync_timing_event(event):
    if event is None:
        return None
    return LOG_PREFIX + json.dumps(event, sort_keys=True, separators=(",", ":"))
