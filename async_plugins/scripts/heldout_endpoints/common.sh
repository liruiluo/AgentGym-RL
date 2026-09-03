#!/usr/bin/env bash

# Shared fail-closed helpers for the four CAMG native held-out endpoints.
# This file is sourced by the route launchers; it never starts a service.

heldout_die() {
  printf 'CAMG held-out launcher error: %s\n' "$*" >&2
  exit 64
}

heldout_require_env() {
  local name=$1
  [[ -n "${!name:-}" ]] || heldout_die "missing environment variable $name"
  # POSIX process environments cannot contain NUL; explicitly reject the two
  # line separators that could otherwise alter logs or generated argv files.
  [[ "${!name}" != *$'\n'* && "${!name}" != *$'\r'* ]] \
    || heldout_die "unsafe environment variable $name"
}

heldout_python() {
  printf '%s\n' "${HELDOUT_RUNTIME_PYTHON:-python3}"
}

heldout_sha256() {
  local path=$1 python
  python=$(heldout_python)
  "$python" - "$path" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 << 20), b""):
        digest.update(block)
print(digest.hexdigest())
PY
}

heldout_assert_file() {
  local path=$1 expected=$2 label=$3
  [[ "$path" = /* && -f "$path" && ! -L "$path" ]] \
    || heldout_die "$label is not an absolute regular file: $path"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
    || heldout_die "$label expected SHA-256 is invalid"
  local observed
  observed=$(heldout_sha256 "$path")
  [[ "$observed" == "$expected" ]] \
    || heldout_die "$label SHA-256 mismatch: $observed != $expected"
}

heldout_assert_executable() {
  local path=$1 label=$2
  # Virtual-environment Python entrypoints are normally symlinks.  Bash's
  # -f/-x checks follow the link and still reject missing, broken, directory,
  # and non-executable targets.
  [[ "$path" = /* && -f "$path" && -x "$path" ]] \
    || heldout_die "$label is not an absolute executable file: $path"
}

heldout_assert_source() {
  local root=$1 expected=$2 label=$3 observed dirty
  [[ "$root" = /* && -d "$root" && ! -L "$root" ]] \
    || heldout_die "$label source root is invalid: $root"
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] \
    || heldout_die "$label source commit is invalid"
  observed=$(git -c "safe.directory=$root" -C "$root" rev-parse HEAD 2>/dev/null) \
    || heldout_die "cannot read $label source HEAD"
  [[ "$observed" == "$expected" ]] \
    || heldout_die "$label source commit mismatch: $observed != $expected"
  dirty=$(git -c "safe.directory=$root" -C "$root" status --porcelain=v1 --untracked-files=all 2>/dev/null) \
    || heldout_die "cannot inspect $label source"
  [[ -z "$dirty" ]] || heldout_die "$label source tree is dirty"
}

heldout_assert_base_contract() {
  local expected_route=$1 expected_port=$2
  local name
  for name in \
    CAMG_HELDOUT_ROUTE_ID CAMG_HELDOUT_ROLE CAMG_HELDOUT_TASK_COUNT \
    CAMG_HELDOUT_ENDPOINT CAMG_HELDOUT_ROUTE_ATTESTATION_SHA256 \
    CAMG_HELDOUT_SOURCE_OUTER_ROOT CAMG_HELDOUT_SOURCE_OUTER_COMMIT \
    CAMG_HELDOUT_SOURCE_INNER_ROOT CAMG_HELDOUT_SOURCE_INNER_COMMIT \
    AMG_MULTITASK_RUN_ID AMG_MULTITASK_ENDPOINT_RUN_DIR \
    AMG_MULTITASK_ENDPOINT_HOST AMG_MULTITASK_ENDPOINT_PORT \
    AMG_MULTITASK_PARENT_PID AMG_MULTITASK_PARENT_START_TICKS; do
    heldout_require_env "$name"
  done
  [[ "$CAMG_HELDOUT_ROUTE_ID" == "$expected_route" ]] \
    || heldout_die "route mismatch: $CAMG_HELDOUT_ROUTE_ID != $expected_route"
  [[ "$CAMG_HELDOUT_ROLE" == heldout ]] || heldout_die "role must be heldout"
  [[ "$CAMG_HELDOUT_TASK_COUNT" =~ ^[1-9][0-9]*$ ]] \
    || heldout_die "task count must be a positive integer"
  [[ "$AMG_MULTITASK_ENDPOINT_HOST" == 127.0.0.1 ]] \
    || heldout_die "endpoint host must be 127.0.0.1"
  [[ "$AMG_MULTITASK_ENDPOINT_PORT" == "$expected_port" ]] \
    || heldout_die "endpoint port mismatch"
  [[ "$CAMG_HELDOUT_ENDPOINT" == "http://127.0.0.1:$expected_port" ]] \
    || heldout_die "verified endpoint URL mismatch"
  [[ "$CAMG_HELDOUT_ROUTE_ATTESTATION_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || heldout_die "route attestation SHA-256 is invalid"
  [[ "$AMG_MULTITASK_ENDPOINT_RUN_DIR" = /* && ! -L "$AMG_MULTITASK_ENDPOINT_RUN_DIR" ]] \
    || heldout_die "endpoint run directory must be an absolute non-symlink path"
  heldout_assert_source \
    "$CAMG_HELDOUT_SOURCE_OUTER_ROOT" "$CAMG_HELDOUT_SOURCE_OUTER_COMMIT" outer
  heldout_assert_source \
    "$CAMG_HELDOUT_SOURCE_INNER_ROOT" "$CAMG_HELDOUT_SOURCE_INNER_COMMIT" inner
  [[ "$(basename -- "$CAMG_HELDOUT_SOURCE_OUTER_ROOT")" == AgentGym-RL ]] \
    || heldout_die "outer source root must be the AgentGym-RL git checkout"
  [[ "$CAMG_HELDOUT_SOURCE_INNER_ROOT" == "$CAMG_HELDOUT_SOURCE_OUTER_ROOT/AgentGym" ]] \
    || heldout_die "inner source root must be the outer checkout's AgentGym submodule"
}

heldout_assert_asset_env() {
  local stem=$1 label=$2
  local path_name="CAMG_HELDOUT_ASSET_${stem}_PATH"
  local sha_name="CAMG_HELDOUT_ASSET_${stem}_SHA256"
  heldout_require_env "$path_name"
  heldout_require_env "$sha_name"
  heldout_assert_file "${!path_name}" "${!sha_name}" "$label"
}

heldout_assert_parent() {
  local python
  python=$(heldout_python)
  "$python" - "$AMG_MULTITASK_PARENT_PID" "$AMG_MULTITASK_PARENT_START_TICKS" <<'PY'
from pathlib import Path
import sys

pid = int(sys.argv[1])
expected = str(sys.argv[2])
raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
observed = raw[raw.rfind(")") + 2 :].split()[19]
if observed != expected:
    raise SystemExit(f"parent start-ticks mismatch: {observed} != {expected}")
PY
}
