#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTER_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

if [[ -n ${PYTHONPATH+x} ]]; then
  echo "launch_amg_fully_async.sh: PYTHONPATH is an identity conflict" >&2
  exit 64
fi

# The launcher's Python is part of the selected, sealed publication.  Parse only
# its path here; launch.py revalidates the complete source lock, runtime bundle,
# model bytes, and sys.executable identity before Hydra or training can run.
VERL_ROOT=""
ENDPOINT_SOURCE_LOCK=""
EXPECT_VERL_ROOT=0
EXPECT_SOURCE_LOCK=0
for ARG in "$@"; do
  if [[ ${EXPECT_VERL_ROOT} -eq 1 ]]; then
    VERL_ROOT=${ARG}
    EXPECT_VERL_ROOT=0
    continue
  fi
  if [[ ${EXPECT_SOURCE_LOCK} -eq 1 ]]; then
    ENDPOINT_SOURCE_LOCK=${ARG}
    EXPECT_SOURCE_LOCK=0
    continue
  fi
  case ${ARG} in
    --verl-root)
      EXPECT_VERL_ROOT=1
      ;;
    --verl-root=*)
      VERL_ROOT=${ARG#--verl-root=}
      ;;
    --endpoint-source-lock)
      EXPECT_SOURCE_LOCK=1
      ;;
    --endpoint-source-lock=*)
      ENDPOINT_SOURCE_LOCK=${ARG#--endpoint-source-lock=}
      ;;
  esac
done
if [[ ${EXPECT_VERL_ROOT} -ne 0 || -z ${VERL_ROOT} || ! -d ${VERL_ROOT}/verl ]]; then
  echo "launch_amg_fully_async.sh: --verl-root must name a veRL source tree" >&2
  exit 64
fi
if [[ ${EXPECT_SOURCE_LOCK} -ne 0 || -z ${ENDPOINT_SOURCE_LOCK} || ! -f ${ENDPOINT_SOURCE_LOCK} || -L ${ENDPOINT_SOURCE_LOCK} ]]; then
  echo "launch_amg_fully_async.sh: --endpoint-source-lock must name a regular publication lock" >&2
  exit 64
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "launch_amg_fully_async.sh: jq is required to select publication training Python" >&2
  exit 69
fi
PUBLICATION_PYTHON=$(jq -er \
  '.training_runtime.python | select(type == "string" and startswith("/"))' \
  "${ENDPOINT_SOURCE_LOCK}") || {
  echo "launch_amg_fully_async.sh: publication has no absolute training_runtime.python" >&2
  exit 65
}
if [[ ! -x ${PUBLICATION_PYTHON} ]]; then
  echo "launch_amg_fully_async.sh: publication training Python is missing" >&2
  exit 66
fi

export PYTHONPATH="${OUTER_ROOT}/async_plugins:${VERL_ROOT}:${OUTER_ROOT}/AgentGym/agentenv:${OUTER_ROOT}/AgentGym/agentenv-openmle-fast"
exec "${PUBLICATION_PYTHON}" -m agentmemorygym_verl.launch --outer-root "${OUTER_ROOT}" "$@"
