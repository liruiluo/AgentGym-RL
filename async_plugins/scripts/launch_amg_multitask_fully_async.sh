#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTER_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

if [[ -n ${PYTHONPATH+x} ]]; then
  echo "launch_amg_multitask_fully_async.sh: PYTHONPATH is an identity conflict" >&2
  exit 64
fi

VERL_ROOT=
SOURCE_LOCK=
EXPECT_VERL_ROOT=0
EXPECT_SOURCE_LOCK=0
for ARG in "$@"; do
  if [[ ${EXPECT_VERL_ROOT} -eq 1 ]]; then
    VERL_ROOT=${ARG}
    EXPECT_VERL_ROOT=0
    continue
  fi
  if [[ ${EXPECT_SOURCE_LOCK} -eq 1 ]]; then
    SOURCE_LOCK=${ARG}
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
    --multitask-source-lock)
      EXPECT_SOURCE_LOCK=1
      ;;
    --multitask-source-lock=*)
      SOURCE_LOCK=${ARG#--multitask-source-lock=}
      ;;
  esac
done

if [[ ${EXPECT_VERL_ROOT} -ne 0 || -z ${VERL_ROOT} || ! -d ${VERL_ROOT}/verl ]]; then
  echo "launch_amg_multitask_fully_async.sh: --verl-root must name a veRL source tree" >&2
  exit 64
fi
if [[ ${EXPECT_SOURCE_LOCK} -ne 0 || -z ${SOURCE_LOCK} || ! -f ${SOURCE_LOCK} || -L ${SOURCE_LOCK} ]]; then
  echo "launch_amg_multitask_fully_async.sh: --multitask-source-lock must name a regular source lock" >&2
  exit 64
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "launch_amg_multitask_fully_async.sh: jq is required to select publication training Python" >&2
  exit 69
fi
PUBLICATION_PYTHON=$(jq -er \
  '.training_runtime.python | select(type == "string" and startswith("/"))' \
  "${SOURCE_LOCK}") || {
  echo "launch_amg_multitask_fully_async.sh: source lock has no absolute training_runtime.python" >&2
  exit 65
}
if [[ ! -x ${PUBLICATION_PYTHON} ]]; then
  echo "launch_amg_multitask_fully_async.sh: publication training Python is missing" >&2
  exit 66
fi

TRL_WHEEL="${OUTER_ROOT}/async_plugins/vendor/trl-0.9.6-py3-none-any.whl"
if [[ ! -f ${TRL_WHEEL} || -L ${TRL_WHEEL} ]]; then
  echo "launch_amg_multitask_fully_async.sh: locked veRL TRL wheel is missing" >&2
  exit 66
fi
export PYTHONPATH="${TRL_WHEEL}:${OUTER_ROOT}/async_plugins:${VERL_ROOT}:${OUTER_ROOT}/AgentGym/agentenv:${OUTER_ROOT}/AgentGym/agentenv-openmle-fast"
exec "${PUBLICATION_PYTHON}" -m agentmemorygym_verl.multitask_orchestrator \
  --outer-root "${OUTER_ROOT}" "$@"
