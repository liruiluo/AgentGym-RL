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
MULTITASK_SOURCE_LOCK=""
EXPECT_VERL_ROOT=0
EXPECT_SOURCE_LOCK=0
EXPECT_MULTITASK_SOURCE_LOCK=0
RESOLVE_ONLY=0
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
  if [[ ${EXPECT_MULTITASK_SOURCE_LOCK} -eq 1 ]]; then
    MULTITASK_SOURCE_LOCK=${ARG}
    EXPECT_MULTITASK_SOURCE_LOCK=0
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
    --multitask-source-lock)
      EXPECT_MULTITASK_SOURCE_LOCK=1
      ;;
    --multitask-source-lock=*)
      MULTITASK_SOURCE_LOCK=${ARG#--multitask-source-lock=}
      ;;
    --resolve-only)
      RESOLVE_ONLY=1
      ;;
  esac
done
if [[ ${EXPECT_VERL_ROOT} -ne 0 || -z ${VERL_ROOT} || ! -d ${VERL_ROOT}/verl ]]; then
  echo "launch_amg_fully_async.sh: --verl-root must name a veRL source tree" >&2
  exit 64
fi
if [[ ${EXPECT_SOURCE_LOCK} -ne 0 || ${EXPECT_MULTITASK_SOURCE_LOCK} -ne 0 ]]; then
  echo "launch_amg_fully_async.sh: source-lock option is missing its path" >&2
  exit 64
fi
if [[ -n ${ENDPOINT_SOURCE_LOCK} && -n ${MULTITASK_SOURCE_LOCK} ]]; then
  echo "launch_amg_fully_async.sh: select exactly one source-lock format" >&2
  exit 64
fi
SOURCE_LOCK=${ENDPOINT_SOURCE_LOCK:-${MULTITASK_SOURCE_LOCK}}
if [[ -z ${SOURCE_LOCK} || ! -f ${SOURCE_LOCK} || -L ${SOURCE_LOCK} ]]; then
  echo "launch_amg_fully_async.sh: a regular endpoint or multitask source lock is required" >&2
  exit 64
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "launch_amg_fully_async.sh: jq is required to select publication training Python" >&2
  exit 69
fi
PUBLICATION_PYTHON=$(jq -er \
  '.training_runtime.python | select(type == "string" and startswith("/"))' \
  "${SOURCE_LOCK}") || {
  echo "launch_amg_fully_async.sh: publication has no absolute training_runtime.python" >&2
  exit 65
}
PUBLICATION_SITE_PACKAGES=$(jq -er \
  '.training_runtime.site_packages | select(type == "string" and startswith("/"))' \
  "${SOURCE_LOCK}") || {
  echo "launch_amg_fully_async.sh: publication has no absolute training_runtime.site_packages" >&2
  exit 65
}
if [[ ! -x ${PUBLICATION_PYTHON} ]]; then
  echo "launch_amg_fully_async.sh: publication training Python is missing" >&2
  exit 66
fi

# CUDA 13 is staged from split runtime/compiler packages on B300.  Restore the
# standard linker name and expose the matching CCCL headers before any SGLang or
# FlashInfer JIT worker starts; otherwise initialization fails late after all
# model shards have already loaded.
CUDA13_TOOLKIT_ROOT=/dev/shm/cuda-13-b300-toolkit
CUDA13_CUDART_VERSIONED=${CUDA13_TOOLKIT_ROOT}/lib64/libcudart.so.13
CUDA13_CUDART_LINKER_NAME=${CUDA13_TOOLKIT_ROOT}/lib64/libcudart.so
CUDA13_CCCL_INCLUDE=${PUBLICATION_SITE_PACKAGES}/flashinfer/data/cccl/libcudacxx/include
if [[ ${RESOLVE_ONLY} -eq 0 ]]; then
  if [[ ! -f ${CUDA13_CUDART_VERSIONED} ]]; then
    echo "launch_amg_fully_async.sh: CUDA 13 libcudart runtime is missing" >&2
    exit 66
  fi
  if [[ ! -e ${CUDA13_CUDART_LINKER_NAME} && ! -L ${CUDA13_CUDART_LINKER_NAME} ]]; then
    ln -s libcudart.so.13 "${CUDA13_CUDART_LINKER_NAME}"
  fi
  if [[ ! -f ${CUDA13_CUDART_LINKER_NAME} ]] || \
     [[ $(readlink -f "${CUDA13_CUDART_LINKER_NAME}") != $(readlink -f "${CUDA13_CUDART_VERSIONED}") ]]; then
    echo "launch_amg_fully_async.sh: CUDA 13 libcudart linker name is invalid" >&2
    exit 66
  fi
  if [[ ! -f ${CUDA13_CCCL_INCLUDE}/nv/target ]]; then
    echo "launch_amg_fully_async.sh: CUDA 13 CCCL headers are missing" >&2
    exit 66
  fi
  export CPATH="${CUDA13_CCCL_INCLUDE}${CPATH:+:${CPATH}}"
fi

TRL_WHEEL="${OUTER_ROOT}/async_plugins/vendor/trl-0.9.6-py3-none-any.whl"
if [[ ! -f ${TRL_WHEEL} || -L ${TRL_WHEEL} ]]; then
  echo "launch_amg_fully_async.sh: locked veRL TRL wheel is missing" >&2
  exit 66
fi
export PYTHONPATH="${TRL_WHEEL}:${OUTER_ROOT}/async_plugins:${VERL_ROOT}:${OUTER_ROOT}/AgentGym/agentenv:${OUTER_ROOT}/AgentGym/agentenv-openmle-fast"
exec "${PUBLICATION_PYTHON}" -m agentmemorygym_verl.launch --outer-root "${OUTER_ROOT}" "$@"
