#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_ZOO_ROOT=${RKNN3_MODEL_ZOO_ROOT:-${HOME}/rknn_proj/rknn3-model-zoo-1.1.0}
TOOLCHAIN_ROOT=${RK1828_TOOLCHAIN_ROOT:-${HOME}/sdk/rk1828/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu}
TOOLCHAIN_PREFIX=${GCC_COMPILER:-${TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu}
CMAKE_BIN=${CMAKE_BIN:-${HOME}/ProgramFiles/anaconda3/envs/tools/bin/cmake}
BUILD_DIR=${REPO_ROOT}/build/rknn_llm_ppl_eval_rk3588_aarch64
INSTALL_DIR=${BUILD_DIR}/install

if [[ ! -d "${MODEL_ZOO_ROOT}/3rdparty/rknpu3" ]]; then
  echo "Invalid RKNN3_MODEL_ZOO_ROOT: ${MODEL_ZOO_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${TOOLCHAIN_PREFIX}-g++" ]]; then
  echo "Cross compiler not found: ${TOOLCHAIN_PREFIX}-g++" >&2
  exit 2
fi
if [[ ! -x "${CMAKE_BIN}" ]]; then
  echo "CMake not found: ${CMAKE_BIN}" >&2
  exit 2
fi

"${CMAKE_BIN}" -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DRKNN3_MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT}" \
  -DTARGET_SOC=rk3588 \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER="${TOOLCHAIN_PREFIX}-gcc" \
  -DCMAKE_CXX_COMPILER="${TOOLCHAIN_PREFIX}-g++" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel 4
"${CMAKE_BIN}" --install "${BUILD_DIR}"

echo "Built: ${INSTALL_DIR}/rknn_llm_ppl_eval"
