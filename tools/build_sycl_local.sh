#!/usr/bin/env bash
# Local helper: build Strata's SYCL engine on this Fedora box without Docker.
# Uses the locally extracted oneAPI Level-Zero loader and ninja (see docs/LOCAL_INTEL_UHD770.md).
set -euo pipefail

ONEAPI=/opt/intel/oneapi
ZE_DIR="${ZE_DIR:-/tmp/opencode/ze/usr/lib64}"
NINJA_DIR="${NINJA_DIR:-/tmp/opencode/ninja/usr/bin}"
GGML_DIR="${GGML_DIR:-/home/otherdrums/llama.cpp}"

source "$ONEAPI/setvars.sh" >/dev/null 2>&1
export PATH="$NINJA_DIR:$PATH"
export LD_LIBRARY_PATH="$ZE_DIR:${LD_LIBRARY_PATH:-}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO/build-sycl}"

cmake -S "$REPO/sycl" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_GGML_DIR="$GGML_DIR" \
  -DSTRATA_SYCL_AOT="${AOT:-}"

cmake --build "$BUILD_DIR" -j "${JOBS:-8}" "$@"
echo "BUILD EXIT $?"
