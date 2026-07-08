#!/usr/bin/env bash
# Build mscclpp_ep_bench: a pure-C++/MPI low-latency EP benchmark that links the
# mscclpp EP LL runtime (MoERuntime) directly. Run ON a build node (needs the
# mscclpp source tree + the installed libmscclpp.so in the torch conda env).
#
# The EP dispatch/combine symbols are only compiled into the nanobind Python
# module (mscclpp_ep_cpp.so), so we recompile the two LL EP translation units
# (moe_runtime.cc + kernels/low_latency.cu) into this binary and link the base
# libmscclpp.so. Flags mirror src/ext/ep/CMakeLists.txt.
set -euo pipefail

SRC=${MSCCLPP_SRC:-/opt/microsoft/mrc/ep/mscclpp}
EP=$SRC/src/ext/ep
CONDA_PREFIX_DIR=${CONDA_PREFIX:-$HOME/miniconda3/envs/torch}
SP=$CONDA_PREFIX_DIR/lib/python*/site-packages
MSCCLPP_LIBDIR=$(ls -d $SP/mscclpp/lib | head -1)
MSCCLPP_INCDIR=$(ls -d $SP/mscclpp/include | head -1)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
CUPTI_INC=$CUDA_HOME/targets/sbsa-linux/include
CUPTI_LIB=$CUDA_HOME/targets/sbsa-linux/lib
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/mscclpp_ep_bench}

# HPCX / MPI (for mpi.h + libmpi). Autodetect if not loaded.
if ! command -v mpicxx >/dev/null 2>&1; then
  HPCX=$(ls -d /opt/hpcx-* 2>/dev/null | head -1)
  [ -n "$HPCX" ] && source "$HPCX/hpcx-init.sh" && hpcx_load
fi
MPI_INC=$(mpicxx --showme:incdirs 2>/dev/null | tr ' ' '\n' | sed 's/^/-I/' | tr '\n' ' ' || true)
MPI_LIB=$(mpicxx --showme:libdirs 2>/dev/null | tr ' ' '\n' | sed 's/^/-L/' | tr '\n' ' ' || true)

INCLUDES=(
  -I"$EP"
  -I"$EP/ht"
  -I"$SRC/include"
  -I"$SRC/src/core/include"
  -I"$SRC/src/ext/include"
  -I"$MSCCLPP_INCDIR"
  -I"$CUPTI_INC"
)
DEFINES=(-DMSCCLPP_USE_CUDA -DEP_DISPATCH_NCCLEP -DNUM_MAX_NVL_PEERS=4)
NVCCFLAGS=(
  -O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda
  -arch=sm_100 -Xcompiler -fPIC
)

echo "== compiling EP translation units + harness =="
set -x
# moe_runtime.cc is a CUDA translation unit (uses device intrinsics via
# gpu_data_types.hpp); force CUDA language with -x cu, as the CMake build does.
nvcc "${NVCCFLAGS[@]}" "${DEFINES[@]}" "${INCLUDES[@]}" $MPI_INC \
  -x cu -c "$EP/moe_runtime.cc" -o /tmp/moe_runtime.o
nvcc "${NVCCFLAGS[@]}" "${DEFINES[@]}" "${INCLUDES[@]}" $MPI_INC \
  -c "$EP/kernels/low_latency.cu" -o /tmp/low_latency.o
nvcc "${NVCCFLAGS[@]}" "${DEFINES[@]}" "${INCLUDES[@]}" $MPI_INC \
  -c "$HERE/mscclpp_ep_bench.cu" -o /tmp/mscclpp_ep_bench.o

nvcc "${NVCCFLAGS[@]}" /tmp/mscclpp_ep_bench.o /tmp/moe_runtime.o /tmp/low_latency.o \
  -o "$OUT" \
  -L"$MSCCLPP_LIBDIR" -lmscclpp \
  -L"$CUPTI_LIB" -lcupti \
  $MPI_LIB -lmpi \
  -lcuda -lcudart \
  -Xlinker -rpath -Xlinker "$MSCCLPP_LIBDIR" \
  -Xlinker -rpath -Xlinker "$CUPTI_LIB"
set +x
echo "== built $OUT =="
