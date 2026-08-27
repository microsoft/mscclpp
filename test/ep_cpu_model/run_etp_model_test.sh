#!/usr/bin/env bash
# Build and run the CPU-side ETP dataflow model (no GPU, no nvcc required).
#
#   CUDA_INCLUDE=/usr/local/cuda/include ./test/ep_cpu_model/run_etp_model_test.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${here}/../.." && pwd)"
cuda_include="${CUDA_INCLUDE:-/usr/local/cuda/include}"
out="${OUT:-/tmp/etp_model_test}"
extra=()
for dir in ${EXTRA_INCLUDE_DIRS:-}; do extra+=("-I${dir}"); done
g++ -std=c++17 -O1 -Wall \
  -I "${root}/include" -I "${root}/src/ext/ep/include" -I "${root}/src/ext/ep" \
  -I "${cuda_include}" "${extra[@]}" \
  "${here}/etp_model_test.cc" -o "${out}"
"${out}"
