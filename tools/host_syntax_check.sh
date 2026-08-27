#!/usr/bin/env bash
# Best-effort host-side syntax/type check of the EP headers without nvcc.
#
# g++ cannot parse `<<<...>>>` launches, so the sources are copied to a scratch
# directory with the launch configurations stripped, and a small shim supplies
# the nvcc builtins used by host-visible code. Device-only bodies guarded by
# MSCCLPP_BULK_AVAILABLE are compiled out, so this is NOT a substitute for a
# real nvcc build; it catches signature, name-lookup and host-path errors only.
#
#   CUDA_INCLUDE=/usr/local/cuda/include ./tools/host_syntax_check.sh
#
# Known pre-existing failure (also on the unmodified feature/ep base):
#   dispatch/common.cuh -> common/quantization.cuh uses mscclpp::to<>, which
#   gpu_data_types.hpp only defines for device compilation.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_include="${CUDA_INCLUDE:-/usr/local/cuda/include}"
scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT
cp -r "${root}/src" "${scratch}/src"
python3 - "${scratch}" <<'PY'
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])
for path in (root / "src/ext/ep").rglob("*"):
    if path.suffix in (".cu", ".cuh", ".hpp", ".cc"):
        text = path.read_text()
        stripped = re.sub(r"<<<[^;]*?>>>", "", text, flags=re.S)
        if stripped != text:
            path.write_text(stripped)
PY
echo 'int main() { return 0; }' > "${scratch}/empty.cc"
status=0
for header in include/config.hpp include/expert_map.hpp common/latency.cuh combine/common.cuh dispatch/common.cuh; do
  printf '%-24s ' "${header}"
  if g++ -std=c++17 -fsyntax-only -x c++ -DMSCCLPP_DEVICE_INLINE=inline \
      -I "${root}/include" -I "${scratch}/src/ext/ep/include" -I "${scratch}/src/ext/ep" \
      -I "${cuda_include}" ${EXTRA_INCLUDE_FLAGS:-} \
      -include "${root}/tools/host_syntax_shim.hpp" \
      -include "${scratch}/src/ext/ep/${header}" "${scratch}/empty.cc" 2>"${scratch}/err"; then
    echo OK
  else
    echo FAILED
    head -8 "${scratch}/err"
    status=1
  fi
done
exit "${status}"
