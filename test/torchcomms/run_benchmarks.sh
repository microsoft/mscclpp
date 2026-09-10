#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Benchmark MSCCL++ allreduce via TorchComms and generate report + figures.
#
# Output:
#   bench_results/torchcomms_raw.json  — raw benchmark data
#   bench_results/report.txt           — formatted table
#   bench_results/latency.png          — latency vs message size
#   bench_results/bandwidth.png        — bus bandwidth vs message size
#
# Prerequisites:
#   - TorchComms backend built: ./build_torchcomm.sh
#   - Conda env activated with torchcomms, matplotlib
#   - TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP set (build_torchcomm.sh prints this)
#
# Usage:
#   ./test/torchcomms/run_benchmarks.sh
#   ./test/torchcomms/run_benchmarks.sh --nproc 2
#   ./test/torchcomms/run_benchmarks.sh --iters 500 --warmup 50

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/bench_results"

NPROC=8
WARMUP=20
ITERS=200
DTYPE="fp32"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc) NPROC="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --outdir) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUT_DIR}"

# Find the TorchComms backend .so
SO_FILE=$(find "${REPO_ROOT}/build-torchcomm/lib" -name "_comms_mscclpp*.so" 2>/dev/null | head -1)
if [[ -z "${SO_FILE}" ]]; then
    echo "ERROR: _comms_mscclpp .so not found. Run ./build_torchcomm.sh first."
    exit 1
fi
export TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP="${SO_FILE}"

TORCHCOMMS_JSON="${OUT_DIR}/torchcomms_raw.json"

echo "=== MSCCL++ AllReduce Benchmark ==="
echo "  GPUs:    ${NPROC}"
echo "  Warmup:  ${WARMUP}"
echo "  Iters:   ${ITERS}"
echo "  Dtype:   ${DTYPE}"
echo "  Output:  ${OUT_DIR}/"
echo ""

# Run TorchComms allreduce benchmark
echo "Benchmarking MSCCL++ via TorchComms..."
torchrun --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/bench_torchcomms.py" \
    --warmup "${WARMUP}" --iters "${ITERS}" --dtype "${DTYPE}" \
    --json-output "${TORCHCOMMS_JSON}" \
    2>/dev/null
echo ""

# Generate report and figures
echo "Generating report and figures..."
python3 "${SCRIPT_DIR}/bench_report.py" \
    --torchcomms-json "${TORCHCOMMS_JSON}" \
    --nproc "${NPROC}" \
    --outdir "${OUT_DIR}"

echo ""
echo "=== Report ==="
cat "${OUT_DIR}/report.txt"
echo ""
echo "Figures: ${OUT_DIR}/latency.png, ${OUT_DIR}/bandwidth.png"
