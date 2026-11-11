#!/bin/bash
# Run the unified EP benchmark for each backend and mode, then write results.json, results.csv
# and summary.md to EP_BENCH_LOG_DIR. All backends run even if one fails; the script exits
# non-zero if any failed. Run install-ep-backends.sh first.
#
# Each backend runs in a separate mpirun so it can load its own NCCL, using the same options
# that `--backend all` applies.
#
# Environment overrides:
#   EP_BENCH_BACKENDS     backends to run (default: "nccl mscclpp deepep")
#   EP_BENCH_MODES        benchmark modes (default: "latency throughput")
#   EP_BENCH_ARGS         workload arguments (default: DeepSeek-V3-style decode workload)
#   EP_BENCH_NP           total MPI ranks (default: GPUs on this node)
#   EP_BENCH_HOSTFILE     MPI hostfile for multi-node runs (default: unset, single node)
#   EP_BENCH_PPN          ranks per node with a hostfile (default: GPUs on this node)
#   EP_BENCH_TCP_IF       Open MPI btl_tcp_if_include value for multi-node runs
#   EP_BENCH_LOG_DIR      where logs and results are written (default: /tmp/ep-benchmark)
#   EP_BACKEND_ENV        file written by install-ep-backends.sh (default: /tmp/ep-backend-env)
#   EP_VERSIONS_FILE      versions JSON written by install-ep-backends.sh
#                         (default: /tmp/ep-backend-versions.json)
#   MASTER_ADDR / MASTER_PORT  torch.distributed rendezvous (default: 127.0.0.1 / 51703)
#   NCCL_IB_DISABLE       default 1; single-node runs use NVLink

set -euo pipefail

NUM_GPUS="$(nvidia-smi --list-gpus | wc -l)"
EP_BENCH_BACKENDS="${EP_BENCH_BACKENDS:-nccl mscclpp deepep}"
EP_BENCH_MODES="${EP_BENCH_MODES:-latency throughput}"
EP_BENCH_ARGS="${EP_BENCH_ARGS:---num-tokens 128 --hidden 7168 --num-experts 256 --num-topk 8}"
EP_BENCH_NP="${EP_BENCH_NP:-${NUM_GPUS}}"
EP_BENCH_PPN="${EP_BENCH_PPN:-${NUM_GPUS}}"
EP_BENCH_LOG_DIR="${EP_BENCH_LOG_DIR:-/tmp/ep-benchmark}"
EP_BACKEND_ENV="${EP_BACKEND_ENV:-/tmp/ep-backend-env}"
EP_VERSIONS_FILE="${EP_VERSIONS_FILE:-/tmp/ep-backend-versions.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-51703}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

DEEPEP_PYTHONPATH=""
DEEPEP_CUDA_HOME=""
if [ -f "${EP_BACKEND_ENV}" ]; then
    # shellcheck disable=SC1090
    source "${EP_BACKEND_ENV}"
fi

MPI_ARGS=(--allow-run-as-root --bind-to none -np "${EP_BENCH_NP}")
if [ -n "${EP_BENCH_HOSTFILE:-}" ]; then
    MPI_ARGS+=(-hostfile "${EP_BENCH_HOSTFILE}" -npernode "${EP_BENCH_PPN}")
fi
if [ -n "${EP_BENCH_TCP_IF:-}" ]; then
    MPI_ARGS+=(-mca btl_tcp_if_include "${EP_BENCH_TCP_IF}")
fi
for var in MASTER_ADDR MASTER_PORT NCCL_IB_DISABLE PATH LD_LIBRARY_PATH; do
    MPI_ARGS+=(-x "${var}")
done

# Options `--backend all` applies to every backend.
mode_args() {
    case "$1" in
        latency) echo "--ep-layout expert_major --combine-mode direct_send" ;;
        *) echo "--combine-mode rank_local_reduce" ;;
    esac
}

mkdir -p "${EP_BENCH_LOG_DIR}"
rm -f "${EP_BENCH_LOG_DIR}"/ep-*.log "${EP_BENCH_LOG_DIR}"/ep-*.status
if [ -f "${EP_VERSIONS_FILE}" ]; then
    cp "${EP_VERSIONS_FILE}" "${EP_BENCH_LOG_DIR}/versions.json"
fi
# Run outside the repository so Python imports the installed packages.
cd /tmp

failed=0
for mode in ${EP_BENCH_MODES}; do
    for backend in ${EP_BENCH_BACKENDS}; do
        log="${EP_BENCH_LOG_DIR}/ep-${mode}-${backend}.log"
        backend_mpi_args=(-x "CUDA_HOME=${CUDA_HOME}")
        if [ "${backend}" = deepep ]; then
            if [ -z "${DEEPEP_PYTHONPATH}" ] || [ -z "${DEEPEP_CUDA_HOME}" ]; then
                echo "ERROR: DeepEP's NCCL or CUDA is unknown; run install-ep-backends.sh first" >&2
                exit 1
            fi
            # DeepEP JIT-compiles kernels with ${CUDA_HOME}/bin/nvcc.
            backend_mpi_args=(-x "PYTHONPATH=${DEEPEP_PYTHONPATH}" -x "CUDA_HOME=${DEEPEP_CUDA_HOME}")
        fi
        echo "=== Unified EP benchmark: mode=${mode}, backend=${backend}, ranks=${EP_BENCH_NP}"
        status=0
        # shellcheck disable=SC2046,SC2086
        timeout -k 10 1800 mpirun "${MPI_ARGS[@]}" "${backend_mpi_args[@]}" \
            python -m mscclpp_benchmark.ep.bench --backend "${backend}" --mode "${mode}" \
            $(mode_args "${mode}") ${EP_BENCH_ARGS} \
            2>&1 | tee "${log}" || status=$?
        echo "${status}" > "${EP_BENCH_LOG_DIR}/ep-${mode}-${backend}.status"
        if [ "${status}" -ne 0 ]; then
            echo "ERROR: backend=${backend} mode=${mode} failed with exit code ${status}" >&2
            failed=1
        fi
    done
done

echo
echo "=== Comparison summary"
python "${SCRIPT_DIR}/summarize_ep_benchmark.py" "${EP_BENCH_LOG_DIR}" \
    --modes "${EP_BENCH_MODES}" --backends "${EP_BENCH_BACKENDS}" \
    --workload "${EP_BENCH_ARGS}" --versions-file "${EP_BENCH_LOG_DIR}/versions.json"

exit "${failed}"
