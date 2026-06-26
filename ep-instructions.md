# EP benchmark instructions and results

This file captures the setup, code changes, launch commands, and results used for the MSCCL++ EP HT and LL benchmarks in this workspace.

## Workspace

Run local commands from:

```bash
cd /home/mahdiehghazi/microbenchmarking/mscclpp
```

The EP README used as the source instruction file is:

```bash
/home/mahdiehghazi/microbenchmarking/mscclpp/src/ext/ep/README.md
```

The host list is:

```bash
/home/mahdiehghazi/microbenchmarking/mscclpp/hosts
```

Node 0 is local (`100.85.8.60`). Remote nodes are launched through SSH. Each node has 4 local GB300 GPUs, so all runs use `--nproc_per_node=4` and `MSCCLPP_EP_LOCAL_WORLD_SIZE=4`.

## Environment setup

Create and activate the Python environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch cupy-cuda13x mpi4py pybind11 blake3 sortedcontainers scikit-build-core nanobind setuptools_scm
```

Build the EP extension:

```bash
. .venv/bin/activate
CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
python -m pip install --no-build-isolation \
    --config-settings=cmake.define.MSCCLPP_EP_NUM_MAX_NVL_PEERS=4 \
    --config-settings=cmake.define.MSCCLPP_EP_DISPATCH_NCCLEP=ON \
    .
```

Notes:

- `MSCCLPP_EP_NUM_MAX_NVL_PEERS=4` matches the 4-GPU-per-node GB300/GB200-style topology.
- `MSCCLPP_EP_DISPATCH_NCCLEP=ON` was required for this checkout because EP sources reference NCCL-EP/direct-path `Config` members guarded by `EP_DISPATCH_NCCLEP`.

## Remote staging

Before multi-node runs, stage the local `.venv` and EP tests to remote nodes:

```bash
cd /home/mahdiehghazi/microbenchmarking/mscclpp

for h in $(sed -n '2,8p' hosts | awk 'NF'); do
  echo "=== staging $h ==="
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" \
    'mkdir -p /home/mahdiehghazi/microbenchmarking/mscclpp/test/python/ext'
  rsync -a --delete .venv "$h:/home/mahdiehghazi/microbenchmarking/mscclpp/"
  rsync -a test/python/ext/ep "$h:/home/mahdiehghazi/microbenchmarking/mscclpp/test/python/ext/"
done
```

For 4-node runs, use `sed -n '2,4p' hosts`; for 8-node runs, use `sed -n '2,8p' hosts`.

## Common runtime settings

Single-node GB300/GB200-style runs used:

```bash
NCCL_NET_PLUGIN=none
NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

Multi-node runs also used:

```bash
NCCL_SOCKET_IFNAME=eth0
MSCCLPP_SOCKET_IFNAME=eth0
GLOO_SOCKET_IFNAME=eth0
MSCCLPP_EP_LOCAL_WORLD_SIZE=4
```

The actual data path is NVLink/NVLS fabric via cuMem fabric IPC, not traditional IB verbs. Logs show `gpuCallocPhysical (fabric-IPC)`, `fabric_ipc=1`, and for HT `NVLS HT multicast: enabled=1`. The `NCCL_IB_HCA` value avoids NCCL probe issues, while EP traffic is handled by MSCCL++.

## Single-node baseline runs

### HT intranode, README shape

```bash
cd /home/mahdiehghazi/microbenchmarking/mscclpp
. .venv/bin/activate

MSCCLPP_EP_INTRA_DIRECT=1 \
NCCL_NET_PLUGIN=none \
NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
MSCCLPP_EP_BENCH=1 \
MSCCLPP_EP_BENCH_TOKENS=4096 \
MSCCLPP_EP_BENCH_HIDDEN=7168 \
MSCCLPP_EP_BENCH_EXPERTS=256 \
MSCCLPP_EP_BENCH_TOPK=8 \
torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port=29600 \
    test/python/ext/ep/test_intranode_multirank.py
```

Result: PASS, dispatch `358.9us` / `654.38 GB/s` aggregate, combine `387.2us` / `606.55 GB/s` aggregate.

### LL intranode, README shape

```bash
cd /home/mahdiehghazi/microbenchmarking/mscclpp
. .venv/bin/activate

NCCL_NET_PLUGIN=none \
NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
MSCCLPP_EP_BENCH=1 \
MSCCLPP_EP_BENCH_TOKENS=128 \
MSCCLPP_EP_BENCH_HIDDEN=7168 \
MSCCLPP_EP_BENCH_EXPERTS=256 \
MSCCLPP_EP_BENCH_TOPK=8 \
timeout 600s torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port=29601 \
    test/python/ext/ep/test_low_latency_multirank.py
```

Result: PASS, dispatch `41.9us` / `1442.76 GB/s` aggregate, combine `35.4us` / `1705.16 GB/s` aggregate.

## Multi-node launcher pattern

Use node 0 locally and launch other node ranks over SSH. For node 0, choose the interface by routing to node 1; for all other nodes, route to node 0. This avoids node 0 advertising loopback (`127.0.0.1`) to NCCL.

Template variables to change:

- `NNODES`: `4` or `8`
- `PORT`: unique master port
- `TEST_SCRIPT`: HT or LL test script
- `EXTRA_ENV`: HT/LL-specific env

```bash
cd /home/mahdiehghazi/microbenchmarking/mscclpp

NNODES=8
PORT=29622
TEST_SCRIPT=test/python/ext/ep/test_low_latency_multirank.py
EXTRA_ENV='MSCCLPP_EP_BENCH=1 MSCCLPP_EP_BENCH_TOKENS=32 MSCCLPP_EP_BENCH_HIDDEN=8192 MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=4'

LOGDIR=/tmp/ep-run-$(date +%s)
mkdir -p "$LOGDIR"
hosts=($(head -n "$NNODES" hosts | awk 'NF'))
master=${hosts[0]}
pids=()

base_cmd='cd /home/mahdiehghazi/microbenchmarking/mscclpp && . .venv/bin/activate && IFACE=$(ip -o route get IFACE_TARGET_PLACEHOLDER | awk '\''{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}'\'') && echo using_iface=$IFACE && export NCCL_SOCKET_IFNAME=$IFACE MSCCLPP_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE NCCL_NET_PLUGIN=none NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 MSCCLPP_EP_LOCAL_WORLD_SIZE=4 EXTRA_ENV_PLACEHOLDER && timeout 1200s torchrun --nnodes=NNODES_PLACEHOLDER --nproc_per_node=4 --node_rank=NODE_RANK_PLACEHOLDER --master_addr=MASTER_ADDR_PLACEHOLDER --master_port=MASTER_PORT_PLACEHOLDER TEST_SCRIPT_PLACEHOLDER'

for rank in $(seq 0 $((NNODES - 1))); do
  h=${hosts[$rank]}
  iface_target=$master
  if [ "$rank" = 0 ]; then iface_target=${hosts[1]}; fi

  cmd=${base_cmd//NODE_RANK_PLACEHOLDER/$rank}
  cmd=${cmd//MASTER_ADDR_PLACEHOLDER/$master}
  cmd=${cmd//MASTER_PORT_PLACEHOLDER/$PORT}
  cmd=${cmd//IFACE_TARGET_PLACEHOLDER/$iface_target}
  cmd=${cmd//NNODES_PLACEHOLDER/$NNODES}
  cmd=${cmd//TEST_SCRIPT_PLACEHOLDER/$TEST_SCRIPT}
  cmd=${cmd//EXTRA_ENV_PLACEHOLDER/$EXTRA_ENV}

  if [ "$rank" = 0 ]; then
    bash -lc "$cmd" >"$LOGDIR/node${rank}-${h}.log" 2>&1 &
  else
    ssh -o BatchMode=yes "$h" "bash -lc $(printf '%q' "$cmd")" >"$LOGDIR/node${rank}-${h}.log" 2>&1 &
  fi
  pids[$rank]=$!
done

status=0
for rank in $(seq 0 $((NNODES - 1))); do
  if ! wait "${pids[$rank]}"; then status=1; fi
done

echo "exit_status=$status"
for f in "$LOGDIR"/*.log; do
  echo "===== $f ====="
  grep -E 'using_iface|cfg|Buffer|dispatch|combine|PASS|Traceback|Assertion|RuntimeError|Error|failed|NCCL|mscclpp_ep|bench' "$f" | tail -180
done
exit "$status"
```

## 4-node README-shape benchmarks

Hosts: `100.85.8.60`, `100.85.8.55`, `100.85.8.67`, `100.85.8.64`.

Workload:

| Mode | Tokens/rank | Hidden | Experts | Top-k |
|---|---:|---:|---:|---:|
| HT | 4096 | 7168 | 256 | 8 |
| LL | 128 | 7168 | 256 | 8 |

Results:

| Mode | Result | Dispatch avg | Dispatch agg BW | Combine avg | Combine agg BW |
|---|---:|---:|---:|---:|---:|
| HT | PASS | 873.9 us | 1075.06 GB/s | 600.7 us | 1564.11 GB/s |
| LL | PASS | 49.3 us | 4786.03 GB/s | 40.5 us | 5824.14 GB/s |

Saved under:

```bash
/home/mahdiehghazi/microbenchmarking/results/ep-4node-20260625
```

## Custom workload: 32 tokens, hidden 8192, 256 experts, top-k 4

The batch/token count is 32 tokens per rank:

| Scope | Ranks | Global tokens |
|---|---:|---:|
| 4 nodes | 16 | 512 |
| 8 nodes | 32 | 1024 |

### HT command settings

HT uses:

```bash
TEST_SCRIPT=test/python/ext/ep/test_internode_multirank.py
EXTRA_ENV='MSCCLPP_EP_DIRECT=1 MSCCLPP_EP_RDMA_RECV=64 MSCCLPP_EP_BENCH=1 MSCCLPP_EP_BENCH_TOKENS=32 MSCCLPP_EP_BENCH_HIDDEN=8192 MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=4'
```

For 4 nodes, `MSCCLPP_EP_RDMA_RECV=64` is not required but is safe. For 8 nodes it was required; the initial 8-node HT attempt without it failed because `num_rdma_bytes` exceeded the non-LL `INT_MAX` cap.

### LL command settings

LL uses:

```bash
TEST_SCRIPT=test/python/ext/ep/test_low_latency_multirank.py
EXTRA_ENV='MSCCLPP_EP_BENCH=1 MSCCLPP_EP_BENCH_TOKENS=32 MSCCLPP_EP_BENCH_HIDDEN=8192 MSCCLPP_EP_BENCH_EXPERTS=256 MSCCLPP_EP_BENCH_TOPK=4'
```

Before the LL hidden=8192 code change, LL failed with:

```text
RuntimeError: Failed: Assertion error .../src/ext/ep/kernels/internode_ll.cu:497 'false && "Unsupported hidden"'
```

## LL hidden=8192 code change

Files changed:

- `src/ext/ep/kernels/launch.cuh`
- `src/ext/ep/kernels/internode_ll.cu`

Change summary:

- Added `8192` to `SWITCH_HIDDEN`.
- Changed the LL PortChannel/wide layout from `(kNumWarpGroups=3, kNumWarpsPerGroup=10)` to `(2, 16)`, giving a 1024-thread block.
- The 1024-thread layout supports BF16 hidden 8192 because `8192 * 2 / sizeof(int4) = 1024` int4 chunks.
- The two-warp-group layout keeps the cooperative grid at `ceil(num_experts / 2)` blocks instead of one block per expert.

After making this code change, rebuild the EP extension and re-stage `.venv` to remote nodes.

## Custom workload final results

| Scope | Mode | Result | Dispatch avg | Dispatch agg BW | Combine avg | Combine agg BW |
|---|---|---:|---:|---:|---:|---:|
| 4 nodes, 16 ranks | HT | PASS | 140.7 us | 59.62 GB/s | 52.4 us | 160.09 GB/s |
| 4 nodes, 16 ranks | LL | PASS | 21.5 us | 1536.19 GB/s | 17.9 us | 1847.88 GB/s |
| 8 nodes, 32 ranks | HT | PASS | 171.0 us | 98.09 GB/s | 53.4 us | 313.95 GB/s |
| 8 nodes, 32 ranks | LL | PASS | 21.8 us | 3084.10 GB/s | 18.3 us | 3666.73 GB/s |

Saved result directories:

```bash
/home/mahdiehghazi/microbenchmarking/results/ep-4node-t32-h8192-e256-topk4-20260625
/home/mahdiehghazi/microbenchmarking/results/ep-8node-t32-h8192-e256-topk4-20260625
/home/mahdiehghazi/microbenchmarking/results/ep-ll-8192-fix-20260625
```

The first two directories include pre-fix LL failure logs for hidden 8192. The `ep-ll-8192-fix-20260625` directory contains post-fix LL validation logs for 4-node and 8-node runs.

## Important troubleshooting notes

- If node 0 computes its network interface by routing to itself, NCCL may advertise `127.0.0.1` and remote ranks can fail with connection refused. Route node 0 to node 1 when choosing `NCCL_SOCKET_IFNAME`.
- For 8-node HT, set `MSCCLPP_EP_RDMA_RECV=64` to keep RDMA buffer sizing under the non-LL `INT_MAX` cap.
- LL hidden sizes are compile-time cases. After this session, `8192` is supported in this checkout; arbitrary hidden sizes such as `8912` are still unsupported unless added and validated separately.
- Top-k 4 is supported by LL (`kNumMaxTopk=9`).
