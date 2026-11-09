# MoE benchmark

Measure sending tokens to experts, running SwiGLU expert computation, and
combining the results. Inputs and weights are synthetic, and routing stays fixed
during each run. This measures one MoE layer, not full-model inference.

## Quick start

From the repository root, in a CUDA 13 environment on Blackwell (SM100/SM103)
GPUs:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[cuda13,moe-benchmark]'
.venv/bin/python python/mscclpp_benchmark/moe_bench/run.py \
  --model small --input decode --routing balanced \
  --warmup 10 --iterations 10 --check
```

- Uses MSCCL++ BF16 communication and FlashInfer MXFP8/MXFP4 expert
  computation. The `moe-benchmark` extra installs the required
  `flashinfer-python==0.7.1` and `nvidia-cutlass-dsl==4.8.0`.
- Runs 16 input tokens per GPU, with eight experts and two experts per token.
- Runs on **one node by default**, using all visible GPUs. Use idle GPUs;
  `CUDA_VISIBLE_DEVICES` is respected.
- Use `--nodelist` for [multiple MNNVL-connected nodes](#multiple-nodes).
  The selected backend's topology and GPU-count limits still apply.
- Starts workers automatically. Do not wrap the command in `torchrun` or `mpirun`.
- `--check` compares every output element with an independent MXFP8/MXFP4
  reference before timing (see [Correctness check](#correctness-check)). The 10
  warmup iterations are excluded from the 10 measured iterations.

## Results

Each run creates `benchmark_results/<run-id>/`:

| File | Contents |
|---|---|
| `run.json` | Settings, hardware/software details, and completion status |
| `summary.csv` | Mean/P50/P95 latency and global token throughput |
| `iterations.csv` | Per-GPU dispatch, compute, combine, and total timings |

Use `--output-dir` to change the location. Analyze only completed runs.
CUDA-event timings exclude warmup but include launch gaps, waits for other GPUs,
and any layout conversion or quantization. Global total is the maximum GPU
duration per iteration, not the sum of GPU times.

### Correctness check

With `--check`, each GPU computes its expected output locally with plain PyTorch:
it regenerates the routed experts' weights, applies the kernels' MXFP4 weight and
MXFP8 activation rounding, runs SwiGLU, and combines routes as the selected
communication backend does. No dispatch, combine, or FlashInfer kernel is used.
Every element must satisfy

```text
|actual - expected| <= atol + rtol * |expected| + 2**-6 * sum_k |w_k * y_k|
```

where `w_k` and `y_k` are the routing weight and expert output of route `k`
(defaults `rtol=0.05`, `atol=0.01`; override with `--rtol` and `--atol`), and
non-finite values fail. The last term allows one BF16 rounding step in each
expert output and destination partial sum, which matters when partial sums
nearly cancel; a wrong or missing route still exceeds it. All GPUs agree on the result; a failure reports the
first mismatching rank, token, and column.

The output is checked three times:

1. **Initial input**, before warmup.
2. **Changed input**: the input is scaled and shifted in place, checked against a
   new reference, then restored. A fixed input could hide stale dispatch data.
3. **Last timed iteration**, against the initial reference, so the measured
   iterations (including graph replays) are covered.

`run.json` records `correctness`, with per-check maximum and mean absolute error,
mismatched elements, and `max_tolerance_fraction` (how much of its tolerance the
worst element used). If a worker fails, for any reason, `run.json` records the
lowest failing rank's exception under `failure`.

### CUDA graphs

`--graph layer` captures dispatch, expert compute, and combine as one CUDA
graph after an eager warmup, then replays it for `--check`, warmup, and every
timed iteration. Phase events are recorded inside the graph, so results keep
the dispatch, compute, combine, and total breakdown; graph launch overhead is
outside the measured total. `run.json` records the scope as
`cuda_graph_scope: dispatch_mlp_combine`. NCCL-EP supports `layer` only in
latency mode.

`--graph mlp` captures only expert compute; dispatch and combine stay eager.

## Configuration

Choose presets from [configs.yaml](configs.yaml):

| Setting | Choices |
|---|---|
| `--model` | `small`, `medium`, `large`, `scaled_small`, `scaled_medium`, `scaled_large` |
| `--input` | `decode` (16 tokens/GPU), `prefill` (1024 tokens/GPU) |
| `--routing` | `balanced`, `uniform`, `hot_pair`, `zipf`, `zipf_skewed`, `hot_experts`, `lognormal` |
| `--backend` | `mscclpp` (default), `nccl` |
| `--mode` | `latency` (default), `throughput` |
| `--graph` | `none` (default), `mlp` (expert compute only), `layer` (whole layer) |

Optional flags override YAML settings. Use `--config` for another YAML file and
`--help` for the full option list. Defaults are 10 warmup iterations, 100 measured
iterations, and seed 42.

To change dimensions or token counts, add presets under the existing `models:`
and `inputs:` sections; keep the other sections:

```yaml
models:
  custom_model:
    d_model: 5120
    intermediate_size: 10240
    num_experts: 16
    top_k: 2
    activation: swiglu
inputs:
  custom_input:
    tokens_per_rank: 128
    dtype: bf16
```

Select them with `--model custom_model --input custom_input`. Expert count must
be divisible by the total GPU count; top-k must be between 1 and
`min(9, num_experts)` (at most 8 for MSCCL++). Supported hidden dimensions are
4096, 4352, 5120, 6656, 7168, 8192, 8704, and 9216; MSCCL++ latency mode does
not support 5120, so the `medium` and `scaled_medium` presets require
`--mode throughput` or `--backend nccl`. MSCCL++ throughput supports at most 128
experts per GPU. Token counts and intermediate dimensions must be positive and
fit available memory and backend limits.

### Routing distributions

Routing is fixed for each run and identical on every launch with the same seed.
Each token always uses `top_k` distinct experts.

| Preset | Expert load |
|---|---|
| `balanced` | Every expert receives the same number of routes (within one) |
| `uniform` | Each token picks `top_k` experts uniformly at random |
| `hot_pair` | Experts 0 and 1 share 80% of the probability (a hotspot on rank 0) |
| `zipf` | Per-rank Zipf popularity (`alpha`, default 1.2): hot, warm, and tail experts, with 15% of each rank's experts unused; ranks stay roughly balanced |
| `zipf_skewed` | Zipf within ranks (20% unused) plus cross-rank skew: rank `r` carries `1 / (r + 1)^0.7` of the load, so rank 0 is heaviest |
| `hot_experts` | Bimodal: about one in sixteen experts per rank is hot, a quarter are unused, and the rest receive small loads |
| `lognormal` | Global long-tail popularity, `LogNormal(0, sigma)` per expert (`sigma`, default 0.85) |

`zipf`, `zipf_skewed`, `hot_experts`, and `lognormal` adapt the MegaMoE
benchmark's distributions (`python/mscclpp/ext/megamoe/benchmark.py` on
`feature/megamoe`). They fix the global number of routes per expert, capped at
one route per token so experts stay distinct, then spread those routes across
all GPUs' tokens. Custom presets can set `alpha` or `sigma`. A `hot_experts`
preset with `expert_ids` and `probability_mass` instead makes the listed experts
hot, which places a hotspot on any rank:

```yaml
routing:
  hot_last_rank:
    kind: hot_experts
    expert_ids: [6, 7]
    probability_mass: 0.8
```

`run.json` records the realized `assignments_per_expert`, `assignments_per_rank`,
and `rank_max_over_mean`.

### Backends and layouts

| Backend | Mode | Supported `--layout` values |
|---|---|---|
| MSCCL++ | latency | `expert_major` (default), `rank_major` |
| MSCCL++ | throughput | `token_major`, `rank_major` |
| NCCL-EP | latency | `expert_major`, `rank_major` |
| NCCL-EP | throughput | `expert_major`, `token_major` |

Unsupported combinations fail rather than selecting a fallback. NCCL-EP
throughput requires an even token count per GPU.

Expert computation uses FlashInfer CuTe grouped GEMMs with MXFP8 activations
and MXFP4 weights (`--compute-backend cutlass_flashinfer`, the only compute
backend). Each expert is a SwiGLU MLP: the first GEMM computes the up and gate
projections and applies `up * silu(gate)` in its epilogue, then the second GEMM
projects back to `d_model`. It requires Blackwell SM100/SM103, CUDA 13 PyTorch,
and model dimensions divisible by 128. Communication remains BF16; activation
quantization is included in compute timing.

### Optional NCCL-EP setup

To use `--backend nccl`, install `nccl-extensions[cu13]`. See
[NVIDIA's installation instructions](https://github.com/NVIDIA/nccl-extensions#installation).

## Multiple nodes

Multi-node runs require **MSCCL++ and one shared MNNVL GPU fabric with working
NVIDIA IMEX**. Ordinary Ethernet/RDMA is not a fallback for GPU communication.

Before deploying, ensure:

- SSH key authentication and verified host keys work for every node.
- Build and target nodes have matching Python version/ABI, CPU/GPU architectures,
  and libc. Targets also need matching GPU counts, Torch/CUDA/MSCCL++ versions,
  and the same native wheel.
- Targets have Python with virtual-environment support, a compatible NVIDIA
  driver, `nvidia-smi`, `ip`, SSH/SCP, `tar`, required system libraries (including
  libnuma), and several GB of free disk space.
- Each listed IPv4 address belongs to a local interface on its target.
  Node-to-node networking allows the master port (default 29500),
  Gloo/MSCCL++ coordination traffic, and IMEX communication.

### 1. Deploy

On the build node, after installation, replace the placeholders with your nodes'
IPv4 addresses and SSH username:

```bash
.venv/bin/python tools/deploy_moe_bench.py \
  --nodelist "<node-1-ip>,<node-2-ip>" --user "<ssh-user>"
```

This builds MSCCL++ once and copies a checked bundle to each node. Targets
install into new virtual environments without downloading packages or compiling
MSCCL++. No shared filesystem is needed. Drivers and system services are not
installed; FlashInfer is bundled, but optional NCCL-EP packages are not.

Deployments are stored under `~/.local/share/mscclpp/moe-bench/` without
overwriting existing ones. Use `--build-only` to create a bundle without
contacting nodes, or `--bundle "<bundle-path>"` to deploy an existing bundle.
Keep the default `--cuda 13`; FlashInfer compute requires CUDA 13 PyTorch.

### 2. Run one distributed workload

From the controller:

```bash
.venv/bin/python python/mscclpp_benchmark/moe_bench/run.py \
  --nodelist "<node-1-ip>,<node-2-ip>" --user "<ssh-user>" \
  --model small --input decode --routing balanced \
  --warmup 10 --iterations 10 --check
```

This uses all visible GPUs on each target, respecting its SSH-session
`CUDA_VISIBLE_DEVICES`. Two four-GPU nodes use eight GPUs in total; the `small`
preset fits this setup. For other sizes, ensure the expert count is divisible
by the total GPU count.

The newest deployment on each target is selected by default. Use `--remote-dir`
to choose an exact deployment or parent directory at the same path on every
node. The controller copies current benchmark Python code and configuration
into a separate run directory; it does not rebuild the native library.

`--timeout` limits distributed worker execution (default: 600 seconds).
Failures cancel the other workers, including orphaned descendants; cleanup
errors appear in node logs. Results and one `node-<rank>.log` per node are saved
on the controller. Remote run directories remain for troubleshooting.
Result downloads replace local files only after transfer succeeds, so an
interrupted download does not overwrite the controller's existing run record.

For CUDA Fabric import errors or `CUDA_ERROR_NOT_SUPPORTED`, ask your
administrator to check IMEX and cross-node memory imports. Matching fabric IDs
alone do not prove communication works.

## Tests

With `pytest` installed:

```bash
.venv/bin/python -m pytest python/mscclpp_benchmark/moe_bench/tests -q
```

GPU tests are opt-in. On two idle Blackwell GPUs, run the MSCCL++ MXFP8/MXFP4
reference checks with:

```bash
GLOO_SOCKET_IFNAME=lo .venv/bin/python -m torch.distributed.run \
  --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc-per-node=2 \
  python/mscclpp_benchmark/moe_bench/tests/test_compute_flashinfer.py mscclpp
```

It covers every mode and layout with eager, `mlp`, and `layer` graphs. Each
`layer` graph is replayed three times with changed inputs, and every replay is
compared against the reference. Replace `mscclpp` with `nccl` to test NCCL-EP.

To check communication alone, bit-exactly and without FlashInfer, replace
`test_compute_flashinfer.py` with `test_exact_communication.py`. Expert compute
becomes an identity copy, each token row encodes its global ID, and routing
weights are `1/top_k`, so every received row and combined output must match
exactly (eager and `layer` graphs, two inputs each).
