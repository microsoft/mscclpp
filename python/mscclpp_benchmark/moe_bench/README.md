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
- `--check` compares the output with an independent MXFP8/MXFP4 reference
  before timing (relative L2 error below 3%). The 10 warmup iterations are
  excluded from the 10 measured iterations.

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

## Configuration

Choose presets from [configs.yaml](configs.yaml):

| Setting | Choices |
|---|---|
| `--model` | `small`, `medium`, `large`, `scaled_small`, `scaled_medium`, `scaled_large` |
| `--input` | `decode` (16 tokens/GPU), `prefill` (1024 tokens/GPU) |
| `--routing` | `balanced`, `uniform`, `hot_pair` |
| `--backend` | `mscclpp` (default), `nccl` |
| `--mode` | `latency` (default), `throughput` |
| `--graph` | `none` (default), `mlp` (capture computation only) |

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
backend). It requires Blackwell SM100/SM103, CUDA 13 PyTorch, and model
dimensions divisible by 128. Communication remains BF16; activation
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
