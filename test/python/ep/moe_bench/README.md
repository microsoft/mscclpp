# MoE benchmark

Measure sending tokens to experts, running SwiGLU expert computation, and
combining the results. Inputs and weights are synthetic, and routing stays fixed
during each run. This measures one MoE layer, not full-model inference.

## Quick start

From the repository root, in a CUDA 13 environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[cuda13,moe-benchmark]'
.venv/bin/python test/python/ep/moe_bench/run.py \
  --model small --input decode --routing balanced \
  --warmup 10 --iterations 10 --check
```

- Uses MSCCL++ communication and PyTorch BF16 computation by default.
- Runs 16 input tokens per GPU, with eight experts and two experts per token.
- Runs on **one node by default**, using all visible GPUs. Use idle GPUs;
  `CUDA_VISIBLE_DEVICES` is respected.
- Use `--nodelist` for [multiple MNNVL-connected nodes](#multiple-nodes).
  The selected backend's topology and GPU-count limits still apply.
- Starts workers automatically. Do not wrap the command in `torchrun` or `mpirun`.
- Checks output against a reference before timing. The 10 warmup iterations
  are excluded from the 10 measured iterations.

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
`min(9, num_experts)`. Supported hidden dimensions are 4096, 4352, 5120, 6656,
7168, 8192, 8704, and 9216. Token counts and intermediate dimensions must be
positive and fit available memory and backend limits.

### Backends and layouts

| Backend | Mode | Supported `--layout` values |
|---|---|---|
| MSCCL++ | latency | `expert_major` (default), `rank_major` |
| MSCCL++ | throughput | `token_major`, `rank_major` |
| NCCL-EP | latency | `expert_major`, `rank_major` |
| NCCL-EP | throughput | `expert_major`, `token_major` |

Unsupported combinations fail rather than selecting a fallback. NCCL-EP
throughput requires an even token count per GPU.

The default `--compute-backend torch` supports BF16 activations and weights.
Optional FlashInfer compute requires Blackwell SM100/SM103, CUDA 13 PyTorch,
and dimensions divisible by 128. Install the
[optional dependencies](../../../../src/ext/ep/README.md#optional-backend-setup), then run:

```bash
.venv/bin/python test/python/ep/moe_bench/run.py \
  --model scaled_small --input decode --routing balanced \
  --compute-backend cutlass_flashinfer \
  --activation-dtype mxfp8 --weight-dtype mxfp4 --graph mlp
```

Communication remains BF16; activation quantization is included in compute
timing. `--check` is available only for Torch BF16, not FlashInfer compute.

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
installed; optional NCCL-EP/FlashInfer packages are not bundled.

Deployments are stored under `~/.local/share/mscclpp/moe-bench/` without
overwriting existing ones. Use `--build-only` to create a bundle without
contacting nodes, or `--bundle "<bundle-path>"` to deploy an existing bundle.
CUDA 13 is the default; use `--cuda 12` with a compatible build environment.

### 2. Run one distributed workload

From the controller:

```bash
.venv/bin/python test/python/ep/moe_bench/run.py \
  --nodelist "<node-1-ip>,<node-2-ip>" --user "<ssh-user>" \
  --model small --input decode --routing balanced \
  --warmup 10 --iterations 10 --check
```

This uses all visible GPUs on each target, respecting its SSH-session
`CUDA_VISIBLE_DEVICES`. Two four-GPU nodes use eight GPUs in total; the `small`
preset fits this setup. For other sizes, ensure the expert count is divisible
by the total GPU count. MSCCL++ throughput supports 2, 4, 8, or 16 total GPUs.

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
.venv/bin/python -m pytest test/python/ep/moe_bench/tests -q
```

GPU tests are skipped by default. See the
[GPU test command](../../../../src/ext/ep/README.md#benchmark-tests) and run it
only on idle GPUs.
