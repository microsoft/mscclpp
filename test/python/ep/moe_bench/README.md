# MoE benchmark

Dispatch -> SwiGLU expert compute -> weighted combine on local GPUs or across
MNNVL-connected nodes.
Communication is BF16. Inputs, weights, and routing are synthetic; routing stays
fixed during each run.

## Install

Run this from the repository root directory (the directory containing `pyproject.toml`):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[cuda13,moe-benchmark]'
```

### Copy a build to other nodes (no clone or shared storage required)

On the build node, after installation:

```bash
.venv/bin/python tools/deploy_moe_bench.py \
  --nodelist 10.0.4.159,10.0.4.69 --user azhpcuser
```

This builds MSCCL++ once, downloads dependency wheels for the default
MSCCL++/PyTorch BF16 backend, and copies the same bundle to every listed node
using SSH/SCP. Each node creates its own virtual environment and installs from
the bundled wheels **without accessing a package registry or compiling MSCCL++**.
The script verifies checksums and imports before reporting success.
It does not copy your existing virtual environment, SSH keys, or Git checkout.

Requirements:

- SSH key authentication and verified host keys for every node. The script does
  not bypass SSH host-key checks.
- Matching Python major/minor version, Python ABI, CPU/GPU architectures and
  libc version on the build and target nodes; incompatible nodes are rejected.
- Python virtual-environment support, a compatible NVIDIA driver, `nvidia-smi`,
  `tar`, and required system libraries (including libnuma) on every node.
  Host drivers and system packages are not copied or installed.
- Enough disk space for the bundle, dependency wheels and installed environment.
  PyTorch/CUDA dependencies require several GB per node.

Installed versions in the build Python environment constrain dependency
resolution. Build-tool dependencies and any missing runtime dependencies are
obtained on the build node; targets install only the bundled wheels. CUDA 13 is
the default; use `--cuda 12` for a compatible CUDA 12 build environment.
Optional FlashInfer and NCCL-EP dependencies are not included.

Each deployment uses a new versioned directory beneath
`~/.local/share/mscclpp/moe-bench/` and prints its path. Existing deployments are
not overwritten. Run from that directory using the same command below.
The bundle records the source commit and whether the build checkout was dirty;
benchmark results retain this provenance even without Git on the target.
Do not change a bundle after building it.

To build without contacting nodes, or reuse an existing bundle:

```bash
.venv/bin/python tools/deploy_moe_bench.py --build-only
.venv/bin/python tools/deploy_moe_bench.py \
  --bundle moe_deploy/<bundle-name>.tar \
  --nodelist 10.0.4.159,10.0.4.69 --user azhpcuser
```

**Deployment only prepares the software.** Use `--nodelist` below to launch one
distributed benchmark across the prepared nodes.

## Run

### Small BF16 workload: 10 measured iterations

From the repository root:

```bash
.venv/bin/python test/python/ep/moe_bench/run.py \
  --model small --input decode --routing balanced \
  --backend mscclpp --mode latency --layout expert_major \
  --compute-backend torch --activation-dtype bf16 --weight-dtype bf16 \
  --graph none --warmup 10 --iterations 10
```

This uses 16 input tokens per GPU, eight experts, and top-k two. It measures a
synthetic MoE layer, not full-model inference. The 10 warmup iterations are
excluded from the 10 measured iterations.

The launcher uses all GPUs visible on the current node and starts one worker per
GPU automatically; do not wrap it in `torchrun` or `mpirun`. Exactly 2 or 4 visible
GPUs are currently supported; other counts raise an error instead of selecting a
subset. Existing `CUDA_VISIBLE_DEVICES` restrictions are respected. There is no
GPU-count option: remove `--gpus` and `environment.num_gpus` from older commands
and YAML files. The detected count is still recorded in the results.
Use `--nodelist` for multiple nodes, as described below.

Use idle GPUs. The launcher automatically uses `127.0.0.1` for PyTorch's local
worker coordination and defaults `GLOO_SOCKET_IFNAME` to `lo`, so no hostname
configuration or IP-address option is needed. An explicitly set
`GLOO_SOCKET_IFNAME` is preserved. These settings affect CPU-side coordination,
not the selected GPU communication backend.

### One workload across N MNNVL-connected nodes

After deployment has finished on every target, run once from the controller:

```bash
.venv/bin/python test/python/ep/moe_bench/run.py \
  --nodelist 10.0.4.159,10.0.4.69 --user azhpcuser \
  --model small --input decode --routing balanced \
  --warmup 10 --iterations 10 --check
```

This starts **one distributed workload**, not independent benchmarks. With four
visible GPUs per node, the example uses eight GPUs in total. Add more distinct
IPv4 addresses to the list for more nodes; each must expose the same GPU count.
All visible GPUs on each target are used, respecting its SSH-session
`CUDA_VISIBLE_DEVICES`. The controller can also be one of the listed targets.
SSH is used for every target, including the current machine.

Requirements and limits:

- Every GPU must belong to the same CUDA IPC/MNNVL fabric domain. Ordinary
  Ethernet/RDMA connectivity is not a fallback for GPU communication.
- NVIDIA IMEX must be configured for that domain and running on the nodes.
  Healthy `nvidia-smi` fabric status and a matching clique ID alone do **not**
  prove that cross-node CUDA Fabric memory imports work.
- Targets need matching GPU counts, Torch/CUDA/MSCCL++ versions and the same
  native wheel checksum. SSH keys and verified host keys must already work.
- `python3`, `ip`, SSH/SCP, and `tar` must be available. The node IP must be
  assigned to a local interface. Node-to-node TCP must allow the master port
  (default 29500), Gloo/MSCCL++ bootstrap traffic, and IMEX communication.
- `--nodelist` currently supports only the MSCCL++ communication backend.
  Optional compute backends must already be installed on every target.
- Expert count must be divisible by the **total** GPU count. The eight-expert
  `small` preset fits two four-GPU nodes; three such nodes need a custom preset
  with, for example, 24 experts. MSCCL++ throughput supports only 2, 4, 8, or 16
  total ranks; latency uses the native runtime's topology/capacity limits.

By default each target uses its newest deployment under
`~/.local/share/mscclpp/moe-bench/`. Use `--remote-dir` to select an exact
deployment directory, or a different parent directory, at the same path on
every node. Missing or incomplete deployments are errors, not skipped nodes.

No shared filesystem is required. The controller copies a small snapshot of its
current benchmark Python code and resolved configuration into a fresh
`<deployment>/runs/<run-id>/` on each target, using that deployment's native
library and virtual environment. It neither rebuilds MSCCL++ nor modifies the
installed benchmark sources. The controller needs Python and PyYAML; Torch runs
on the targets. Gloo and MSCCL++ bootstrap use the interface assigned to each
listed IP, rather than loopback.

`--check` compares Torch BF16 output against an independent seeded expert
reference on every rank before warmup/timing. It is not available for MX
compute. Use `--timeout` to bound worker execution; a failure or timeout cancels
the other workers. Rank-zero results are copied back to the controller alongside
one `node-<rank>.log` per node. Remote run directories remain for diagnostics.

If a cross-node run fails with a missing `/tmp/mscclpp_bootstrap_*.sock`, a Fabric
import warning, or `CUDA_ERROR_NOT_SUPPORTED`, have the system administrator
verify cross-node CUDA Fabric imports and the IMEX domain before retrying.
Check `systemctl status nvidia-imex`, its logs, and
`/etc/nvidia-imex/nodes_config.cfg` where that service uses the default config.
The launcher does not install drivers or configure/start system services.

### Optional Blackwell MX compute

After installing the optional FlashInfer dependencies from the
[backend setup details](../../../../src/ext/ep/README.md):

```bash
.venv/bin/python test/python/ep/moe_bench/run.py \
  --model scaled_small --input decode --routing balanced \
  --backend mscclpp --mode latency --layout expert_major \
  --compute-backend cutlass_flashinfer \
  --activation-dtype mxfp8 --weight-dtype mxfp4 \
  --graph mlp --warmup 200 --iterations 100
```

## Options

Omitted optional flags use the YAML settings. Defaults below are from the supplied
`configs.yaml`.

| Option | Values | Default |
|---|---|---|
| `--model` | `small`, `medium`, `large`, `scaled_small`, `scaled_medium`, `scaled_large`, or a custom model preset | Required |
| `--input` | `decode` (16 tokens/GPU), `prefill` (1024), or a custom input preset | Required |
| `--routing` | `balanced`, `uniform`, `hot_pair`, or a custom routing preset | Required |
| `--backend` | `mscclpp`, `nccl` | `mscclpp` |
| `--mode` | `latency`, `throughput` | `latency` |
| `--layout` | `expert_major`, `rank_major`, `token_major` | `expert_major` |
| `--compute-backend` | `torch`, `cutlass_flashinfer` | `torch` |
| `--activation-dtype` | `bf16`, `mxfp8` | `bf16` |
| `--weight-dtype` | `bf16`, `mxfp4` | `bf16` |
| `--graph` | `none`, `mlp` (capture local compute, not communication) | `none` |
| `--warmup` | Integer >= 0 | `10` |
| `--iterations` | Integer >= 1 | `100` |
| `--seed` | Integer from 0 through 2^63 - 1 | `42` |
| `--output-dir` | Results directory | `benchmark_results` |
| `--config` | YAML file path | `configs.yaml` beside `run.py` |
| `--nodelist` | Comma-separated node IPv4 addresses | Local execution |
| `--user` | SSH username | Current user |
| `--remote-dir` | Remote deployment directory or parent | `.local/share/mscclpp/moe-bench` |
| `--master-port` | TCP port, 1 through 65535 | `29500` |
| `--timeout` | Distributed worker timeout, seconds > 0 | `600` |
| `--check` | Validate Torch BF16 output before timing | Disabled |
| `--help` | Show CLI help | |

Only these compute pairs are supported:
`torch`: BF16/BF16; `cutlass_flashinfer`: MXFP8/MXFP4 (activation/weight).
MXFP8 activation quantization happens after BF16 dispatch, inside measured compute.

| Backend | Mode | Supported layouts |
|---|---|---|
| MSCCL++ | latency | `expert_major`, `rank_major` |
| MSCCL++ | throughput | `token_major`, `rank_major` |
| NCCL-EP | latency | `expert_major`, `rank_major` |
| NCCL-EP | throughput | `expert_major`, `token_major` (native `FLAT`) |

Unsupported combinations raise an error. NCCL-EP throughput requires an even
token count per GPU. FlashInfer compute requires compatible Blackwell hardware;
dependency requirements are listed in the backend setup details above.

## Change dimensions and token count

Edit this directory's `configs.yaml`. Add entries under the existing `models:`
and `inputs:` mappings, for example:

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

Keep the other YAML sections; do not replace the entire file with this snippet.
Then use `--model custom_model --input custom_input` in the run command.
Dimensions and token counts do not have direct CLI overrides.

- Hidden dimension: `4096`, `4352`, `5120`, `6656`, `7168`, `8192`, `8704`, or `9216`.
- Intermediate dimension: positive; divisible by 128 for FlashInfer compute.
- Expert count: positive and divisible by GPU count.
- Top-k: from 1 through `min(9, num_experts)`.
- Token count: positive and subject to GPU memory/backend capacity limits.

The `environment:`, `compute:`, and `benchmark:` sections set defaults for CLI
options. Routing presets live under `routing:`; `hot_pair` uses expert IDs `[0, 1]`
and probability mass `0.8`.

## Results

Each run creates `<output-dir>/<timestamp>_<unique-id>/`. Relative output paths
are resolved from the current working directory.

- `run.json`: configuration, invocation, hardware/software metadata, and status.
- `summary.csv`: mean/P50/P95 latency and global token throughput.
- `iterations.csv`: per-GPU dispatch, compute, combine, and total timing samples.

Timing uses CUDA events, excludes warmup, and includes launch gaps and peer waits.
Compute timing includes required layout conversion, quantization, and local
reduction. Global total is the maximum rank duration per iteration, not a sum of
rank times.
Distributed metadata includes node addresses, deployments, worker commands,
global ranks, hostnames, local GPU indices, and reference-check status. Measured
samples include every global rank (80 rows for eight GPUs and 10 iterations).

## Code layout

- `communication_backends/`: lazy communication factory and MSCCL++ / NCCL-EP adapters.
- `compute_backends/`: lazy compute factory, `torch_reference.py`, and optional `flashinfer.py`.
- `workload.py`: shared seeded input and expert-weight generation.
- `memory.py`: conservative tensor memory estimates.
- `layout.py`: native dispatch-layout adaptation for expert compute.
- `run.py` / `worker.py`: launcher and timed distributed pipeline.
- `multinode.py`: SSH deployment discovery, distributed launch, cancellation, and result collection.
- `config.py` / `configs.yaml`, `routing.py`, `results.py`: configuration, fixed routing, and output.
- `tests/`: CPU tests and opt-in distributed reference tests.

## Tests

Unit tests and opt-in distributed reference tests live in `tests/`. From the
repository root, run CPU coverage with:

```bash
.venv/bin/python -m pytest test/python/ep/moe_bench/tests -q
```

GPU tests are skipped by default. On idle GPUs, enable the relevant coverage
using the [development test commands](../../../../src/ext/ep/README.md).
