# MoE benchmark

Single-node dispatch -> SwiGLU expert compute -> weighted combine on 2 or 4 GPUs.
Communication is BF16. Inputs, weights, and routing are synthetic; routing stays
fixed during each run.

## Run

From the repository root, using an environment with MSCCL++ and the selected
backend dependencies installed (see [setup details](../../../../src/ext/ep/README.md)):

```bash
.venv/bin/python test/python/ep/moe_bench/run.py \
  --model scaled_small --input decode --routing balanced \
  --backend mscclpp --mode latency --layout expert_major \
  --compute-backend cutlass_flashinfer \
  --activation-dtype mxfp8 --weight-dtype mxfp4 \
  --gpus 4 --graph mlp --warmup 200 --iterations 100
```

The launcher starts one worker per GPU automatically. Use idle GPUs.
For the PyTorch BF16 reference, replace the compute options with
`--compute-backend torch --activation-dtype bf16 --weight-dtype bf16`.

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
| `--gpus` | `2`, `4` | `4` |
| `--graph` | `none`, `mlp` (capture local compute, not communication) | `none` |
| `--warmup` | Integer >= 0 | `10` |
| `--iterations` | Integer >= 1 | `100` |
| `--seed` | Integer from 0 through 2^63 - 1 | `42` |
| `--output-dir` | Results directory | `benchmark_results` |
| `--config` | YAML file path | `configs.yaml` beside `run.py` |
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
dependency requirements are listed in the setup details above.

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

## Code layout

- `communication_backends/`: lazy communication factory and MSCCL++ / NCCL-EP adapters.
- `compute_backends/`: lazy compute factory, `torch_reference.py`, and optional `flashinfer.py`.
- `workload.py`: shared seeded input and expert-weight generation.
- `memory.py`: conservative tensor memory estimates.
- `layout.py`: native dispatch-layout adaptation for expert compute.
- `run.py` / `worker.py`: launcher and timed distributed pipeline.
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
