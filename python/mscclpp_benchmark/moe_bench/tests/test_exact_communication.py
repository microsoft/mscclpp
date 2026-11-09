# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Opt-in bit-exact dispatch/combine test with identity experts (no FlashInfer).

Each token row encodes its global ID, and routing weights are 1/top_k, so every
dispatched row, BF16 partial, and combined output is exactly representable.
Any misrouted, missing, duplicated, or stale row changes the result.
"""

import itertools
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

if __package__:
    from ..config import SUPPORTED_LAYOUTS
    from .test_moe_bench import assert_idle_gpus, tiny_config
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import SUPPORTED_LAYOUTS
    from test_moe_bench import assert_idle_gpus, tiny_config


class IdentityExperts:
    """Expert compute that returns each dispatched row unchanged."""

    metadata = {
        "compute_backend": "identity",
        "compute_implementation": "identity",
        "mlp_timed_operations": "copy",
    }

    def __call__(self, tokens, out):
        out.copy_(tokens)
        return out


def encoded_rows(token_ids, hidden, device):
    """Row ``token`` holds ``8 * id + column % 8``; exact in BF16 for IDs < 32."""
    columns = torch.arange(hidden, device=device) % 8
    return (8 * token_ids[:, None] + columns[None, :]).to(torch.bfloat16)


@pytest.mark.parametrize("backend", ["mscclpp", "nccl"])
def test_exact_communication(backend):
    if os.environ.get("MOE_BENCH_GPU_TESTS") != "1":
        pytest.skip("Set MOE_BENCH_GPU_TESTS=1 on two idle GPUs")
    assert_idle_gpus()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            backend,
        ],
        check=True,
        timeout=900,
    )


@torch.inference_mode()
def exact_communication(backend):
    import torch.distributed as dist
    import worker

    worker.create_mlp = lambda *args: IdentityExperts()
    rank, world, device = worker.initialize()
    pipeline = None
    pipelines = 0
    try:
        combinations = [(mode, layout) for mode, layouts in SUPPORTED_LAYOUTS[backend].items() for layout in layouts]
        for (mode, layout), graph, routing in itertools.product(
            combinations,
            (False, "layer"),
            (
                {"kind": "balanced"},
                {"kind": "uniform"},
                {"kind": "hot_experts", "expert_ids": [0, 1], "probability_mass": 0.999},
            ),
        ):
            if backend == "nccl" and mode == "throughput" and graph == "layer":
                continue
            config = tiny_config()
            config["environment"].update(communication_backend=backend, cuda_graph=graph, mode=mode, layout=layout)
            if backend == "nccl" and mode == "throughput":
                config["input"]["tokens_per_rank"] = 4
            config["routing"] = routing
            tokens = config["input"]["tokens_per_rank"]
            top_k = config["model"]["top_k"]
            hidden = config["model"]["d_model"]
            if rank == 0:
                print(f"Exact {backend}/{mode}/{layout}, graph={graph}, routing={routing['kind']}", flush=True)
            pipeline = worker.Pipeline(config, rank, device)
            assert (pipeline.layer_graph is not None) == (graph == "layer")
            all_ids = [None] * world
            dist.all_gather_object(all_ids, pipeline.ids.cpu())
            local_experts = len(pipeline.counts)
            for revision in range(2):
                # New IDs each revision: replay must consume the new input, not stale rows.
                offset = revision * world * tokens
                token_ids = offset + rank * tokens + torch.arange(tokens, device=device)
                pipeline.x.copy_(encoded_rows(token_ids, hidden, device))
                pipeline.weights.fill_(1.0 / top_k)
                pipeline.step()
                torch.cuda.synchronize()
                # Weights sum to one, so the combined row must be the original row, bit for bit.
                assert torch.equal(
                    pipeline.output, pipeline.x
                ), f"rank {rank}: combine mismatch at {(pipeline.output != pipeline.x).nonzero()[:4].tolist()}"
                if not getattr(pipeline.backend, "adapt_layout", False):
                    received = pipeline.native_recv
                    for expert in range(local_experts):
                        count = pipeline.counts[expert]
                        global_expert = rank * local_experts + expert
                        expected = sorted(
                            offset + source * tokens + token
                            for source, ids in enumerate(all_ids)
                            for token in (ids == global_expert).any(dim=1).nonzero().flatten().tolist()
                        )
                        rows = received[expert, :count]
                        actual = sorted((rows[:, 0].float() / 8).long().tolist())
                        assert actual == expected, f"rank {rank} expert {expert}: received {actual}, want {expected}"
                        ids = (rows[:, 0].float() / 8).long()
                        assert torch.equal(rows, encoded_rows(ids, hidden, device)), f"rank {rank} expert {expert}"
            dist.barrier()
            pipeline.close()
            pipeline = None
            pipelines += 1
        if rank == 0:
            print(f"Exact communication {backend}: {pipelines} pipelines, {2 * pipelines} replays passed", flush=True)
    finally:
        try:
            if pipeline is not None:
                pipeline.close()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    exact_communication(sys.argv[1])
