# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Internal torchrun worker for the reference MoE benchmark."""

import argparse
from contextlib import ExitStack
from datetime import timedelta
import json
import os
from pathlib import Path
import platform

import torch
import torch.distributed as dist

if __package__:
    from .communication_backends import create_backend, preflight_backend
    from .compute_backends import create_mlp
    from .config import validate_config
    from .layout import LayoutMLP, assignment_counts, local_route_ids
    from .memory import estimate_tensor_bytes
    from .results import save_results
    from .routing import make_routing
    from .workload import make_input
else:
    from communication_backends import create_backend, preflight_backend
    from compute_backends import create_mlp
    from config import validate_config
    from layout import LayoutMLP, assignment_counts, local_route_ids
    from memory import estimate_tensor_bytes
    from results import save_results
    from routing import make_routing
    from workload import make_input


def initialize():
    if int(os.environ["LOCAL_WORLD_SIZE"]) != int(os.environ["WORLD_SIZE"]):
        raise ValueError("MoE benchmark supports single-node torchrun launches only")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


class Pipeline:
    """Fixed-routing native dispatch, layout-aware expert compute, and combine."""

    @torch.inference_mode()
    def __init__(self, config, rank, device):
        self.config = config
        model, inputs, environment = config["model"], config["input"], config["environment"]
        world = environment["num_gpus"]
        if dist.get_world_size() != world:
            raise ValueError("Launched world size does not match environment.num_gpus")
        ids, weights = make_routing(config, rank)
        counts = torch.bincount(ids.flatten(), minlength=model["num_experts"])
        dist.all_reduce(counts)
        self.global_counts = counts.tolist()
        local_experts = model["num_experts"] // world
        self.counts = self.global_counts[rank * local_experts : (rank + 1) * local_experts]
        estimate = estimate_tensor_bytes(config, self.counts)
        free, _ = torch.cuda.mem_get_info(device)
        budgets = [None] * world
        dist.all_gather_object(budgets, {"rank": rank, "estimated_tensor_bytes": estimate, "free_bytes": free})
        if any(item["estimated_tensor_bytes"] > item["free_bytes"] * 0.8 for item in budgets):
            raise RuntimeError(f"Tensor estimate exceeds 80% of available GPU memory; reduce workload: {budgets}")
        self.memory_estimates = budgets
        # Compile optional kernels before entering any native GPU collectives.
        # Exchange failures over Gloo so peers cannot enter dispatch while one
        # rank has failed dependency loading, allocation, or compilation.
        error = None
        try:
            preflight_backend(config)
            self.x = make_input(config, rank, device)
            # Throughput's fixed-routing cache checks tensor version counters.
            with torch.inference_mode(False):
                self.ids, self.weights = ids.to(device), weights.to(device)
            shape = (local_experts, world * inputs["tokens_per_rank"], model["d_model"])
            self.recv = torch.empty(shape, dtype=torch.bfloat16, device=device)
            self.expert_output = torch.empty_like(self.recv)
            self.output = torch.empty_like(self.x)
            self.mlp = create_mlp(config, rank, self.counts, device)
            if environment["compute_backend"] != "torch":
                self.recv.zero_()
                self.mlp(self.recv, self.expert_output)
                torch.cuda.synchronize()
        except (RuntimeError, ImportError, ValueError, OSError, AssertionError) as exception:
            error = f"rank {rank}: {type(exception).__name__}: {exception}"
        errors = [None] * world
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(f"Runtime/compute initialization failed before communication: {errors}")
        with ExitStack() as resources:
            self.backend = create_backend(
                config, rank, self.x, self.ids, self.weights, self.recv, self.expert_output, self.output
            )
            resources.callback(self.close)
            error = None
            try:
                self._initialize_compute(environment["cuda_graph"])
            except (RuntimeError, ValueError, AssertionError) as exception:
                error = f"rank {rank}: {type(exception).__name__}: {exception}"
                print(f"Compute warmup/capture failed: {error}", flush=True)
            dist.all_gather_object(errors, error)
            if any(errors):
                raise RuntimeError(f"Compute warmup/capture failed: {errors}")
            resources.pop_all()

    @torch.inference_mode()
    def _initialize_compute(self, graph_mode):
        # Fixed routing permits count reads outside timing, without padded GEMMs
        # or device-to-host count synchronization inside every iteration.
        tokens, counts = self.backend.dispatch()
        error = None
        try:
            if getattr(self.backend, "adapt_layout", False):
                counts = assignment_counts(local_route_ids(self.backend), len(self.counts))
                self.mlp = LayoutMLP(self.mlp, self.counts, self.recv, self.expert_output, self.backend)
            self.native_recv = tokens
            self.native_signature = self._dispatch_signature(tokens)
            actual_counts = counts.cpu().tolist()
            if actual_counts != self.counts:
                raise RuntimeError(f"Dispatch counts {actual_counts} differ from fixed routing {self.counts}")
            self.mlp(tokens, self.expert_output)
            torch.cuda.synchronize()
        except (RuntimeError, ValueError, AssertionError) as exception:
            error = f"rank {dist.get_rank()}: {type(exception).__name__}: {exception}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        if any(errors):
            # Dispatch/combine are paired native collectives. Drain the pair on
            # every rank before reporting local conversion/compute failures.
            self.backend.combine_buffer.zero_()
        self.backend.combine()
        torch.cuda.synchronize()
        if any(errors):
            raise RuntimeError(f"Dispatched-layout compute initialization failed: {errors}")
        self.mlp_graph = None
        if graph_mode == "mlp":
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    self.mlp(self.native_recv, self.expert_output)
            torch.cuda.current_stream().wait_stream(capture_stream)
            self.mlp_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.mlp_graph, stream=capture_stream):
                self.mlp(self.native_recv, self.expert_output)
            torch.cuda.synchronize()

    def _dispatch_signature(self, tokens):
        signature = (tokens.data_ptr(), tokens.shape, tokens.stride())
        if getattr(self.backend, "adapt_layout", False):
            signature += (self.backend.route_ids.data_ptr(), self.backend.route_weights.data_ptr())
        return signature

    @torch.inference_mode()
    def step(self, events=None):
        stream = torch.cuda.current_stream()
        if events is not None:
            events[0].record(stream)
        tokens, _ = self.backend.dispatch()
        if self.mlp_graph is not None and self._dispatch_signature(tokens) != self.native_signature:
            raise RuntimeError("Native dispatch buffers changed; MLP graph replay would use stale input/metadata")
        if events is not None:
            events[1].record(stream)
        if self.mlp_graph is None:
            self.mlp(tokens, self.expert_output)
        else:
            self.mlp_graph.replay()
        if events is not None:
            events[2].record(stream)
        self.backend.combine()
        if events is not None:
            events[3].record(stream)
        return self.output

    def close(self):
        self.mlp_graph = None
        self.native_recv = None
        self.backend.close()

    def load_statistics(self):
        world = self.config["environment"]["num_gpus"]
        local_experts = len(self.global_counts) // world
        per_rank = [sum(self.global_counts[rank * local_experts : (rank + 1) * local_experts]) for rank in range(world)]
        return {
            "assignments_per_expert": self.global_counts,
            "owner_rank_per_expert": [expert // local_experts for expert in range(len(self.global_counts))],
            "assignments_per_rank": per_rank,
            "rank_max_over_mean": max(per_rank) / (sum(per_rank) / world),
        }


def make_timing_events(iterations):
    events = [[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(iterations)]
    stream = torch.cuda.current_stream()
    # PyTorch creates the native CUDA event on its first record, not construction.
    # Materialize every event before warmup so timed iterations only record it.
    for boundaries in events:
        for event in boundaries:
            event.record(stream)
    stream.synchronize()
    return events


def run(config, run_dir):
    validate_config(config)
    rank, world, device = initialize()
    pipeline = None
    try:
        pipeline = Pipeline(config, rank, device)
        settings = config["benchmark"]
        events = make_timing_events(settings["iterations"])
        for _ in range(settings["warmup"]):
            pipeline.step(events[0])
            events[0][-1].synchronize()
            dist.barrier()

        for boundaries in events:
            # Rank alignment and completion synchronization are outside events.
            dist.barrier()
            pipeline.step(boundaries)
            boundaries[-1].synchronize()
        local_latencies = [
            {
                "dispatch_us": start.elapsed_time(dispatched) * 1000,
                "mlp_us": dispatched.elapsed_time(computed) * 1000,
                "combine_us": computed.elapsed_time(end) * 1000,
                "total_us": start.elapsed_time(end) * 1000,
            }
            for start, dispatched, computed, end in events
        ]
        latencies = [None] * world if rank == 0 else None
        dist.gather_object(local_latencies, latencies, dst=0)
        devices = [None] * world
        props = torch.cuda.get_device_properties(device)
        dist.all_gather_object(
            devices,
            {
                "rank": rank,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": [props.major, props.minor],
            },
        )
        if rank == 0:
            metadata = {
                "gpus": devices,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                **pipeline.backend.metadata,
                **pipeline.mlp.metadata,
                "cuda_graph_scope": "mlp_only" if pipeline.mlp_graph is not None else "none",
                "bf16_reduced_precision_reduction": False,
                "tensor_memory_estimates": pipeline.memory_estimates,
                "environment": {
                    key: value
                    for key, value in os.environ.items()
                    if key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
                    or key.startswith("MSCCLPP_EP_")
                    or key.startswith("NCCL_")
                },
            }
            save_results(run_dir, latencies, config, metadata, pipeline.load_statistics())
        dist.barrier()
    finally:
        try:
            if pipeline is not None:
                pipeline.close()
        finally:
            pipeline = None
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    with (args.run_dir / "run.json").open() as stream:
        config = json.load(stream)["config"]
    run(config, args.run_dir)


if __name__ == "__main__":
    main()
