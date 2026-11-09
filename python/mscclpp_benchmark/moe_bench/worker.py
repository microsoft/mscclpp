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
    from .results import save_results, write_json
    from .routing import make_routing
    from .workload import make_input
else:
    from communication_backends import create_backend, preflight_backend
    from compute_backends import create_mlp
    from config import validate_config
    from layout import LayoutMLP, assignment_counts, local_route_ids
    from memory import estimate_tensor_bytes
    from results import save_results, write_json
    from routing import make_routing
    from workload import make_input


def initialize():
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    world = int(os.environ["WORLD_SIZE"])
    if not 0 <= local_rank < local_world or local_world < 1 or world < local_world or world % local_world:
        raise ValueError("Invalid torchrun local/global rank configuration")
    if torch.cuda.device_count() < local_world:
        raise ValueError("Local worker count exceeds visible GPUs")
    torch.cuda.set_device(local_rank)
    # MSCCL++ binds each communicator to one stream, so setup, eager steps, and
    # full-layer graph capture share this non-default stream.
    torch.cuda.set_stream(torch.cuda.Stream())
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
            self.ids, self.weights = ids.to(device), weights.to(device)
            shape = (local_experts, world * inputs["tokens_per_rank"], model["d_model"])
            self.recv = torch.empty(shape, dtype=torch.bfloat16, device=device)
            self.expert_output = torch.empty_like(self.recv)
            self.output = torch.empty_like(self.x)
            self.mlp = create_mlp(config, rank, self.counts, device)
            # Compile kernels before entering native collectives.
            self.recv.zero_()
            self.mlp(self.recv, self.expert_output)
            torch.cuda.synchronize()
        except (RuntimeError, ImportError, ValueError, OSError, AssertionError) as exception:
            error = f"rank {rank}: {type(exception).__name__}: {exception}"
        errors = [None] * world
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(f"Runtime/compute initialization failed before communication: {errors}")
        self.mlp_graph = self.layer_graph = self.phase_events = None
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
        elif graph_mode == "layer":
            stream = torch.cuda.current_stream()
            # External events stay timing-capable when recorded inside the graph.
            self.phase_events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
            for event in self.phase_events:
                event.record(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                tokens = self._layer(self.phase_events)
            if self._dispatch_signature(tokens) != self.native_signature:
                raise RuntimeError("Captured dispatch buffers differ from the eager native buffers")
            self.layer_graph = graph

    def _dispatch_signature(self, tokens):
        signature = (tokens.data_ptr(), tokens.shape, tokens.stride())
        if getattr(self.backend, "adapt_layout", False):
            signature += (self.backend.route_ids.data_ptr(), self.backend.route_weights.data_ptr())
        return signature

    def _layer(self, events=None):
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
        return tokens

    @torch.inference_mode()
    def step(self, events=None):
        """Run one layer; with a layer graph, phase events are captured inside it."""
        if self.layer_graph is None:
            self._layer(events)
        else:
            self.layer_graph.replay()
        return self.output

    def read_latencies(self, events):
        """Return one iteration's phase latencies after its stream has completed."""
        start, dispatched, computed, end = events if self.layer_graph is None else self.phase_events
        return {
            "dispatch_us": start.elapsed_time(dispatched) * 1000,
            "mlp_us": dispatched.elapsed_time(computed) * 1000,
            "combine_us": computed.elapsed_time(end) * 1000,
            "total_us": start.elapsed_time(end) * 1000,
        }

    def close(self):
        if getattr(self, "layer_graph", None) is not None:
            torch.cuda.synchronize()
        self.mlp_graph = self.layer_graph = self.phase_events = None
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


def check_output(actual, expected, rank, world, rtol, atol, label, magnitude=None):
    """Compare every rank's output element-wise; all ranks agree on the verdict."""
    if __package__:
        from .reference import compare_outputs
    else:
        from reference import compare_outputs
    stats, first = compare_outputs(actual, expected, rtol, atol, magnitude)
    ranks = [None] * world
    dist.all_gather_object(ranks, {"rank": rank, **stats, "first_mismatch": first})
    total = sum(item["total_elements"] for item in ranks)
    summary = {
        "max_tolerance_fraction": max(item["max_tolerance_fraction"] for item in ranks),
        "max_abs_error": max(item["max_abs_error"] for item in ranks),
        "mean_abs_error": sum(item["mean_abs_error"] * item["total_elements"] for item in ranks) / total,
        "mismatched_elements": sum(item["mismatched_elements"] for item in ranks),
        "non_finite_elements": sum(item["non_finite_elements"] for item in ranks),
        "total_elements": total,
        "max_rank_relative_l2": max(
            float("inf") if item["relative_l2"] is None else item["relative_l2"] for item in ranks
        ),
    }
    if summary["mismatched_elements"] or summary["non_finite_elements"]:
        failed = next(item for item in ranks if item["first_mismatch"] is not None or item["non_finite_elements"])
        raise RuntimeError(
            f"Distributed output ({label}) differs from the independent MXFP8/MXFP4 reference: "
            f"{summary['mismatched_elements']}/{total} elements outside rtol={rtol}, atol={atol} "
            f"({summary['non_finite_elements']} non-finite); max_abs_error={summary['max_abs_error']:.4g}; "
            f"first mismatch on rank {failed['rank']}: {failed['first_mismatch']}"
        )
    if rank == 0:
        print(
            f"Reference check passed ({label}): 0/{total} elements outside rtol={rtol}, atol={atol}; "
            f"worst element used {summary['max_tolerance_fraction']:.1%} of its tolerance; "
            f"max_abs_error={summary['max_abs_error']:.4g}, mean_abs_error={summary['mean_abs_error']:.4g}",
            flush=True,
        )
    return summary


@torch.inference_mode()
def check_before_timing(pipeline, config, rank, world, rtol, atol):
    """Check the configured input, then a changed input, and restore the original.

    A changed input catches stale dispatch buffers that a fixed input would hide.
    Returns the original input's reference and contribution magnitudes for the
    check after timing.
    """
    if __package__:
        from .reference import reference_output
    else:
        from reference import reference_output
    pipeline.step()
    torch.cuda.synchronize()
    expected, magnitude = reference_output(config, pipeline.x, pipeline.ids, pipeline.weights, True)
    checks = {"initial": check_output(pipeline.output, expected, rank, world, rtol, atol, "initial input", magnitude)}
    original = pipeline.x.clone()
    pipeline.x.mul_(0.61).add_(0.17)
    pipeline.step()
    torch.cuda.synchronize()
    changed, changed_magnitude = reference_output(config, pipeline.x, pipeline.ids, pipeline.weights, True)
    checks["changed_input"] = check_output(
        pipeline.output, changed, rank, world, rtol, atol, "changed input", changed_magnitude
    )
    pipeline.x.copy_(original)
    torch.cuda.synchronize()
    return (expected, magnitude), checks


def run(config, run_dir, check=False, rtol=None, atol=None):
    validate_config(config)
    rank, world, device = initialize()
    pipeline = None
    try:
        pipeline = Pipeline(config, rank, device)
        correctness = None
        if check:
            if __package__:
                from .reference import CHECK_ATOL, CHECK_RTOL, ROUNDING_SLACK
            else:
                from reference import CHECK_ATOL, CHECK_RTOL, ROUNDING_SLACK
            rtol = CHECK_RTOL if rtol is None else rtol
            atol = CHECK_ATOL if atol is None else atol
            (expected, magnitude), checks = check_before_timing(pipeline, config, rank, world, rtol, atol)
        settings = config["benchmark"]
        events = make_timing_events(settings["iterations"])
        stream = torch.cuda.current_stream()
        for _ in range(settings["warmup"]):
            pipeline.step(events[0])
            stream.synchronize()
            dist.barrier()

        local_latencies = []
        for boundaries in events:
            # Rank alignment and completion synchronization are outside events.
            dist.barrier()
            pipeline.step(boundaries)
            stream.synchronize()
            local_latencies.append(pipeline.read_latencies(boundaries))
        if check:
            # The input is restored, so the last timed iteration must match the original reference.
            checks["after_timing"] = check_output(
                pipeline.output, expected, rank, world, rtol, atol, "last timed iteration", magnitude
            )
            del expected, magnitude
            correctness = {
                "status": "passed",
                "rtol": rtol,
                "atol": atol,
                "rounding_slack": ROUNDING_SLACK,
                "max_tolerance_fraction": max(item["max_tolerance_fraction"] for item in checks.values()),
                "max_abs_error": max(item["max_abs_error"] for item in checks.values()),
                "checks": checks,
            }
        latencies = [None] * world if rank == 0 else None
        dist.gather_object(local_latencies, latencies, dst=0)
        devices = [None] * world
        props = torch.cuda.get_device_properties(device)
        dist.all_gather_object(
            devices,
            {
                "rank": rank,
                "hostname": platform.node(),
                "local_rank": device.index,
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
                "reference_check": "passed" if check else "not_requested",
                "correctness": correctness,
                **pipeline.backend.metadata,
                **pipeline.mlp.metadata,
                "cuda_graph_scope": (
                    "dispatch_mlp_combine"
                    if pipeline.layer_graph is not None
                    else "mlp_only" if pipeline.mlp_graph is not None else "none"
                ),
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
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with (args.run_dir / "run.json").open() as stream:
        document = json.load(stream)
    tolerance = document.get("check_tolerance") or {}
    try:
        run(document["config"], args.run_dir, args.check, tolerance.get("rtol"), tolerance.get("atol"))
    except Exception as error:
        # The launcher only sees exit codes; leave the reason for run.json.
        rank = int(os.environ.get("RANK", "0"))
        write_json(
            args.run_dir / f"worker-failure-rank{rank}.json",
            {"rank": rank, "type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
