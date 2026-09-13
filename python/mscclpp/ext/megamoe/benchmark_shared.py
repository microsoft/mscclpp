# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic routed-first native MegaMoE layer benchmark; launch with torchrun."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

MODES = ("routed-only", "shared-only", "serial", "overlap")
INCLUDED = {
    "routed-only": ["synthetic_router", "squash", "routed_input_staging", "routed_experts", "unsquash"],
    "shared-only": ["shared_input_staging", "shared_expert"],
    "serial": [
        "synthetic_router",
        "squash",
        "routed_input_staging",
        "routed_experts",
        "shared_input_staging",
        "shared_expert",
        "unsquash",
        "bf16_add",
    ],
}
INCLUDED["overlap"] = INCLUDED["serial"] + ["routed_entry_start_gate", "stream_fork_join"]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32, help="tokens per rank; zero is supported")
    parser.add_argument("--original-hidden", type=int, default=8704)
    parser.add_argument("--hidden", type=int, default=4096, help="squashed routed hidden width")
    parser.add_argument("--intermediate", type=int, default=4352, help="routed post-SwiGLU width")
    parser.add_argument("--experts", type=int, default=None, help="global routed experts (default 16 * WORLD_SIZE)")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--shared-intermediate", type=int, default=2048, help="shared post-SwiGLU width")
    parser.add_argument("--route-sm-margin", type=int, default=32)
    parser.add_argument("--shared-sms", type=int, default=32, help="requested shared persistent CTA count")
    parser.add_argument("--post-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rms-eps", type=float, default=1e-6, help="finite positive post-RMSNorm epsilon")
    parser.add_argument("--residual-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--graph-batch", type=int, default=5, help="same-input layer invocations per graph replay")
    parser.add_argument("--warmup", type=int, default=5, help="eager and graph warmups per schedule")
    parser.add_argument("--iterations", type=int, default=30, help="timed graph replays per schedule")
    parser.add_argument("--check", action="store_true", help="untimed oracles and eager/graph schedule equivalence")
    parser.add_argument("--reference-relative-l2", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trace-path", default=None, help="rank-0 Chrome trace; all ranks profile three layer replays")
    parser.add_argument(
        "--trace-eager", action="store_true", help="trace eager launches with host stage labels instead"
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--bootstrap-port", type=int, default=None, help="native TCP port (default MASTER_PORT+1)")
    args = parser.parse_args(argv)
    if args.tokens < 0:
        parser.error("tokens must be nonnegative")
    if min(args.graph_batch, args.warmup, args.iterations) < 1:
        parser.error("graph-batch, warmup, and iterations must be positive")
    if args.route_sm_margin < 0 or args.shared_sms < 2 or args.shared_sms % 2:
        parser.error("route-sm-margin must be nonnegative; shared-sms must be a positive even count >= 2")
    if not math.isfinite(args.reference_relative_l2) or args.reference_relative_l2 <= 0:
        parser.error("reference-relative-l2 must be finite and positive")
    if not math.isfinite(args.rms_eps) or args.rms_eps <= 0:
        parser.error("rms-eps must be finite and positive")
    if args.bootstrap_port is not None and not 1 <= args.bootstrap_port <= 65535:
        parser.error("bootstrap-port must be in [1, 65535]")
    if args.trace_eager and not args.trace_path:
        parser.error("--trace-eager requires --trace-path")
    return args


def _scope_report(args):
    included = {mode: stages.copy() for mode, stages in INCLUDED.items()}
    suffix = ["post_rmsnorm"] if args.post_norm else ["postnorm_bypass_bf16_copy"]
    if args.residual:
        suffix.append("residual_add")
    for mode in ("routed-only", "serial", "overlap"):
        included[mode].extend(suffix)
    excluded = [
        "attention",
        "prenorm",
        "residual_gather",
        "dropout",
        "stochastic_rounding",
        "other_model_layers",
        "real_model_weights",
    ]
    if not args.post_norm:
        excluded.append("postnorm")
    if not args.residual:
        excluded.append("residual")
    scope = "synthetic_router_squash_routed_unsquash_plus_native_shared_bf16_add"
    if args.post_norm:
        scope += "_post_rmsnorm"
    if args.residual:
        scope += "_residual"
    return {
        "scope": scope,
        "included_by_mode": included,
        "excluded": excluded,
        "postprocess": {
            "post_norm": args.post_norm,
            "residual": args.residual,
            "epsilon": args.rms_eps,
            "norm_implementation": "Torch FP32-input/FP32-weight rms_norm then BF16 conversion",
            "gamma": "deterministic FP32 linspace(0.75, 1.25, original_hidden)",
            "normalized_dtype": "BF16",
            "residual_dtype": args.residual_dtype.upper(),
            "final_dtype": args.residual_dtype.upper() if args.residual else "BF16",
            "residual_source": "separate synthetic skip initialized from BF16 input; not a reconstructed prenorm residual",
            "producer_updates": "input and residual sample updates are untimed; stored skip is read on every invocation",
            "order": "BF16 branch sum -> optional post-RMSNorm/BF16 handoff -> optional residual add",
            "shared_only": "component timer; excludes layer postnorm and residual",
            "disabled_norm": "BF16 pass-through copy, not RMSNorm",
        },
    }


def _configs(args, rank, world, physical_sms):
    from .api import MegaMoEConfig

    route_ctas = physical_sms - args.route_sm_margin
    if route_ctas < 2 or route_ctas % 2 or args.shared_sms > physical_sms:
        raise ValueError("SM margins must request even CTA counts >= 2, not exceeding physical SM count")
    routed = MegaMoEConfig(
        rank=rank,
        world_size=world,
        max_tokens=max(1, args.tokens),
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_experts=args.experts if args.experts is not None else 16 * world,
        top_k=args.top_k,
        sm_margin=args.route_sm_margin,
    )
    shared = MegaMoEConfig(
        rank=0,
        world_size=1,
        max_tokens=max(1, args.tokens),
        hidden=args.original_hidden,
        intermediate=args.shared_intermediate,
        num_experts=1,
        top_k=1,
        sm_margin=physical_sms - args.shared_sms,
    )
    return routed, shared


def _assert_ctas(routed, shared, physical_sms, args):
    expected = (physical_sms - args.route_sm_margin, args.shared_sms)
    actual = (routed.cta_count, shared.cta_count)
    if actual != expected:
        raise RuntimeError(f"requested routed/shared CTA counts {expected}, but native occupancy selected {actual}")


def _phase(rank, phase, **fields):
    print(
        json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "rank": rank, "phase": phase, **fields}),
        flush=True,
    )


class _Postprocess:
    """Torch post-RMSNorm followed by a separate, caller-provided skip connection.

    ``residual`` is copied once into owned storage, in the requested residual
    dtype. Update ``self.residual`` explicitly on the producer stream when an
    input sample changes; forwards never copy the input into it or accumulate
    outputs back into it. Such producer updates are outside layer timing.

    RMSNorm uses FP32 input, FP32 gamma, and FP32 arithmetic, then an explicit
    BF16 handoff before residual addition. The normalized/output buffers and
    gamma have stable addresses. Torch allocates the FP32 RMSNorm result inside
    each invocation (inside the graph pool during capture). With normalization
    disabled, a BF16 pass-through copy still refreshes ``normalized``.
    """

    def __init__(self, residual, *, epsilon=1e-6, post_norm=True, add_residual=True, residual_dtype="fp32"):
        import torch

        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("epsilon must be finite and positive")
        if not isinstance(post_norm, bool) or not isinstance(add_residual, bool):
            raise ValueError("post_norm and add_residual must be bool")
        if residual_dtype not in ("fp32", "bf16"):
            raise ValueError("residual_dtype must be 'fp32' or 'bf16'")
        if (
            not isinstance(residual, torch.Tensor)
            or residual.ndim != 2
            or residual.shape[-1] < 1
            or not residual.is_floating_point()
            or residual.requires_grad
        ):
            raise ValueError("residual must be an inference-only floating-point [T,H] tensor with H > 0")
        self.epsilon = float(epsilon)
        self.post_norm = post_norm
        self.residual_enabled = add_residual
        self.residual_dtype = residual_dtype
        dtype = torch.float32 if residual_dtype == "fp32" else torch.bfloat16
        self.residual = torch.empty(residual.shape, dtype=dtype, device=residual.device)
        self.residual.copy_(residual)
        self.weight = torch.linspace(0.75, 1.25, residual.shape[-1], dtype=torch.float32, device=residual.device)
        self.fp32_input = torch.empty(residual.shape, dtype=torch.float32, device=residual.device)
        self.normalized = torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)
        self.output = (
            torch.empty(residual.shape, dtype=dtype, device=residual.device) if add_residual else self.normalized
        )

    def normalize(self, branch_output):
        """Write the BF16 normalized buffer, or refresh it with a bypass copy."""
        import torch
        import torch.nn.functional as functional

        if (
            not isinstance(branch_output, torch.Tensor)
            or branch_output.shape != self.normalized.shape
            or branch_output.device != self.normalized.device
            or not branch_output.is_floating_point()
            or branch_output.requires_grad
        ):
            raise ValueError(
                "branch_output must be an inference-only floating tensor matching the residual shape/device"
            )
        if self.post_norm and branch_output.numel():
            self.fp32_input.copy_(branch_output)
            # Matching FP32 operands avoid the mixed BF16/FP32 RMSNorm fallback;
            # casting gamma to BF16 instead would change the model's numerics.
            result = functional.rms_norm(self.fp32_input, (self.weight.numel(),), weight=self.weight, eps=self.epsilon)
            self.normalized.copy_(result)
        else:
            self.normalized.copy_(branch_output)
        return self.normalized

    def add_residual(self):
        """Finalize the last normalized buffer without modifying the skip input."""
        import torch

        if self.residual_enabled:
            torch.add(self.normalized, self.residual, out=self.output)
        return self.output

    def forward(self, branch_output):
        """Compute postnorm(branch_output), then add the stored residual."""
        self.normalize(branch_output)
        return self.add_residual()

    def _reference_normalized(self, branch_output):
        import torch

        if branch_output.shape != self.normalized.shape:
            raise ValueError("reference branch_output shape must match the residual")
        source = branch_output.detach().to(device="cpu", dtype=torch.float32)
        if self.post_norm:
            inverse_rms = torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + self.epsilon)
            source = source * inverse_rms * self.weight.detach().cpu()
        return source.to(torch.bfloat16)

    def reference(self, branch_output, residual=None):
        """Independent CPU FP32 formula, BF16 handoff, then selected-dtype add."""
        normalized = self._reference_normalized(branch_output)
        if not self.residual_enabled:
            return normalized
        residual = self.residual if residual is None else residual
        if residual.shape != normalized.shape:
            raise ValueError("reference residual shape must match branch_output")
        skip = residual.detach().to(device="cpu", dtype=self.residual.dtype).float()
        return (normalized.float() + skip).to(self.output.dtype)


class _Frontend:
    """Preallocated Torch projections/metadata, also usable by CPU shape tests."""

    def __init__(self, inputs, routed_config, *, postprocess=None):
        import torch

        self.inputs = inputs
        tokens, original_hidden = inputs.shape
        hidden, experts, top_k = routed_config.hidden, routed_config.num_experts, routed_config.top_k
        device = inputs.device

        def empty(shape, dtype=torch.bfloat16):
            return torch.empty(shape, device=device, dtype=dtype)

        self.router_weight = empty((original_hidden, experts), torch.float32).normal_(std=original_hidden**-0.5)
        self.squash_weight = empty((original_hidden, hidden)).normal_(std=original_hidden**-0.5)
        self.unsquash_weight = empty((hidden, original_hidden)).normal_(std=hidden**-0.5)
        self.fp32_input = empty(inputs.shape, torch.float32)
        self.logits = empty((tokens, experts), torch.float32)
        self.selected_logits = empty((tokens, top_k), torch.float32)
        self.indices64 = empty((tokens, top_k), torch.int64)
        self.ids = empty((tokens, top_k), torch.int32)
        self.scores = empty((tokens, top_k), torch.float32)
        self.squashed = empty((tokens, hidden))
        self.routed_output = empty((tokens, hidden))
        self.unsquashed = empty(inputs.shape)
        self.shared_output = empty(inputs.shape)
        self.combined = empty(inputs.shape)
        self.postprocess = postprocess if postprocess is not None else _Postprocess(inputs)
        self.output = self.postprocess.output
        self.shared_ids = torch.zeros((tokens, 1), device=device, dtype=torch.int32)
        self.shared_scores = torch.ones((tokens, 1), device=device, dtype=torch.float32)

    def router(self):
        import torch

        self.fp32_input.copy_(self.inputs)
        torch.mm(self.fp32_input, self.router_weight, out=self.logits)
        torch.topk(self.logits, self.ids.shape[1], dim=-1, out=(self.selected_logits, self.indices64))
        self.ids.copy_(self.indices64)
        torch.softmax(self.selected_logits, dim=-1, out=self.scores)

    def squash(self):
        import torch

        torch.mm(self.inputs, self.squash_weight, out=self.squashed)

    def unsquash(self):
        import torch

        torch.mm(self.routed_output, self.unsquash_weight, out=self.unsquashed)

    def combine(self):
        import torch

        torch.add(self.unsquashed, self.shared_output, out=self.combined)


class _Layer:
    def __init__(self, frontend, routed, shared):
        import torch

        self.frontend, self.routed, self.shared = frontend, routed, shared
        self.device = frontend.inputs.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.shared_stream = torch.cuda.Stream(device=self.device)
        self.shared_done = torch.cuda.Event()
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.shared_stream.wait_stream(torch.cuda.current_stream(self.device))
        # Materialize the event before any capture. Each overlap invocation rejoins
        # before the next invocation can overwrite native workspaces or metadata.
        self.shared_done.record(torch.cuda.current_stream(self.device))
        self.annotate = False

    def _stage(self, name):
        import torch

        return torch.profiler.record_function(f"megamoe/{name}") if self.annotate else nullcontext()

    def launch(self, mode):
        import torch

        if mode not in MODES:
            raise ValueError(f"unknown schedule: {mode}")
        f = self.frontend
        with torch.cuda.stream(self.stream):
            if mode != "shared-only":
                with self._stage("router"):
                    f.router()
                with self._stage("squash"):
                    f.squash()
                with self._stage("routed"):
                    self.routed.forward(
                        f.squashed,
                        f.ids,
                        f.scores,
                        output=f.routed_output,
                        stream=self.stream,
                        signal_start=mode == "overlap",
                    )
            if mode != "routed-only":
                shared_stream = self.shared_stream if mode == "overlap" else self.stream
                with self._stage("shared"), torch.cuda.stream(shared_stream):
                    if mode == "overlap":
                        # The reset-event wait also orders original-input producers
                        # on the main stream; the value wait gates on kernel ENTRY.
                        self.routed.wait_until_started(shared_stream)
                    self.shared.forward_shared(f.inputs, output=f.shared_output, stream=shared_stream)
                    if mode == "overlap":
                        self.shared_done.record(shared_stream)
            if mode != "shared-only":
                with self._stage("unsquash"):
                    f.unsquash()
            if mode in ("serial", "overlap"):
                with self._stage("combine"):
                    if mode == "overlap":
                        self.stream.wait_event(self.shared_done)
                    f.combine()
            if mode == "shared-only":
                return f.shared_output
            branch_output = f.unsquashed if mode == "routed-only" else f.combined
            with self._stage("postnorm" if f.postprocess.post_norm else "postnorm_bypass_copy"):
                f.postprocess.normalize(branch_output)
            if f.postprocess.residual_enabled:
                with self._stage("residual"):
                    return f.postprocess.add_residual()
            return f.postprocess.add_residual()


def _shared_reference(inputs, weights):
    """Local E1 oracle: FP32 GEMMs/SwiGLU, BF16 activation and FC2 handoffs.

    Unlike the routed reference this function never touches the global process
    group. It accepts CPU tensors as well as CUDA tensors.
    """
    import torch
    import torch.nn.functional as functional
    from .quantization import dequantize_mxfp8

    fc1, sf1, fc2, sf2 = weights
    if inputs.ndim != 2 or fc1.ndim != 3 or fc2.ndim != 3 or fc1.shape[0] != 1 or fc2.shape[0] != 1:
        raise ValueError("shared reference requires [T,H] input and exactly one expert")
    hidden, intermediate = inputs.shape[1], fc2.shape[2]
    if tuple(fc1.shape) != (1, 2 * intermediate, hidden) or tuple(fc2.shape) != (1, hidden, intermediate):
        raise ValueError("shared reference FC1/FC2 shapes do not match the unsquashed input")
    first = dequantize_mxfp8(fc1[0], sf1[0], dtype=torch.float32)
    second = dequantize_mxfp8(fc2[0], sf2[0], dtype=torch.float32)
    gate, up = (inputs.float() @ first.T).chunk(2, dim=-1)
    activation = (functional.silu(gate) * up).to(torch.bfloat16)
    return (activation.float() @ second.T).to(torch.bfloat16)


def _error_stats(actual, reference):
    import torch

    actual, reference = actual.float(), reference.to(actual.device).float()
    if actual.shape != reference.shape:
        raise AssertionError(f"reference shape mismatch: {actual.shape} != {reference.shape}")
    if not bool(torch.isfinite(actual).all() & torch.isfinite(reference).all()):
        raise AssertionError("nonfinite native or reference output")
    error = actual - reference
    return {
        "relative_l2": (error.norm() / reference.norm().clamp_min(1e-12)).item(),
        "max_abs": error.abs().max().item() if error.numel() else 0.0,
        "mean_abs": error.abs().mean().item() if error.numel() else 0.0,
    }


def _capture(layer, mode, repetitions, barrier=lambda: None):
    import torch

    layer.launch(mode)
    torch.cuda.synchronize(layer.device)
    barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=layer.stream):
        for _ in range(repetitions):
            layer.launch(mode)
    return graph


def _check(layer, routed_weights, shared_weights, args, barrier):
    import torch
    from .benchmark import _reference

    f = layer.frontend
    graphs = {}
    for mode in ("serial", "overlap"):
        _phase(layer.routed.config.rank, "check_capture", mode=mode)
        graphs[mode] = _capture(layer, mode, args.graph_batch, barrier)
    _phase(layer.routed.config.rank, "check_cpu_reference_setup")
    routed_weights_cpu = [weight.cpu() for weight in routed_weights]
    shared_weights_cpu = [weight.cpu() for weight in shared_weights]
    unsquash_weight_cpu = f.unsquash_weight.cpu()
    original = f.inputs.clone()
    original_residual = f.postprocess.residual.clone()
    layer.stream.wait_stream(torch.cuda.current_stream(layer.device))
    results = []
    previous_ids = None
    for sample in range(2):
        _phase(layer.routed.config.rank, "check_sample", sample=sample)
        # Alter graph inputs on their producer stream, not the consumer stream.
        with torch.cuda.stream(layer.stream):
            f.inputs.copy_(original if sample == 0 else -original)
            f.postprocess.residual.copy_(original_residual if sample == 0 else -original_residual + 0.0001)
        serial = layer.launch("serial")
        layer.stream.synchronize()
        expected = serial.clone()
        combined, normalized = f.combined.clone(), f.postprocess.normalized.clone()
        shared_output, routed_output = f.shared_output.clone(), f.routed_output.clone()
        ids = f.ids.cpu()
        if previous_ids is not None and f.inputs.shape[0] and layer.routed.config.num_experts > 1:
            if torch.equal(ids, previous_ids):
                raise AssertionError("changed synthetic input did not change routing")
        previous_ids = ids
        # The routed oracle intentionally uses the actual global Gloo group.
        # The shared oracle is local even when the benchmark runs at EP32.
        route_reference = _reference(layer.routed.config, f.squashed.cpu(), ids, f.scores.cpu(), routed_weights_cpu)
        shared_reference = _shared_reference(f.inputs.cpu(), shared_weights_cpu)
        combined_reference = (route_reference @ unsquash_weight_cpu) + shared_reference
        normalized_reference = f.postprocess._reference_normalized(combined_reference)
        final_reference = f.postprocess.reference(combined_reference)
        metrics = {
            "routed": _error_stats(routed_output, route_reference),
            "shared": _error_stats(shared_output, shared_reference),
            "combined": _error_stats(combined, combined_reference),
            "normalized": _error_stats(normalized, normalized_reference),
            "final": _error_stats(expected, final_reference),
        }
        for branch, error in metrics.items():
            if error["relative_l2"] > args.reference_relative_l2:
                raise AssertionError(f"{branch} sample {sample}: relative L2 {error} exceeds tolerance")
        if f.inputs.numel():
            if not bool(torch.count_nonzero(shared_output)):
                raise AssertionError("shared branch produced no contribution")
            if torch.equal(combined, f.unsquashed):
                raise AssertionError("shared contribution disappeared from the BF16 branch add")
        # Copies/oracle tensor producers above use the current stream. Make the
        # dependency explicit before overwriting their source buffers.
        layer.stream.wait_stream(torch.cuda.current_stream(layer.device))
        overlap = layer.launch("overlap")
        layer.stream.synchronize()
        torch.testing.assert_close(overlap, expected, rtol=0, atol=0)
        torch.testing.assert_close(f.combined, combined, rtol=0, atol=0)
        torch.testing.assert_close(f.postprocess.normalized, normalized, rtol=0, atol=0)
        for mode, graph in graphs.items():
            with torch.cuda.stream(layer.stream):
                graph.replay()
            layer.stream.synchronize()
            torch.testing.assert_close(f.output, expected, rtol=0, atol=0)
            torch.testing.assert_close(f.combined, combined, rtol=0, atol=0)
            torch.testing.assert_close(f.postprocess.normalized, normalized, rtol=0, atol=0)
        results.append({"sample": sample, "oracles": metrics, "eager_and_graph_schedules_bitwise_equal": True})
    with torch.cuda.stream(layer.stream):
        f.inputs.copy_(original)
        f.postprocess.residual.copy_(original_residual)
    layer.stream.synchronize()
    del graph
    graphs.clear()
    gc.collect()
    return {
        "samples": results,
        "reference_device": "cpu",
        "changed_inputs_checked": True,
        "changed_residuals_checked": f.postprocess.residual_enabled,
        "zero_tokens": not bool(f.inputs.shape[0]),
    }


def _time_mode(layer, mode, args, barrier, rank):
    import torch

    _phase(rank, "warmup", mode=mode)
    for _ in range(args.warmup):
        layer.launch(mode)
    _phase(rank, "capture", mode=mode, graph_batch=args.graph_batch)
    graph = _capture(layer, mode, args.graph_batch, barrier)
    with torch.cuda.stream(layer.stream):
        for _ in range(args.warmup):
            graph.replay()
    layer.stream.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    _phase(rank, "timing", mode=mode, iterations=args.iterations)
    for _ in range(args.iterations):
        barrier()
        with torch.cuda.stream(layer.stream):
            begin.record()
            graph.replay()
            end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / args.graph_batch)
    _phase(rank, "timing_done", mode=mode, median_us=statistics.median(samples))
    del graph
    gc.collect()
    return samples


def _trace_summary(trace, routed_ctas, shared_ctas, expected_pairs=3):
    """Extract actual GPU intervals, not CPU launch or profiler-range timestamps."""
    rows = []
    for event in trace.get("traceEvents", []):
        if event.get("cat") != "kernel" or "megamoe" not in event.get("name", "").lower():
            continue
        args = event.get("args", {})
        grid = args.get("grid")
        if not isinstance(grid, (list, tuple)) or len(grid) != 3:
            raise RuntimeError("native trace kernel lacks a three-dimensional grid")
        rows.append(
            {
                "name": event["name"],
                "start_us": event["ts"],
                "duration_us": event["dur"],
                "end_us": event["ts"] + event["dur"],
                "stream": args.get("stream"),
                "grid": list(grid),
            }
        )
    rows.sort(key=lambda row: row["start_us"])
    if routed_ctas == shared_ctas:
        raise RuntimeError("trace grid attribution requires distinct routed/shared CTA counts")
    routed = [row for row in rows if row["grid"] == [routed_ctas, 1, 1]]
    shared = [row for row in rows if row["grid"] == [shared_ctas, 1, 1]]
    if len(rows) != 2 * expected_pairs or len(routed) != expected_pairs or len(shared) != expected_pairs:
        raise RuntimeError(
            f"expected {expected_pairs} routed/shared trace pairs, observed {len(routed)}/{len(shared)} "
            f"({len(rows)} native kernel rows)"
        )
    pairs = []
    for index, (route, local) in enumerate(zip(routed, shared)):
        if route["stream"] is None or local["stream"] is None or route["stream"] == local["stream"]:
            raise RuntimeError("trace does not show separate native routed/shared streams")
        if local["start_us"] < route["start_us"]:
            raise RuntimeError(f"trace pair {index}: shared kernel started before routed kernel")
        if index + 1 < len(routed) and max(route["end_us"], local["end_us"]) > routed[index + 1]["start_us"]:
            raise RuntimeError("trace shows concurrent reuse across layer invocations")
        overlap = max(0.0, min(route["end_us"], local["end_us"]) - local["start_us"])
        pairs.append(
            {
                "iteration": index,
                "routed": route,
                "shared": local,
                "shared_starts_after_routed_us": local["start_us"] - route["start_us"],
                "overlap_us": overlap,
                "paired_span_us": max(route["end_us"], local["end_us"]) - route["start_us"],
            }
        )
    return {
        "scope": "profiled layer iterations, including possible first-sample outliers; not steady-state latency",
        "native_kernel_rows": len(rows),
        "paired_iterations": len(pairs),
        "routed_ctas": routed_ctas,
        "shared_ctas": shared_ctas,
        "routed_streams": sorted({row["stream"] for row in routed}),
        "shared_streams": sorted({row["stream"] for row in shared}),
        "routed_first": True,
        "multistream": True,
        "overlapped_iterations": sum(pair["overlap_us"] > 0 for pair in pairs),
        "total_overlap_us": sum(pair["overlap_us"] for pair in pairs),
        "total_paired_span_us": sum(pair["paired_span_us"] for pair in pairs),
        "total_shared_duration_us": sum(row["duration_us"] for row in shared),
        "pairs": pairs,
    }


def _trace(layer, path, barrier, rank, eager=False):
    import torch

    _phase(rank, "trace_capture", iterations=3)
    graph = None if eager else _capture(layer, "overlap", 1, barrier)
    with torch.cuda.stream(layer.stream):
        layer.launch("overlap") if eager else graph.replay()
    layer.stream.synchronize()
    barrier()
    # Every rank profiles; profiling only rank zero can stall its active peers.
    layer.annotate = eager
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as p:
        for _ in range(3):
            with torch.profiler.record_function("megamoe/routed_first_overlap"), torch.cuda.stream(layer.stream):
                layer.launch("overlap") if eager else graph.replay()
        layer.stream.synchronize()
    layer.annotate = False
    barrier()
    _phase(rank, "trace_export", path=path if rank == 0 else None)
    result = None
    if rank == 0:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        p.export_chrome_trace(str(destination))
        with destination.open() as trace_file:
            result = _trace_summary(json.load(trace_file), layer.routed.cta_count, layer.shared.cta_count)
        result["launch_mode"] = "eager_with_host_stage_labels" if eager else "single_layer_graph_replay"
    del graph, p
    gc.collect()
    return result


def _summarize_samples(reports, mode):
    samples = [
        max(report["samples_us"][mode][i] for report in reports) for i in range(len(reports[0]["samples_us"][mode]))
    ]
    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "samples": samples,
    }


def main(argv=None):
    args = _parse_args(argv)
    import torch
    import torch.distributed as dist
    from mscclpp import Communicator, TcpBootstrap
    from .api import MegaMoE, is_available
    from .benchmark import _weights

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not all(os.environ.get(name) for name in ("MASTER_ADDR", "MASTER_PORT")):
        raise RuntimeError("set explicit MASTER_ADDR and MASTER_PORT, normally using torchrun")
    port = args.bootstrap_port if args.bootstrap_port is not None else int(os.environ["MASTER_PORT"]) + 1
    if not 1 <= port <= 65535:
        raise ValueError("native bootstrap port must be in [1, 65535]")
    if not torch.cuda.is_available() or torch.version.hip or not is_available():
        raise RuntimeError("requires the native MegaMoE build and SM100 CUDA GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("native MegaMoE requires SM100")
    # The router/reference explicitly request FP32, not implicit TF32 matmuls.
    torch.backends.cuda.matmul.allow_tf32 = False
    physical_sms = torch.cuda.get_device_properties(device).multi_processor_count
    route_config, shared_config = _configs(args, rank, world, physical_sms)
    if args.trace_path and physical_sms - args.route_sm_margin == args.shared_sms:
        raise ValueError("--trace-path requires distinct routed/shared CTA counts for grid attribution")
    _phase(rank, "initialization_start", world_size=world, physical_sms=physical_sms)
    start = time.perf_counter()
    dist.init_process_group("gloo", rank=rank, world_size=world)
    bootstrap = TcpBootstrap.create(rank, world)
    bootstrap.initialize(f"{os.environ['MASTER_ADDR']}:{port}")
    communicator = Communicator(bootstrap)
    _phase(rank, "routed_weights")
    torch.manual_seed(args.seed + rank)
    routed_weights = _weights(route_config, device)
    context_start = time.perf_counter()
    routed = MegaMoE(route_config, communicator, *routed_weights)
    route_init_ms = (time.perf_counter() - context_start) * 1000
    _phase(rank, "shared_weights")
    # This is NOT a rank/world override on the global communicator: the shared
    # expert owns a separate one-rank bootstrap, registration, and workspace.
    shared_bootstrap = TcpBootstrap.create(0, 1)
    shared_bootstrap.initialize(TcpBootstrap.create_unique_id())
    shared_communicator = Communicator(shared_bootstrap)
    torch.manual_seed(args.seed + 100000)
    shared_weights = _weights(shared_config, device)
    context_start = time.perf_counter()
    shared = MegaMoE(shared_config, shared_communicator, *shared_weights)
    shared_init_ms = (time.perf_counter() - context_start) * 1000
    _assert_ctas(routed, shared, physical_sms, args)
    torch.manual_seed(args.seed + 200000 + rank)
    inputs = torch.randn((args.tokens, args.original_hidden), device=device, dtype=torch.bfloat16)
    # Replicate synthetic projection/router parameters across ranks.
    torch.manual_seed(args.seed + 300000)
    postprocess = _Postprocess(
        inputs,
        epsilon=args.rms_eps,
        post_norm=args.post_norm,
        add_residual=args.residual,
        residual_dtype=args.residual_dtype,
    )
    frontend = _Frontend(inputs, route_config, postprocess=postprocess)
    layer = _Layer(frontend, routed, shared)
    torch.cuda.synchronize(device)
    initialization_ms = (time.perf_counter() - start) * 1000
    _phase(
        rank,
        "initialization_done",
        initialization_ms=initialization_ms,
        routed_ctas=routed.cta_count,
        shared_ctas=shared.cta_count,
    )
    correctness = None
    if args.check:
        _phase(rank, "check_start")
        correctness = _check(layer, routed_weights, shared_weights, args, dist.barrier)
        _phase(rank, "check_done")
    del routed_weights, shared_weights
    samples = {mode: _time_mode(layer, mode, args, dist.barrier, rank) for mode in MODES}
    trace = _trace(layer, args.trace_path, dist.barrier, rank, args.trace_eager) if args.trace_path else None
    report = {
        "rank": rank,
        "hardware": torch.cuda.get_device_name(device),
        "physical_sms": physical_sms,
        "routed_ctas": routed.cta_count,
        "shared_ctas": shared.cta_count,
        "workspace_bytes": {"routed": routed.workspace_bytes, "shared": shared.workspace_bytes},
        "initialization_ms": initialization_ms,
        "native_context_initialization_ms": {"routed": route_init_ms, "shared": shared_init_ms},
        "samples_us": samples,
        "median_us": {mode: statistics.median(values) for mode, values in samples.items()},
        "correctness": correctness,
    }
    reports = [None] * world
    dist.all_gather_object(reports, report)
    if rank == 0:
        summaries = {mode: _summarize_samples(reports, mode) for mode in MODES}
        result = {
            "implementation": "mscclpp-native-cuda-megamoe-shared",
            **_scope_report(args),
            "not_full_sglang_model_parity": True,
            "configuration": {**vars(args), "experts": route_config.num_experts, "world_size": world},
            "routed_config": vars(route_config),
            "shared_config": vars(shared_config),
            "shared_execution_path": "unweighted local expert; direct input/output, no dispatch or top-k combine",
            "dtypes": {
                "inputs_projections_branch_sum": "BF16",
                "postnorm_math_and_weight": "FP32",
                "normalized_output": "BF16",
                "residual": args.residual_dtype.upper(),
                "final_output": args.residual_dtype.upper() if args.residual else "BF16",
                "router_linear_logits_selected_softmax": "FP32",
                "routing_ids": "int32",
                "native_expert_weights": "MXFP8 E4M3FN with E8M0 K32 scales",
            },
            "input_mode": "both native branches staged on every invocation, inside the measured graph",
            "graph_samples": "same input reused; router and both native staging copies execute every invocation",
            "sm_partition": "soft persistent CTA/SM capacity caps, not fixed SM IDs or exclusive SM ownership",
            "timing": (
                "CUDA events around graph replay, including fork/join; microseconds per complete layer invocation"
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "latency_us_max_across_ranks": summaries,
            "serial_same_cap_over_overlap_speedup": summaries["serial"]["median"] / summaries["overlap"]["median"],
            "comparison_caveat": (
                "same-cap native serial baseline, not a production shared kernel or previous router-free run"
            ),
            "trace": trace,
            "ranks": reports,
        }
        text = json.dumps(result, indent=2)
        print(text, flush=True)
        if args.json_output:
            path = Path(args.json_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
    torch.cuda.synchronize(device)
    bootstrap.barrier()
    shared_bootstrap.barrier()
    _phase(rank, "teardown")
    # Capture helpers explicitly release their graphs before returning. Retain
    # both bootstraps until all ranks have destroyed their native contexts.
    del layer, frontend, postprocess, inputs, routed, shared
    gc.collect()
    bootstrap.barrier()
    del communicator, shared_communicator
    gc.collect()
    del bootstrap, shared_bootstrap
    gc.collect()
    dist.destroy_process_group()
    _phase(rank, "complete")


if __name__ == "__main__":
    main()
