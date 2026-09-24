# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic full-layer benchmark with FlashInfer routed and MSCCL++ shared experts."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

from .benchmark_flashinfer import (
    _activate_flashinfer_root,
    _configure_single_node_transport,
    _load_knobs,
    _runtime_versions,
    _source_info,
    _weights as _flashinfer_weights,
)
from .benchmark_shared import (
    MODES,
    _Frontend,
    _Postprocess,
    _check,
    _parse_args as _parse_shared_args,
    _phase,
    _scope_report as _native_scope_report,
    _summarize_samples,
    _time_mode,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e5m2", action="store_true")
    parser.add_argument("--gate-up-clamp", type=float, default=-1.0)
    parser.add_argument("--fast-math", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--in-kernel-fc2-reduce", action="store_true")
    knobs = parser.add_mutually_exclusive_group()
    knobs.add_argument("--autotune", action="store_true", help="run FlashInfer's collective knob tuner at warmup")
    knobs.add_argument("--knobs-json", help="inline JSON object or path to JSON with pinned FlashInfer kernel knobs")
    parser.add_argument(
        "--flashinfer-root",
        default=os.environ.get("FLASHINFER_ROOT"),
        help="FlashInfer source checkout; defaults to FLASHINFER_ROOT or sibling ../flashinfer",
    )
    parser.add_argument(
        "--flashinfer-cache-root",
        default=os.environ.get("FLASHINFER_BENCHMARK_CACHE_ROOT"),
        help="persistent FlashInfer cache root; each global rank uses a separate subdirectory",
    )
    options = {argument.split("=", 1)[0] for argument in (argv if argv is not None else os.sys.argv[1:])}
    args = _parse_shared_args(argv, parser=parser)
    if not math.isfinite(args.gate_up_clamp):
        parser.error("gate-up-clamp must be finite; negative disables clamping")
    if args.e5m2 and "--reference-relative-l2" not in options:
        args.reference_relative_l2 = 0.12
    if args.in_kernel_fc2_reduce and args.check:
        parser.error("--check requires deterministic FC2 reduction; omit --in-kernel-fc2-reduce")
    if args.trace_path:
        parser.error("FlashInfer shared benchmark does not support --trace-path")
    if args.tuned_profile or args.cache_dir or args.bootstrap_port is not None:
        parser.error("--tuned-profile, --cache-dir, and --bootstrap-port apply only to the native routed backend")
    if args.route_sm_margin != 32:
        parser.error("FlashInfer does not expose the native routed SM-margin control; keep --route-sm-margin=32")
    if args.hidden < 32 or args.hidden % 32:
        parser.error("hidden must be a positive multiple of 32 for FlashInfer")
    if args.intermediate < 64 or args.intermediate % 64:
        parser.error("intermediate must be a positive multiple of 64 for FlashInfer")
    if args.original_hidden < 128 or args.original_hidden % 128:
        parser.error("original-hidden must be a positive multiple of 128 for the native shared expert")
    if args.shared_intermediate < 128 or args.shared_intermediate % 128:
        parser.error("shared-intermediate must be a positive multiple of 128")
    return args


@dataclass(frozen=True)
class _RoutedConfig:
    rank: int
    world_size: int
    max_tokens: int
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    gate_up_clamp: float

    @property
    def local_experts(self):
        return self.num_experts // self.world_size


def _configs(args, rank, world, physical_sms):
    from .api import MegaMoEConfig

    if args.experts is None:
        experts = 16 * world
    else:
        experts = args.experts
    if experts < 1 or experts % world:
        raise ValueError("global routed experts must be positive and divisible by world size")
    if not 1 <= args.top_k <= min(32, experts):
        raise ValueError("top-k must be in [1, min(32, experts)]")
    if args.shared_sms < 2 or args.shared_sms % 2 or args.shared_sms > physical_sms:
        raise ValueError("shared-sms must be an even count in [2, physical SM count]")
    capacity = max(1, args.tokens)
    routed = _RoutedConfig(
        rank=rank,
        world_size=world,
        max_tokens=capacity,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_experts=experts,
        top_k=args.top_k,
        gate_up_clamp=args.gate_up_clamp,
    )
    shared = MegaMoEConfig(
        rank=0,
        world_size=1,
        max_tokens=capacity,
        hidden=args.original_hidden,
        intermediate=args.shared_intermediate,
        num_experts=1,
        top_k=1,
        sm_margin=physical_sms - args.shared_sms,
    )
    return routed, shared


def _scope_report(args):
    report = _native_scope_report(args)
    report["scope"] = "synthetic_router_squash_flashinfer_routed_unsquash_plus_native_shared_bf16_add"
    report["included_by_mode"]["overlap"] = [
        stage if stage != "routed_entry_start_gate" else "routed_enqueue_first_producer_ready_event"
        for stage in report["included_by_mode"]["overlap"]
    ]
    report["overlap_policy"] = {
        "routed_first": "FlashInfer routed work is enqueued before the native shared expert",
        "shared_gate": "shared stream waits for original-input/router/squash producers, not routed kernel entry",
        "limitation": "FlashInfer public API exposes neither a kernel-entry signal nor a routed CTA/SM cap",
    }
    return report


def _routed_reference(config, inputs, ids, scores, weights):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional

    def gather(tensor):
        gathered = [torch.empty_like(tensor) for _ in range(config.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered)

    all_inputs, all_ids, all_scores = gather(inputs), gather(ids), gather(scores)
    partial = torch.zeros(
        all_inputs.shape[0],
        config.top_k,
        config.hidden,
        dtype=torch.float32,
    )
    w13, w2 = weights
    for expert in range(config.local_experts):
        token, slot = torch.where(all_ids == config.rank * config.local_experts + expert)
        if not token.numel():
            continue
        gate, up = (all_inputs[token].float() @ w13[expert].float().T).chunk(2, dim=-1)
        if config.gate_up_clamp >= 0:
            gate = gate.clamp(max=config.gate_up_clamp)
            up = up.clamp(-config.gate_up_clamp, config.gate_up_clamp)
        activated = (functional.silu(gate) * up * all_scores[token, slot, None]).to(torch.bfloat16)
        partial[token, slot] = (activated.float() @ w2[expert].float().T).to(torch.bfloat16).float()
    dist.all_reduce(partial)
    first_token = config.rank * inputs.shape[0]
    return partial[first_token : first_token + inputs.shape[0]].sum(dim=1).to(torch.bfloat16)


class _FlashInferRouted:
    def __init__(self, config, layer, tensors, output):
        self.config = config
        self.layer = layer
        self.tensors = tensors
        self.output = output
        self.output_pointer = output.data_ptr()

    def forward(self):
        output = self.layer.forward(self.tensors, return_workspace_view=True)
        if output.data_ptr() != self.output_pointer:
            raise RuntimeError("FlashInfer workspace output address changed after warmup")
        return output

    def destroy(self):
        self.layer.destroy()


class _Layer:
    def __init__(self, frontend, routed, shared, stream):
        import torch

        self.frontend, self.routed, self.shared = frontend, routed, shared
        self.device = frontend.inputs.device
        self.stream = stream
        self.shared_stream = torch.cuda.Stream(device=self.device)
        self.routed_inputs_ready = torch.cuda.Event()
        self.shared_done = torch.cuda.Event()
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.shared_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.routed_inputs_ready.record(torch.cuda.current_stream(self.device))
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
                if mode == "overlap":
                    self.routed_inputs_ready.record(self.stream)
                with self._stage("routed"):
                    routed_output = self.routed.forward()
                    if routed_output.data_ptr() != f.routed_output.data_ptr():
                        raise RuntimeError("FlashInfer routed output view changed")
            if mode != "routed-only":
                shared_stream = self.shared_stream if mode == "overlap" else self.stream
                with self._stage("shared"), torch.cuda.stream(shared_stream):
                    if mode == "overlap":
                        shared_stream.wait_event(self.routed_inputs_ready)
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


def main(argv=None):
    args = _parse_args(argv)
    flashinfer_root = _activate_flashinfer_root(args.flashinfer_root)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _configure_single_node_transport(world, int(os.environ.get("LOCAL_WORLD_SIZE", world)))
    if args.flashinfer_cache_root:
        cache = Path(args.flashinfer_cache_root).expanduser().resolve() / f"rank-{rank}"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = str(cache)
        os.environ["XDG_CACHE_HOME"] = str(cache / ".cache")

    import torch
    import torch.distributed as dist
    import flashinfer
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from mscclpp import Communicator, TcpBootstrap
    from .api import MegaMoE, is_available
    from .benchmark import _weights as _native_weights

    if not all(os.environ.get(name) for name in ("MASTER_ADDR", "MASTER_PORT")):
        raise RuntimeError("set explicit MASTER_ADDR and MASTER_PORT, normally using torchrun")
    if not torch.cuda.is_available() or torch.version.hip or not is_available():
        raise RuntimeError("requires FlashInfer, the native MegaMoE build, and GB200 GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("this benchmark supports GB200 (SM100) only")
    torch.backends.cuda.matmul.allow_tf32 = False
    physical_sms = torch.cuda.get_device_properties(device).multi_processor_count
    _phase(rank, "initialization_start", world_size=world, physical_sms=physical_sms)
    start = time.perf_counter()
    dist.init_process_group("gloo", rank=rank, world_size=world)
    nccl_group = dist.new_group(backend="nccl")
    route_config, shared_config = _configs(args, rank, world, physical_sms)
    clamp = None if args.gate_up_clamp < 0 else args.gate_up_clamp
    requested_knobs = "auto" if args.autotune else _load_knobs(args.knobs_json)
    kernel_config = Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=args.intermediate,
        top_k=args.top_k,
        kind="bf16_mxfp8_e5m2" if args.e5m2 else "bf16_mxfp8_e4m3",
        gate_up_clamp=clamp,
        fast_math=args.fast_math,
        enable_in_kernel_fc2_reduce=args.in_kernel_fc2_reduce,
        knobs=requested_knobs,
    )
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))

    _phase(rank, "routed_weights")
    routed_weights = _flashinfer_weights(
        rank,
        route_config.local_experts,
        route_config.hidden,
        route_config.intermediate,
        device,
        seed=args.seed,
    )
    construction_start = time.perf_counter()
    with torch.cuda.stream(stream):
        flash_layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world,
                rank=rank,
                stream=stream.cuda_stream,
                device=local_rank,
                process_group=nccl_group,
            ),
            fleet_params=FleetParams(
                num_experts=route_config.num_experts,
                max_tokens_per_rank=route_config.max_tokens,
                token_hidden_size=route_config.hidden,
            ),
            weights=MoEWeightPack(w13=routed_weights[0], w2=routed_weights[1]),
            backend=MegaConfig(
                megakernel=kernel_config,
                quantize_input=True,
                preprocess_weights=True,
            ),
        )
    flash_construction_ms = (time.perf_counter() - construction_start) * 1000

    _phase(rank, "shared_weights")
    shared_bootstrap = TcpBootstrap.create(0, 1)
    shared_bootstrap.initialize(TcpBootstrap.create_unique_id())
    shared_communicator = Communicator(shared_bootstrap)
    torch.manual_seed(args.seed + 100000)
    shared_weights = _native_weights(shared_config, device)
    context_start = time.perf_counter()
    shared = MegaMoE(shared_config, shared_communicator, *shared_weights)
    shared_init_ms = (time.perf_counter() - context_start) * 1000
    if shared.cta_count != args.shared_sms:
        raise RuntimeError(f"requested {args.shared_sms} shared CTAs, native occupancy selected {shared.cta_count}")

    torch.manual_seed(args.seed + 200000 + rank)
    inputs = torch.randn((args.tokens, args.original_hidden), device=device, dtype=torch.bfloat16)
    torch.manual_seed(args.seed + 300000)
    postprocess = _Postprocess(
        inputs,
        epsilon=args.rms_eps,
        post_norm=args.post_norm,
        add_residual=args.residual,
        residual_dtype=args.residual_dtype,
    )
    frontend = _Frontend(
        inputs,
        route_config,
        postprocess=postprocess,
        router_allow_tf32=args.router_allow_tf32,
        router_weight_layout=args.router_weight_layout,
        router_probability_order=args.router_probability_order,
    )
    with torch.cuda.stream(stream):
        frontend.router()
        frontend.squash()
    stream.synchronize()
    dist.barrier()
    routed_tensors = MoEEpTensors(
        hidden_states=frontend.squashed,
        topk_ids=frontend.indices64,
        topk_weights=frontend.scores,
    )
    _phase(rank, "flashinfer_collective_warmup")
    warmup_start = time.perf_counter()
    with torch.cuda.stream(stream):
        flash_layer.warmup(routed_tensors)
        routed_output = flash_layer.forward(routed_tensors, return_workspace_view=True)
    stream.synchronize()
    dist.barrier()
    flash_warmup_ms = (time.perf_counter() - warmup_start) * 1000
    frontend.routed_output = routed_output
    routed = _FlashInferRouted(route_config, flash_layer, routed_tensors, routed_output)
    layer = _Layer(frontend, routed, shared, stream)
    torch.cuda.synchronize(device)
    initialization_ms = (time.perf_counter() - start) * 1000
    _phase(rank, "initialization_done", initialization_ms=initialization_ms, shared_ctas=shared.cta_count)

    correctness = None
    if args.check:
        _phase(rank, "check_start")
        correctness = _check(
            layer,
            routed_weights,
            shared_weights,
            args,
            dist.barrier,
            routed_reference=_routed_reference,
        )
        _phase(rank, "check_done")
    del routed_weights, shared_weights
    samples = {mode: _time_mode(layer, mode, args, dist.barrier, rank) for mode in args.modes}
    report = {
        "rank": rank,
        "physical_sms": physical_sms,
        "shared_ctas": shared.cta_count,
        "workspace_bytes": {"routed": None, "shared": shared.workspace_bytes},
        "initialization_ms": initialization_ms,
        "context_initialization_ms": {
            "flashinfer_construction": flash_construction_ms,
            "flashinfer_warmup_compile": flash_warmup_ms,
            "native_shared": shared_init_ms,
        },
        "samples_us": samples,
        "median_us": {mode: statistics.median(values) for mode, values in samples.items()},
        "correctness": correctness,
    }
    reports = [None] * world
    dist.all_gather_object(reports, report)
    if rank == 0:
        summaries = {mode: _summarize_samples(reports, mode) for mode in args.modes}
        result = {
            "implementation": "flashinfer-routed-plus-mscclpp-native-shared-megamoe",
            **_scope_report(args),
            "not_full_sglang_model_parity": True,
            "configuration": {
                **vars(args),
                "experts": route_config.num_experts,
                "world_size": world,
                "kind": kernel_config.kind,
                "knobs": requested_knobs,
            },
            "routed_config": {
                "fleet": asdict(route_config),
                "backend": asdict(kernel_config),
                "output": "FlashInfer symmetric-workspace view consumed directly by unsquash",
                "routed_sm_cap": None,
            },
            "shared_config": vars(shared_config),
            "shared_execution_path": "MSCCL++ unweighted local expert; direct input/output",
            "flashinfer": _source_info(flashinfer_root, flashinfer),
            "runtime_versions": _runtime_versions(),
            "communication_environment": {
                name: os.environ.get(name)
                for name in ("NCCL_IB_DISABLE", "NCCL_NET", "NCCL_MNNVL_ENABLE", "NVSHMEM_REMOTE_TRANSPORT")
            },
            "dtypes": {
                "inputs_projections_branch_sum": "BF16",
                "flashinfer_routed_weights": "MXFP8 E4M3/E5M2 with E8M0 K32 scales",
                "native_shared_weights": "MXFP8 E4M3FN with E8M0 K32 scales",
                "postnorm_math_and_weight": "FP32",
                "normalized_output": "BF16",
                "residual": args.residual_dtype.upper(),
                "final_output": args.residual_dtype.upper() if args.residual else "BF16",
            },
            "input_mode": "both expert branches stage inputs on every invocation inside the measured graph",
            "graph_samples": "same input reused; router, projections, both staging paths, and postprocess run each time",
            "sm_partition": (
                "native shared branch requests a persistent CTA cap; FlashInfer routed backend exposes no equivalent "
                "public SM cap or exclusive partition"
            ),
            "timing": "CUDA events around graph replay; microseconds per complete synthetic layer invocation",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "latency_us_max_across_ranks": summaries,
            "serial_over_overlap_speedup": (
                summaries["serial"]["median"] / summaries["overlap"]["median"]
                if {"serial", "overlap"} <= summaries.keys()
                else None
            ),
            "comparison_caveat": (
                "frontend and native shared branch match benchmark_shared; FlashInfer overlap is enqueue-first rather "
                "than native kernel-entry-gated and routed SM capacity is not capped"
            ),
            "ranks": reports,
        }
        text = json.dumps(result, indent=2)
        print(text, flush=True)
        if args.json_output:
            path = Path(args.json_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")

    torch.cuda.synchronize(device)
    dist.barrier()
    shared_bootstrap.barrier()
    _phase(rank, "teardown")
    del layer, frontend, postprocess, inputs
    routed.destroy()
    del routed, shared
    gc.collect()
    dist.barrier()
    shared_bootstrap.barrier()
    del shared_communicator, shared_bootstrap
    dist.destroy_process_group(nccl_group)
    dist.destroy_process_group()
    _phase(rank, "complete")


if __name__ == "__main__":
    main()
