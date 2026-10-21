# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optional NCCL-EP native LL/HT adapters (no MPI bootstrap)."""

from contextlib import ExitStack
from importlib.metadata import version

import torch
import torch.distributed as dist


def preflight(config):
    """Check optional APIs before any rank enters a native collective."""
    try:
        import nccl.core as core
        import nccl.ep as ep
    except ImportError as error:
        raise RuntimeError(
            "NCCL-EP requires nccl.core and nccl.ep. Install NVIDIA's "
            "'nccl-extensions[cu13]' (or [cu12] matching PyTorch), with a matching "
            "CUDA toolkit/nvcc; see src/ext/ep/README.md. MSCCL++ does not require this extra."
        ) from error
    environment = config["environment"]
    mode, layout = environment.get("mode", "latency"), environment.get("layout", "expert_major")
    algorithm = ep.Algorithm.LOW_LATENCY
    if mode == "throughput":
        algorithm = getattr(ep.Algorithm, "THROUGHPUT", getattr(ep.Algorithm, "HIGH_THROUGHPUT", None))
        if algorithm is None:
            raise RuntimeError("Installed NCCL-EP does not expose a throughput algorithm; upgrade nccl-extensions")
        if config["input"]["tokens_per_rank"] % 2:
            raise ValueError(
                "NCCL-EP throughput benchmark requires even tokens_per_rank for native S2D transfer alignment"
            )
    native_layout = "FLAT" if layout == "token_major" else layout.upper()
    if not hasattr(ep.Layout, native_layout):
        raise RuntimeError(f"Installed NCCL-EP does not expose Layout.{native_layout}; upgrade nccl-extensions")
    if (mode == "throughput" or layout != "expert_major") and not hasattr(ep, "ExpertIdKind"):
        raise RuntimeError("Installed NCCL-EP lacks explicit received expert-ID numbering; upgrade nccl-extensions")
    return core, ep, algorithm, native_layout


class NcclBackend:
    def __init__(self, config, rank, x, ids, weights, recv, expert_output, output):
        core, ep, algorithm, native_layout = preflight(config)

        model = config["model"]
        world = config["environment"]["num_gpus"]
        self.mode = config["environment"].get("mode", "latency")
        self.layout_name = config["environment"].get("layout", "expert_major")
        throughput = self.mode == "throughput"
        self.adapt_layout = throughput or self.layout_name == "rank_major"
        self.route_id_base = 0
        self.route_valid_rows = None
        local_experts = model["num_experts"] // world
        capacity = world * config["input"]["tokens_per_rank"]
        if throughput and self.layout_name == "expert_major":
            capacity *= min(model["top_k"], local_experts)
        # Keep tensors and descriptors alive: native handles cache their addresses.
        self.x, self.ids, self.weights = x, ids, weights
        self.recv, self.expert_output, self.output = recv, expert_output, output
        self.counts = torch.empty(model["num_experts"] // world, dtype=torch.int32, device=x.device)
        self.metadata = {
            "communication_backend": "nccl",
            "nccl4py_version": version("nccl4py"),
            "nccl_ep_version": version("nccl-extensions"),
            "nccl_version": str(core.get_version().libnccl.version),
            "communication_algorithm": getattr(algorithm, "name", str(algorithm)).lower().replace("-", "_"),
            "communication_mode": self.mode,
            "dispatch_layout": self.layout_name,
            "native_dispatch_layout": native_layout.lower(),
            "combine_mode": (
                "preweighted_expert_rows"
                if throughput and self.layout_name == "expert_major"
                else "preweighted_local_reduce" if self.adapt_layout else "weighted_topk"
            ),
        }
        with ExitStack() as resources:
            uid = [core.get_unique_id() if rank == 0 else None]
            dist.broadcast_object_list(uid, src=0)
            self.comm = core.Communicator.init(nranks=world, rank=rank, unique_id=uid[0])
            resources.callback(self.comm.destroy)
            self.group = ep.Group.create(
                self.comm,
                ep.GroupConfig(
                    algorithm=algorithm,
                    num_experts=model["num_experts"],
                    max_dispatch_tokens_per_rank=config["input"]["tokens_per_rank"],
                    max_recv_tokens_per_rank=capacity,
                    max_token_bytes=model["d_model"] * 2,
                    alloc=ep.AllocConfig(),
                ),
            )
            resources.callback(self.group.destroy)
            self.ids_tensor = ep.Tensor(ids)
            handle_layout = None
            if throughput:
                self.offsets = torch.empty_like(self.counts)
                self.total = torch.empty(1, dtype=torch.int32, device=x.device)
                handle_layout = ep.LayoutInfo(
                    expert_counters=ep.Tensor(self.counts),
                    recv_total_counter=ep.Tensor(self.total),
                    expert_offsets=ep.Tensor(self.offsets) if self.layout_name == "expert_major" else None,
                    recv_topk_idx_kind=ep.ExpertIdKind.LOCAL,
                )
            self.handle = self.group.create_handle(
                getattr(ep.Layout, native_layout),
                self.ids_tensor,
                layout_info=handle_layout,
                config=ep.HandleConfig(),
                stream=torch.cuda.current_stream().cuda_stream,
            )
            resources.callback(self.handle.destroy)
            resources.callback(torch.cuda.synchronize, x.device)
            self.combine_buffer = expert_output
            if self.adapt_layout:
                shape = (
                    (capacity, model["d_model"])
                    if throughput
                    else (world, config["input"]["tokens_per_rank"], model["d_model"])
                )
                self.recv = torch.empty(shape, dtype=x.dtype, device=x.device)
                self.combine_buffer = torch.empty_like(self.recv)
                if throughput and self.layout_name == "expert_major":
                    self.route_weights = torch.empty((capacity, 1), dtype=torch.float32, device=x.device)
                    self.route_ids = torch.full((capacity, 1), -1, dtype=torch.int32, device=x.device)
                    for expert, (offset, count) in enumerate(
                        zip(self.offsets.cpu().tolist(), self.counts.cpu().tolist())
                    ):
                        self.route_ids[offset : offset + count] = expert
                    output_ids = None
                else:
                    self.route_weights = torch.empty(
                        (*shape[:-1], model["top_k"]), dtype=torch.float32, device=x.device
                    )
                    self.route_ids = torch.full(
                        self.route_weights.shape, -1, dtype=torch.int64 if throughput else torch.int32, device=x.device
                    )
                    output_ids = ep.Tensor(self.route_ids)
                    if throughput:
                        self.route_valid_rows = torch.arange(capacity, device=x.device) < self.total[0]
                self.dispatch_inputs = ep.DispatchInputs(tokens=ep.Tensor(x), topk_weights=ep.Tensor(weights))
                self.dispatch_outputs = ep.DispatchOutputs(
                    tokens=ep.Tensor(self.recv),
                    topk_weights=ep.Tensor(
                        self.route_weights.view(-1)
                        if throughput and self.layout_name == "expert_major"
                        else self.route_weights
                    ),
                    topk_idx=output_ids,
                )
                if throughput:
                    self.layout = None
                else:
                    self.rank_counts = torch.empty(world, dtype=torch.int32, device=x.device)
                    self.layout = ep.LayoutInfo(
                        src_rank_counters=ep.Tensor(self.rank_counts), recv_topk_idx_kind=ep.ExpertIdKind.LOCAL
                    )
            else:
                self.dispatch_inputs = ep.DispatchInputs(tokens=ep.Tensor(x))
                self.dispatch_outputs = ep.DispatchOutputs(tokens=ep.Tensor(recv))
                self.layout = ep.LayoutInfo(expert_counters=ep.Tensor(self.counts))
            self.dispatch_config = ep.DispatchConfig()
            self.combine_inputs = ep.CombineInputs(tokens=ep.Tensor(self.combine_buffer))
            self.combine_outputs = ep.CombineOutputs(
                tokens=ep.Tensor(output), topk_weights=None if self.adapt_layout else ep.Tensor(weights)
            )
            self.combine_config = ep.CombineConfig()
            self._resources = resources.pop_all()

    def dispatch(self):
        self.handle.dispatch(
            self.dispatch_inputs,
            self.dispatch_outputs,
            layout_info=self.layout,
            config=self.dispatch_config,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        if self.layout_name == "rank_major" and self.route_valid_rows is None:
            positions = torch.arange(self.x.shape[0], device=self.x.device)
            self.route_valid_rows = (positions[None, :] < self.rank_counts[:, None]).flatten()
        return self.recv, None if self.adapt_layout else self.counts

    def combine(self):
        self.handle.combine(
            self.combine_inputs,
            self.combine_outputs,
            config=self.combine_config,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    def close(self):
        self._resources.close()
