# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MSCCL++ native latency/throughput communication."""

import os

import torch
import torch.distributed as dist


class MscclppBackend:
    def __init__(self, config, rank, x, ids, weights, recv, expert_output, output):
        import mscclpp
        import mscclpp.ep as ep

        model = config["model"]
        environment = config["environment"]
        world = environment["num_gpus"]
        self.mode = environment.get("mode", "latency")
        self.layout = environment.get("layout", "expert_major")
        self.adapt_layout = self.layout != "expert_major"
        self.route_id_base = rank * (model["num_experts"] // world) if self.mode == "latency" else 0
        self.group = mscclpp.CommGroup(torch_group=dist.distributed_c10d._get_default_group())
        if self.group.nranks_per_ipc_domain != world:
            raise RuntimeError(
                "MoE benchmark requires all GPUs to share one CUDA IPC/MNNVL domain; "
                f"domain has {self.group.nranks_per_ipc_domain} of {world} ranks"
            )
        self.comm = ep.MoECommunicator(
            ep.MoECommunicatorConfig(
                comm=self.group,
                num_experts=model["num_experts"],
                hidden_size=model["d_model"],
                topk=model["top_k"],
                max_tokens_per_rank=config["input"]["tokens_per_rank"],
                device=x.device,
                mode=getattr(ep.MoEMode, self.mode.upper()),
                output_layout=getattr(ep.DispatchLayout, self.layout.upper()),
                combine_mode=ep.CombineMode.RANK_LOCAL_REDUCE,
            )
        )
        if not self.comm.is_available():
            raise RuntimeError(f"MSCCL++ {self.mode} EP is unavailable for this configuration/topology")
        self.comm.initialize()
        self.x, self.ids, self.weights = x, ids, weights
        self.recv, self.expert_output, self.output = recv, expert_output, output
        self.handle = None
        self.prepare_handle = None
        self.prepare_counts = None
        self.combine_buffer = expert_output
        self.route_valid_rows = None
        self.num_recv_tokens = None
        if self.mode == "throughput":
            if self.layout == "rank_major":
                self.prepare_counts = torch.empty(world, dtype=torch.int32, device=x.device)
            else:
                # Token-major rows are a prefix of the capacity-sized pool whose
                # length the native runtime does not expose.
                self.num_recv_tokens = _received_tokens(ids, rank, world, model["num_experts"] // world)
        multinode = int(os.environ.get("LOCAL_WORLD_SIZE", world)) < world
        self.metadata = {
            "communication_backend": "mscclpp",
            "mscclpp_version": mscclpp.__version__,
            "ipc_domain_ranks": self.group.nranks_per_ipc_domain,
            "communication_transport": "cuda_ipc_mnnvl" if multinode else "cuda_ipc",
            "communication_algorithm": self.mode,
            "communication_mode": self.mode,
            "dispatch_layout": self.layout,
            "native_dispatch_layout": self.layout,
            "combine_mode": "preweighted_local_reduce" if self.adapt_layout else "rank_local_reduce",
        }

    def dispatch(self):
        kwargs = {"output_buffer": self.recv} if not self.adapt_layout else {}
        if self.mode == "throughput":
            # Routing is fixed for the run, so prepare once and reuse it, like
            # NCCL-EP's cached handle.
            if self.prepare_handle is None:
                self.prepare_handle = self.comm.prepare(self.ids, output_count=self.prepare_counts)
            kwargs["prepare_handle"] = self.prepare_handle
        dispatched, self.handle = self.comm.dispatch(self.x, self.ids, self.weights, **kwargs)
        if self.adapt_layout:
            self.route_ids, self.route_weights = dispatched.topk_ids, dispatched.weights
            if self.combine_buffer is self.expert_output:
                if self.layout == "rank_major":
                    positions = torch.arange(dispatched.tokens.shape[1], device=self.x.device)
                    received = dispatched.layout.num_tokens_per_rank
                    self.route_valid_rows = (positions[None, :] < received[:, None]).flatten()
                else:
                    positions = torch.arange(dispatched.tokens.shape[0], device=self.x.device)
                    self.route_valid_rows = positions < self.num_recv_tokens
                self.combine_buffer = dispatched.combine_input_buffer
            return dispatched.tokens, None
        return dispatched.tokens, dispatched.layout.num_tokens_per_expert

    def combine(self):
        self.comm.combine(self.combine_buffer, self.handle, out=self.output)

    def close(self):
        if self.comm is not None:
            torch.cuda.synchronize(self.x.device)
            self.handle = self.prepare_handle = None
            self.route_ids = self.route_weights = self.route_valid_rows = None
            self.combine_buffer = self.expert_output
            self.comm = None
            self.group = None


def _received_tokens(ids, rank, world, local_experts):
    """Return how many distinct tokens all ranks send to ``rank``."""
    destinations = torch.zeros(world, dtype=torch.int64)
    owners = ids.detach().cpu().div(local_experts, rounding_mode="floor")
    for destination in range(world):
        destinations[destination] = int((owners == destination).any(dim=1).sum())
    dist.all_reduce(destinations)
    return int(destinations[rank])
