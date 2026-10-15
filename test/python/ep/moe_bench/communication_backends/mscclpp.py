# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MSCCL++ native latency/throughput communication."""

import torch
import torch.distributed as dist


class MscclppBackend:
    def __init__(self, config, rank, x, ids, weights, recv, expert_output, output):
        import mscclpp
        import mscclpp.ep as ep

        model = config["model"]
        environment = config["environment"]
        self.mode = environment.get("mode", "latency")
        self.layout = environment.get("layout", "expert_major")
        self.adapt_layout = self.layout != "expert_major"
        self.route_id_base = rank * (model["num_experts"] // environment["num_gpus"]) if self.mode == "latency" else 0
        self.group = mscclpp.CommGroup(torch_group=dist.distributed_c10d._get_default_group())
        self.comm = ep.MoECommunicator(
            comm=self.group,
            num_experts=model["num_experts"],
            num_local_experts=model["num_experts"] // config["environment"]["num_gpus"],
            hidden_size=model["d_model"],
            topk=model["top_k"],
            max_tokens_per_rank=config["input"]["tokens_per_rank"],
            mode=getattr(ep.MoEMode, self.mode.upper()),
            output_layout=getattr(ep.DispatchLayout, self.layout.upper()),
            combine_mode=ep.CombineMode.RANK_LOCAL_REDUCE,
        )
        if not self.comm.is_available():
            raise RuntimeError(f"MSCCL++ {self.mode} EP is unavailable for this configuration/topology")
        if self.comm.is_internode():
            raise RuntimeError("MoE benchmark requires one GPU IPC domain on a single node")
        self.x, self.ids, self.weights = x, ids, weights
        self.recv, self.expert_output, self.output = recv, expert_output, output
        self.handle = None
        self.combine_buffer = expert_output
        self.route_valid_rows = None
        self.metadata = {
            "communication_backend": "mscclpp",
            "mscclpp_version": mscclpp.__version__,
            "communication_algorithm": self.mode,
            "communication_mode": self.mode,
            "dispatch_layout": self.layout,
            "native_dispatch_layout": self.layout,
            "combine_mode": "preweighted_local_reduce" if self.adapt_layout else "rank_local_reduce",
        }

    def dispatch(self):
        kwargs = {"output_buffer": self.recv} if not self.adapt_layout else {}
        if self.mode == "throughput":
            kwargs["previous_handle"] = self.handle
        dispatched, self.handle = self.comm.dispatch(self.x, self.ids, self.weights, **kwargs)
        if self.adapt_layout:
            self.route_ids, self.route_weights = dispatched.topk_ids, dispatched.weights
            if self.combine_buffer is self.expert_output:
                if self.layout == "rank_major":
                    capacity = self.x.shape[0]
                    positions = torch.arange(capacity, device=self.x.device)
                    self.route_valid_rows = (
                        positions[None, :] < dispatched.layout.num_tokens_per_rank[:, None]
                    ).flatten()
                self.combine_buffer = (
                    dispatched.combine_input_buffer
                    if dispatched.combine_input_buffer is not None
                    else torch.empty_like(dispatched.tokens)
                )
            return dispatched.tokens, None
        return dispatched.tokens, dispatched.layout.num_tokens_per_expert

    def combine(self):
        self.comm.combine(self.combine_buffer, self.handle, out=self.output)

    def close(self):
        if self.comm is not None:
            torch.cuda.synchronize(self.x.device)
            self.handle = None
            self.route_ids = self.route_weights = self.route_valid_rows = None
            self.combine_buffer = self.expert_output
            self.comm = None
            self.group = None
