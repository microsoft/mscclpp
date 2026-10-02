# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MSCCL++ latency, expert-major communication for the reference pipeline."""

import torch
import torch.distributed as dist


class MscclppBackend:
    def __init__(self, config, rank, x, ids, weights, recv, expert_output, output):
        import mscclpp
        import mscclpp.ep as ep

        model = config["model"]
        self.group = mscclpp.CommGroup(torch_group=dist.distributed_c10d._get_default_group())
        self.comm = ep.MoECommunicator(
            comm=self.group,
            num_experts=model["num_experts"],
            num_local_experts=model["num_experts"] // config["environment"]["num_gpus"],
            hidden_size=model["d_model"],
            topk=model["top_k"],
            max_tokens_per_rank=config["input"]["tokens_per_rank"],
            mode=ep.MoEMode.LATENCY,
            output_layout=ep.DispatchLayout.EXPERT_MAJOR,
            combine_mode=ep.CombineMode.RANK_LOCAL_REDUCE,
        )
        if not self.comm.is_available():
            raise RuntimeError("MSCCL++ latency EP is unavailable for this configuration")
        if self.comm.is_internode():
            raise RuntimeError("MoE benchmark requires one GPU IPC domain on a single node")
        self.x, self.ids, self.weights = x, ids, weights
        self.recv, self.expert_output, self.output = recv, expert_output, output
        self.handle = None
        self.metadata = {
            "communication_backend": "mscclpp",
            "mscclpp_version": mscclpp.__version__,
            "communication_algorithm": "latency",
            "dispatch_layout": "expert_major",
            "combine_mode": "rank_local_reduce",
        }

    def dispatch(self):
        dispatched, self.handle = self.comm.dispatch(self.x, self.ids, self.weights, output_buffer=self.recv)
        return dispatched.tokens, dispatched.layout.num_tokens_per_expert

    def combine(self):
        self.comm.combine(self.expert_output, self.handle, out=self.output)

    def close(self):
        if self.comm is not None:
            torch.cuda.synchronize(self.x.device)
            self.handle = None
            self.comm = None
            self.group = None
