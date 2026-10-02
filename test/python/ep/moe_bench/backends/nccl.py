# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optional NCCL-EP low-latency expert-major adapter (no MPI bootstrap)."""

from contextlib import ExitStack
from importlib.metadata import version

import torch
import torch.distributed as dist


class NcclBackend:
    def __init__(self, config, rank, x, ids, weights, recv, expert_output, output):
        try:
            import nccl.core as core
            import nccl.ep as ep
        except ImportError as error:
            raise RuntimeError(
                "NCCL-EP requires nccl.core and nccl.ep. Install NVIDIA's "
                "'nccl-extensions[cu13]' (or [cu12] matching PyTorch), with a matching "
                "CUDA toolkit/nvcc; see src/ext/ep/README.md. MSCCL++ does not require this extra."
            ) from error

        model = config["model"]
        world = config["environment"]["num_gpus"]
        # Keep tensors and descriptors alive: native handles cache their addresses.
        self.x, self.ids, self.weights = x, ids, weights
        self.recv, self.expert_output, self.output = recv, expert_output, output
        self.counts = torch.empty(model["num_experts"] // world, dtype=torch.int32, device=x.device)
        self.metadata = {
            "communication_backend": "nccl",
            "nccl4py_version": version("nccl4py"),
            "nccl_ep_version": version("nccl-extensions"),
            "nccl_version": str(core.get_version().libnccl.version),
            "communication_algorithm": "low_latency",
            "dispatch_layout": "expert_major",
            "combine_mode": "weighted_topk",
        }
        with ExitStack() as resources:
            uid = [core.get_unique_id() if rank == 0 else None]
            dist.broadcast_object_list(uid, src=0)
            self.comm = core.Communicator.init(nranks=world, rank=rank, unique_id=uid[0])
            resources.callback(self.comm.destroy)
            self.group = ep.Group.create(
                self.comm,
                ep.GroupConfig(
                    algorithm=ep.Algorithm.LOW_LATENCY,
                    num_experts=model["num_experts"],
                    max_dispatch_tokens_per_rank=config["input"]["tokens_per_rank"],
                    max_recv_tokens_per_rank=world * config["input"]["tokens_per_rank"],
                    max_token_bytes=model["d_model"] * 2,
                    alloc=ep.AllocConfig(),
                ),
            )
            resources.callback(self.group.destroy)
            self.ids_tensor = ep.Tensor(ids)
            self.handle = self.group.create_handle(
                ep.Layout.EXPERT_MAJOR,
                self.ids_tensor,
                layout_info=None,
                config=ep.HandleConfig(),
                stream=torch.cuda.current_stream().cuda_stream,
            )
            resources.callback(self.handle.destroy)
            resources.callback(torch.cuda.synchronize, x.device)
            self.dispatch_inputs = ep.DispatchInputs(tokens=ep.Tensor(x))
            self.dispatch_outputs = ep.DispatchOutputs(tokens=ep.Tensor(recv))
            self.layout = ep.LayoutInfo(expert_counters=ep.Tensor(self.counts))
            self.dispatch_config = ep.DispatchConfig()
            self.combine_inputs = ep.CombineInputs(tokens=ep.Tensor(expert_output))
            self.combine_outputs = ep.CombineOutputs(tokens=ep.Tensor(output), topk_weights=ep.Tensor(weights))
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
        return self.recv, self.counts

    def combine(self):
        self.handle.combine(
            self.combine_inputs,
            self.combine_outputs,
            config=self.combine_config,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    def close(self):
        self._resources.close()
