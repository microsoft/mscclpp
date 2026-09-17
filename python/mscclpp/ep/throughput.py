# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
# branch ``chhwang/dev-atomic-add-cleanup``. Licensed under the MIT License.
"""Throughput-mode context."""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional

import torch

from mscclpp.ep._cpp import DispatchLayout, MoEMode, create_moe_runtime
from mscclpp.ep.context import Context
from mscclpp.ep.runtime import Runtime, requires_initialized
from mscclpp.ep.types import (
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    MoECommunicatorConfig,
    QuantConfig,
    _ThroughputCombineContext,
)
from mscclpp.ep.utils import (
    current_stream_ptr as _stream_ptr,
    ptr as _ptr,
    resolve_expert_placement,
    resolve_num_blocks,
    tensor_from_pointer,
)


class ThroughputContext(Context):
    """Throughput-mode context."""

    def __init__(
        self,
        config: MoECommunicatorConfig,
    ) -> None:
        comm = config.comm
        if comm is None:
            raise ValueError("mode=THROUGHPUT requires an mscclpp.CommGroup via comm=")
        output_layout = config.output_layout
        if output_layout is None:
            output_layout = DispatchLayout.TOKEN_MAJOR
        dispatch_blocks, combine_blocks = resolve_num_blocks(
            config.num_blocks,
            default=(20, 20),
            scalar_combine_offset=0,
        )
        if dispatch_blocks <= 0:
            raise ValueError("dispatch block count must be positive in throughput mode")
        if combine_blocks <= 0:
            raise ValueError("combine block count must be positive in throughput mode")
        max_hidden_bytes = config.hidden_size * torch.empty((), dtype=torch.bfloat16).element_size()

        self.rank: int = comm.my_rank
        self.group_size: int = comm.nranks
        self.world_size = comm.nranks
        self.comm = comm
        self.local_rank = torch.cuda.current_device()
        self.device = torch.device("cuda", self.local_rank)
        self.initialized = False
        self.mode = MoEMode.THROUGHPUT
        self.output_layout = output_layout
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.max_tokens_per_rank = config.max_tokens_per_rank
        self.dispatch_blocks = dispatch_blocks
        self.combine_blocks = combine_blocks
        self.max_hidden_bytes = max_hidden_bytes
        self.enable_overlap = config.enable_overlap

        if self.output_layout not in (DispatchLayout.TOKEN_MAJOR, DispatchLayout.RANK_MAJOR):
            raise NotImplementedError("THROUGHPUT mode supports TOKEN_MAJOR or RANK_MAJOR output")
        if config.invalid_token_expert_id is not None:
            raise ValueError("invalid_token_expert_id is only supported in latency mode")
        self.num_local_experts, self.local_expert_start = resolve_expert_placement(
            num_experts=self.num_experts,
            world_size=self.world_size,
            rank=self.rank,
            num_local_experts=config.num_local_experts,
            local_expert_start=config.local_expert_start,
        )
        if config.quant is not None:
            raise NotImplementedError("throughput quantized dispatch (scales) is not implemented yet")

        self.expert_alignment = config.expert_alignment


class ThroughputRuntime(Runtime):
    """Throughput-optimized runtime using bounded communication resources."""

    context: ThroughputContext

    def __init__(self, context: ThroughputContext) -> None:
        cpp_runtime = create_moe_runtime(
            context.comm.communicator,
            context.mode,
            max_tokens_per_rank=context.max_tokens_per_rank,
            max_hidden_bytes=context.max_hidden_bytes,
            output_layout=context.output_layout,
        )
        super().__init__(context, cpp_runtime)

    @requires_initialized
    def dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        quant: Optional[QuantConfig],
        *,
        output_buffer: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream],
        previous_handle: Optional[DispatchHandle],
        runtime_max_tokens_per_rank: Optional[int],
    ) -> tuple[DispatchOutput, DispatchHandle]:
        mode_context = self.context
        del output_buffer
        if runtime_max_tokens_per_rank is not None:
            raise ValueError("runtime_max_tokens_per_rank is only supported by latency rank-major dispatch")
        stream_scope = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with stream_scope:
            self._validate_dispatch(input, topk_ids, weights, quant)
            implicit_weights = weights is None
            if weights is None:
                weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=topk_ids.device)
            cache = previous_handle._dispatch_cache if previous_handle is not None else None
            if cache is not None and not self._cache_matches(cache, input, topk_ids, weights, implicit_weights):
                cache = None
            if cache is not None:
                num_tokens_per_rank = cache["num_tokens_per_rank"]
                num_tokens_per_expert = cache["num_tokens_per_expert"]
                is_token_in_rank = cache["is_token_in_rank"]
            else:
                (
                    num_tokens_per_rank,
                    num_tokens_per_expert,
                    is_token_in_rank,
                ) = self._compute_counts(topk_ids, mode_context.num_experts)
            if cache is not None:
                (
                    recv_x,
                    _,
                    _runtime_recv_topk_idx,
                    _runtime_recv_topk_weights,
                    _runtime_num_recv_tokens_per_expert_list,
                    rank_prefix_matrix,
                    _,
                    send_head,
                ) = self._dispatch_throughput(
                    input,
                    None,
                    None,
                    None,
                    None,
                    is_token_in_rank,
                    None,
                    cache["num_recv_tokens"],
                    cache["rank_prefix_matrix"],
                    cache["channel_prefix_matrix"],
                    mode_context.expert_alignment,
                )
                del (
                    _runtime_recv_topk_idx,
                    _runtime_recv_topk_weights,
                    _runtime_num_recv_tokens_per_expert_list,
                )
                recv_topk_idx = cache["recv_topk_idx"]
                recv_topk_weights = cache["recv_topk_weights"]
                num_recv_tokens_per_expert_list = cache["num_recv_tokens_per_expert_list"]
                combine_context = _ThroughputCombineContext(
                    recv_topk_weights=recv_topk_weights,
                    send_head=send_head,
                )
                dispatch_cache = cache
            else:
                (
                    recv_x,
                    _,
                    recv_topk_idx,
                    recv_topk_weights,
                    num_recv_tokens_per_expert_list,
                    rank_prefix_matrix,
                    channel_prefix_matrix,
                    send_head,
                ) = self._dispatch_throughput(
                    input,
                    None,
                    topk_ids,
                    weights,
                    num_tokens_per_rank,
                    is_token_in_rank,
                    num_tokens_per_expert,
                    0,
                    None,
                    None,
                    mode_context.expert_alignment,
                )
                combine_context = _ThroughputCombineContext(
                    recv_topk_weights=recv_topk_weights,
                    send_head=send_head,
                )
                dispatch_cache = {
                    "num_tokens_per_rank": num_tokens_per_rank,
                    "num_tokens_per_expert": num_tokens_per_expert,
                    "is_token_in_rank": is_token_in_rank,
                    "rank_prefix_matrix": rank_prefix_matrix,
                    "channel_prefix_matrix": channel_prefix_matrix,
                    "num_recv_tokens": int(recv_x.size(0)),
                    "recv_topk_idx": recv_topk_idx,
                    "recv_topk_weights": recv_topk_weights,
                    "num_recv_tokens_per_expert_list": num_recv_tokens_per_expert_list,
                    "context_id": id(mode_context),
                    "num_tokens": int(input.size(0)),
                    "device": input.device,
                    "topk_ids_ptr": topk_ids.data_ptr(),
                    "topk_ids_version": topk_ids._version,
                    "implicit_weights": implicit_weights,
                    "weights_ptr": 0 if implicit_weights else weights.data_ptr(),
                    "weights_version": 0 if implicit_weights else weights._version,
                }
            recv_tokens_per_rank = None
            if mode_context.output_layout == DispatchLayout.RANK_MAJOR:
                recv_prefix = rank_prefix_matrix[:, mode_context.rank]
                recv_tokens_per_rank = torch.diff(
                    recv_prefix, prepend=torch.zeros((1,), dtype=recv_prefix.dtype, device=recv_prefix.device)
                )
            output_info = DispatchOutputInfo(
                layout=DispatchLayoutInfo(
                    kind=mode_context.output_layout,
                    num_tokens_per_expert=num_recv_tokens_per_expert_list,
                    num_tokens_per_rank=recv_tokens_per_rank,
                ),
                quant=None,
            )
            dispatch_out = DispatchOutput(
                tokens=recv_x,
                quant=output_info.quant,
                layout=output_info.layout,
                topk_ids=recv_topk_idx,
                weights=recv_topk_weights,
            )
            handle = DispatchHandle(output_info=output_info, _context=combine_context, _dispatch_cache=dispatch_cache)
            return dispatch_out, handle

    @requires_initialized
    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream],
    ) -> torch.Tensor:
        mode_context = self.context
        stream_scope = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with stream_scope:
            self._validate_combine(expert_output, handle)
            context = handle._context
            topk_weights = context.recv_topk_weights
            send_head = context.send_head
            num_input_tokens, hidden = int(expert_output.size(0)), int(expert_output.size(1))
            num_output_tokens = int(send_head.size(0))
            num_topk = int(topk_weights.size(1)) if topk_weights is not None else 0
            combined_x = torch.empty((num_output_tokens, hidden), dtype=torch.bfloat16, device="cuda")
            combined_topk_weights = (
                torch.empty((num_output_tokens, num_topk), dtype=torch.float32, device="cuda")
                if topk_weights is not None
                else None
            )
            self.cpp_runtime.combine(
                _ptr(combined_x),
                _ptr(combined_topk_weights),
                _ptr(expert_output),
                _ptr(topk_weights),
                _ptr(send_head),
                num_input_tokens,
                num_output_tokens,
                hidden,
                num_topk,
                expert_output.element_size(),
                mode_context.combine_blocks,
                _stream_ptr(),
            )
            if out is not None:
                out.copy_(combined_x)
                return out
            return combined_x

    def _compute_counts(self, topk_idx: torch.Tensor, num_experts: int):
        """Return per-rank, per-expert, and token-membership routing metadata.

        This is routing metadata consumed by dispatch; it is unrelated to
        ``DispatchLayout`` (the memory layout of the dispatch output).
        """
        mode_context = self.context
        assert topk_idx.dim() == 2 and topk_idx.is_contiguous()
        num_tokens, num_topk = int(topk_idx.size(0)), int(topk_idx.size(1))

        num_tokens_per_rank = torch.empty((mode_context.group_size,), dtype=torch.int32, device="cuda")
        num_tokens_per_expert = torch.empty((num_experts,), dtype=torch.int32, device="cuda")
        is_token_in_rank = torch.empty((num_tokens, mode_context.group_size), dtype=torch.bool, device="cuda")

        self.cpp_runtime.prepare(
            _ptr(num_tokens_per_rank),
            _ptr(num_tokens_per_expert),
            _ptr(is_token_in_rank),
            _ptr(topk_idx),
            num_tokens,
            num_topk,
            num_experts,
            _stream_ptr(),
        )
        return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank

    def _dispatch_throughput(
        self,
        x: torch.Tensor,
        x_scales: Optional[torch.Tensor],
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        num_tokens_per_rank: Optional[torch.Tensor],
        is_token_in_rank: torch.Tensor,
        num_tokens_per_expert: Optional[torch.Tensor],
        cached_num_recv_tokens: int,
        cached_rank_prefix_matrix: Optional[torch.Tensor],
        cached_channel_prefix_matrix: Optional[torch.Tensor],
        expert_alignment: int,
    ):
        """Run throughput dispatch and return combine metadata."""
        mode_context = self.context
        assert x.dim() == 2 and x.is_contiguous()
        cached_mode = cached_rank_prefix_matrix is not None
        num_tokens, hidden = int(x.size(0)), int(x.size(1))
        x_element_size = x.element_size()
        num_channels = mode_context.dispatch_blocks

        num_topk = int(topk_idx.size(1)) if topk_idx is not None else 0
        num_scales = 0
        if x_scales is not None:
            num_scales = 1 if x_scales.dim() == 1 else int(x_scales.size(1))

        # ----- Phase A: notify (non-cached) or reuse cached layout -----
        if cached_mode:
            num_recv_tokens = cached_num_recv_tokens
            rank_prefix_matrix = cached_rank_prefix_matrix
            channel_prefix_matrix = cached_channel_prefix_matrix
            num_recv_tokens_per_expert_list: List[int] = []
            num_experts = 0
        else:
            assert num_tokens_per_rank is not None and num_tokens_per_expert is not None
            num_experts = int(num_tokens_per_expert.size(0))
            num_local_experts = num_experts // mode_context.group_size
            rank_prefix_matrix = torch.empty(
                (mode_context.group_size, mode_context.group_size), dtype=torch.int32, device="cuda"
            )
            channel_prefix_matrix = torch.empty(
                (mode_context.group_size, num_channels), dtype=torch.int32, device="cuda"
            )
            num_recv_per_expert_host = torch.empty((num_local_experts,), dtype=torch.int32, device="cpu")
            num_recv_tokens = self.cpp_runtime.notify(
                _ptr(rank_prefix_matrix),
                _ptr(channel_prefix_matrix),
                _ptr(num_recv_per_expert_host),
                _ptr(num_tokens_per_rank),
                _ptr(num_tokens_per_expert),
                _ptr(is_token_in_rank),
                num_tokens,
                num_experts,
                expert_alignment,
                mode_context.dispatch_blocks,
                _stream_ptr(),
            )
            num_recv_tokens_per_expert_list = num_recv_per_expert_host.tolist()

        if mode_context.output_layout == DispatchLayout.RANK_MAJOR:
            num_recv_tokens = mode_context.group_size * mode_context.max_tokens_per_rank

        # ----- Phase B: allocate recv outputs (or view the recv pool) -----
        recv_x = self._allocate_recv(num_recv_tokens, hidden)
        send_head = torch.empty((num_tokens, mode_context.group_size), dtype=torch.int32, device="cuda")
        recv_topk_idx = (
            torch.empty((num_recv_tokens, num_topk), dtype=torch.int64, device="cuda") if topk_idx is not None else None
        )
        recv_topk_weights = (
            torch.empty((num_recv_tokens, num_topk), dtype=torch.float32, device="cuda")
            if topk_weights is not None
            else None
        )
        recv_x_scales = (
            torch.empty((num_recv_tokens, num_scales), dtype=torch.float32, device="cuda")
            if x_scales is not None
            else None
        )

        self.cpp_runtime.dispatch(
            _ptr(recv_x),
            _ptr(recv_x_scales),
            _ptr(recv_topk_idx),
            _ptr(recv_topk_weights),
            _ptr(send_head),
            _ptr(x),
            _ptr(x_scales),
            _ptr(topk_idx),
            _ptr(topk_weights),
            _ptr(is_token_in_rank),
            _ptr(rank_prefix_matrix),
            _ptr(channel_prefix_matrix),
            num_tokens,
            hidden,
            num_topk,
            num_scales,
            num_experts,
            x_element_size,
            num_recv_tokens,
            cached_mode,
            mode_context.dispatch_blocks,
            _stream_ptr(),
        )
        return (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            send_head,
        )

    def _allocate_recv(
        self,
        num_recv_tokens: int,
        hidden: int,
    ) -> torch.Tensor:
        """Return this rank's direct receive-pool view."""
        mode_context = self.context
        pool_ptr = self.cpp_runtime.dispatch_output_buffer_ptr()
        if pool_ptr == 0:
            raise RuntimeError("throughput receive-pool capacity exceeded")
        _, recv_x = tensor_from_pointer(
            pool_ptr,
            (num_recv_tokens, hidden),
            torch.bfloat16,
            mode_context.device,
            self.cpp_runtime,
        )
        return recv_x

    def _cache_matches(self, cache, input, topk_ids, weights, implicit_weights) -> bool:
        mode_context = self.context
        return (
            cache.get("context_id") == id(mode_context)
            and cache.get("num_tokens") == int(input.size(0))
            and cache.get("device") == input.device
            and cache.get("topk_ids_ptr") == topk_ids.data_ptr()
            and cache.get("topk_ids_version") == topk_ids._version
            and cache.get("implicit_weights") == implicit_weights
            and (implicit_weights or cache.get("weights_ptr") == weights.data_ptr())
            and (implicit_weights or cache.get("weights_version") == weights._version)
        )

    def _validate_dispatch(self, input, topk_ids, weights, quant) -> None:
        mode_context = self.context
        if quant is not None:
            raise NotImplementedError("throughput dispatch does not support quantized input scales yet")
        if input.dim() != 2 or not input.is_contiguous():
            raise ValueError("input must be a contiguous [num_tokens, hidden] tensor")
        if input.device.type != "cuda" or input.dtype != torch.bfloat16:
            raise ValueError("throughput dispatch input must be a CUDA BF16 tensor")
        if input.size(1) != mode_context.hidden_size:
            raise ValueError(f"input hidden size {input.size(1)} != configured {mode_context.hidden_size}")
        if input.size(0) > mode_context.max_tokens_per_rank:
            raise ValueError("input token count exceeds max_tokens_per_rank")
        if topk_ids.dim() != 2 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be a contiguous [num_tokens, topk] tensor")
        if topk_ids.device != input.device or topk_ids.dtype != torch.int64:
            raise ValueError("topk_ids must be an int64 CUDA tensor on the same device as input")
        if topk_ids.shape != (input.size(0), mode_context.topk):
            raise ValueError("topk_ids shape must be [input.size(0), topk]")
        if weights is not None:
            if weights.dim() != 2 or not weights.is_contiguous():
                raise ValueError("weights must be a contiguous [num_tokens, topk] tensor")
            if weights.device != input.device or weights.dtype != torch.float32:
                raise ValueError("weights must be a float32 CUDA tensor on the same device as input")
            if weights.shape != topk_ids.shape:
                raise ValueError("weights shape must match topk_ids")

    def _validate_combine(self, expert_output, handle) -> None:
        mode_context = self.context
        if not isinstance(handle, DispatchHandle) or not isinstance(handle._context, _ThroughputCombineContext):
            raise TypeError("handle must be a DispatchHandle returned by dispatch")
        if expert_output.dim() != 2 or not expert_output.is_contiguous():
            raise ValueError("expert_output must be a contiguous [total_recv_tokens, hidden] tensor")
        if expert_output.size(1) != mode_context.hidden_size:
            raise ValueError(
                f"expert_output hidden size {expert_output.size(1)} != configured {mode_context.hidden_size}"
            )
