# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
# branch ``chhwang/dev-atomic-add-cleanup``. Licensed under the MIT License.
"""Fabric-domain high-throughput backend for the high-level MoE communicator.

The C++ runtime follows the low-latency resource model: it reuses the existing
MSCCL++ communicator and writes directly into peer receive pools through a
torch-free raw-pointer boundary. Dynamic receive sizing uses a two-phase
``notify_dispatch`` then ``dispatch`` protocol. Cached dispatches reuse the
previous routing matrices and receive count.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from ._cpp import Config, DispatchLayout, MoEMode, _cpp
from .types import (
    DispatchHandle,
    DispatchLayoutInfo,
    DispatchOutput,
    DispatchOutputInfo,
    HighThroughputCombineContext,
    HighThroughputDispatchHandle,
    MoECommunicatorConfig,
    QuantConfig,
)
from .utils import (
    bf16_view as _bf16_view,
    current_stream_ptr as _stream_ptr,
    ptr as _ptr,
    resolve_expert_placement,
)


class HighThroughputRuntime:
    """Core high-throughput expert-parallel (EP) communication runtime.

    ``comm`` provides the initialized MSCCL++ communicator used to exchange and
    map the intranode physical symmetric buffers.
    """

    #: Default number of SMs reserved for comms kernels. Matches DeepEP.
    num_sms: int = 20

    def __init__(
        self,
        comm: Any,
        max_hidden_bytes: int,
        config: Config,
    ) -> None:
        self.rank: int = comm.my_rank
        self.group_size: int = comm.nranks
        self.comm = comm
        self.runtime = _cpp.ExpertParallelRuntime(comm.communicator, max_hidden_bytes, config)

    # ------------------------------------------------------------------
    # Sanity helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self.runtime.is_available()

    def is_internode_available(self) -> bool:
        return self.runtime.is_internode_available()

    # ------------------------------------------------------------------
    # Dispatch layout
    # ------------------------------------------------------------------

    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int):
        """Return per-rank, per-expert, and token-membership routing metadata."""
        assert topk_idx.dim() == 2 and topk_idx.is_contiguous()
        num_tokens, num_topk = int(topk_idx.size(0)), int(topk_idx.size(1))

        num_tokens_per_rank = torch.empty((self.group_size,), dtype=torch.int32, device="cuda")
        num_tokens_per_expert = torch.empty((num_experts,), dtype=torch.int32, device="cuda")
        is_token_in_rank = torch.empty((num_tokens, self.group_size), dtype=torch.bool, device="cuda")

        self.runtime.layout(
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

    # ------------------------------------------------------------------
    # Dispatch (two-phase) + combine
    # ------------------------------------------------------------------

    def dispatch(
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
        """Run high-throughput dispatch and return outputs plus combine metadata."""
        assert x.dim() == 2 and x.is_contiguous()
        cached_mode = cached_rank_prefix_matrix is not None
        num_tokens, hidden = int(x.size(0)), int(x.size(1))
        x_element_size = x.element_size()
        num_channels = self.runtime.get_dispatch_num_channels(x_element_size)

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
            num_local_experts = num_experts // self.group_size
            rank_prefix_matrix = torch.empty((self.group_size, self.group_size), dtype=torch.int32, device="cuda")
            channel_prefix_matrix = torch.empty((self.group_size, num_channels), dtype=torch.int32, device="cuda")
            num_recv_per_expert_host = torch.empty((num_local_experts,), dtype=torch.int32, device="cpu")
            num_recv_tokens = self.runtime.notify_dispatch(
                _ptr(rank_prefix_matrix),
                _ptr(channel_prefix_matrix),
                _ptr(num_recv_per_expert_host),
                _ptr(num_tokens_per_rank),
                _ptr(num_tokens_per_expert),
                _ptr(is_token_in_rank),
                num_tokens,
                num_experts,
                x_element_size,
                expert_alignment,
                _stream_ptr(),
            )
            num_recv_tokens_per_expert_list = num_recv_per_expert_host.tolist()

        # ----- Phase B: allocate recv outputs (or view the recv pool) -----
        recv_x = self._alloc_recv_x(num_tokens, num_recv_tokens, hidden, x_element_size)
        send_head = torch.empty((num_tokens, self.group_size), dtype=torch.int32, device="cuda")
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

        self.runtime.dispatch(
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

    def _alloc_recv_x(self, num_tokens: int, num_recv_tokens: int, hidden: int, x_element_size: int) -> torch.Tensor:
        """Return this rank's direct receive-pool view."""
        pool_ptr = self.runtime.resolve_recv_x_buffer(num_tokens, num_recv_tokens, hidden, x_element_size)
        if pool_ptr == 0:
            raise RuntimeError("high-throughput direct receive-pool capacity exceeded")
        return _bf16_view(pool_ptr, num_recv_tokens, hidden, owner=self)

    def combine(
        self,
        x: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        send_head: torch.Tensor,
    ):
        """Returns ``(combined_x, combined_topk_weights|None)``."""
        assert x.dim() == 2 and x.is_contiguous()
        num_tokens, hidden = int(x.size(0)), int(x.size(1))
        num_recv_tokens = int(send_head.size(0))
        num_topk = int(topk_weights.size(1)) if topk_weights is not None else 0
        combined_x = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        combined_topk_weights = (
            torch.empty((num_recv_tokens, num_topk), dtype=torch.float32, device="cuda")
            if topk_weights is not None
            else None
        )
        self.runtime.combine(
            _ptr(combined_x),
            _ptr(combined_topk_weights),
            _ptr(x),
            _ptr(topk_weights),
            _ptr(send_head),
            num_tokens,
            num_recv_tokens,
            hidden,
            num_topk,
            x.element_size(),
            _stream_ptr(),
        )
        return combined_x, combined_topk_weights


class HighThroughputBackend:
    """Backend implementation for ``MoEMode.HIGH_THROUGHPUT``."""

    def __init__(self, config: MoECommunicatorConfig, output_layout: DispatchLayout) -> None:
        comm = config.comm
        if comm is None:
            raise ValueError("mode=HIGH_THROUGHPUT requires an mscclpp.CommGroup via comm=")
        if Config is None or not hasattr(_cpp, "ExpertParallelRuntime"):
            raise ImportError(
                "mscclpp_ep_cpp was built without the high-throughput EP backend. "
                "Rebuild with -DMSCCLPP_BUILD_EXT_EP=ON and ensure Config/ExpertParallelRuntime are exported."
            )

        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = torch.cuda.current_device()
        self.device = torch.device("cuda", self.local_rank)
        self.mode = MoEMode.HIGH_THROUGHPUT
        self.output_layout = output_layout

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.max_tokens_per_rank = config.max_tokens_per_rank
        self.num_sms = config.num_sms
        self.enable_overlap = config.enable_overlap

        if self.output_layout != DispatchLayout.TOKEN_MAJOR:
            raise NotImplementedError("HT mode currently supports only DispatchLayout.TOKEN_MAJOR")
        if config.invalid_token_expert_id is not None:
            raise ValueError("invalid_token_expert_id is only supported in low-latency mode")

        self.num_local_experts, self.local_expert_start = resolve_expert_placement(
            num_experts=self.num_experts,
            world_size=self.world_size,
            rank=self.rank,
            num_local_experts=config.num_local_experts,
            local_expert_start=config.local_expert_start,
        )

        if config.quant is not None:
            raise NotImplementedError("HT quantized dispatch (scales) is not implemented yet")

        self.expert_alignment = config.expert_alignment
        self._cfg = Config(self.num_sms)
        hidden_bytes = self.hidden_size * torch.empty((), dtype=torch.bfloat16).element_size()
        self._runtime = HighThroughputRuntime(
            comm,
            max_hidden_bytes=hidden_bytes,
            config=self._cfg,
        )

    def is_available(self) -> bool:
        return self._runtime.is_available()

    def is_internode_available(self) -> bool:
        return self._runtime.is_internode_available()

    def is_internode(self) -> bool:
        return self._runtime.is_internode_available()

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
    ) -> tuple[DispatchOutput, DispatchHandle]:
        del output_buffer
        if stream is not None:
            with torch.cuda.stream(stream):
                return self._dispatch(input, topk_ids, weights, quant, previous_handle)
        return self._dispatch(input, topk_ids, weights, quant, previous_handle)

    def _dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        quant: Optional[QuantConfig],
        previous_handle: Optional[DispatchHandle],
    ) -> tuple[DispatchOutput, DispatchHandle]:
        self._validate_dispatch_inputs(input, topk_ids, weights, quant)
        implicit_weights = weights is None
        if weights is None:
            weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=topk_ids.device)

        cache = getattr(previous_handle, "_dispatch_cache", None) if previous_handle is not None else None
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
            ) = self._runtime.get_dispatch_layout(topk_ids, self.num_experts)

        if cache is not None:
            (
                recv_x,
                _recv_x_scales,
                _runtime_recv_topk_idx,
                _runtime_recv_topk_weights,
                _runtime_num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                _channel_prefix_matrix,
                send_head,
            ) = self._runtime.dispatch(
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
                self.expert_alignment,
            )
            del _runtime_recv_topk_idx, _runtime_recv_topk_weights, _runtime_num_recv_tokens_per_expert_list
            recv_topk_idx = cache["recv_topk_idx"]
            recv_topk_weights = cache["recv_topk_weights"]
            num_recv_tokens_per_expert_list = cache["num_recv_tokens_per_expert_list"]
            combine_context = HighThroughputCombineContext(
                recv_topk_weights=recv_topk_weights,
                send_head=send_head,
            )
            dispatch_cache = cache
        else:
            (
                recv_x,
                _recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                channel_prefix_matrix,
                send_head,
            ) = self._runtime.dispatch(
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
                self.expert_alignment,
            )
            combine_context = HighThroughputCombineContext(
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
                "backend_id": id(self),
                "num_tokens": int(input.size(0)),
                "device": input.device,
                "topk_ids_ptr": topk_ids.data_ptr(),
                "topk_ids_version": topk_ids._version,
                "implicit_weights": implicit_weights,
                "weights_ptr": 0 if implicit_weights else weights.data_ptr(),
                "weights_version": 0 if implicit_weights else weights._version,
            }

        output_info = DispatchOutputInfo(
            layout=DispatchLayoutInfo(
                kind=self.output_layout,
                num_tokens_per_expert=num_recv_tokens_per_expert_list,
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
        handle = HighThroughputDispatchHandle(output_info=output_info, combine_context=combine_context)
        # The torch-free HT runtime orders its work on the caller's CUDA stream
        # (no separate event handle), so there is nothing to attach here.
        handle._event = None  # type: ignore[attr-defined]
        handle._dispatch_cache = dispatch_cache  # type: ignore[attr-defined]
        return dispatch_out, handle

    def _cache_matches(self, cache, input, topk_ids, weights, implicit_weights) -> bool:
        return (
            cache.get("backend_id") == id(self)
            and cache.get("num_tokens") == int(input.size(0))
            and cache.get("device") == input.device
            and cache.get("topk_ids_ptr") == topk_ids.data_ptr()
            and cache.get("topk_ids_version") == topk_ids._version
            and cache.get("implicit_weights") == implicit_weights
            and (implicit_weights or cache.get("weights_ptr") == weights.data_ptr())
            and (implicit_weights or cache.get("weights_version") == weights._version)
        )

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream],
    ) -> torch.Tensor:
        if stream is not None:
            with torch.cuda.stream(stream):
                return self._combine(expert_output, handle, out)
        return self._combine(expert_output, handle, out)

    def _combine(
        self, expert_output: torch.Tensor, handle: DispatchHandle, out: Optional[torch.Tensor]
    ) -> torch.Tensor:
        self._validate_combine_inputs(expert_output, handle)
        context = handle.combine_context
        combined_x, _combined_w = self._runtime.combine(
            expert_output,
            context.recv_topk_weights,
            context.send_head,
        )
        if out is not None:
            out.copy_(combined_x)
            return out
        return combined_x

    def _validate_dispatch_inputs(self, input, topk_ids, weights, quant) -> None:
        if quant is not None:
            raise NotImplementedError("HT dispatch does not support quantized input scales yet")
        if input.dim() != 2 or not input.is_contiguous():
            raise ValueError("input must be a contiguous [num_tokens, hidden] tensor")
        if input.device.type != "cuda" or input.dtype != torch.bfloat16:
            raise ValueError("HT dispatch input must be a CUDA BF16 tensor")
        if input.size(1) != self.hidden_size:
            raise ValueError(f"input hidden size {input.size(1)} != configured {self.hidden_size}")
        if input.size(0) > self.max_tokens_per_rank:
            raise ValueError("input token count exceeds max_tokens_per_rank")
        if topk_ids.dim() != 2 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be a contiguous [num_tokens, topk] tensor")
        if topk_ids.device != input.device or topk_ids.dtype != torch.int64:
            raise ValueError("topk_ids must be an int64 CUDA tensor on the same device as input")
        if topk_ids.shape != (input.size(0), self.topk):
            raise ValueError("topk_ids shape must be [input.size(0), topk]")
        if weights is not None:
            if weights.dim() != 2 or not weights.is_contiguous():
                raise ValueError("weights must be a contiguous [num_tokens, topk] tensor")
            if weights.device != input.device or weights.dtype != torch.float32:
                raise ValueError("weights must be a float32 CUDA tensor on the same device as input")
            if weights.shape != topk_ids.shape:
                raise ValueError("weights shape must match topk_ids")

    def _validate_combine_inputs(self, expert_output, handle) -> None:
        if not isinstance(handle, HighThroughputDispatchHandle):
            raise TypeError("handle must be a DispatchHandle returned by dispatch")
        if expert_output.dim() != 2 or not expert_output.is_contiguous():
            raise ValueError("expert_output must be a contiguous [total_recv_tokens, hidden] tensor")
        if expert_output.size(1) != self.hidden_size:
            raise ValueError(f"expert_output hidden size {expert_output.size(1)} != configured {self.hidden_size}")
