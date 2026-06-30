# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
# branch ``chhwang/dev-atomic-add-cleanup``. Licensed under the MIT License.
"""Python frontend for the MSCCL++ Expert-Parallel extension.

``MoECommunicator`` is the high-level API. The backend is selected by
:attr:`MoECommunicatorConfig.mode` (a :class:`MoEMode` enum):

* ``MoEMode.LOW_LATENCY`` — decode path. Wraps the C++ ``MoERuntime`` (RDMA +
  CUDA-IPC PortChannel). Output layout ``DispatchLayout.EXPERT_MAJOR``; the
  caller pre-allocates the recv buffer.
* ``MoEMode.HIGH_THROUGHPUT`` — prefill path. Wraps the DeepEP-style
  :class:`mscclpp.ext.ep.Buffer` (NVLink intranode + RDMA internode HT kernels,
  with the GB200 TMA direct-gather combine + all-sender dispatch). Output layout
  ``DispatchLayout.FLAT`` grouped by local expert id; intranode vs internode is
  selected internally from the RDMA buffer-size hint.

The two backends are independent C++ runtimes. LL takes an ``mscclpp.CommGroup``
via ``comm=``; HT takes a ``torch.distributed`` process group via ``group=``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

try:
    import mscclpp_ep_cpp as _cpp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mscclpp_ep_cpp is not available. Build mscclpp with -DMSCCLPP_BUILD_EXT_EP=ON "
        "or install with `pip install .[ep]`."
    ) from exc

from .buffer import Config, ExpertParallelRuntime

DispatchLayout = _cpp.DispatchLayout
MoEMode = _cpp.MoEMode


@dataclass
class MoECommunicatorConfig:
    """Configuration for the high-level MoE dispatch/combine API."""

    # Communication. ``comm`` (mscclpp.CommGroup) drives the LL backend; ``group``
    # (torch.distributed ProcessGroup) drives the HT backend.
    comm: Optional[Any] = None
    group: Optional[dist.ProcessGroup] = None
    device: Optional[Union[torch.device, int]] = None

    # Expert topology
    num_experts: int = 0
    num_local_experts: Optional[int] = None
    local_expert_start: Optional[int] = None

    # Model shape and capacity
    hidden_size: int = 0
    topk: int = 0
    max_tokens_per_rank: int = 0
    max_recv_tokens_per_rank: Optional[int] = None

    # Runtime mode and output layout
    mode: MoEMode = MoEMode.LOW_LATENCY
    output_layout: Optional[DispatchLayout] = None

    # Quantization defaults
    input_dtype: Optional[torch.dtype] = None
    quant_format: Optional[str] = None

    # Transport / launch tuning
    num_rdma_qps_per_rank: int = 12
    num_sms: int = 20
    enable_overlap: bool = False

    # HT-only buffer/launch tuning (advanced)
    expert_alignment: int = 1
    nvl_chunked_send: int = 8
    nvl_chunked_recv: int = 256
    rdma_chunked_send: int = 16
    rdma_chunked_recv: int = 128


@dataclass
class QuantScales:
    local: Optional[torch.Tensor] = None
    global_scale: Optional[torch.Tensor] = None
    format: Optional[str] = None
    block_size: Optional[int] = None


@dataclass
class DispatchOutput:
    tokens: torch.Tensor
    scales: Optional[QuantScales]
    num_tokens_per_expert: Union[torch.Tensor, List[int]]
    expert_offsets: Optional[torch.Tensor] = None
    layout: DispatchLayout = DispatchLayout.FLAT


@dataclass
class DispatchHandle:
    """Opaque dispatch metadata consumed by :meth:`MoECommunicator.combine`.

    LL keeps ``src_info`` / ``layout_range`` (EXPERT_MAJOR). HT keeps the
    transport-tagged ``combine_meta`` bundle (FLAT); :meth:`combine` is the only
    reader.
    """

    topk_ids: torch.Tensor
    weights: torch.Tensor
    num_experts: int
    num_tokens: int
    hidden_size: int
    num_local_experts: int
    local_expert_start: int
    layout: DispatchLayout
    output_scales: Optional[QuantScales] = None
    # --- LL backend metadata ---
    src_info: Optional[torch.Tensor] = None
    layout_range: Optional[torch.Tensor] = None
    num_max_dispatch_tokens_per_rank: int = 0
    # --- HT backend metadata ---
    is_internode: bool = False
    combine_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommOverlapConfig:
    op: str
    level: str = "op"
    stream: Optional[torch.cuda.Stream] = None
    wait_event: Optional[torch.cuda.Event] = None
    signal: Optional[torch.Tensor] = None
    num_comm_sms: Optional[int] = None
    block_m: Optional[int] = None
    block_ready_value: Optional[int] = None


class _MoERuntime:
    """Private low-level low-latency runtime wrapper (wraps ``_cpp.MoERuntime``)."""

    num_sms: int = 20

    def __init__(
        self,
        comm: Any,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        mode: MoEMode = MoEMode.LOW_LATENCY,
        num_qps_per_rank: int = 12,
    ) -> None:
        if not isinstance(mode, MoEMode):
            raise TypeError("mode must be a MoEMode")
        self.mode = mode
        if self.mode != MoEMode.LOW_LATENCY:
            raise NotImplementedError("_MoERuntime supports only MoEMode.LOW_LATENCY")

        self.rank: int = comm.my_rank
        self.group_size: int = comm.nranks
        self.comm = comm
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.num_qps_per_rank = num_qps_per_rank

        self._cpp_runtime = _cpp.MoERuntime(comm.communicator, num_nvl_bytes, num_rdma_bytes, mode)
        if num_qps_per_rank <= 0:
            raise ValueError("num_qps_per_rank must be > 0")

    def is_available(self) -> bool:
        return self._cpp_runtime.is_available()

    def is_internode_available(self) -> bool:
        return self._cpp_runtime.is_internode_available()

    def get_local_device_id(self) -> int:
        return self._cpp_runtime.get_local_device_id()

    def get_num_rdma_ranks(self) -> int:
        return self._cpp_runtime.get_num_rdma_ranks()

    def get_rdma_rank(self) -> int:
        return self._cpp_runtime.get_rdma_rank()

    def get_root_rdma_rank(self, global_: bool) -> int:
        return self._cpp_runtime.get_root_rdma_rank(global_)


class _CompletionRequest:
    """Request handle returned by the HT ``*_async`` methods."""

    def __init__(self, event, result):
        self._event = event
        self._result = result

    def wait(self):
        if self._event is not None:
            try:
                self._event.current_stream_wait()
            except AttributeError:
                torch.cuda.current_stream().synchronize()
        return self._result


class MoECommunicator:
    """High-level MoE communicator for dispatch/combine.

    ``mode=MoEMode.LOW_LATENCY`` selects the LL backend (EXPERT_MAJOR);
    ``mode=MoEMode.HIGH_THROUGHPUT`` selects the HT backend (FLAT).
    """

    def __init__(self, config: Optional[MoECommunicatorConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either MoECommunicatorConfig or keyword arguments, not both")
        if config is None:
            config = MoECommunicatorConfig(**kwargs)

        if config.device is not None:
            torch.cuda.set_device(config.device)

        if not isinstance(config.mode, MoEMode):
            raise TypeError("MoECommunicatorConfig.mode must be a MoEMode")
        self.mode = config.mode
        self.output_layout = _resolve_output_layout(config.output_layout, self.mode)

        # ---- shared shape / capacity ----
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.max_tokens_per_rank = config.max_tokens_per_rank
        if self.num_experts <= 0 or self.hidden_size <= 0 or self.topk <= 0 or self.max_tokens_per_rank <= 0:
            raise ValueError("num_experts, hidden_size, topk, and max_tokens_per_rank must be positive")

        self.num_sms = config.num_sms
        self.enable_overlap = config.enable_overlap

        if self.mode == MoEMode.LOW_LATENCY:
            self._init_ll(config)
        else:
            self._init_ht(config)

    # ------------------------------------------------------------------
    # Backend construction
    # ------------------------------------------------------------------

    def _resolve_placement(self) -> None:
        if self.num_local_experts is None:
            if self.num_experts % self.world_size != 0:
                raise ValueError("num_experts must be divisible by world_size for even contiguous placement")
            self.num_local_experts = self.num_experts // self.world_size
        if self.num_local_experts * self.world_size != self.num_experts:
            raise NotImplementedError("only even contiguous expert placement is currently supported")
        if self.local_expert_start is None:
            self.local_expert_start = self.rank * self.num_local_experts

    def _init_ll(self, config: MoECommunicatorConfig) -> None:
        from mscclpp._core import CommGroup  # local import: only LL needs it

        comm = config.comm
        if comm is None:
            if config.group is None:
                raise ValueError("mode=LOW_LATENCY requires an mscclpp.CommGroup via comm= (or a torch group=)")
            comm = CommGroup(torch_group=config.group)
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = torch.cuda.current_device()
        self.device = torch.device("cuda", self.local_rank)

        if self.output_layout != DispatchLayout.EXPERT_MAJOR:
            raise NotImplementedError("low-latency mode currently supports only DispatchLayout.EXPERT_MAJOR")
        if self.num_experts % self.world_size != 0:
            raise ValueError("low-latency mode requires num_experts divisible by world_size")

        self.num_local_experts = config.num_local_experts
        self.local_expert_start = config.local_expert_start
        self._resolve_placement()

        if config.max_recv_tokens_per_rank not in (None, self.max_tokens_per_rank):
            raise NotImplementedError("low-latency mode currently uses max_tokens_per_rank as recv capacity")
        if config.input_dtype not in (None, torch.bfloat16):
            raise NotImplementedError("low-latency dispatch currently supports BF16 input only")

        self.quant_format = _normalize_quant_format(config.quant_format)
        if self.quant_format not in (None, "fp8_e4m3"):
            raise NotImplementedError(f"unsupported low-latency quant_format: {config.quant_format}")
        self.dispatch_requires_quantization = self.quant_format is not None

        self._is_internode = True
        num_rdma_bytes = _get_low_latency_rdma_size_hint(
            self.max_tokens_per_rank, self.hidden_size, self.world_size, self.num_experts
        )
        self._dispatch_scales: Optional[torch.Tensor] = None
        self._dispatch_src_info: Optional[torch.Tensor] = None
        self._dispatch_layout_range: Optional[torch.Tensor] = None
        self._dispatch_count: Optional[torch.Tensor] = None

        self._runtime = _MoERuntime(
            comm,
            num_nvl_bytes=0,
            num_rdma_bytes=num_rdma_bytes,
            mode=self.mode,
            num_qps_per_rank=config.num_rdma_qps_per_rank,
        )

    def _init_ht(self, config: MoECommunicatorConfig) -> None:
        group = config.group
        if group is None:
            raise ValueError("mode=HIGH_THROUGHPUT requires a torch.distributed ProcessGroup via group=")
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        self.local_rank = torch.cuda.current_device()
        self.device = torch.device("cuda", self.local_rank)

        if self.output_layout != DispatchLayout.FLAT:
            raise NotImplementedError("HT mode currently supports only DispatchLayout.FLAT")

        self.num_local_experts = config.num_local_experts
        self.local_expert_start = config.local_expert_start
        self._resolve_placement()

        if config.input_dtype not in (None, torch.bfloat16):
            raise NotImplementedError("HT dispatch currently supports BF16 input only")
        if config.quant_format is not None:
            raise NotImplementedError("HT quantized dispatch (scales) is not implemented yet")

        self.expert_alignment = config.expert_alignment

        # Config(num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv). The C++ size
        # hints return 0 RDMA bytes when world_size <= NUM_MAX_NVL_PEERS, which is
        # exactly the intranode/internode boundary — so derive the transport from
        # the hint instead of hardcoding the NVL peer count.
        self._cfg = Config(
            self.num_sms,
            config.nvl_chunked_send,
            config.nvl_chunked_recv,
            config.rdma_chunked_send,
            config.rdma_chunked_recv,
        )
        hidden_bytes = self.hidden_size * torch.tensor([], dtype=torch.bfloat16).element_size()
        num_nvl_bytes = self._cfg.get_nvl_buffer_size_hint(hidden_bytes, self.world_size)
        num_rdma_bytes = self._cfg.get_rdma_buffer_size_hint(hidden_bytes, self.world_size)
        self._is_internode = num_rdma_bytes > 0

        self._buffer = ExpertParallelRuntime(
            group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=config.num_rdma_qps_per_rank,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._runtime.is_available() if self.mode == MoEMode.LOW_LATENCY else self._buffer.is_available()

    def is_internode_available(self) -> bool:
        if self.mode == MoEMode.LOW_LATENCY:
            return self._runtime.is_internode_available()
        return self._buffer.is_internode_available()

    def is_internode(self) -> bool:
        return self._is_internode

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        input: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        scales: Optional[QuantScales] = None,
        *,
        output_buffer: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
        previous_handle: Optional[DispatchHandle] = None,
    ) -> Tuple[DispatchOutput, DispatchHandle]:
        if self.mode == MoEMode.LOW_LATENCY:
            return self._dispatch_ll(input, topk_ids, weights, scales, output_buffer, stream)
        return self._dispatch_ht(input, topk_ids, weights, scales, previous_handle)

    def _dispatch_ll(self, input, topk_ids, weights, scales, output_buffer, stream):
        self._validate_dispatch_inputs(input, topk_ids, weights, scales, output_buffer)
        if weights is None:
            weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=topk_ids.device)

        out_buf, packed_scales, src_info, layout_range, count = self._get_dispatch_output_tensors(output_buffer)
        self._runtime._cpp_runtime.dispatch(
            input.data_ptr(),
            topk_ids.data_ptr(),
            out_buf.data_ptr(),
            0 if packed_scales is None else packed_scales.data_ptr(),
            src_info.data_ptr(),
            layout_range.data_ptr(),
            count.data_ptr(),
            input.size(0),
            self.hidden_size,
            self.topk,
            self.max_tokens_per_rank,
            self.num_experts,
            self.dispatch_requires_quantization,
            self.output_layout,
            _cuda_stream_ptr(stream),
        )
        output_scales = None
        if packed_scales is not None:
            output_scales = QuantScales(local=packed_scales, format="fp8_e4m3", block_size=128)
        dispatch_out = DispatchOutput(
            tokens=out_buf,
            scales=output_scales,
            num_tokens_per_expert=count,
            expert_offsets=None,
            layout=self.output_layout,
        )
        handle = DispatchHandle(
            topk_ids=topk_ids,
            weights=weights,
            num_experts=self.num_experts,
            num_tokens=input.size(0),
            hidden_size=self.hidden_size,
            num_local_experts=self.num_local_experts,
            local_expert_start=self.local_expert_start,
            layout=self.output_layout,
            output_scales=output_scales,
            src_info=src_info,
            layout_range=layout_range,
            num_max_dispatch_tokens_per_rank=self.max_tokens_per_rank,
        )
        return dispatch_out, handle

    def _dispatch_ht(self, input, topk_ids, weights, scales, previous_handle):
        self._validate_dispatch_inputs_ht(input, topk_ids, weights, scales)
        if weights is None:
            weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=topk_ids.device)

        cache = getattr(previous_handle, "_dispatch_cache", None) if previous_handle is not None else None
        if cache is not None:
            num_tokens_per_rank = cache["num_tokens_per_rank"]
            num_tokens_per_rdma_rank = cache["num_tokens_per_rdma_rank"]
            num_tokens_per_expert = cache["num_tokens_per_expert"]
            is_token_in_rank = cache["is_token_in_rank"]
        else:
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
            ) = self._buffer.get_dispatch_layout(topk_ids, self.num_experts)

        if self._is_internode:
            (
                recv_x,
                _recv_x_scales,
                _recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                _rdma_channel_prefix_matrix,
                _gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                _recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
            ) = self._buffer.internode_dispatch(
                input,
                None,
                topk_ids,
                weights,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                0,
                None,
                None,
                None,
                None,
                self.expert_alignment,
                self._cfg,
            )
            combine_meta = {
                "recv_topk_weights": recv_topk_weights,
                "src_meta": recv_src_meta,
                "is_token_in_rank": is_token_in_rank,
                "recv_rdma_channel_prefix_matrix": recv_rdma_channel_prefix_matrix,
                "recv_rdma_rank_prefix_sum": recv_rdma_rank_prefix_sum,
                "recv_gbl_channel_prefix_matrix": recv_gbl_channel_prefix_matrix,
                "send_rdma_head": send_rdma_head,
                "send_nvl_head": send_nvl_head,
            }
            dispatch_cache = (
                cache
                if cache is not None
                else {
                    "num_tokens_per_rank": num_tokens_per_rank,
                    "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
                    "num_tokens_per_expert": num_tokens_per_expert,
                    "is_token_in_rank": is_token_in_rank,
                }
            )
        elif cache is not None:
            (
                recv_x,
                _recv_x_scales,
                _recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                _channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                send_head,
            ) = self._buffer.intranode_dispatch(
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
                self._cfg,
            )
            combine_meta = {
                "recv_topk_weights": recv_topk_weights,
                "src_idx": recv_src_idx,
                "rank_prefix_matrix": rank_prefix_matrix,
                "recv_channel_prefix_matrix": recv_channel_prefix_matrix,
                "send_head": send_head,
            }
            dispatch_cache = cache
        else:
            (
                recv_x,
                _recv_x_scales,
                _recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                send_head,
            ) = self._buffer.intranode_dispatch(
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
                self._cfg,
            )
            combine_meta = {
                "recv_topk_weights": recv_topk_weights,
                "src_idx": recv_src_idx,
                "rank_prefix_matrix": rank_prefix_matrix,
                "recv_channel_prefix_matrix": recv_channel_prefix_matrix,
                "send_head": send_head,
            }
            dispatch_cache = {
                "num_tokens_per_rank": num_tokens_per_rank,
                "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
                "num_tokens_per_expert": num_tokens_per_expert,
                "is_token_in_rank": is_token_in_rank,
                "rank_prefix_matrix": rank_prefix_matrix,
                "channel_prefix_matrix": channel_prefix_matrix,
                "num_recv_tokens": int(recv_x.size(0)),
            }

        dispatch_out = DispatchOutput(
            tokens=recv_x,
            scales=None,
            num_tokens_per_expert=num_recv_tokens_per_expert_list,
            expert_offsets=_exclusive_cumsum(num_recv_tokens_per_expert_list),
            layout=DispatchLayout.FLAT,
        )
        handle = DispatchHandle(
            topk_ids=topk_ids,
            weights=weights,
            num_experts=self.num_experts,
            num_tokens=int(input.size(0)),
            hidden_size=self.hidden_size,
            num_local_experts=self.num_local_experts,
            local_expert_start=self.local_expert_start,
            layout=DispatchLayout.FLAT,
            output_scales=None,
            is_internode=self._is_internode,
            combine_meta=combine_meta,
        )
        # The torch-free HT runtime orders its work on the caller's CUDA stream
        # (no separate event handle), so there is nothing to attach here.
        handle._event = None  # type: ignore[attr-defined]
        handle._dispatch_cache = dispatch_cache  # type: ignore[attr-defined]
        return dispatch_out, handle

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
        *,
        out: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        if self.mode == MoEMode.LOW_LATENCY:
            return self._combine_ll(expert_output, handle, out, stream)
        return self._combine_ht(expert_output, handle, out)

    def _combine_ll(self, expert_output, handle, out, stream):
        self._validate_combine_inputs(expert_output, handle, out)
        combine_requires_dequantization = _requires_dequantization(expert_output)
        x_scales = None
        if combine_requires_dequantization:
            if handle.output_scales is None or handle.output_scales.local is None:
                raise ValueError("FP8 expert_output requires scales captured in the dispatch handle")
            x_scales = handle.output_scales.local
        if out is None:
            out = torch.empty((handle.num_tokens, self.hidden_size), dtype=torch.bfloat16, device=expert_output.device)
        self._runtime._cpp_runtime.combine(
            expert_output.data_ptr(),
            0 if x_scales is None else x_scales.data_ptr(),
            handle.topk_ids.data_ptr(),
            handle.weights.data_ptr(),
            handle.src_info.data_ptr(),
            handle.layout_range.data_ptr(),
            out.data_ptr(),
            handle.num_tokens,
            self.hidden_size,
            handle.weights.size(1),
            handle.num_max_dispatch_tokens_per_rank,
            handle.num_experts,
            combine_requires_dequantization,
            _cuda_stream_ptr(stream),
        )
        return out

    def _combine_ht(self, expert_output, handle, out):
        self._validate_combine_inputs_ht(expert_output, handle)
        m = handle.combine_meta
        if handle.is_internode:
            combined_x, _combined_w = self._buffer.internode_combine(
                expert_output,
                m["recv_topk_weights"],
                m["src_meta"],
                m["is_token_in_rank"],
                m["recv_rdma_channel_prefix_matrix"],
                m["recv_rdma_rank_prefix_sum"],
                m["recv_gbl_channel_prefix_matrix"],
                m["send_rdma_head"],
                m["send_nvl_head"],
                self._cfg,
            )
        else:
            combined_x, _combined_w = self._buffer.intranode_combine(
                expert_output,
                m["recv_topk_weights"],
                m["src_idx"],
                m["rank_prefix_matrix"],
                m["recv_channel_prefix_matrix"],
                m["send_head"],
                self._cfg,
            )
        if out is not None:
            out.copy_(combined_x)
            return out
        return combined_x

    # ------------------------------------------------------------------
    # Optional async / overlap APIs (HT only)
    # ------------------------------------------------------------------

    def dispatch_async(self, *args, **kwargs):
        raise NotImplementedError("dispatch_async is not implemented for MoECommunicator yet")

    def combine_async(self, *args, **kwargs):
        raise NotImplementedError("combine_async is not implemented for MoECommunicator yet")

    def create_overlap_config(
        self, op: str, *, handle: Optional[DispatchHandle] = None, level: str = "op"
    ) -> CommOverlapConfig:
        if op not in ("dispatch", "combine"):
            raise ValueError("op must be 'dispatch' or 'combine'")
        if level != "op":
            raise NotImplementedError("block-level overlap is not implemented yet")
        if op == "combine" and handle is None:
            raise ValueError("combine overlap config requires a DispatchHandle")
        return CommOverlapConfig(op=op, level=level)

    # ------------------------------------------------------------------
    # LL output tensors + validation
    # ------------------------------------------------------------------

    def _get_dispatch_output_tensors(self, output_buffer: torch.Tensor):
        device = output_buffer.device
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if self._dispatch_src_info is None or self._dispatch_src_info.device != device:
            self._dispatch_src_info = torch.empty(
                (self.num_local_experts, slots_per_expert), dtype=torch.int32, device=device
            )
            self._dispatch_layout_range = torch.empty(
                (self.num_local_experts, self.world_size), dtype=torch.int64, device=device
            )
            self._dispatch_count = torch.empty((self.num_local_experts,), dtype=torch.int32, device=device)
            self._dispatch_scales = None
            if self.dispatch_requires_quantization:
                num_scales = self.hidden_size // 128
                scales_storage = torch.empty(
                    (self.num_local_experts, num_scales, slots_per_expert), dtype=torch.float32, device=device
                )
                self._dispatch_scales = scales_storage.transpose(1, 2)
        return (
            output_buffer,
            self._dispatch_scales,
            self._dispatch_src_info,
            self._dispatch_layout_range,
            self._dispatch_count,
        )

    def _validate_dispatch_inputs(self, input, topk_ids, weights, scales, output_buffer) -> None:
        if output_buffer is None:
            raise ValueError("output_buffer is required for low-latency dispatch")
        if scales is not None and (scales.local is not None or scales.global_scale is not None):
            raise NotImplementedError("low-latency dispatch does not support quantized input scales yet")
        if input.dim() != 2 or not input.is_contiguous():
            raise ValueError("input must be a contiguous [num_tokens, hidden_size] tensor")
        if input.device.type != "cuda" or input.dtype != torch.bfloat16:
            raise ValueError("low-latency dispatch input must be a CUDA BF16 tensor")
        if input.size(1) != self.hidden_size:
            raise ValueError(f"input hidden size {input.size(1)} does not match configured {self.hidden_size}")
        if input.size(0) > self.max_tokens_per_rank:
            raise ValueError("input token count exceeds max_tokens_per_rank")
        if topk_ids.dim() != 2 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be a contiguous [num_tokens, topk] tensor")
        if topk_ids.device != input.device or topk_ids.dtype != torch.int64:
            raise ValueError("topk_ids must be an int64 CUDA tensor on the same device as input")
        if topk_ids.shape != (input.size(0), self.topk):
            raise ValueError("topk_ids shape must match [input.size(0), configured topk]")
        if weights is not None:
            if weights.dim() != 2 or not weights.is_contiguous():
                raise ValueError("weights must be a contiguous [num_tokens, topk] tensor")
            if weights.device != input.device or weights.dtype != torch.float32:
                raise ValueError("weights must be a float32 CUDA tensor on the same device as input")
            if weights.shape != topk_ids.shape:
                raise ValueError("weights shape must match topk_ids")
        expected_dtype = torch.float8_e4m3fn if self.dispatch_requires_quantization else torch.bfloat16
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if self.output_layout == DispatchLayout.EXPERT_MAJOR:
            expected_shape = (self.num_local_experts, slots_per_expert, self.hidden_size)
        else:
            expected_shape = (self.num_local_experts * slots_per_expert, self.hidden_size)
        if output_buffer.dim() != len(expected_shape) or not output_buffer.is_contiguous():
            raise ValueError(f"output_buffer must be a contiguous {self.output_layout} tensor")
        if output_buffer.device != input.device or output_buffer.dtype != expected_dtype:
            raise ValueError(f"output_buffer must be a {expected_dtype} CUDA tensor on the same device as input")
        if tuple(output_buffer.shape) != expected_shape:
            raise ValueError(f"output_buffer shape must be {expected_shape}")

    def _validate_combine_inputs(self, expert_output, handle, out) -> None:
        if handle.num_experts != self.num_experts or handle.hidden_size != self.hidden_size:
            raise ValueError("DispatchHandle does not belong to this MoECommunicator configuration")
        slots_per_expert = self.world_size * self.max_tokens_per_rank
        if handle.layout == DispatchLayout.EXPERT_MAJOR:
            expected_shape = (self.num_local_experts, slots_per_expert, self.hidden_size)
        else:
            expected_shape = (self.num_local_experts * slots_per_expert, self.hidden_size)
        if expert_output.dim() != len(expected_shape) or not expert_output.is_contiguous():
            raise ValueError("expert_output must keep dispatch output's contiguous layout")
        if tuple(expert_output.shape) != expected_shape:
            raise ValueError(f"expert_output shape must be {expected_shape}")
        if expert_output.dtype not in (torch.bfloat16, getattr(torch, "float8_e4m3fn", None)):
            raise ValueError("expert_output must be BF16 or FP8 E4M3")
        if out is not None:
            expected_out_shape = (handle.num_tokens, self.hidden_size)
            if tuple(out.shape) != expected_out_shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
                raise ValueError(f"out must be a contiguous BF16 tensor with shape {expected_out_shape}")

    # ------------------------------------------------------------------
    # HT validation
    # ------------------------------------------------------------------

    def _validate_dispatch_inputs_ht(self, input, topk_ids, weights, scales) -> None:
        if scales is not None and (scales.local is not None or scales.global_scale is not None):
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

    def _validate_combine_inputs_ht(self, expert_output, handle) -> None:
        if not isinstance(handle, DispatchHandle):
            raise TypeError("handle must be a DispatchHandle returned by dispatch")
        if expert_output.dim() != 2 or not expert_output.is_contiguous():
            raise ValueError("expert_output must be a contiguous [total_recv_tokens, hidden] tensor")
        if expert_output.size(1) != self.hidden_size:
            raise ValueError(f"expert_output hidden size {expert_output.size(1)} != configured {self.hidden_size}")
        if handle.is_internode != self._is_internode:
            raise ValueError("handle transport does not match this communicator")


def _resolve_output_layout(layout: Optional[DispatchLayout], mode: MoEMode) -> DispatchLayout:
    if layout is None:
        return DispatchLayout.EXPERT_MAJOR if mode == MoEMode.LOW_LATENCY else DispatchLayout.FLAT
    if not isinstance(layout, DispatchLayout):
        raise TypeError("MoECommunicatorConfig.output_layout must be a DispatchLayout")
    return layout


def _cuda_stream_ptr(stream: Optional[torch.cuda.Stream]) -> int:
    return (stream if stream is not None else torch.cuda.current_stream()).cuda_stream


def _normalize_quant_format(fmt: Optional[str]) -> Optional[str]:
    if fmt is None:
        return None
    normalized = fmt.lower().replace("-", "_")
    if normalized in ("fp8", "fp8_e4m3", "f8e4m3", "float8_e4m3fn"):
        return "fp8_e4m3"
    return normalized


def _requires_dequantization(tensor: torch.Tensor) -> bool:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    return fp8_dtype is not None and tensor.dtype == fp8_dtype


def _exclusive_cumsum(counts: Union[torch.Tensor, List[int]]) -> torch.Tensor:
    if isinstance(counts, torch.Tensor):
        flat = counts.to(torch.int64).flatten()
        zero = torch.zeros(1, dtype=torch.int64, device=flat.device)
        return torch.cat([zero, torch.cumsum(flat, dim=0)])
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + int(c))
    return torch.tensor(offsets, dtype=torch.int64)


def _get_low_latency_rdma_size_hint(
    num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int
) -> int:
    return _cpp.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)
