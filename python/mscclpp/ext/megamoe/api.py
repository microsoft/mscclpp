# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Torch tensor adapter for the pointer-only native MegaMoE binding."""

from dataclasses import dataclass
import math


def is_available() -> bool:
    """Whether this MSCCL++ build includes MegaMoE (not a hardware capability test)."""
    from mscclpp import _mscclpp

    return bool(getattr(_mscclpp, "megamoe_available", lambda: False)())


@dataclass(frozen=True)
class MegaMoEConfig:
    """Immutable dimensions for a collectively constructed routed-expert context.

    ``intermediate`` is the post-SwiGLU width. Experts are assigned in contiguous
    ``num_experts // world_size`` blocks. ``sm_margin`` reserves SMs, not a fraction:
    32 on a 152-SM GB200 permits at most 120 persistent CTAs.
    """

    rank: int
    world_size: int
    max_tokens: int
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    sm_margin: int = 0
    weight_e5m2: bool = False
    gate_up_clamp: float = -1.0

    def __post_init__(self):
        for name in (
            "rank",
            "world_size",
            "max_tokens",
            "hidden",
            "intermediate",
            "num_experts",
            "top_k",
            "sm_margin",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**31 - 1:
                raise ValueError(f"{name} must be a nonnegative 32-bit integer")
        if not 1 <= self.world_size <= 72 or not self.rank < self.world_size:
            raise ValueError("world_size must be in [1, 72] and rank in [0, world_size)")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if any(value < 128 or value % 128 or value > (2**31 - 1) // 2 for value in (self.hidden, self.intermediate)):
            raise ValueError("hidden and intermediate must be multiples of 128 within signed 32-bit indexing")
        if self.num_experts < 1 or self.num_experts % self.world_size:
            raise ValueError("num_experts must be positive and divisible by world_size")
        if not 1 <= self.top_k <= min(32, self.num_experts):
            raise ValueError("top_k must be in [1, min(32, num_experts)]")
        if self.world_size * self.max_tokens * self.top_k > 2**31 - 1:
            raise ValueError("routing capacity exceeds signed 32-bit indexing")
        if not isinstance(self.weight_e5m2, bool):
            raise ValueError("weight_e5m2 must be bool")
        if not isinstance(self.gate_up_clamp, (int, float)) or not math.isfinite(self.gate_up_clamp):
            raise ValueError("gate_up_clamp must be finite; negative disables clamping")

    @property
    def local_experts(self) -> int:
        """Number of contiguous experts owned by this rank."""
        return self.num_experts // self.world_size


def _tensor(tensor, name, shape, dtype, device):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.device != device or not tensor.is_cuda:
        raise ValueError(f"{name} must be on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous in canonical row-major order")
    if tensor.requires_grad:
        raise ValueError("MegaMoE is an inference-only operation and does not implement autograd")


class MegaMoE:
    """Routed SwiGLU experts with CUDA Graph-compatible MSCCL++ communication.

    Construction is collective, outside graph capture, on one SM100 GPU per rank
    in the same active NVLink fabric. ``fc1`` and ``fc2`` are canonical FP8 tensors
    [local_experts, 2*I, H] and [local_experts, H, I]. The first I FC1 rows are gate
    and the second I rows are up. Scales are uint8 E8M0 tensors [local_experts, M,
    K//32]; no backend-specific swizzle is accepted. Weights are packed into
    context-owned allocations, so original weight tensors may be released after
    construction.

    Contexts and input views must outlive all ranks' outstanding launches and
    graphs. Do not run two forwards or graph replays concurrently on one context.
    Order use on different streams explicitly with CUDA events. Shared experts,
    squash, unsquash, residuals, and router-weight normalization are not included.
    """

    def __init__(self, config: MegaMoEConfig, communicator, fc1, fc1_scale, fc2, fc2_scale, *, stream=None, tag=17920):
        if not isinstance(config, MegaMoEConfig):
            raise TypeError("config must be a MegaMoEConfig")
        if not is_available():
            raise RuntimeError(
                "MSCCL++ was built without native MegaMoE. Rebuild with "
                "-DMSCCLPP_BUILD_EXT_MEGAMOE=ON using CUDA nvcc/ptxas >=13.3 and CUTLASS."
            )
        import torch
        from mscclpp import _mscclpp

        if not torch.cuda.is_available() or torch.version.hip:
            raise RuntimeError("Native MegaMoE requires NVIDIA CUDA and an SM100 GPU; ROCm is unsupported")
        self.config = config
        self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (10, 0):
            raise RuntimeError("Native MegaMoE currently requires an SM100 GPU (for example GB200)")
        if config.sm_margin > torch.cuda.get_device_properties(self.device).multi_processor_count - 2:
            raise ValueError("sm_margin must leave at least two SMs")
        stream = self._stream(stream)
        with torch.cuda.stream(stream):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Construct MegaMoE before CUDA Graph capture")
        e, h, i = config.local_experts, config.hidden, config.intermediate
        dtype = torch.float8_e5m2 if config.weight_e5m2 else torch.float8_e4m3fn
        _tensor(fc1, "fc1", (e, 2 * i, h), dtype, self.device)
        _tensor(fc1_scale, "fc1_scale", (e, 2 * i, h // 32), torch.uint8, self.device)
        _tensor(fc2, "fc2", (e, h, i), dtype, self.device)
        _tensor(fc2_scale, "fc2_scale", (e, h, i // 32), torch.uint8, self.device)
        native_config = _mscclpp.CppMegaMoeConfig()
        for name, value in vars(config).items():
            setattr(native_config, name, value)
        self._native = _mscclpp.CppMegaMoeContext.create(
            communicator,
            native_config,
            fc1.data_ptr(),
            fc1_scale.data_ptr(),
            fc2.data_ptr(),
            fc2_scale.data_ptr(),
            stream.cuda_stream,
            tag,
        )

    def _stream(self, stream):
        import torch

        if stream is None:
            return torch.cuda.current_stream(self.device)
        if not isinstance(stream, torch.cuda.Stream) or stream.device != self.device:
            raise ValueError(f"stream must be a torch.cuda.Stream on {self.device}")
        return stream

    @property
    def cta_count(self) -> int:
        """Actual persistent CTA count after SM-margin and occupancy limits."""
        return self._native.cta_count

    @property
    def workspace_bytes(self) -> dict:
        """Symmetric/private byte sizes (packed weights are not included)."""
        return {"symmetric": self._native.symmetric_bytes, "private": self._native.private_bytes}

    def input_view(self, num_tokens=None):
        """Return a zero-copy BF16 [T,H] Torch view that owns the native context.

        A squash/projection kernel may write directly here before ``forward`` on
        the same stream. Passing this exact view skips the input staging copy.
        Taking slices preserves the underlying DLPack storage owner.
        """
        import torch

        if num_tokens is None:
            num_tokens = self.config.max_tokens
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
            raise TypeError("num_tokens must be an integer")
        return torch.utils.dlpack.from_dlpack(self._native.input_dlpack(num_tokens))

    def forward(self, inputs, topk_ids, topk_weights, *, output=None, stream=None, validate_routing=False):
        """Enqueue routed experts and return BF16 [T,H] output on ``stream``.

        ``inputs`` is BF16, ``topk_ids`` is int32, and ``topk_weights`` is float32.
        IDs must be in [0, num_experts); weights must be finite. Every rank must
        participate in the same launch order, but token counts may differ or be
        zero. Optional value validation synchronizes and is forbidden in capture;
        ordinary validation of shape/dtype/contiguity is always performed.

        Pass an output allocated before capture for stable graph replay storage.
        The caller must preserve graph inputs/outputs and this context across
        replays. Stream dependencies for producer tensors remain caller-owned.
        """
        import torch

        stream = self._stream(stream)
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise ValueError("inputs must be a rank-2 torch.Tensor")
        t, h = inputs.shape
        if t > self.config.max_tokens:
            raise ValueError("input token count exceeds max_tokens")
        _tensor(inputs, "inputs", (t, self.config.hidden), torch.bfloat16, self.device)
        _tensor(topk_ids, "topk_ids", (t, self.config.top_k), torch.int32, self.device)
        _tensor(topk_weights, "topk_weights", (t, self.config.top_k), torch.float32, self.device)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            if validate_routing:
                if torch.cuda.is_current_stream_capturing():
                    raise ValueError("validate_routing=True is not CUDA Graph capturable")
                if not bool(((topk_ids >= 0) & (topk_ids < self.config.num_experts)).all()):
                    raise ValueError("routing IDs must be in [0, num_experts)")
                if not bool(torch.isfinite(topk_weights).all()):
                    raise ValueError("router weights must be finite")
            if output is None:
                output = torch.empty((t, h), dtype=torch.bfloat16, device=self.device)
            _tensor(output, "output", (t, h), torch.bfloat16, self.device)
            # Record allocator use on the launch stream without a host wait.
            for tensor in (inputs, topk_ids, topk_weights, output):
                tensor.record_stream(stream)
            self._native.forward(
                inputs.data_ptr(),
                topk_ids.data_ptr(),
                topk_weights.data_ptr(),
                output.data_ptr(),
                t,
                stream.cuda_stream,
            )
        return output

    __call__ = forward
