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
    weight_mxfp4: bool = False

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
        if not isinstance(self.weight_mxfp4, bool):
            raise ValueError("weight_mxfp4 must be bool")
        if self.weight_e5m2 and self.weight_mxfp4:
            raise ValueError("weight_e5m2 and weight_mxfp4 are mutually exclusive")
        if self.weight_mxfp4 and self.hidden < 4096:
            raise ValueError("MXFP4/MXFP8 requires hidden >= 4096")
        if self.weight_mxfp4 and (self.world_size, self.num_experts, self.top_k) == (1, 1, 1):
            raise ValueError("MXFP4/MXFP8 currently supports routed experts only")
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

    Construction is collective, outside graph capture, on one SM100-family GPU per rank
    in the same active NVLink fabric. By default, ``fc1`` and ``fc2`` are canonical
    FP8 tensors [local_experts, 2*I, H] and [local_experts, H, I]. With
    ``weight_mxfp4=True``, they are packed uint8 [local_experts, 2*I, H//2] and
    [local_experts, H, I//2], with the even K element in the lower nibble. The
    first I FC1 rows are gate and the second I rows are up. Scales are canonical
    uint8 E8M0 tensors [local_experts, M, K//32]. Weights are packed into
    context-owned allocations, so original tensors may be released after construction.

    Contexts and input views must outlive all ranks' outstanding launches and
    graphs. Do not run two forwards or graph replays concurrently on one context.
    Order use on different streams explicitly with CUDA events. Shared experts,
    squash, unsquash, residuals, and router-weight normalization are not included.

    ``kernel`` selects a routed ``KernelConfig``, ``W4A8KernelConfig``, or
    prepared ``CompiledKernel``.
    Preparing modules before collective construction is recommended; neither
    forward nor graph replay compiles or tunes. The default uses the builtin
    kernel without a compiler dependency. Local shared experts remain unchanged.
    """

    def __init__(
        self,
        config: MegaMoEConfig,
        communicator,
        fc1,
        fc1_scale,
        fc2,
        fc2_scale,
        *,
        stream=None,
        tag=17920,
        kernel=None,
    ):
        if not isinstance(config, MegaMoEConfig):
            raise TypeError("config must be a MegaMoEConfig")
        if not is_available():
            raise RuntimeError(
                "MSCCL++ was built without native MegaMoE. Rebuild with "
                "-DMSCCLPP_BUILD_EXT_MEGAMOE=ON using CUDA nvcc/ptxas >=13.0 and CUTLASS."
            )
        import torch
        from mscclpp import _mscclpp
        from .jit import CompiledKernel, KernelConfig, W4A8KernelConfig, compile_kernel

        if not torch.cuda.is_available() or torch.version.hip:
            raise RuntimeError("Native MegaMoE requires NVIDIA CUDA and an SM100-family GPU; ROCm is unsupported")
        self.config = config
        self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) not in ((10, 0), (10, 3), (10, 7)):
            raise RuntimeError("Native MegaMoE currently requires an SM100-family GPU")
        if config.sm_margin > torch.cuda.get_device_properties(self.device).multi_processor_count - 2:
            raise ValueError("sm_margin must leave at least two SMs")
        stream = self._stream(stream)
        with torch.cuda.stream(stream):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Construct MegaMoE before CUDA Graph capture")
        e, h, i = config.local_experts, config.hidden, config.intermediate
        dtype = (
            torch.uint8 if config.weight_mxfp4 else (torch.float8_e5m2 if config.weight_e5m2 else torch.float8_e4m3fn)
        )
        fc1_shape = (e, 2 * i, h // 2) if config.weight_mxfp4 else (e, 2 * i, h)
        fc2_shape = (e, h, i // 2) if config.weight_mxfp4 else (e, h, i)
        _tensor(fc1, "fc1", fc1_shape, dtype, self.device)
        _tensor(fc1_scale, "fc1_scale", (e, 2 * i, h // 32), torch.uint8, self.device)
        _tensor(fc2, "fc2", fc2_shape, dtype, self.device)
        _tensor(fc2_scale, "fc2_scale", (e, h, i // 32), torch.uint8, self.device)
        if kernel is None:
            kernel = KernelConfig()
        if not isinstance(kernel, (KernelConfig, W4A8KernelConfig, CompiledKernel)):
            raise TypeError("kernel must be KernelConfig, W4A8KernelConfig, CompiledKernel, or None")
        selected = kernel if isinstance(kernel, (KernelConfig, W4A8KernelConfig)) else kernel.config
        if config.weight_mxfp4:
            if isinstance(selected, KernelConfig) and selected != KernelConfig():
                raise ValueError("MXFP4/MXFP8 requires the builtin kernel or a W4A8KernelConfig")
        elif isinstance(selected, W4A8KernelConfig):
            raise ValueError("W4A8KernelConfig requires weight_mxfp4=True")
        if (config.world_size, config.num_experts, config.top_k) == (1, 1, 1) and selected != KernelConfig():
            raise ValueError("JIT specialization applies to routed experts; the local shared kernel is fixed")
        if isinstance(kernel, (KernelConfig, W4A8KernelConfig)):
            kernel = compile_kernel(kernel)
        self._kernel = kernel
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
            kernel.path,
            "" if kernel.key == "builtin" else kernel.key,
        )

    @property
    def kernel_config(self):
        """Routed specialization selected before context construction."""
        return self._kernel.config

    @property
    def effective_kernel_config(self):
        """Actual tile and pipeline values used by the native kernel."""
        policy = {
            "tile_m": self._native.kernel_tile_m,
            "tile_n": self._native.kernel_tile_n,
            "tile_k": self._native.kernel_tile_k,
            "load_stages": self._native.kernel_load_stages,
            "transform_stages": self._native.kernel_transform_stages,
        }
        if self.config.weight_mxfp4:
            policy.update(
                {
                    "num_warps": self._native.kernel_num_warps,
                    "transfer_registers": self._native.kernel_transfer_registers,
                    "load_warps": self._native.kernel_load_warps,
                    "split_pipelines": self._native.kernel_split_pipelines,
                    "epilogue_tokens": self._native.kernel_epilogue_tokens,
                    "epilogue_warps": self._native.kernel_epilogue_warps,
                    "epilogue_registers": self._native.kernel_epilogue_registers,
                    "dispatch_chunk": self._native.kernel_dispatch_chunk,
                    "dispatch_warps": self._native.kernel_dispatch_warps,
                    "dispatch_stages": self._native.kernel_dispatch_stages,
                    "fixed_token_count": self._native.kernel_fixed_token_count,
                }
            )
        return policy

    @property
    def kernel_id(self):
        """Content-addressed JIT key, or ``builtin`` for the default kernel."""
        return self._native.kernel_id

    @property
    def shared_bytes(self):
        """Dynamic shared-memory bytes per CTA for the selected native kernel."""
        return self._native.shared_bytes

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

    def wait_until_started(self, stream):
        """Order ``stream`` after the latest signaled forward enters its kernel.

        Call after ``forward(..., signal_start=True)``. This waits for kernel entry,
        not output readiness, and uses a non-resident CUDA stream memory wait.
        It is graph-capturable. Join this consumer before another forward/replay
        reuses the context; ordinary producer-input ordering remains caller-owned.
        """
        import torch

        stream = self._stream(stream)
        with torch.cuda.device(self.device):
            self._native.wait_until_started(stream.cuda_stream)

    def forward_shared(self, inputs, *, output=None, stream=None):
        """Enqueue an unweighted local SwiGLU expert without routing metadata.

        Requires ``world_size=1``, ``num_experts=1``, and ``top_k=1``.
        Uses the same FP32 activation arithmetic, BF16 handoff/output, clamping,
        and stream/lifetime rules as ``forward`` with ID zero and weight one.
        Input staging is retained unless ``inputs`` is an exact ``input_view``
        alias. No token dispatch or final top-k reduction is needed.
        """
        import torch

        if (self.config.world_size, self.config.num_experts, self.config.top_k) != (1, 1, 1):
            raise ValueError("forward_shared requires world_size=1, num_experts=1, top_k=1")
        stream = self._stream(stream)
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise ValueError("inputs must be a rank-2 torch.Tensor")
        tokens = inputs.shape[0]
        if tokens > self.config.max_tokens:
            raise ValueError("input token count exceeds max_tokens")
        _tensor(inputs, "inputs", (tokens, self.config.hidden), torch.bfloat16, self.device)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            if output is None:
                output = torch.empty((tokens, self.config.hidden), dtype=torch.bfloat16, device=self.device)
            _tensor(output, "output", (tokens, self.config.hidden), torch.bfloat16, self.device)
            inputs.record_stream(stream)
            output.record_stream(stream)
            self._native.forward_shared(inputs.data_ptr(), output.data_ptr(), tokens, stream.cuda_stream)
        return output

    def forward(
        self, inputs, topk_ids, topk_weights, *, output=None, stream=None, validate_routing=False, signal_start=False
    ):
        """Enqueue routed experts and return BF16 [T,H] output on ``stream``.

        ``inputs`` is BF16, ``topk_ids`` is int32, and ``topk_weights`` is float32.
        IDs must be in [0, num_experts); weights must be finite. Every rank must
        participate in the same launch order, but token counts may differ or be
        zero. Optional value validation synchronizes and is forbidden in capture;
        ordinary validation of shape/dtype/contiguity is always performed.
        W4A8 ``fixed_token_count=True`` is a captured-decode opt-in: captured
        input row counts must match across ranks, but masked routes may differ.
        Use expert ID -1 for unused slots, including every slot of a padded row.

        Pass an output allocated before capture for stable graph replay storage.
        The caller must preserve graph inputs/outputs and this context across
        replays. Stream dependencies for producer tensors remain caller-owned.

        ``signal_start=True`` enables ``wait_until_started`` for a subsequent
        consumer stream. It adds a device-flag reset/event before kernel launch;
        the default path does not enqueue these operations.
        """
        import torch

        stream = self._stream(stream)
        if not isinstance(signal_start, bool):
            raise TypeError("signal_start must be bool")
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
                signal_start,
            )
        return output

    __call__ = forward
