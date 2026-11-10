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
    gate_up_clamp: float = -1.0
    weight_dtype: str = "mxfp4"
    activation_dtype: str = "mxfp8"
    activation_scale: float | None = None
    combine_route_weights: bool = False

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
        if not isinstance(self.gate_up_clamp, (int, float)) or not math.isfinite(self.gate_up_clamp):
            raise ValueError("gate_up_clamp must be finite; negative disables clamping")
        if not isinstance(self.combine_route_weights, bool):
            raise ValueError("combine_route_weights must be a bool")
        if not isinstance(self.weight_dtype, str) or self.weight_dtype not in ("mxfp4", "mxfp8", "fp8_nv"):
            raise ValueError("weight_dtype must be 'mxfp4', 'mxfp8', or 'fp8_nv'")
        if not isinstance(self.activation_dtype, str) or self.activation_dtype not in (
            "mxfp8",
            "fp8_static",
            "bf16",
        ):
            raise ValueError("activation_dtype must be 'mxfp8', 'fp8_static', or 'bf16'")
        if self.weight_dtype == "mxfp8" and self.activation_dtype == "mxfp8" and self.activation_scale is None:
            object.__setattr__(self, "activation_dtype", "bf16")
        elif self.activation_dtype == "bf16" and self.weight_dtype == "mxfp4":
            object.__setattr__(self, "weight_dtype", "mxfp8")
        if self.weight_dtype in ("mxfp8", "fp8_nv"):
            static = self.activation_dtype == "fp8_static" or (
                self.activation_dtype == "mxfp8" and self.activation_scale is not None
            )
            if static:
                if (self.world_size, self.num_experts, self.top_k) != (1, 1, 1) or self.combine_route_weights:
                    raise ValueError(
                        f"weight_dtype='{self.weight_dtype}' with static FP8 activations (W8A8) supports a single "
                        "local expert only: world_size=1, num_experts=1, top_k=1, combine_route_weights=False"
                    )
            elif self.weight_dtype == "fp8_nv":
                raise ValueError("weight_dtype='fp8_nv' requires activation_dtype='fp8_static' (W8A8)")
            elif self.activation_dtype != "bf16" or self.activation_scale is not None:
                raise ValueError(
                    "weight_dtype='mxfp8' requires activation_dtype='bf16' without activation_scale, "
                    "or activation_dtype='fp8_static' for a single local expert"
                )
        else:
            if self.hidden < 4096:
                raise ValueError("MXFP4/MXFP8 requires hidden >= 4096")
            if (self.world_size, self.num_experts, self.top_k) == (1, 1, 1):
                raise ValueError("MXFP4/MXFP8 currently supports routed experts only")
        if self.activation_scale is not None:
            if (
                isinstance(self.activation_scale, bool)
                or not isinstance(self.activation_scale, (int, float))
                or not math.isfinite(self.activation_scale)
                or self.activation_scale <= 0
            ):
                raise ValueError("activation_scale must be a finite positive float")
            object.__setattr__(self, "activation_scale", float(self.activation_scale))
            if self.activation_dtype == "mxfp8":
                object.__setattr__(self, "activation_dtype", "fp8_static")
        elif self.activation_dtype == "fp8_static":
            object.__setattr__(self, "activation_scale", 1.0)

    @property
    def is_w8a16(self) -> bool:
        """Whether weights use MXFP8 (E4M3 + E8M0 K32) and activations use BF16."""
        return self.weight_dtype == "mxfp8" and self.activation_dtype == "bf16"

    @property
    def is_w8a8(self) -> bool:
        """Whether a single local expert uses FP8 weights (MXFP8 or fp8_nv) and static-FP8 E4M3 activations."""
        return self.weight_dtype in ("mxfp8", "fp8_nv") and self.activation_dtype == "fp8_static"

    @property
    def is_static_fp8(self) -> bool:
        """Whether activations and combine use FP8 E4M3 with per-tensor static scales."""
        return self.activation_dtype == "fp8_static"

    @property
    def local_experts(self) -> int:
        """Number of contiguous experts owned by this rank."""
        return self.num_experts // self.world_size


@dataclass(frozen=True)
class SharedExpertConfig:
    """Dimensions specific to one fused local fp8_nv shared expert."""

    intermediate: int
    gate_up_clamp: float = -1.0
    hidden: int | None = None

    def __post_init__(self):
        if (
            isinstance(self.intermediate, bool)
            or not isinstance(self.intermediate, int)
            or self.intermediate < 128
            or self.intermediate % 128
            or self.intermediate > (2**31 - 1) // 2
        ):
            raise ValueError("shared intermediate must be a positive multiple of 128 within signed 32-bit indexing")
        if not isinstance(self.gate_up_clamp, (int, float)) or not math.isfinite(self.gate_up_clamp):
            raise ValueError("shared gate_up_clamp must be finite; negative disables clamping")
        if self.hidden is not None and (
            isinstance(self.hidden, bool)
            or not isinstance(self.hidden, int)
            or self.hidden < 128
            or self.hidden % 128
            or self.hidden > (2**31 - 1) // 2
        ):
            raise ValueError("shared hidden must be a positive multiple of 128 within signed 32-bit indexing")


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
    in the same active NVLink fabric. ``fc1`` and ``fc2`` are packed uint8
    [local_experts, 2*I, H//2] and
    [local_experts, H, I//2], with the even K element in the lower nibble. The
    first I FC1 rows are gate and the second I rows are up. Scales are canonical
    uint8 E8M0 tensors [local_experts, M, K//32]. MXFP8 weights are float8_e4m3fn
    [local_experts, 2*I, H] and [local_experts, H, I] with the same E8M0 scales;
    ``fp8_nv`` weights use those value tensors with one float32 scale per matrix,
    ``fc1_scale``/``fc2_scale`` of shape [local_experts]. Weights are packed into
    context-owned allocations, so original tensors may be released after construction.

    Contexts and input views must outlive all ranks' outstanding launches and
    graphs. Do not run two forwards or graph replays concurrently on one context.
    Order use on different streams explicitly with CUDA events. Shared experts,
    squash, unsquash, residuals, and router-weight normalization are not included.

    ``kernel`` selects a ``W4A8KernelConfig`` or prepared ``CompiledKernel``.
    Preparing modules before collective construction is recommended; neither
    forward nor graph replay compiles or tunes. The default uses the builtin
    kernel without a compiler dependency. Local shared experts remain unchanged.

    ``shared_workspace=other`` reuses another context's symmetric/private GPU
    workspace, peer mappings, counters, input views, and execution-start signal,
    but allocates separate packed weights and weight-specific kernel plans.
    Device, communicator, config, and kernel specialization must match. Sharing
    mode and the original workspace owner must agree across ranks. All aliases
    must execute serially in the same global context-use order across ranks,
    including graph replays and writes to input views. Use one stream or explicit
    CUDA-event ordering; join start-signal consumers before any alias runs again.
    A context need only outlive its own graphs/launches: aliases retain the runtime
    even if the original Python context is deleted. The default remains independent.
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
        shared_workspace: "MegaMoE | None" = None,
        shared_config: SharedExpertConfig | None = None,
        shared_fc1=None,
        shared_fc1_scale=None,
        shared_fc2=None,
        shared_fc2_scale=None,
    ):
        if not isinstance(config, MegaMoEConfig):
            raise TypeError("config must be a MegaMoEConfig")
        if shared_workspace is not None and not isinstance(shared_workspace, MegaMoE):
            raise TypeError("shared_workspace must be a MegaMoE or None")
        shared_values = (shared_fc1, shared_fc1_scale, shared_fc2, shared_fc2_scale)
        if (shared_config is None) != all(value is None for value in shared_values):
            raise ValueError("shared_config and all four shared weight tensors must be supplied together")
        if shared_config is not None and not isinstance(shared_config, SharedExpertConfig):
            raise TypeError("shared_config must be a SharedExpertConfig or None")
        if shared_config is not None and any(value is None for value in shared_values):
            raise ValueError("shared_config and all four shared weight tensors must be supplied together")
        if not is_available():
            raise RuntimeError(
                "MSCCL++ was built without native MegaMoE. Rebuild with "
                "-DMSCCLPP_BUILD_EXT_MEGAMOE=ON using CUDA nvcc/ptxas >=13.0 and CUTLASS."
            )
        import torch
        from mscclpp import _mscclpp
        from .jit import CompiledKernel, W4A8KernelConfig, compile_kernel, load_cached_kernel

        if not torch.cuda.is_available() or torch.version.hip:
            raise RuntimeError("Native MegaMoE requires NVIDIA CUDA and an SM100-family GPU; ROCm is unsupported")
        self.config = config
        self.shared_config = shared_config
        self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) not in ((10, 0), (10, 3), (10, 7)):
            raise RuntimeError("Native MegaMoE currently requires an SM100-family GPU")
        stream = self._stream(stream)
        with torch.cuda.stream(stream):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Construct MegaMoE before CUDA Graph capture")
        e, h, i = config.local_experts, config.hidden, config.intermediate
        mxfp8_weights = config.weight_dtype in ("mxfp8", "fp8_nv")
        fc1_shape = (e, 2 * i, h if mxfp8_weights else h // 2)
        fc2_shape = (e, h, i if mxfp8_weights else i // 2)
        weight_dtype = torch.float8_e4m3fn if mxfp8_weights else torch.uint8
        _tensor(fc1, "fc1", fc1_shape, weight_dtype, self.device)
        _tensor(fc2, "fc2", fc2_shape, weight_dtype, self.device)
        if config.weight_dtype == "fp8_nv":
            # One FP32 scale per weight matrix: dequantized weight = fp8 value * scale.
            _tensor(fc1_scale, "fc1_scale", (e,), torch.float32, self.device)
            _tensor(fc2_scale, "fc2_scale", (e,), torch.float32, self.device)
        else:
            _tensor(fc1_scale, "fc1_scale", (e, 2 * i, h // 32), torch.uint8, self.device)
            _tensor(fc2_scale, "fc2_scale", (e, h, i // 32), torch.uint8, self.device)
        if kernel is None:
            kernel = load_cached_kernel("builtin")
        if not isinstance(kernel, (W4A8KernelConfig, CompiledKernel)):
            raise TypeError("kernel must be W4A8KernelConfig, CompiledKernel, or None")
        if mxfp8_weights and (isinstance(kernel, W4A8KernelConfig) or kernel.key != "builtin"):
            raise ValueError("W4A8KernelConfig requires weight_dtype='mxfp4'")
        if shared_config is not None:
            if config.weight_dtype != "mxfp4" or config.activation_dtype != "fp8_static":
                raise ValueError(
                    "fused shared mode requires routed weight_dtype='mxfp4' and activation_dtype='fp8_static'"
                )
            if isinstance(kernel, W4A8KernelConfig) or kernel.key != "builtin":
                raise ValueError("fused shared mode currently requires the builtin kernel")
            sh = shared_config.hidden or h
            si = shared_config.intermediate
            _tensor(shared_fc1, "shared_fc1", (1, 2 * si, sh), torch.float8_e4m3fn, self.device)
            _tensor(shared_fc2, "shared_fc2", (1, sh, si), torch.float8_e4m3fn, self.device)
            _tensor(shared_fc1_scale, "shared_fc1_scale", (1,), torch.float32, self.device)
            _tensor(shared_fc2_scale, "shared_fc2_scale", (1,), torch.float32, self.device)
        selected = kernel if isinstance(kernel, W4A8KernelConfig) else kernel.config
        cluster_size = 2 if mxfp8_weights else selected.cluster_size
        if config.sm_margin > torch.cuda.get_device_properties(self.device).multi_processor_count - cluster_size:
            raise ValueError(f"sm_margin must leave at least {cluster_size} SMs for the selected cluster")
        if isinstance(kernel, W4A8KernelConfig):
            kernel = compile_kernel(kernel)
        self._kernel = kernel
        native_config = _mscclpp.CppMegaMoeConfig()
        for name in (
            "rank",
            "world_size",
            "max_tokens",
            "hidden",
            "intermediate",
            "num_experts",
            "top_k",
            "sm_margin",
            "gate_up_clamp",
            "combine_route_weights",
        ):
            setattr(native_config, name, getattr(config, name))
        native_config.weight_dtype = getattr(
            _mscclpp.CppMegaMoeWeightDtype,
            {"mxfp4": "Mxfp4", "mxfp8": "Mxfp8", "fp8_nv": "Fp8Nv"}[config.weight_dtype],
        )
        native_config.activation_dtype = (
            _mscclpp.CppMegaMoeActivationDtype.Bf16
            if config.is_w8a16
            else (
                _mscclpp.CppMegaMoeActivationDtype.Fp8Static
                if config.is_static_fp8
                else _mscclpp.CppMegaMoeActivationDtype.Mxfp8
            )
        )
        native_config.activation_scale = float(config.activation_scale or 1.0)
        if shared_config is None:
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
                None if shared_workspace is None else shared_workspace._native,
            )
        else:
            native_shared_config = _mscclpp.CppMegaMoeSharedExpertConfig()
            native_shared_config.hidden = shared_config.hidden or config.hidden
            native_shared_config.intermediate = shared_config.intermediate
            native_shared_config.gate_up_clamp = shared_config.gate_up_clamp
            self._native = _mscclpp.CppMegaMoeContext.create_fused_shared(
                communicator,
                native_config,
                fc1.data_ptr(),
                fc1_scale.data_ptr(),
                fc2.data_ptr(),
                fc2_scale.data_ptr(),
                native_shared_config,
                shared_fc1.data_ptr(),
                shared_fc1_scale.data_ptr(),
                shared_fc2.data_ptr(),
                shared_fc2_scale.data_ptr(),
                stream.cuda_stream,
                tag,
                None if shared_workspace is None else shared_workspace._native,
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
            "expert_major_tasks": self._native.kernel_expert_major_tasks,
            "cache_weights": self._native.kernel_cache_weights,
            "adaptive_cache_weights": self._native.kernel_adaptive_cache_weights,
            "cluster_size": self._native.kernel_cluster_size,
        }
        if self.config.is_w8a16:
            policy["transform_stages"] = self._native.kernel_transform_stages
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
        """Symmetric/private runtime sizes, not incremental bytes for workspace aliases."""
        return {"symmetric": self._native.symmetric_bytes, "private": self._native.private_bytes}

    @property
    def workspace_id(self) -> int:
        """Private workspace pointer, equal exactly for live contexts sharing their runtime."""
        return self._native.workspace_id

    @property
    def packed_weight_pointers(self) -> tuple[int, int, int, int]:
        """Context-owned packed FC1, FC1-scale, FC2, and FC2-scale pointers."""
        return tuple(self._native.packed_weight_pointers)

    def input_view(self, num_tokens=None):
        """Return a zero-copy BF16 [T,H] Torch view that owns the native context.

        A squash/projection kernel may write directly here before ``forward`` on
        the same stream. Passing this exact view skips the input staging copy.
        Taking slices preserves the underlying DLPack storage owner.
        Workspace aliases return views of the same storage; order all writes and
        launches across those contexts explicitly.
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
        For workspace aliases, the latest forward/signal and consumer-join rule
        apply to the entire shared runtime, not to an individual weight bank.
        """
        import torch

        stream = self._stream(stream)
        with torch.cuda.device(self.device):
            self._native.wait_until_started(stream.cuda_stream)

    def forward_shared(self, inputs, *, output=None, stream=None):
        """Enqueue one unweighted local expert and return BF16 [T,H] output on ``stream``.

        The expert uses MXFP8 weights with BF16 activations (W8A16), or MXFP8/fp8_nv weights with static-FP8
        activations (W8A8). W8A8 quantizes a 16-byte-aligned ``inputs`` directly; otherwise ``inputs`` is first
        staged into the context.
        """
        import torch

        if (
            self.config.weight_dtype not in ("mxfp8", "fp8_nv")
            or self.config.combine_route_weights
            or (
                self.config.world_size,
                self.config.num_experts,
                self.config.top_k,
            )
            != (1, 1, 1)
        ):
            raise ValueError(
                "forward_shared requires weight_dtype='mxfp8' or 'fp8_nv', world_size=1, num_experts=1, and top_k=1"
            )
        stream = self._stream(stream)
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise ValueError("inputs must be a rank-2 torch.Tensor")
        t, h = inputs.shape
        if t > self.config.max_tokens:
            raise ValueError("input token count exceeds max_tokens")
        _tensor(inputs, "inputs", (t, self.config.hidden), torch.bfloat16, self.device)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            if output is None:
                output = torch.empty((t, h), dtype=torch.bfloat16, device=self.device)
            _tensor(output, "output", (t, h), torch.bfloat16, self.device)
            inputs.record_stream(stream)
            output.record_stream(stream)
            self._native.forward_shared(inputs.data_ptr(), output.data_ptr(), t, stream.cuda_stream)
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

        if self.config.is_w8a8:
            raise ValueError(
                "W8A8 (weight_dtype='mxfp8' or 'fp8_nv', activation_dtype='fp8_static') supports forward_shared only"
            )
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

    def forward_fused_shared(
        self,
        inputs,
        topk_ids,
        topk_weights,
        *,
        shared_inputs=None,
        output=None,
        shared_output=None,
        stream=None,
        validate_routing=False,
        signal_start=False,
    ):
        """Enqueue routed and fused local shared experts and return their separate BF16 outputs.

        ``inputs`` has routed width ``config.hidden``. ``shared_inputs`` has
        ``shared_config.hidden`` and is required when that width differs; equal-width
        paths reuse ``inputs`` and its quantized buffer.
        """
        import torch

        if self.shared_config is None:
            raise ValueError("forward_fused_shared requires construction with shared_config and shared weights")
        stream = self._stream(stream)
        if not isinstance(signal_start, bool):
            raise TypeError("signal_start must be bool")
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise ValueError("inputs must be a rank-2 torch.Tensor")
        t, h = inputs.shape
        if t > self.config.max_tokens:
            raise ValueError("input token count exceeds max_tokens")
        _tensor(inputs, "inputs", (t, self.config.hidden), torch.bfloat16, self.device)
        shared_hidden = self.shared_config.hidden or self.config.hidden
        if shared_inputs is None:
            if shared_hidden != self.config.hidden:
                raise ValueError("shared_inputs is required when the fused shared hidden width differs")
            shared_inputs = inputs
        _tensor(shared_inputs, "shared_inputs", (t, shared_hidden), torch.bfloat16, self.device)
        if shared_hidden == self.config.hidden and shared_inputs.data_ptr() != inputs.data_ptr():
            raise ValueError("equal-width fused paths must share the same input allocation")
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
            if shared_output is None:
                shared_output = torch.empty((t, shared_hidden), dtype=torch.bfloat16, device=self.device)
            _tensor(output, "output", (t, h), torch.bfloat16, self.device)
            _tensor(shared_output, "shared_output", (t, shared_hidden), torch.bfloat16, self.device)
            if t and output.data_ptr() == shared_output.data_ptr():
                raise ValueError("output and shared_output must be disjoint")
            for tensor in (inputs, shared_inputs, topk_ids, topk_weights, output, shared_output):
                tensor.record_stream(stream)
            self._native.forward_fused_shared(
                inputs.data_ptr(),
                shared_inputs.data_ptr(),
                topk_ids.data_ptr(),
                topk_weights.data_ptr(),
                output.data_ptr(),
                shared_output.data_ptr(),
                t,
                stream.cuda_stream,
                signal_start,
            )
        return output, shared_output

    __call__ = forward
