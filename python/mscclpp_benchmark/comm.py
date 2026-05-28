# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
_ALLREDUCE_COLLECTIVE = "allreduce"
_ALLGATHER_COLLECTIVE = "allgather"
_mscclpp_module = None

from mscclpp_benchmark.gpu import current_device, device_name, set_device
from mscclpp_benchmark.tuning_config import HardwareProfile, TunedConfig, TunedConfigStore, normalize_sku


def _mscclpp():
    global _mscclpp_module
    if _mscclpp_module is None:
        import mscclpp
        import mscclpp.ext

        _mscclpp_module = mscclpp
    return _mscclpp_module


class Buffer:
    def __init__(
        self,
        nbytes: int | None = None,
        *,
        dtype: str | Any = "float16",
        shape: tuple[int, ...] | None = None,
        buffer: Any | None = None,
    ) -> None:
        self.dtype = dtype
        self.element_size = _dtype_size(dtype)
        if buffer is None:
            if nbytes is None:
                if shape is None:
                    raise ValueError("Either nbytes or shape is required")
                nbytes = _numel(shape) * self.element_size
            _ensure_device()
            buffer = _mscclpp().RawGpuBuffer(int(nbytes))
        self.buffer = buffer
        self.nbytes = int(buffer.bytes())
        self.shape = shape if shape is not None else (self.nbytes // self.element_size,)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return _numel(self.shape)

    def data_ptr(self) -> int:
        return int(self.buffer.data())


class _AllReduceOp:
    def __init__(self, comm: "Comm", x: Any, *, symmetric_memory: bool = False) -> None:
        self._comm = comm
        self._x = x
        self._symmetric_memory = symmetric_memory

    def __call__(self, **_: Any) -> Any:
        self._comm.run(self._x, symmetric_memory=self._symmetric_memory)
        return self._x


class _AllGatherOp:
    def __init__(self, comm: "Comm", x: Any, *, dim: int, y: Any | None = None, symmetric_memory: bool = False) -> None:
        shape = _shape(x)
        if len(shape) == 0:
            raise ValueError("MSCCL++ allgather requires a non-scalar buffer")
        if dim % len(shape) != 0:
            raise NotImplementedError("Raw-buffer allgather currently supports only dim=0")
        if y is None:
            y_shape = (comm._scale() * shape[0], *shape[1:])
            y = Buffer(dtype=_dtype(x), shape=y_shape)
        self._comm = comm
        self._x = x
        self.y = y
        self._symmetric_memory = symmetric_memory

    def __call__(self, **_: Any) -> Any:
        self._comm.run(
            self._x,
            collective=_ALLGATHER_COLLECTIVE,
            output_tensor=self.y,
            symmetric_memory=self._symmetric_memory,
        )
        return self.y


class Comm:
    """Runtime MSCCL++ wrapper that owns algorithm handles and execution without Torch/CuPy tensors."""

    def __init__(
        self,
        comm_group: Any,
        scratch_buffer_size: int = 1 << 27,
        *,
        config_store: "TunedConfigStore | None" = None,
        hardware_profile: HardwareProfile | None = None,
    ) -> None:
        self._comm_group = comm_group
        self._mpi_comm = getattr(comm_group, "_mpi_comm", None)
        self._rank = comm_group.my_rank
        self._closed = False
        _ensure_device()
        self._mscclpp = _mscclpp()
        self._scratch_buffer = self._mscclpp.RawGpuBuffer(scratch_buffer_size)
        self._config_store = TunedConfigStore.empty() if config_store is None else config_store
        self._hardware_profile = (
            _detect_hardware_profile(scale=self._scale()) if hardware_profile is None else hardware_profile
        )
        self._default_config_warning_keys: set[tuple[str, str, str, int]] = set()

        algorithms = self._mscclpp.ext.AlgorithmCollectionBuilder().build_default_algorithms(
            scratch_buffer=self._scratch_buffer.data(),
            scratch_buffer_size=self._scratch_buffer.bytes(),
            rank=self._rank,
        )
        self._algorithms_by_collective: dict[str, dict[str, Any]] = {}
        for algorithm in algorithms:
            self._algorithms_by_collective.setdefault(algorithm.collective, {})[algorithm.name] = algorithm

    @property
    def comm_group(self) -> Any:
        return self._comm_group

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def nranks(self) -> int:
        return self._comm_group.nranks

    @property
    def algorithms(self) -> dict[str, dict[str, Any]]:
        return self._algorithms_by_collective

    @property
    def hardware_profile(self) -> HardwareProfile:
        return self._hardware_profile

    def make_allreduce(self, x: Any, *, symmetric_memory: bool = False) -> _AllReduceOp:
        return _AllReduceOp(self, x, symmetric_memory=symmetric_memory)

    def make_allgather(self, x: Any, dim: int, y: Any | None = None, *, symmetric_memory: bool = False) -> _AllGatherOp:
        return _AllGatherOp(self, x, dim=dim, y=y, symmetric_memory=symmetric_memory)

    def _scale(self) -> int:
        if self._mpi_comm is not None:
            return int(self._mpi_comm.Get_size())
        return 1

    def resolve_config(self, case: Any, *, symmetric_memory: bool = False) -> TunedConfig:
        dtype_override = getattr(getattr(case, "dtype_spec", None), "mscclpp_dtype", None)
        accum_dtype = getattr(getattr(case, "dtype_spec", None), "accum_dtype", None) or dtype_override
        symmetric_memory = symmetric_memory or bool(getattr(case, "symmetric_memory", False))
        return self._resolve_config(
            case.collective,
            case.input,
            dtype_override=dtype_override,
            accum_dtype=accum_dtype,
            symmetric_memory=symmetric_memory,
        )

    def _resolve_config(
        self,
        collective: str,
        buffer: Any,
        *,
        dtype_override: Any | None = None,
        accum_dtype: Any | None = None,
        symmetric_memory: bool = False,
    ) -> TunedConfig:
        tuned_config = self._config_store.select(self._hardware_profile, collective, _nbytes(buffer))
        if tuned_config is not None and tuned_config.algorithm in self._algorithms_by_collective.get(collective, {}):
            return tuned_config

        if self._rank == 0:
            dim = int(_shape(buffer)[1]) if len(_shape(buffer)) > 1 else 1
            warning_key = (
                collective,
                str(dtype_override if dtype_override is not None else _dtype(buffer)),
                str(
                    accum_dtype
                    if accum_dtype is not None
                    else dtype_override if dtype_override is not None else _dtype(buffer)
                ),
                dim,
            )
            if warning_key not in self._default_config_warning_keys:
                self._default_config_warning_keys.add(warning_key)
                logger.warning(
                    "MSCCL++ default config: no tuning for collective=%s profile=%s dtype=%s accum=%s dim=%s; perf may be poor",
                    collective,
                    self._hardware_profile,
                    warning_key[1],
                    warning_key[2],
                    dim,
                )
        return _default_tuned_config(
            collective,
            _nbytes(buffer),
            self._algorithms_by_collective,
            symmetric_memory=symmetric_memory,
        )

    def run(
        self,
        buffer: Any,
        config: TunedConfig | None = None,
        stream: Any | None = None,
        *,
        collective: str = _ALLREDUCE_COLLECTIVE,
        output_tensor: Any | None = None,
        dtype_override: Any | None = None,
        accum_dtype: Any | None = None,
        symmetric_memory: bool = False,
    ) -> int:
        if self._closed:
            raise RuntimeError("Cannot use a closed MSCCL++ comm")

        raise_on_error = True
        if hasattr(buffer, "input") and hasattr(buffer, "output") and hasattr(buffer, "dtype_spec"):
            case = buffer
            buffer = case.input
            output_tensor = case.output
            collective = case.collective
            dtype_override = case.dtype_spec.mscclpp_dtype
            accum_dtype = case.dtype_spec.accum_dtype or dtype_override
            symmetric_memory = symmetric_memory or bool(getattr(case, "symmetric_memory", False))
            raise_on_error = False

        if collective not in self._algorithms_by_collective:
            raise RuntimeError(f"No supported MSCCL++ {collective} algorithm is available")

        if config is None:
            config = self._resolve_config(
                collective,
                buffer,
                dtype_override=dtype_override,
                accum_dtype=accum_dtype,
                symmetric_memory=symmetric_memory,
            )
        symmetric_memory = symmetric_memory or config.symmetric_memory
        algorithm = self._algorithms_by_collective[collective][config.algorithm]
        output = buffer if output_tensor is None else output_tensor
        dtype = dtype_override if dtype_override is not None else _dtype_to_mscclpp(_dtype(buffer))
        accum = accum_dtype if accum_dtype is not None else dtype
        ret = algorithm.execute(
            comm=self._comm_group.communicator,
            input_buffer=_data_ptr(buffer),
            output_buffer=_data_ptr(output),
            input_size=_nbytes(buffer),
            output_size=_nbytes(output),
            dtype=dtype,
            op=self._mscclpp.ReduceOp.SUM if collective == _ALLREDUCE_COLLECTIVE else self._mscclpp.ReduceOp.NOP,
            stream=_stream_ptr(stream),
            nblocks=config.nblocks or 0,
            nthreads_per_block=config.nthreads or 0,
            symmetric_memory=symmetric_memory,
            accum_dtype=accum,
        )
        if ret != 0 and raise_on_error:
            raise RuntimeError(f"MSCCL++ {collective} failed on rank {self._rank} with error code {ret}")
        return ret

    def reset(self, config: TunedConfig | None = None) -> None:
        if config is not None:
            for algorithms_by_name in self._algorithms_by_collective.values():
                algorithm = algorithms_by_name.get(config.algorithm)
                if algorithm is not None:
                    algorithm.reset()
                    return
        for algorithms_by_name in self._algorithms_by_collective.values():
            for algorithm in algorithms_by_name.values():
                algorithm.reset()

    def close(self) -> None:
        self.reset()
        self._algorithms_by_collective = {}
        self._scratch_buffer = None
        self._closed = True
        self._mscclpp.ext.AlgorithmCollectionBuilder.reset()


def _numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _dtype_size(dtype: Any) -> int:
    dtype_name = _dtype_name(dtype)
    if dtype_name in {"float16", "bfloat16"}:
        return 2
    if dtype_name in {"float32", "int32", "uint32"}:
        return 4
    if dtype_name in {"uint8", "float8_e4m3b15", "float8_e4m3fn", "float8_e4m3fnuz"}:
        return 1
    raise ValueError(f"Unknown data type size for {dtype}")


def _dtype_name(dtype: Any) -> str:
    if isinstance(dtype, str):
        return dtype.strip().lower().replace("-", "_")
    name = str(dtype).rsplit(".", 1)[-1]
    return name.strip().lower().replace("-", "_")


def _dtype_to_mscclpp(dtype: Any) -> Any:
    dtype_name = _dtype_name(dtype)
    mapping = {
        "float16": _mscclpp().DataType.float16,
        "float32": _mscclpp().DataType.float32,
        "int32": _mscclpp().DataType.int32,
        "uint8": _mscclpp().DataType.uint8,
        "float8_e4m3b15": _mscclpp().DataType.float8_e4m3b15,
        "float8_e4m3fn": _mscclpp().DataType.float8_e4m3fn,
        "float8_e4m3fnuz": _mscclpp().DataType.float8_e4m3fnuz,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unknown data type: {dtype}") from exc


def _data_ptr(buffer: Any) -> int:
    if hasattr(buffer, "data_ptr"):
        data_ptr = buffer.data_ptr
        return int(data_ptr() if callable(data_ptr) else data_ptr)
    if hasattr(buffer, "data"):
        data = buffer.data
        if callable(data):
            return int(data())
        if hasattr(data, "ptr"):
            return int(data.ptr)
    raise TypeError(f"Cannot get device pointer from {type(buffer)!r}")


def _stream_ptr(stream: Any | None) -> int:
    if stream is None:
        return 0
    return int(getattr(stream, "ptr", stream))


def _nbytes(buffer: Any) -> int:
    if hasattr(buffer, "nbytes"):
        return int(buffer.nbytes)
    if hasattr(buffer, "bytes"):
        value = buffer.bytes
        return int(value() if callable(value) else value)
    raise TypeError(f"Cannot get byte size from {type(buffer)!r}")


def _shape(buffer: Any) -> tuple[int, ...]:
    shape = getattr(buffer, "shape", None)
    if shape is None:
        return (_nbytes(buffer) // _dtype_size(_dtype(buffer)),)
    return tuple(int(dim) for dim in shape)


def _dtype(buffer: Any) -> Any:
    dtype = getattr(buffer, "dtype", None)
    if dtype is None:
        return "uint8"
    return dtype


def _detect_hardware_profile(*, scale: int) -> HardwareProfile:
    try:
        sku = device_name()
    except Exception:
        sku = "UNKNOWN"
    return HardwareProfile(sku=normalize_sku(sku), scale=scale)


def _ensure_device() -> None:
    set_device(current_device())


def _default_tuned_config(
    collective: str,
    message_size: int,
    algorithms_by_collective: dict[str, dict[str, Any]],
    *,
    symmetric_memory: bool = False,
) -> TunedConfig:
    if collective == _ALLGATHER_COLLECTIVE:
        return TunedConfig("default_allgather_fullmesh2", symmetric_memory=symmetric_memory)
    available = algorithms_by_collective.get(collective, {})
    if symmetric_memory and _mscclpp().is_nvls_supported() and "default_allreduce_nvls_zero_copy" in available:
        return TunedConfig("default_allreduce_nvls_zero_copy", symmetric_memory=True)
    if message_size <= 512 * 1024 and "default_allreduce_packet" in available:
        return TunedConfig("default_allreduce_packet", symmetric_memory=symmetric_memory)
    if "default_allreduce_rsag_zero_copy" in available:
        return TunedConfig("default_allreduce_rsag_zero_copy", symmetric_memory=symmetric_memory)
    if available:
        return TunedConfig(next(iter(available)), symmetric_memory=symmetric_memory)
    raise RuntimeError(f"No MSCCL++ algorithm is available for {collective}")
