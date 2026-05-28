# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

import cupy as cp
from mpi4py import MPI

_mscclpp_module = None

from mscclpp_benchmark.comm import Comm
from mscclpp_benchmark.gpu import capture_graph, init_runtime
from mscclpp_benchmark.tuner import OfflineTuner
from mscclpp_benchmark.tuning_config import HardwareProfile, TunedConfig, TunedConfigStore, normalize_sku

_ALLREDUCE = "allreduce"
_ALLGATHER = "allgather"
_DEFAULT_BATCH_SIZES = (
    1,
    2,
    3,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    3584,
    4096,
)
_DEFAULT_CANDIDATE_NBLOCKS = (1, 4, 8, 16, 24, 32, 48, 56, 64)
_DEFAULT_CANDIDATE_NTHREADS = (256, 512, 768, 1024)


def _mscclpp():
    global _mscclpp_module
    if _mscclpp_module is None:
        import mscclpp
        import mscclpp.ext

        _mscclpp_module = mscclpp
    return _mscclpp_module


@dataclass(frozen=True)
class DTypeSpec:
    name: str
    cupy_dtype: Any
    mscclpp_dtype: Any
    accum_dtype: Any | None = None
    supports_reduction_correctness: bool = True
    fp8_format: str | None = None


@dataclass(frozen=True)
class CandidateSpec:
    algorithm: str
    min_message_size: int | None = None
    max_message_size: int | None = None
    max_nblocks: int | None = None
    min_nthreads: int | None = None
    requires_nvls: bool = False
    requires_symmetric_memory: bool = False


@dataclass
class BenchmarkCase:
    collective: str
    message_size: int
    total_size: int
    input: cp.ndarray
    output: cp.ndarray
    dtype_spec: DTypeSpec
    allgather_mode: str


def config_accum_dtype(case: BenchmarkCase) -> Any:
    return case.dtype_spec.accum_dtype or case.dtype_spec.mscclpp_dtype


def _device_name() -> str:
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props.get("name", "UNKNOWN")
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def _detect_hardware_profile(scale: int) -> HardwareProfile:
    return HardwareProfile(sku=normalize_sku(_device_name()), scale=scale)


def _parse_dtype(dtype_name: str) -> DTypeSpec:
    mscclpp = _mscclpp()
    normalized = dtype_name.strip().lower().replace("-", "_")
    if normalized in {"float16", "fp16", "half"}:
        return DTypeSpec("float16", cp.float16, mscclpp.DataType.float16)
    if normalized in {"float32", "fp32", "float"}:
        return DTypeSpec("float32", cp.float32, mscclpp.DataType.float32)
    if normalized in {"int32", "i32"}:
        return DTypeSpec("int32", cp.int32, mscclpp.DataType.int32)
    if normalized in {"uint8", "u8"}:
        return DTypeSpec("uint8", cp.uint8, mscclpp.DataType.uint8)
    if normalized in {"float8_e4m3fn", "fp8_e4m3fn"}:
        return DTypeSpec(
            "float8_e4m3fn",
            cp.uint8,
            mscclpp.DataType.float8_e4m3fn,
            accum_dtype=mscclpp.DataType.float16,
            fp8_format="e4m3fn",
        )
    if normalized in {"float8_e4m3fnuz", "fp8_e4m3fnuz"}:
        return DTypeSpec(
            "float8_e4m3fnuz",
            cp.uint8,
            mscclpp.DataType.float8_e4m3fnuz,
            accum_dtype=mscclpp.DataType.float16,
            fp8_format="e4m3fnuz",
        )
    if normalized in {"float8_e4m3b15", "fp8_e4m3b15"}:
        return DTypeSpec(
            "float8_e4m3b15",
            cp.uint8,
            mscclpp.DataType.float8_e4m3b15,
            accum_dtype=mscclpp.DataType.float32,
            fp8_format="e4m3b15",
        )
    raise ValueError(
        f"Unsupported dtype {dtype_name!r}; use float16, float32, int32, uint8, "
        "float8_e4m3fn, float8_e4m3fnuz, or float8_e4m3b15"
    )


def _dtype_is_float(dtype: Any) -> bool:
    return dtype in (cp.float16, cp.float32)


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _parse_int_list(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    values = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    if not values or values[0] <= 0:
        raise ValueError(f"Expected a comma-separated list of positive integers, got {raw!r}")
    return values


def _candidate_specs(collective: str, message_size: int) -> tuple[CandidateSpec, ...]:
    if collective == _ALLGATHER:
        return (CandidateSpec("default_allgather_fullmesh2", max_nblocks=32),)
    if collective != _ALLREDUCE:
        raise ValueError(f"Unsupported collective: {collective}")
    if message_size <= 512 * 1024:
        return (
            CandidateSpec("default_allreduce_nvls_packet", max_nblocks=16, requires_nvls=True),
            CandidateSpec("default_allreduce_packet", max_nblocks=56),
            CandidateSpec("default_allreduce_allpair_packet", max_nblocks=56),
        )
    if message_size <= 4 * 1024 * 1024:
        return (
            CandidateSpec("default_allreduce_packet", max_nblocks=56),
            CandidateSpec("default_allreduce_allpair_packet", max_nblocks=56),
            CandidateSpec("default_allreduce_rsag_zero_copy"),
            CandidateSpec("default_allreduce_fullmesh", max_nblocks=64),
        )
    return (
        CandidateSpec("default_allreduce_rsag_zero_copy"),
        CandidateSpec("default_allreduce_fullmesh", max_nblocks=64),
    )


def _candidate_algorithms(comm: Comm, case: BenchmarkCase) -> list[tuple[Any, CandidateSpec]]:
    available = comm.algorithms.get(case.collective, {})
    candidates: list[tuple[Any, CandidateSpec]] = []
    seen: set[str] = set()
    for candidate in _candidate_specs(case.collective, case.message_size):
        if candidate.requires_nvls and not _mscclpp().is_nvls_supported():
            continue
        if candidate.min_message_size is not None and case.message_size < candidate.min_message_size:
            continue
        if candidate.max_message_size is not None and case.message_size > candidate.max_message_size:
            continue
        algorithm = available.get(candidate.algorithm)
        if algorithm is None or algorithm.name in seen:
            continue
        seen.add(algorithm.name)
        candidates.append((algorithm, candidate))
    if candidates:
        return candidates
    return [(algorithm, CandidateSpec(algorithm.name)) for algorithm in available.values()]


def _make_case(
    *,
    collective: str,
    nelems: int,
    dtype_spec: DTypeSpec,
    comm_group: Any,
    allgather_mode: str,
) -> BenchmarkCase:
    if collective == _ALLREDUCE:
        memory = _mscclpp().GpuBuffer(nelems, dtype=dtype_spec.cupy_dtype)
        return BenchmarkCase(
            collective=collective,
            message_size=memory.nbytes,
            total_size=memory.nbytes,
            input=memory,
            output=memory,
            dtype_spec=dtype_spec,
            allgather_mode=allgather_mode,
        )

    if collective != _ALLGATHER:
        raise ValueError(f"Unsupported collective: {collective}")

    if allgather_mode == "in-place":
        output = _mscclpp().GpuBuffer(nelems * comm_group.nranks, dtype=dtype_spec.cupy_dtype)
        start = comm_group.my_rank * nelems
        input_buffer = output[start : start + nelems]
    elif allgather_mode == "out-of-place":
        input_buffer = _mscclpp().GpuBuffer(nelems, dtype=dtype_spec.cupy_dtype)
        output = _mscclpp().GpuBuffer(nelems * comm_group.nranks, dtype=dtype_spec.cupy_dtype)
    else:
        raise ValueError(f"Unsupported allgather mode: {allgather_mode}")

    return BenchmarkCase(
        collective=collective,
        message_size=input_buffer.nbytes,
        total_size=output.nbytes,
        input=input_buffer,
        output=output,
        dtype_spec=dtype_spec,
        allgather_mode=allgather_mode,
    )


def _fill_case_for_benchmark(case: BenchmarkCase, rank: int) -> None:
    if case.collective == _ALLREDUCE:
        case.input.fill(0)
        return
    _fill_allgather_input(case, rank)
    case.output.fill(0)
    if case.allgather_mode == "in-place":
        _fill_allgather_input(case, rank)


def _fill_case_for_correctness(case: BenchmarkCase, rank: int, iteration: int) -> None:
    value = iteration * MPI.COMM_WORLD.size + rank
    if case.collective == _ALLREDUCE:
        case.input.fill(_dtype_value(case, value))
        return
    case.output.fill(0)
    case.input.fill(_dtype_value(case, value))


def _fill_allgather_input(case: BenchmarkCase, rank: int) -> None:
    value = rank + 1
    case.input.fill(_dtype_value(case, value))


def _dtype_value(case: BenchmarkCase, value: int) -> int:
    if case.dtype_spec.fp8_format is not None:
        return _encode_fp8_scalar(case.dtype_spec.fp8_format, _correctness_numeric_value(case, value))
    if case.dtype_spec.cupy_dtype == cp.uint8:
        return value % 256
    return value


def _correctness_numeric_value(case: BenchmarkCase, value: int) -> float:
    if case.dtype_spec.fp8_format is None:
        return float(value)
    scale = max(64, MPI.COMM_WORLD.size * MPI.COMM_WORLD.size)
    return float(value + 1) / float(scale)


def _check_correctness(
    comm: Comm,
    case: BenchmarkCase,
    config: TunedConfig,
    *,
    raise_on_unsupported: bool = True,
    niter: int = 1,
) -> bool:
    if case.collective == _ALLREDUCE and not case.dtype_spec.supports_reduction_correctness:
        if not raise_on_unsupported:
            return True
        raise ValueError(
            f"Correctness checking for {case.collective} with {case.dtype_spec.name} is not implemented; "
            "use --skip-correctness or a numeric dtype"
        )

    all_ok = True
    for iteration in range(niter):
        _fill_case_for_correctness(case, comm.rank, iteration)
        comm.comm_group.barrier()
        ret = comm.run(case, config)
        cp.cuda.runtime.deviceSynchronize()
        comm.comm_group.barrier()
        if ret != 0:
            return False

        expected = _expected_output(case, comm.nranks, iteration)
        local_ok = _compare_output(case.output, expected)
        all_ok = all_ok and local_ok

        if not local_ok:
            mismatch = _mismatch_mask(case.output, expected)
            print(
                "not close: "
                f"iter={iteration}, rank={comm.rank}, output={case.output[mismatch][0]}, "
                f"expected={expected[mismatch][0]}",
                flush=True,
            )

    return bool(MPI.COMM_WORLD.allreduce(all_ok, op=MPI.LAND))


def _expected_output(case: BenchmarkCase, nranks: int, iteration: int):
    if case.collective == _ALLREDUCE:
        if case.dtype_spec.fp8_format is not None:
            expected_numeric = sum(
                _correctness_numeric_value(case, iteration * MPI.COMM_WORLD.size + rank) for rank in range(nranks)
            )
            return cp.full_like(case.output, _encode_fp8_scalar(case.dtype_spec.fp8_format, expected_numeric))
        expected_value = sum(iteration * MPI.COMM_WORLD.size + rank for rank in range(nranks))
        return cp.full_like(case.output, _dtype_value(case, expected_value))

    expected = cp.empty_like(case.output)
    chunk = case.input.size
    for rank in range(nranks):
        expected[rank * chunk : (rank + 1) * chunk].fill(_dtype_value(case, iteration * MPI.COMM_WORLD.size + rank))
    return expected


def _compare_output(output, expected) -> bool:
    if _dtype_is_float(output.dtype.type):
        return bool(cp.allclose(output, expected, rtol=1.0e-2, atol=2).item())
    return bool(cp.all(output == expected).item())


def _mismatch_mask(output, expected):
    if _dtype_is_float(output.dtype.type):
        return ~cp.isclose(output, expected, rtol=1.0e-2, atol=2)
    return output != expected


_FP8_POSITIVE_TABLES: dict[str, list[tuple[int, float]]] = {}


def _encode_fp8_scalar(fp8_format: str, value: float) -> int:
    if value < 0:
        raise ValueError("FP8 correctness values are expected to be non-negative")
    table = _FP8_POSITIVE_TABLES.setdefault(fp8_format, _build_fp8_positive_table(fp8_format))
    return min(table, key=lambda item: abs(item[1] - value))[0]


def _build_fp8_positive_table(fp8_format: str) -> list[tuple[int, float]]:
    table = []
    for byte in range(128):
        value = _decode_fp8_positive(fp8_format, byte)
        if not math.isnan(value):
            table.append((byte, value))
    return table


def _decode_fp8_positive(fp8_format: str, byte: int) -> float:
    exp = (byte >> 3) & 0xF
    mant = byte & 0x7
    if fp8_format == "e4m3fn" and exp == 0xF and mant == 0x7:
        return float("nan")
    if exp == 0 and mant == 0:
        return 0.0
    if fp8_format == "e4m3fn":
        return math.ldexp(mant / 8.0, -6) if exp == 0 else math.ldexp(1.0 + mant / 8.0, exp - 7)
    if fp8_format == "e4m3fnuz":
        return math.ldexp(mant / 8.0, -7) if exp == 0 else math.ldexp(1.0 + mant / 8.0, exp - 8)
    if fp8_format == "e4m3b15":
        return math.ldexp(mant / 8.0, -14) if exp == 0 else math.ldexp(1.0 + mant / 8.0, exp - 15)
    raise ValueError(f"Unknown FP8 format: {fp8_format}")


def _try_measure_case(
    comm: Comm,
    case: BenchmarkCase,
    config: TunedConfig,
    *,
    n_warmup: int,
    n_graph_launches: int,
    n_ops_per_graph: int,
) -> float | None:
    try:
        return _measure_case(
            comm,
            case,
            config,
            n_warmup=n_warmup,
            n_graph_launches=n_graph_launches,
            n_ops_per_graph=n_ops_per_graph,
        )
    except Exception as exc:
        if comm.rank == 0:
            print(
                f"[skip] {config.algorithm} nb={config.nblocks} nt={config.nthreads} "
                f"size={case.message_size}: {type(exc).__name__}: {exc}",
                flush=True,
            )
        return None


def _measure_case(
    comm: Comm,
    case: BenchmarkCase,
    config: TunedConfig,
    *,
    n_warmup: int,
    n_graph_launches: int,
    n_ops_per_graph: int,
) -> float:
    _fill_case_for_benchmark(case, comm.rank)
    comm.comm_group.barrier()
    if comm.run(case, config) != 0:
        raise RuntimeError("algorithm returned non-zero status")
    cp.cuda.runtime.deviceSynchronize()
    comm.comm_group.barrier()

    stream = cp.cuda.Stream(non_blocking=True)
    graph = None

    def capture_ops() -> None:
        for _ in range(n_ops_per_graph):
            ret = comm.run(case, config, stream)
            if ret != 0:
                raise RuntimeError("algorithm returned non-zero status during graph capture")

    try:
        with stream:
            graph = capture_graph(stream, capture_ops)

        for _ in range(n_warmup):
            graph.launch(stream)
        stream.synchronize()
        comm.comm_group.barrier()

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        for _ in range(n_graph_launches):
            graph.launch(stream)
        end.record(stream)
        end.synchronize()

        elapsed_us = cp.cuda.get_elapsed_time(start, end) * 1000.0 / (n_graph_launches * n_ops_per_graph)
        return float(MPI.COMM_WORLD.allreduce(elapsed_us, op=MPI.MAX))
    finally:
        if graph is not None:
            graph.close()


def _bandwidth_gbps(num_bytes: int, time_us: float) -> float:
    return num_bytes / time_us / 1e3


def _busbw_factor(collective: str, nranks: int) -> float:
    if nranks <= 1:
        return 1.0
    if collective == _ALLREDUCE:
        return 2 * (nranks - 1) / nranks
    if collective == _ALLGATHER:
        return (nranks - 1) / nranks
    raise ValueError(f"Unsupported collective: {collective}")


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    sep_line = "-+-".join("-" * width for width in widths)
    row_lines = [" | ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark MSCCL++ collectives without PyTorch dependencies")
    parser.add_argument("--collective", choices=(_ALLREDUCE, _ALLGATHER), default=_ALLREDUCE)
    parser.add_argument("--d-model", type=int, default=5120)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-sizes", help="Comma-separated batch sizes; default uses the benchmark sweep")
    parser.add_argument("--allgather-mode", choices=("in-place", "out-of-place"), default="in-place")
    parser.add_argument("--config-path", help="Optional MSCCL++ tuned config JSON")
    parser.add_argument("--write-config", help="Write autotuned configs to this JSON path")
    parser.add_argument("--autotune", action="store_true", help="Tune each benchmark size before timing it")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--correctness-iters", type=int, default=1)
    parser.add_argument("--scratch-buffer-size", type=int, default=1 << 27)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup graph replays before benchmark timing")
    parser.add_argument("--graph-launches", type=int, default=10, help="Timed graph replays")
    parser.add_argument("--iterations", type=int, default=100, help="Collective operations captured per CUDA graph")
    parser.add_argument("--tune-warmup", type=int, default=2)
    parser.add_argument("--tune-graph-launches", type=int, default=3)
    parser.add_argument("--tune-iterations", type=int, default=20)
    parser.add_argument("--candidate-nblocks", help="Comma-separated nblocks tuning candidates")
    parser.add_argument("--candidate-nthreads", help="Comma-separated nthreads tuning candidates")
    parser.add_argument("--symmetric-memory", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "d_model",
        "scratch_buffer_size",
        "graph_launches",
        "iterations",
        "tune_graph_launches",
        "tune_iterations",
        "correctness_iters",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or args.tune_warmup < 0:
        raise ValueError("warmup counts must be non-negative")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    init_runtime()

    local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    try:
        visible_devices = cp.cuda.runtime.getDeviceCount()
        if visible_devices <= 0:
            raise RuntimeError("MSCCL++ benchmark requires at least one visible GPU")
        cp.cuda.Device(local_comm.Get_rank() % visible_devices).use()
    finally:
        local_comm.Free()

    dtype_spec = _parse_dtype(args.dtype)
    batch_sizes = _parse_int_list(args.batch_sizes, _DEFAULT_BATCH_SIZES)
    candidate_nblocks = _parse_int_list(args.candidate_nblocks, _DEFAULT_CANDIDATE_NBLOCKS)
    candidate_nthreads = _parse_int_list(args.candidate_nthreads, _DEFAULT_CANDIDATE_NTHREADS)

    comm_group = _mscclpp().CommGroup(MPI.COMM_WORLD)
    setattr(comm_group, "_mpi_comm", MPI.COMM_WORLD)
    hardware_profile = _detect_hardware_profile(comm_group.nranks)
    config_store = TunedConfigStore.load_path(args.config_path) if args.config_path else TunedConfigStore.empty()
    comm = Comm(
        comm_group,
        config_store=config_store,
        hardware_profile=hardware_profile,
        scratch_buffer_size=args.scratch_buffer_size,
    )
    tuner = OfflineTuner(
        comm,
        candidate_nblocks=candidate_nblocks,
        candidate_nthreads=candidate_nthreads,
        n_warmup=args.tune_warmup,
        n_graph_launches=args.tune_graph_launches,
        n_ops_per_graph=args.tune_iterations,
        candidate_algorithms=_candidate_algorithms,
        check_correctness=_check_correctness,
        measure=_try_measure_case,
        symmetric_memory=args.symmetric_memory,
    )

    rows: list[list[str]] = []
    try:
        if comm.rank == 0:
            print(
                f"MSCCL++ {args.collective} benchmark: profile={hardware_profile} dtype={dtype_spec.name} "
                f"graph_launches={args.graph_launches} iterations={args.iterations}",
                flush=True,
            )

        for batch_size in batch_sizes:
            nelems = batch_size * args.d_model
            case = _make_case(
                collective=args.collective,
                nelems=nelems,
                dtype_spec=dtype_spec,
                comm_group=comm_group,
                allgather_mode=args.allgather_mode,
            )
            config = tuner.tune(case) if args.autotune else comm.resolve_config(case)
            if args.autotune:
                config_store.upsert(hardware_profile, args.collective, case.message_size, config)

            correctness = "SKIP"
            if not args.skip_correctness:
                correctness = "PASS" if _check_correctness(comm, case, config, niter=args.correctness_iters) else "FAIL"
                comm.reset(config)
                if correctness != "PASS":
                    raise RuntimeError(
                        f"Correctness failed for batch_size={batch_size}, message_size={case.message_size}, "
                        f"config={config}"
                    )

            time_us = _measure_case(
                comm,
                case,
                config,
                n_warmup=args.warmup,
                n_graph_launches=args.graph_launches,
                n_ops_per_graph=args.iterations,
            )
            comm.reset(config)

            algbw = _bandwidth_gbps(case.total_size, time_us)
            busbw = algbw * _busbw_factor(args.collective, comm_group.nranks)
            rows.append(
                [
                    str(batch_size),
                    _human_size(case.message_size),
                    _human_size(case.total_size),
                    config.algorithm,
                    str(config.nblocks or "auto"),
                    str(config.nthreads or "auto"),
                    f"{time_us:.2f}",
                    f"{algbw:.2f}",
                    f"{busbw:.2f}",
                    correctness,
                ]
            )
            if comm.rank == 0:
                print(".", end="", flush=True)

        if args.write_config and comm.rank == 0:
            config_store.write_path(args.write_config)
            print(f"\nWrote tuned config to {args.write_config}", flush=True)

        if comm.rank == 0:
            print(
                "\n"
                + _format_table(
                    [
                        "batch",
                        "msg",
                        "total",
                        "algorithm",
                        "nblocks",
                        "nthreads",
                        "time_us",
                        "algBW_GB/s",
                        "busBW_GB/s",
                        "check",
                    ],
                    rows,
                ),
                flush=True,
            )
    finally:
        comm_group.barrier()
        cp.cuda.runtime.deviceSynchronize()
        comm.close()


if __name__ == "__main__":
    main()
