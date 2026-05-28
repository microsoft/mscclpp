# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import cupy as cp
from mpi4py import MPI

_mscclpp_module = None

from mscclpp_benchmark.comm import Comm
from mscclpp_benchmark.correctness import (
    CorrectnessStats,
    check_correctness as _check_correctness,
    fill_case_for_benchmark as _fill_case_for_benchmark,
)
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
    supported_skus: tuple[str, ...] | None = None
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
    symmetric_memory: bool = False


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


def _with_accum_type(dtype_spec: DTypeSpec, accum_type: str | None) -> DTypeSpec:
    if accum_type is None:
        return dtype_spec

    mscclpp = _mscclpp()
    normalized = accum_type.strip().lower().replace("-", "_")
    if normalized in {"native", "same", "auto"}:
        accum_dtype = dtype_spec.mscclpp_dtype
    elif normalized in {"float16", "fp16", "half"}:
        accum_dtype = mscclpp.DataType.float16
    elif normalized in {"float32", "fp32", "float"}:
        accum_dtype = mscclpp.DataType.float32
    else:
        raise ValueError(f"Unsupported accum type {accum_type!r}; use native, float16, or float32")

    return DTypeSpec(
        name=dtype_spec.name,
        cupy_dtype=dtype_spec.cupy_dtype,
        mscclpp_dtype=dtype_spec.mscclpp_dtype,
        accum_dtype=accum_dtype,
        supports_reduction_correctness=dtype_spec.supports_reduction_correctness,
        fp8_format=dtype_spec.fp8_format,
    )


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


def _candidate_specs(
    collective: str, message_size: int, *, symmetric_memory: bool = False
) -> tuple[CandidateSpec, ...]:
    if collective == _ALLGATHER:
        return (CandidateSpec("default_allgather_fullmesh2", max_nblocks=64, supported_skus=("MI300X",)),)
    if collective != _ALLREDUCE:
        raise ValueError(f"Unsupported collective: {collective}")
    if message_size <= 512 * 1024:
        candidates = (
            CandidateSpec(
                "default_allreduce_nvls_packet",
                max_nblocks=16,
                supported_skus=("H100", "GB300"),
                requires_nvls=True,
            ),
            CandidateSpec("default_allreduce_packet", max_nblocks=56),
            CandidateSpec("default_allreduce_allpair_packet", max_nblocks=56),
        )
    elif message_size <= 4 * 1024 * 1024:
        candidates = (
            CandidateSpec("default_allreduce_packet", max_nblocks=56),
            CandidateSpec("default_allreduce_allpair_packet", max_nblocks=56),
            CandidateSpec("default_allreduce_rsag_zero_copy"),
            CandidateSpec("default_allreduce_fullmesh", max_nblocks=64, supported_skus=("MI300X",)),
        )
    else:
        candidates = (
            CandidateSpec("default_allreduce_rsag_zero_copy"),
            CandidateSpec("default_allreduce_fullmesh", max_nblocks=64, supported_skus=("MI300X",)),
        )
    if symmetric_memory:
        return (
            CandidateSpec(
                "default_allreduce_nvls_zero_copy",
                max_nblocks=32,
                supported_skus=("H100", "GB300"),
                requires_nvls=True,
                requires_symmetric_memory=True,
            ),
            *candidates,
        )
    return candidates


def _candidate_algorithms(comm: Comm, case: BenchmarkCase) -> list[tuple[Any, CandidateSpec]]:
    available = comm.algorithms.get(case.collective, {})
    candidates: list[tuple[Any, CandidateSpec]] = []
    seen: set[str] = set()
    symmetric_memory = case.symmetric_memory
    profile = getattr(comm, "hardware_profile", None)
    filtered_out = False
    for candidate in _candidate_specs(case.collective, case.message_size, symmetric_memory=symmetric_memory):
        if not _candidate_supports_profile(candidate, profile):
            filtered_out = True
            continue
        if candidate.requires_nvls and not _mscclpp().is_nvls_supported():
            filtered_out = True
            continue
        if candidate.requires_symmetric_memory and not symmetric_memory:
            filtered_out = True
            continue
        if candidate.min_message_size is not None and case.message_size < candidate.min_message_size:
            filtered_out = True
            continue
        if candidate.max_message_size is not None and case.message_size > candidate.max_message_size:
            filtered_out = True
            continue
        algorithm = available.get(candidate.algorithm)
        if algorithm is None or algorithm.name in seen:
            continue
        seen.add(algorithm.name)
        candidates.append((algorithm, candidate))
    if candidates:
        return candidates
    if filtered_out:
        return []
    return [(algorithm, CandidateSpec(algorithm.name)) for algorithm in available.values()]


def _candidate_supports_profile(candidate: CandidateSpec, profile: HardwareProfile | None) -> bool:
    if candidate.supported_skus is None:
        return True
    sku = None if profile is None else profile.sku
    if not sku or sku == "UNKNOWN":
        return True
    return sku in candidate.supported_skus


def _make_case(
    *,
    collective: str,
    nelems: int,
    dtype_spec: DTypeSpec,
    comm_group: Any,
    allgather_mode: str,
    symmetric_memory: bool = False,
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
            symmetric_memory=symmetric_memory,
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
        symmetric_memory=symmetric_memory,
    )


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


def _format_stat(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def _format_mismatches(stats: CorrectnessStats | None) -> str:
    if stats is None or stats.total == 0:
        return "-"
    return f"{stats.mismatches}/{stats.total}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark MSCCL++ collectives without PyTorch dependencies")
    parser.add_argument("--collective", choices=(_ALLREDUCE, _ALLGATHER), default=_ALLREDUCE)
    parser.add_argument("--d-model", type=int, default=5120)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--accum-type", help="Accumulation type for reductions: native, float16, or float32")
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

    dtype_spec = _with_accum_type(_parse_dtype(args.dtype), args.accum_type)
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
                symmetric_memory=args.symmetric_memory,
            )
            config = tuner.tune(case) if args.autotune else comm.resolve_config(case)
            if config is None:
                continue
            if args.autotune:
                config_store.upsert(hardware_profile, args.collective, case.message_size, config)

            correctness = "SKIP"
            correctness_stats: CorrectnessStats | None = None
            if not args.skip_correctness:
                correctness_stats = _check_correctness(comm, case, config, niter=args.correctness_iters)
                correctness = "PASS" if correctness_stats else "FAIL"
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
                    _format_stat(None if correctness_stats is None else correctness_stats.max_abs_diff),
                    _format_stat(None if correctness_stats is None else correctness_stats.mean_abs_diff),
                    _format_mismatches(correctness_stats),
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
                        "max_diff",
                        "mean_diff",
                        "mismatch",
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
