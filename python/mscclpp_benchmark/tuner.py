# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from mscclpp_benchmark.tuning_config import TunedConfig


class OfflineTuner:
    def __init__(
        self,
        comm: Any,
        *,
        candidate_nblocks: Iterable[int],
        candidate_nthreads: Iterable[int],
        n_warmup: int,
        n_graph_launches: int,
        n_ops_per_graph: int,
        candidate_algorithms: Callable[[Any, Any], list[tuple[Any, Any]]],
        check_correctness: Callable[..., bool],
        measure: Callable[..., float | None],
        symmetric_memory: bool,
    ) -> None:
        self.comm = comm
        self.candidate_nblocks = tuple(candidate_nblocks)
        self.candidate_nthreads = tuple(candidate_nthreads)
        self.n_warmup = n_warmup
        self.n_graph_launches = n_graph_launches
        self.n_ops_per_graph = n_ops_per_graph
        self._candidate_algorithms = candidate_algorithms
        self._check_correctness = check_correctness
        self._measure = measure
        self._symmetric_memory = symmetric_memory

    def tune(self, case: Any) -> TunedConfig | None:
        best_config: TunedConfig | None = None
        best_time_us = float("inf")
        symmetric_memory = bool(getattr(case, "symmetric_memory", self._symmetric_memory))
        candidates = self._candidate_algorithms(self.comm, case)
        if not candidates:
            if self.comm.rank == 0:
                print(
                    f"[skip] no supported tuning candidates for collective={case.collective} "
                    f"size={case.message_size}",
                    flush=True,
                )
            return None
        for algorithm, candidate_spec in candidates:
            for nblocks in self.candidate_nblocks:
                if candidate_spec.max_nblocks is not None and nblocks > candidate_spec.max_nblocks:
                    continue
                for nthreads in self.candidate_nthreads:
                    if candidate_spec.min_nthreads is not None and nthreads < candidate_spec.min_nthreads:
                        continue
                    config = TunedConfig(
                        algorithm=algorithm.name,
                        nblocks=nblocks,
                        nthreads=nthreads,
                        symmetric_memory=symmetric_memory,
                    )
                    if not self._check_correctness(self.comm, case, config):
                        self.comm.reset(config)
                        continue
                    self.comm.reset(config)
                    time_us = self._measure(
                        self.comm,
                        case,
                        config,
                        n_warmup=self.n_warmup,
                        n_graph_launches=self.n_graph_launches,
                        n_ops_per_graph=self.n_ops_per_graph,
                    )
                    self.comm.reset(config)
                    if time_us is None or time_us >= best_time_us:
                        continue
                    best_time_us = time_us
                    best_config = TunedConfig(
                        algorithm=algorithm.name,
                        nblocks=nblocks,
                        nthreads=nthreads,
                        symmetric_memory=symmetric_memory,
                        time_us=time_us,
                    )
        if best_config is None:
            return self.comm.resolve_config(case)
        return best_config


def _normalize_name(name: str | None) -> str:
    if not name:
        return "native"
    return name.strip().lower().replace("-", "_")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate offline MSCCL++ tuned configs")
    parser.add_argument("--collective", choices=("allreduce", "allgather"), default="allreduce")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--accum-type")
    parser.add_argument("--sku", default="runtime", help="Used only for the default output filename")
    parser.add_argument("--scale", type=int, help="Expected MPI world size")
    parser.add_argument("--batch-sizes")
    parser.add_argument("--output")
    parser.add_argument("--scratch-buffer-size", type=int, default=1 << 27)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup graph replays during tuning")
    parser.add_argument("--graph-launches", type=int, default=10, help="Timed graph replays during tuning")
    parser.add_argument(
        "--ops-per-graph", type=int, default=100, help="Collective ops captured per graph during tuning"
    )
    parser.add_argument("--candidate-nblocks")
    parser.add_argument("--candidate-nthreads")
    parser.add_argument("--symmetric-memory", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    return parser


def _default_output_path(args: argparse.Namespace) -> str:
    accum = _normalize_name(args.accum_type)
    return (
        "mscclpp_tuned_"
        f"{_normalize_name(args.collective)}_"
        f"{_normalize_name(args.sku)}_"
        f"s{args.scale or 'runtime'}_"
        f"d{args.dim}_"
        f"dtype_{_normalize_name(args.dtype)}_"
        f"accum_{accum}.json"
    )


def _bench_collective_args(args: argparse.Namespace) -> list[str]:
    output = args.output or _default_output_path(args)
    bench_args = [
        "--collective",
        args.collective,
        "--d-model",
        str(args.dim),
        "--dtype",
        args.dtype,
        "--autotune",
        "--write-config",
        output,
        "--scratch-buffer-size",
        str(args.scratch_buffer_size),
        "--tune-warmup",
        str(args.warmup),
        "--tune-graph-launches",
        str(args.graph_launches),
        "--tune-iterations",
        str(args.ops_per_graph),
        "--warmup",
        "0",
        "--graph-launches",
        "1",
        "--iterations",
        "1",
    ]
    if args.batch_sizes:
        bench_args += ["--batch-sizes", args.batch_sizes]
    if args.accum_type:
        bench_args += ["--accum-type", args.accum_type]
    if args.candidate_nblocks:
        bench_args += ["--candidate-nblocks", args.candidate_nblocks]
    if args.candidate_nthreads:
        bench_args += ["--candidate-nthreads", args.candidate_nthreads]
    if args.symmetric_memory:
        bench_args.append("--symmetric-memory")
    if args.skip_correctness:
        bench_args.append("--skip-correctness")
    return bench_args


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.scale is not None:
        from mpi4py import MPI

        world_size = MPI.COMM_WORLD.Get_size()
        if world_size != args.scale:
            raise ValueError(f"MSCCL++ tuning scale mismatch: expected MPI world size {args.scale}, got {world_size}")

    from mscclpp_benchmark.bench_collective import main as bench_collective_main

    bench_collective_main(_bench_collective_args(args))
    if args.output is None:
        print(f"Wrote tuned config to {Path(_default_output_path(args)).resolve()}", flush=True)


if __name__ == "__main__":
    main()
