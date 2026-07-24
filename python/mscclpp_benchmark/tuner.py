# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Callable, Iterable

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

    def tune(self, case: Any) -> TunedConfig | None:
        best_config: TunedConfig | None = None
        best_time_us = float("inf")
        symmetric_memory = bool(getattr(case, "symmetric_memory", False))
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
                    config = TunedConfig(
                        algorithm=algorithm.name,
                        nblocks=nblocks,
                        nthreads=nthreads,
                        symmetric_memory=symmetric_memory,
                    )
                    if not self._check_correctness(self.comm, case, config):
                        self.comm.reset(config)
                        continue
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
