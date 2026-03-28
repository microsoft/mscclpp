# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> torchrun --nnodes=1 --nproc_per_node=8  customized_comm_with_tuning.py

import os

import ipaddress

import netifaces as ni
import torch

import mscclpp
import mscclpp.ext
import mscclpp.utils as mscclpp_utils


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_algorithms(scratch_buffer: torch.Tensor, rank: int) -> mscclpp.AlgorithmCollection:
    builder = mscclpp.ext.AlgorithmCollectionBuilder()
    return builder.build_default_algorithms(
        scratch_buffer=scratch_buffer.data_ptr(),
        scratch_buffer_size=scratch_buffer.nbytes,
        rank=rank,
    )


def _interfaces_for_ip(ip: str):
    target = ipaddress.ip_address(ip)
    for interface in ni.interfaces():
        addresses = ni.ifaddresses(interface)
        if ni.AF_INET in addresses:
            for link in addresses[ni.AF_INET]:
                if "addr" in link and ipaddress.ip_address(link["addr"]) == target:
                    return interface
    return None


def _to_mscclpp_reduce_op(op: torch.distributed.ReduceOp) -> mscclpp.ReduceOp:
    if op == torch.distributed.ReduceOp.SUM:
        return mscclpp.ReduceOp.SUM
    elif op == torch.distributed.ReduceOp.MIN:
        return mscclpp.ReduceOp.MIN
    raise ValueError(f"unsupported op: {op}")


def _round_to_power_of_two(size: int) -> int:
    """Round *size* up to the next power of 2, clamped to [1024, 256 MB]."""
    if size < 1024:
        return 1024
    if size > 256 * 1024 * 1024:
        return 256 * 1024 * 1024
    return 1 << (size - 1).bit_length()


def _make_symmetric_tensor(size_bytes: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a tensor backed by ``RawGpuBuffer`` (symmetric memory)."""
    dlpack = mscclpp.RawGpuBuffer(size_bytes).to_dlpack(data_type=str(dtype))
    return torch.utils.dlpack.from_dlpack(dlpack)


# ---------------------------------------------------------------------------
# CustomizedComm — clean API: all_reduce, all_gather, barrier
# ---------------------------------------------------------------------------


class CustomizedComm:
    """Customized communicator exposing **all_reduce**, **all_gather**, and **barrier**.

    * ``__init__`` only sets up communication and allocates scratch buffers —
      **no tuning happens at construction time**.
    * Tuning is performed lazily: the first call for a given collective +
      message-size triggers auto-tuning for that size, and the result is
      cached for all subsequent calls.
    * ``dtype`` / ``accum_dtype`` are **not** class-level attributes — they
      are derived from the tensors passed at call-time.
    """

    # -- Tuning hyper-parameters -------------------------------------------
    _TUNE_N_WARMUP = 5
    _TUNE_N_GRAPH_LAUNCHES = 10
    _TUNE_N_OPS_PER_GRAPH = 100
    _CANDIDATE_NBLOCKS = [4, 8, 16, 24, 32, 48, 64, 128]
    _CANDIDATE_NTHREADS = [512, 768, 1024]

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        comm: mscclpp.CommGroup,
        symmetric_memory: bool = False,
    ):
        """Set up the communicator.  **No tuning happens here.**

        Args:
            comm: The CommGroup to use.
            symmetric_memory: Whether **user** buffers passed to ``all_reduce``
                / ``all_gather`` are allocated with symmetric memory
                (``mscclpp.RawGpuBuffer``).  Default ``False``.
                Internal buffers (scratch, barrier, tune) always use symmetric
                memory.
        """
        # Comm basics
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.symmetric_memory = symmetric_memory
        self._nvls_supported = mscclpp.is_nvls_supported()

        # Scratch buffer (128 MB, symmetric memory)
        self._scratch_buffer = _make_symmetric_tensor(1 << 27, torch.float16)

        # Barrier tensor — symmetric memory so _barrier_internal can use
        # symmetric_memory=True safely.
        self._barrier_tensor = _make_symmetric_tensor(4096, torch.float32)

        # Load & index algorithms
        algorithms = _load_algorithms(scratch_buffer=self._scratch_buffer, rank=self.rank)
        self._algorithms = self._index_algorithms(algorithms)

        # Tune configs:  {collective: {rounded_size: (algo, nblocks, nthreads)}}
        self._tune_configs: dict[str, dict[int, tuple]] = {
            "allreduce": {},
            "allgather": {},
        }

        # Lazily allocated tune buffer / time tensor
        self._tune_buffer = None
        self._time_tensor = None

    # -------------------------------------------------------- algo indexing

    @staticmethod
    def _index_algorithms(algorithms):
        """Return ``{(collective, name): algo}`` for fast look-up."""
        return {(a.collective, a.name): a for a in algorithms}

    def _get_algo(self, collective: str, name: str):
        return self._algorithms.get((collective, name))

    # --------------------------------------------------- default configs

    def _default_allreduce_config(self):
        """Safe fallback config used for internal barrier / timing sync."""
        nvls_pkt = self._get_algo("allreduce", "default_allreduce_nvls_packet")
        if self._nvls_supported and nvls_pkt is not None:
            return (nvls_pkt, 0, 0)
        return (self._get_algo("allreduce", "default_allreduce_packet"), 0, 0)

    # -------------------------------------------------- low-level execute

    def _execute_allreduce(
        self,
        tensor,
        algo,
        nblocks,
        nthreads,
        op=mscclpp.ReduceOp.SUM,
        stream=None,
        accum_dtype=None,
        symmetric_memory=True,
    ):
        """Execute an allreduce.  Default ``symmetric_memory=True`` (safe for
        internal buffers which are all RawGpuBuffer)."""
        cuda_stream = stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream
        ret = algo.execute(
            comm=self.comm.communicator,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=tensor.nbytes,
            output_size=tensor.nbytes,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
            op=op,
            stream=cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            symmetric_memory=symmetric_memory,
            accum_dtype=accum_dtype,
        )
        if ret != 0:
            print(f"Rank {self.rank}: Algo {algo.name} failed with error {ret}")
        return ret

    def _execute_allgather(
        self,
        input_tensor,
        output_tensor,
        algo,
        nblocks,
        nthreads,
        stream=None,
        symmetric_memory=None,
    ):
        if symmetric_memory is None:
            symmetric_memory = self.symmetric_memory
        cuda_stream = stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream
        ret = algo.execute(
            comm=self.comm.communicator,
            input_buffer=input_tensor.data_ptr(),
            output_buffer=output_tensor.data_ptr(),
            input_size=input_tensor.nbytes,
            output_size=output_tensor.nbytes,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(input_tensor.dtype),
            op=mscclpp.ReduceOp.NOP,
            stream=cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            symmetric_memory=symmetric_memory,
        )
        if ret != 0:
            print(f"Rank {self.rank}: Allgather Algo {algo.name} failed with error {ret}")
        return ret

    def _barrier_internal(self):
        """Barrier using default config — safe to call during tuning.

        Uses symmetric_memory=True because _barrier_tensor is RawGpuBuffer.
        """
        algo, nb, nt = self._default_allreduce_config()
        self._execute_allreduce(self._barrier_tensor, algo, nb, nt)

    # ------------------------------------------------------ lazy tuning

    def _get_tune_buffer(self):
        """Allocate shared tune buffer and timing tensor on first use."""
        if self._tune_buffer is None:
            self._tune_buffer = _make_symmetric_tensor(1 << 27, torch.float16)
            self._tune_buffer.normal_()
            # Timing sync tensor: float32, one element per rank (symmetric)
            self._time_tensor = _make_symmetric_tensor(4096, torch.float32)
        return self._tune_buffer

    def _allreduce_candidates(self, size: int):
        """Return candidate algorithms for allreduce at *size* bytes."""
        algos = []
        if size <= 4 * 1024 * 1024:
            nvls_pkt = self._get_algo("allreduce", "default_allreduce_nvls_packet")
            if self._nvls_supported and nvls_pkt is not None:
                algos.append(nvls_pkt)
            pkt = self._get_algo("allreduce", "default_allreduce_packet")
            if pkt is not None:
                algos.append(pkt)
            # allpair_pkt = self._get_algo("allreduce", "default_allreduce_allpair_packet")
            # if allpair_pkt is not None:
            #     algos.append(allpair_pkt)
        if size >= 512 * 1024:
            nvls_zc = self._get_algo("allreduce", "default_allreduce_nvls_zero_copy")
            if self._nvls_supported and self.symmetric_memory and nvls_zc is not None:
                algos.append(nvls_zc)
            rsag = self._get_algo("allreduce", "default_allreduce_rsag_zero_copy")
            if rsag is not None:
                algos.append(rsag)
        # On ROCm/HIP, fullmesh is a primary algorithm across all sizes
        if torch.version.hip is not None:
            fm = self._get_algo("allreduce", "default_allreduce_fullmesh")
            if fm is not None:
                algos.append(fm)
        return algos

    def _allgather_candidates(self):
        """Return candidate algorithms for allgather."""
        algo = self._get_algo("allgather", "default_allgather_fullmesh2")
        return [algo] if algo is not None else []

    @staticmethod
    def _nblocks_limit(algo_name: str) -> int:
        limits = {
            "default_allreduce_nvls_packet": 16,
            "default_allreduce_packet": 56,
            "default_allreduce_allpair_packet": 56,
            "default_allreduce_fullmesh": 64,
            "default_allgather_fullmesh2": 32,
        }
        return limits.get(algo_name, 128)

    # -- single-size tuning core --

    def _tune_size(self, collective: str, target_size: int):
        """Auto-tune one *(collective, target_size)* pair and cache the result."""
        tune_buf = self._get_tune_buffer()
        time_tensor = self._time_tensor  # symmetric, float32

        candidates = (
            self._allreduce_candidates(target_size)
            if collective == "allreduce"
            else self._allgather_candidates()
        )

        best_time = float("inf")
        best_config = None
        used_algos = set()

        for algo in candidates:
            nb_limit = self._nblocks_limit(algo.name)
            for nb in self._CANDIDATE_NBLOCKS:
                if nb > nb_limit:
                    continue
                for nt in self._CANDIDATE_NTHREADS:
                    # Feasibility check
                    if collective == "allreduce":
                        ret = self._run_tune_allreduce(algo, tune_buf, target_size, nb, nt)
                    else:
                        ret = self._run_tune_allgather(algo, tune_buf, target_size, nb, nt)
                    torch.cuda.synchronize()
                    if ret != 0:
                        continue
                    used_algos.add(algo)

                    # Warmup
                    for _ in range(self._TUNE_N_WARMUP):
                        if collective == "allreduce":
                            self._run_tune_allreduce(algo, tune_buf, target_size, nb, nt)
                        else:
                            self._run_tune_allgather(algo, tune_buf, target_size, nb, nt)
                    self._barrier_internal()

                    # CUDA-graph benchmark
                    capture_stream = torch.cuda.Stream()
                    capture_stream.wait_stream(torch.cuda.current_stream())

                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=capture_stream):
                        for _ in range(self._TUNE_N_OPS_PER_GRAPH):
                            if collective == "allreduce":
                                self._run_tune_allreduce(algo, tune_buf, target_size, nb, nt)
                            else:
                                self._run_tune_allgather(algo, tune_buf, target_size, nb, nt)

                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record(capture_stream)
                    with torch.cuda.stream(capture_stream):
                        for _ in range(self._TUNE_N_GRAPH_LAUNCHES):
                            g.replay()
                    end_ev.record(capture_stream)
                    end_ev.synchronize()

                    elapsed = start_ev.elapsed_time(end_ev)

                    # Cross-rank timing sync via default allreduce config.
                    # time_tensor is RawGpuBuffer → symmetric_memory=True.
                    time_tensor.fill_(elapsed)
                    torch.cuda.current_stream().wait_stream(capture_stream)
                    d_algo, d_nb, d_nt = self._default_allreduce_config()
                    self._execute_allreduce(
                        time_tensor, d_algo, d_nb, d_nt, symmetric_memory=True
                    )
                    avg_time = time_tensor[self.rank].item() / self.world_size

                    if avg_time < best_time:
                        best_time = avg_time
                        best_config = (algo, nb, nt)

        if best_config:
            self._tune_configs[collective][target_size] = best_config
            if self.rank == 0:
                total_ops = self._TUNE_N_GRAPH_LAUNCHES * self._TUNE_N_OPS_PER_GRAPH
                print(
                    f"[tune] {collective} size={target_size}: "
                    f"{best_config[0].name} nblocks={best_config[1]} nthreads={best_config[2]} "
                    f"time={(best_time / total_ops) * 1000:.2f} us",
                    flush=True,
                )
        else:
            # Store a safe fallback so we never tune the same size twice
            if collective == "allreduce":
                self._tune_configs[collective][target_size] = self._default_allreduce_config()
            else:
                ag_algos = self._allgather_candidates()
                self._tune_configs[collective][target_size] = (ag_algos[0], 32, 512) if ag_algos else None
        torch.cuda.synchronize()

    # -- tuning execute helpers --
    # It's safe to use symmetric_memory=True since we reuse the same buffer for tuning.
    def _run_tune_allreduce(self, algo, tune_buf, size, nblocks, nthreads):
        return algo.execute(
            comm=self.comm.communicator,
            input_buffer=tune_buf.data_ptr(),
            output_buffer=tune_buf.data_ptr(),
            input_size=size,
            output_size=size,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(tune_buf.dtype),
            op=mscclpp.ReduceOp.SUM,
            stream=torch.cuda.current_stream().cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            symmetric_memory=True,
        )

    def _run_tune_allgather(self, algo, tune_buf, per_rank_size, nblocks, nthreads):
        total_size = per_rank_size * self.world_size
        output_ptr = tune_buf.data_ptr()
        input_ptr = output_ptr + self.rank * per_rank_size
        return algo.execute(
            comm=self.comm.communicator,
            input_buffer=input_ptr,
            output_buffer=output_ptr,
            input_size=per_rank_size,
            output_size=total_size,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(tune_buf.dtype),
            op=mscclpp.ReduceOp.NOP,
            stream=torch.cuda.current_stream().cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            symmetric_memory=False,
        )

    # ================================================================
    # Public API: all_reduce, all_gather, barrier
    # ================================================================

    def all_reduce(self, tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM, stream=None, accum_dtype=None):
        """In-place allreduce.  Auto-tunes on first call per message size.

        Args:
            tensor: The tensor to reduce in-place.
            op: Reduction operation (default: SUM).
            stream: CUDA stream (default: current stream).
            accum_dtype: Accumulation data type (e.g. ``mscclpp.DataType.float32``
                for high-precision FP8 accumulation).  ``None`` uses the default.
        """
        target_size = _round_to_power_of_two(tensor.nbytes)
        if target_size not in self._tune_configs["allreduce"]:
            self._tune_size("allreduce", target_size)
        algo, nblocks, nthreads = self._tune_configs["allreduce"][target_size]
        self._execute_allreduce(
            tensor,
            algo,
            nblocks,
            nthreads,
            op=_to_mscclpp_reduce_op(op),
            stream=stream,
            accum_dtype=accum_dtype,
            symmetric_memory=self.symmetric_memory,
        )

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None):
        """Allgather.  Auto-tunes on first call per per-rank message size.

        For in-place operation, *input_tensor* should be this rank's shard
        inside *output_tensor*.

        Args:
            output_tensor: Output buffer (size = per_rank_size * world_size).
            input_tensor: This rank's contribution.
            stream: CUDA stream (default: current stream).
        """
        per_rank_size = input_tensor.nbytes
        target_size = _round_to_power_of_two(per_rank_size)
        if target_size not in self._tune_configs["allgather"]:
            self._tune_size("allgather", target_size)
        algo, nblocks, nthreads = self._tune_configs["allgather"][target_size]
        self._execute_allgather(
            input_tensor,
            output_tensor,
            algo,
            nblocks,
            nthreads,
            stream=stream,
            symmetric_memory=self.symmetric_memory,
        )

    def barrier(self):
        """GPU barrier implemented as an allreduce on a small tensor.

        Uses the internal barrier path (default allreduce config, symmetric
        memory) — does **not** go through the tuning path.
        """
        self._barrier_internal()

    # --------------------------------------------------------------- cleanup

    def destroy(self):
        self._algorithms.clear()
        self._tune_configs = {"allreduce": {}, "allgather": {}}
        self._tune_buffer = None
        self._time_tensor = None
        self._barrier_tensor = None
        self._scratch_buffer = None
        self.comm = None


# ---------------------------------------------------------------------------
# Standalone benchmark utilities  (use CustomizedComm for the collective calls)
# ---------------------------------------------------------------------------


def benchmark_allreduce(
    comm: CustomizedComm,
    dtype=torch.float16,
    accum_dtype=None,
    n_warmup=10,
    n_graph_launches=10,
    n_iter_per_graph=100,
):
    """Benchmark allreduce across a range of message sizes.

    Args:
        comm: A ``CustomizedComm`` instance.
        dtype: Tensor dtype to benchmark with.
        accum_dtype: Accumulation dtype passed through to ``all_reduce``.
        n_warmup / n_graph_launches / n_iter_per_graph: timing parameters.
    """
    low = 5 * 1024
    high = 80 * 1024 * 1024
    sizes = []
    curr = low
    while curr <= high:
        sizes.append(curr)
        curr *= 2

    if comm.rank == 0:
        print(f"\n{'=' * 60}")
        print("Allreduce Benchmark")
        print(f"{'=' * 60}")
        print(f"{'Size (Bytes)':<20} {'Time (us)':<20} {'AlgoBW (GB/s)':<20}")

    capture_stream = torch.cuda.Stream()

    bench_buf = _make_symmetric_tensor(1 << 27, dtype)
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        bench_buf.normal_()
    else:
        bench_buf.fill_(0)

    for size in sizes:
        n_elements = size // bench_buf.element_size()
        tensor = bench_buf[:n_elements]

        # Warmup (also triggers lazy tuning for this size)
        comm.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, accum_dtype=accum_dtype)
        torch.cuda.synchronize()

        capture_stream.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            for _ in range(n_iter_per_graph):
                comm.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, accum_dtype=accum_dtype)

        with torch.cuda.stream(capture_stream):
            for _ in range(n_warmup):
                g.replay()
            comm.barrier()
        capture_stream.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record(capture_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(n_graph_launches):
                g.replay()
        end_ev.record(capture_stream)
        end_ev.synchronize()

        elapsed_ms = start_ev.elapsed_time(end_ev)
        avg_ms = elapsed_ms / (n_graph_launches * n_iter_per_graph)
        time_us = avg_ms * 1000
        alg_bw = size / (avg_ms * 1e-3) if avg_ms > 0 else 0

        if comm.rank == 0:
            print(f"{size:<20} {time_us:<20.2f} {alg_bw / 1e9:<20.2f}")


def benchmark_allgather(
    comm: CustomizedComm,
    dtype=torch.float16,
    n_warmup=10,
    n_graph_launches=10,
    n_iter_per_graph=100,
):
    """Benchmark allgather across a range of per-rank message sizes.

    Args:
        comm: A ``CustomizedComm`` instance.
        dtype: Tensor dtype to benchmark with.
        n_warmup / n_graph_launches / n_iter_per_graph: timing parameters.
    """
    low = 5 * 1024
    high = 80 * 1024 * 1024
    sizes = []
    curr = low
    while curr <= high:
        sizes.append(curr)
        curr *= 2

    if comm.rank == 0:
        print(f"\n{'=' * 60}")
        print("Allgather Benchmark")
        print(f"{'=' * 60}")
        print(f"{'Per-Rank Size (B)':<25} {'Total Size (B)':<25} {'Time (us)':<20} {'AlgoBW (GB/s)':<20}")

    capture_stream = torch.cuda.Stream()

    ag_buf = _make_symmetric_tensor(1 << 27, dtype)
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        ag_buf.normal_()
    else:
        ag_buf.fill_(0)

    for per_rank_size in sizes:
        total_size = per_rank_size * comm.world_size
        if total_size > ag_buf.nbytes:
            break

        n_total = total_size // ag_buf.element_size()
        n_per_rank = per_rank_size // ag_buf.element_size()
        output_tensor = ag_buf[:n_total]
        input_tensor = output_tensor[comm.rank * n_per_rank : (comm.rank + 1) * n_per_rank]

        # Warmup (also triggers lazy tuning for this size)
        comm.all_gather(output_tensor, input_tensor)
        torch.cuda.synchronize()

        capture_stream.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            for _ in range(n_iter_per_graph):
                comm.all_gather(output_tensor, input_tensor)

        with torch.cuda.stream(capture_stream):
            for _ in range(n_warmup):
                g.replay()
            comm.barrier()
        capture_stream.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record(capture_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(n_graph_launches):
                g.replay()
        end_ev.record(capture_stream)
        end_ev.synchronize()

        elapsed_ms = start_ev.elapsed_time(end_ev)
        avg_ms = elapsed_ms / (n_graph_launches * n_iter_per_graph)
        time_us = avg_ms * 1000
        alg_bw = total_size / (avg_ms * 1e-3) if avg_ms > 0 else 0

        if comm.rank == 0:
            print(f"{per_rank_size:<25} {total_size:<25} {time_us:<20.2f} {alg_bw / 1e9:<20.2f}")


# ---------------------------------------------------------------------------
# Bootstrap & main
# ---------------------------------------------------------------------------


def init_dist() -> mscclpp.CommGroup:
    """Bootstrap via torch.distributed (works seamlessly with torchrun).

    Falls back to manual interfaceIpPortTrio if MSCCLPP_MASTER_ADDR is set.
    """
    master_addr = os.environ.get("MSCCLPP_MASTER_ADDR")
    if master_addr is not None:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        master_port = os.environ["MSCCLPP_MASTER_PORT"]
        interface = _interfaces_for_ip(master_addr)
        if interface is None:
            raise ValueError(f"Cannot find network interface for IP address {master_addr}")
        trio = f"{interface}:{master_addr}:{master_port}"
        return mscclpp.CommGroup(interfaceIpPortTrio=trio, rank=rank, size=world)
    else:
        import torch.distributed as dist

        dist.init_process_group(backend="gloo")
        group = dist.group.WORLD
        return mscclpp.CommGroup(torch_group=group)


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)

    # Configure dtype and accumulation type via env vars.
    # Example: DTYPE=float8_e4m3fn ACCUM_DTYPE=float32
    dtype_str = os.environ.get("DTYPE", "float16")
    dtype = getattr(torch, dtype_str, torch.float16)

    accum_dtype_map = {
        "float32": mscclpp.DataType.float32,
        "float16": mscclpp.DataType.float16,
        "float8_e4m3fn": mscclpp.DataType.float16,
    }
    accum_dtype_str = os.environ.get("ACCUM_DTYPE", None)
    accum_dtype = accum_dtype_map.get(accum_dtype_str) if accum_dtype_str else None

    comm_group = init_dist()
    custom_comm = CustomizedComm(comm_group)

    # Allreduce benchmark (lazy tuning kicks in per size)
    benchmark_allreduce(custom_comm, dtype=dtype, accum_dtype=accum_dtype, n_warmup=5, n_graph_launches=10, n_iter_per_graph=100)
    custom_comm.barrier()
    torch.cuda.synchronize()

    # Allgather benchmark (lazy tuning kicks in per size)
    benchmark_allgather(custom_comm, dtype=dtype, n_warmup=5, n_graph_launches=10, n_iter_per_graph=100)
    custom_comm.barrier()
    torch.cuda.synchronize()

    custom_comm.destroy()
    print(f"rank {local} All-reduce and all-gather operations completed successfully.")


if __name__ == "__main__":
    main()
