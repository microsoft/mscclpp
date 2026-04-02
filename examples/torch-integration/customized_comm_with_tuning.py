# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# torchrun --nnodes=1 --nproc_per_node=8 examples/torch-integration/customized_comm_with_tuning.py

import os
import ipaddress

import netifaces as ni
import torch
import mscclpp
import mscclpp.ext
import mscclpp.utils as mscclpp_utils


# -- Helpers ------------------------------------------------------------------

def _make_tensor(size_bytes: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a tensor backed by RawGpuBuffer (symmetric memory)."""
    # PyTorch's from_dlpack does not support certain float8 DLPack type codes.
    # Work around by importing as uint8 and reinterpreting via .view().
    _DLPACK_UNSUPPORTED = (torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                           torch.float8_e5m2, torch.float8_e5m2fnuz)
    if dtype in _DLPACK_UNSUPPORTED:
        dlpack = mscclpp.RawGpuBuffer(size_bytes).to_dlpack(data_type=str(torch.uint8))
        return torch.utils.dlpack.from_dlpack(dlpack).view(dtype)
    dlpack = mscclpp.RawGpuBuffer(size_bytes).to_dlpack(data_type=str(dtype))
    return torch.utils.dlpack.from_dlpack(dlpack)


def _load_algorithms(scratch: torch.Tensor, rank: int):
    return mscclpp.ext.AlgorithmCollectionBuilder().build_default_algorithms(
        scratch_buffer=scratch.data_ptr(), scratch_buffer_size=scratch.nbytes, rank=rank,
    )


def _interfaces_for_ip(ip: str):
    target = ipaddress.ip_address(ip)
    for iface in ni.interfaces():
        addrs = ni.ifaddresses(iface)
        if ni.AF_INET in addrs:
            for link in addrs[ni.AF_INET]:
                if "addr" in link and ipaddress.ip_address(link["addr"]) == target:
                    return iface
    return None


def _to_mscclpp_op(op) -> mscclpp.ReduceOp:
    if op == torch.distributed.ReduceOp.SUM:
        return mscclpp.ReduceOp.SUM
    if op == torch.distributed.ReduceOp.MIN:
        return mscclpp.ReduceOp.MIN
    raise ValueError(f"unsupported op: {op}")


def _round_pow2(size: int) -> int:
    """Round up to next power-of-2, clamped to [1024, 256 MB]."""
    size = max(size, 1024)
    size = min(size, 256 << 20)
    return 1 << (size - 1).bit_length()


# -- CustomizedComm -----------------------------------------------------------

class CustomizedComm:
    """Exposes all_reduce, all_gather, barrier with lazy per-size tuning."""

    _TUNE_N_WARMUP = 5
    _TUNE_N_GRAPH_LAUNCHES = 10
    _TUNE_N_OPS_PER_GRAPH = 100
    _CANDIDATE_NBLOCKS = [4, 8, 16, 24, 32, 48, 64, 128]
    _CANDIDATE_NTHREADS = [512, 768, 1024]
    _NBLOCKS_LIMIT = {
        "default_allreduce_nvls_packet": 16,
        "default_allreduce_packet": 56,
        "default_allreduce_allpair_packet": 56,
        "default_allreduce_fullmesh": 64,
        "default_allgather_fullmesh2": 32,
    }

    def __init__(self, comm: mscclpp.CommGroup, symmetric_memory: bool = False):
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.symmetric_memory = symmetric_memory
        self._nvls = mscclpp.is_nvls_supported()

        self._scratch = _make_tensor(1 << 27, torch.float16)
        self._barrier_tensor = _make_tensor(4096, torch.float32)

        algos = _load_algorithms(self._scratch, self.rank)
        self._algos = {(a.collective, a.name): a for a in algos}

        # {collective: {rounded_size: (algo, nblocks, nthreads)}}
        self._tune_cache: dict[str, dict[int, tuple]] = {"allreduce": {}, "allgather": {}}
        self._tune_buf = None
        self._time_buf = None

    def _algo(self, collective: str, name: str):
        return self._algos.get((collective, name))

    def _default_ar_config(self):
        """Fallback allreduce config for barrier / timing sync."""
        pkt = self._algo("allreduce", "default_allreduce_nvls_packet")
        if self._nvls and pkt:
            return (pkt, 0, 0)
        return (self._algo("allreduce", "default_allreduce_packet"), 0, 0)

    # -- low-level execute --

    def _exec_ar(self, tensor, algo, nb, nt, op=mscclpp.ReduceOp.SUM,
                 stream=None, accum_dtype=None, sym=True):
        s = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
        ret = algo.execute(
            comm=self.comm.communicator,
            input_buffer=tensor.data_ptr(), output_buffer=tensor.data_ptr(),
            input_size=tensor.nbytes, output_size=tensor.nbytes,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
            op=op, stream=s, nblocks=nb, nthreads_per_block=nt,
            symmetric_memory=sym, accum_dtype=accum_dtype,
        )
        if ret != 0:
            print(f"Rank {self.rank}: {algo.name} failed ({ret})")
        return ret

    def _exec_ag(self, inp, out, algo, nb, nt, stream=None, sym=None):
        if sym is None:
            sym = self.symmetric_memory
        s = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
        ret = algo.execute(
            comm=self.comm.communicator,
            input_buffer=inp.data_ptr(), output_buffer=out.data_ptr(),
            input_size=inp.nbytes, output_size=out.nbytes,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(inp.dtype),
            op=mscclpp.ReduceOp.NOP, stream=s, nblocks=nb, nthreads_per_block=nt,
            symmetric_memory=sym,
        )
        if ret != 0:
            print(f"Rank {self.rank}: AG {algo.name} failed ({ret})")
        return ret

    def _barrier_internal(self):
        a, nb, nt = self._default_ar_config()
        self._exec_ar(self._barrier_tensor, a, nb, nt, sym=True)

    # -- lazy tuning --

    def _ensure_tune_bufs(self):
        if self._tune_buf is None:
            self._tune_buf = _make_tensor(1 << 27, torch.float16)
            self._tune_buf.normal_()
            self._time_buf = _make_tensor(4096, torch.float32)
        return self._tune_buf

    def _ar_candidates(self, size: int):
        out = []
        if size <= 4 << 20:
            a = self._algo("allreduce", "default_allreduce_nvls_packet")
            if self._nvls and a:
                out.append(a)
            a = self._algo("allreduce", "default_allreduce_packet")
            if a:
                out.append(a)
            a = self._algo("allreduce", "default_allreduce_allpair_packet")
            if a:
                out.append(a)
        if size >= 512 << 10:
            a = self._algo("allreduce", "default_allreduce_nvls_zero_copy")
            if self._nvls and self.symmetric_memory and a:
                out.append(a)
            a = self._algo("allreduce", "default_allreduce_rsag_zero_copy")
            if a:
                out.append(a)
        if torch.version.hip is not None:
            a = self._algo("allreduce", "default_allreduce_fullmesh")
            if a:
                out.append(a)
        return out

    def _ag_candidates(self):
        a = self._algo("allgather", "default_allgather_fullmesh2")
        return [a] if a else []

    def _run_tune(self, collective, algo, buf, size, nb, nt):
        """Single tune invocation for either collective."""
        if collective == "allreduce":
            return algo.execute(
                comm=self.comm.communicator,
                input_buffer=buf.data_ptr(), output_buffer=buf.data_ptr(),
                input_size=size, output_size=size,
                dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(buf.dtype),
                op=mscclpp.ReduceOp.SUM, stream=torch.cuda.current_stream().cuda_stream,
                nblocks=nb, nthreads_per_block=nt, symmetric_memory=True,
            )
        else:
            total = size * self.world_size
            out_ptr = buf.data_ptr()
            return algo.execute(
                comm=self.comm.communicator,
                input_buffer=out_ptr + self.rank * size, output_buffer=out_ptr,
                input_size=size, output_size=total,
                dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(buf.dtype),
                op=mscclpp.ReduceOp.NOP, stream=torch.cuda.current_stream().cuda_stream,
                nblocks=nb, nthreads_per_block=nt, symmetric_memory=False,
            )

    def _tune_size(self, collective: str, target_size: int):
        """Auto-tune one (collective, target_size) pair and cache result."""
        buf = self._ensure_tune_bufs()
        cands = self._ar_candidates(target_size) if collective == "allreduce" else self._ag_candidates()

        best_time, best_cfg = float("inf"), None
        used = set()
        run = lambda a, nb, nt: self._run_tune(collective, a, buf, target_size, nb, nt)

        for algo in cands:
            nb_limit = self._NBLOCKS_LIMIT.get(algo.name, 128)
            for nb in self._CANDIDATE_NBLOCKS:
                if nb > nb_limit:
                    continue
                for nt in self._CANDIDATE_NTHREADS:
                    # Feasibility — sync result across ranks so all agree
                    ret = run(algo, nb, nt)
                    torch.cuda.synchronize()
                    self._time_buf[0] = float(ret)
                    self._exec_ar(self._time_buf[:1], *self._default_ar_config(), sym=True)
                    if self._time_buf[0].item() != 0:
                        continue
                    used.add(algo)

                    # Warmup
                    for _ in range(self._TUNE_N_WARMUP):
                        run(algo, nb, nt)

                    # CUDA-graph timed benchmark
                    cs = torch.cuda.Stream()
                    cs.wait_stream(torch.cuda.current_stream())
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=cs):
                        for _ in range(self._TUNE_N_OPS_PER_GRAPH):
                            run(algo, nb, nt)

                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(cs)
                    with torch.cuda.stream(cs):
                        for _ in range(self._TUNE_N_GRAPH_LAUNCHES):
                            g.replay()
                    end.record(cs)
                    end.synchronize()
                    elapsed = start.elapsed_time(end)

                    # Cross-rank timing sync
                    self._time_buf.fill_(elapsed)
                    torch.cuda.current_stream().wait_stream(cs)
                    self._exec_ar(self._time_buf, *self._default_ar_config(), sym=True)
                    avg = self._time_buf[self.rank].item() / self.world_size

                    if avg < best_time:
                        best_time, best_cfg = avg, (algo, nb, nt)

        if best_cfg:
            self._tune_cache[collective][target_size] = best_cfg
            if self.rank == 0:
                n = self._TUNE_N_GRAPH_LAUNCHES * self._TUNE_N_OPS_PER_GRAPH
                print(f"[tune] {collective} size={target_size}: {best_cfg[0].name} "
                      f"nb={best_cfg[1]} nt={best_cfg[2]} time={best_time / n * 1000:.2f}us", flush=True)
        else:
            fb = self._default_ar_config() if collective == "allreduce" else (
                (self._ag_candidates()[0], 32, 512) if self._ag_candidates() else None)
            self._tune_cache[collective][target_size] = fb

        torch.cuda.synchronize()
        self._barrier_internal()
        for a in used:
            a.reset()

    # -- public API --

    def all_reduce(self, tensor, op=torch.distributed.ReduceOp.SUM, stream=None, accum_dtype=None):
        sz = _round_pow2(tensor.nbytes)
        if sz not in self._tune_cache["allreduce"]:
            self._tune_size("allreduce", sz)
        a, nb, nt = self._tune_cache["allreduce"][sz]
        self._exec_ar(tensor, a, nb, nt, op=_to_mscclpp_op(op),
                      stream=stream, accum_dtype=accum_dtype, sym=self.symmetric_memory)

    def all_gather(self, output_tensor, input_tensor, stream=None):
        sz = _round_pow2(input_tensor.nbytes)
        if sz not in self._tune_cache["allgather"]:
            self._tune_size("allgather", sz)
        a, nb, nt = self._tune_cache["allgather"][sz]
        self._exec_ag(input_tensor, output_tensor, a, nb, nt,
                      stream=stream, sym=self.symmetric_memory)

    def barrier(self):
        self._barrier_internal()

    def destroy(self):
        self._algos.clear()
        self._tune_cache = {"allreduce": {}, "allgather": {}}
        self._tune_buf = self._time_buf = self._barrier_tensor = self._scratch = self.comm = None


# -- Benchmarks (standalone) --------------------------------------------------

def _bench_sizes(low=5 * 1024, high=80 << 20):
    sizes, c = [], low
    while c <= high:
        sizes.append(c)
        c *= 2
    return sizes


def benchmark_allreduce(comm: CustomizedComm, dtype=torch.float16, accum_dtype=None,
                        n_warmup=10, n_graph_launches=10, n_iter=100):
    sizes = _bench_sizes()
    if comm.rank == 0:
        print(f"\n{'='*60}\nAllreduce Benchmark\n{'='*60}")
        print(f"{'Nelements':<18} {'Size(B)':<18} {'Time(us)':<18} {'AlgoBW(GB/s)':<18}")

    cs = torch.cuda.Stream()
    buf = _make_tensor(1 << 27, dtype)
    buf.normal_() if dtype in (torch.float16, torch.float32, torch.bfloat16) else buf.fill_(0)

    for size in sizes:
        nelems = size // buf.element_size()
        t = buf[:size // buf.element_size()]
        comm.all_reduce(t, accum_dtype=accum_dtype)
        torch.cuda.synchronize()

        cs.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=cs):
            for _ in range(n_iter):
                comm.all_reduce(t, accum_dtype=accum_dtype)
        with torch.cuda.stream(cs):
            for _ in range(n_warmup):
                g.replay()
            comm.barrier()
        cs.synchronize()

        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(cs)
        with torch.cuda.stream(cs):
            for _ in range(n_graph_launches):
                g.replay()
        e.record(cs); e.synchronize()

        ms = s.elapsed_time(e) / (n_graph_launches * n_iter)
        if comm.rank == 0:
            print(f"{nelems:<18} {size:<18} {ms*1000:<18.2f} {size/(ms*1e-3)/1e9:<18.2f}")


def benchmark_allgather(comm: CustomizedComm, dtype=torch.float16,
                        n_warmup=10, n_graph_launches=10, n_iter=100):
    sizes = _bench_sizes()
    if comm.rank == 0:
        print(f"\n{'='*60}\nAllgather Benchmark\n{'='*60}")
        print(f"{'PerRank(B)':<18} {'Total(B)':<18} {'Time(us)':<18} {'AlgoBW(GB/s)':<18}")

    cs = torch.cuda.Stream()
    buf = _make_tensor(1 << 27, dtype)
    buf.normal_() if dtype in (torch.float16, torch.float32, torch.bfloat16) else buf.fill_(0)

    for prs in sizes:
        total = prs * comm.world_size
        if total > buf.nbytes:
            break
        nt = total // buf.element_size()
        npr = prs // buf.element_size()
        out = buf[:nt]
        inp = out[comm.rank * npr : (comm.rank + 1) * npr]

        comm.all_gather(out, inp)
        torch.cuda.synchronize()

        cs.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=cs):
            for _ in range(n_iter):
                comm.all_gather(out, inp)
        with torch.cuda.stream(cs):
            for _ in range(n_warmup):
                g.replay()
            comm.barrier()
        cs.synchronize()

        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(cs)
        with torch.cuda.stream(cs):
            for _ in range(n_graph_launches):
                g.replay()
        e.record(cs); e.synchronize()

        ms = s.elapsed_time(e) / (n_graph_launches * n_iter)
        if comm.rank == 0:
            print(f"{prs:<18} {total:<18} {ms*1000:<18.2f} {total/(ms*1e-3)/1e9:<18.2f}")


# -- Bootstrap & main ---------------------------------------------------------

def init_dist() -> mscclpp.CommGroup:
    addr = os.environ.get("MSCCLPP_MASTER_ADDR")
    if addr:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        port = os.environ["MSCCLPP_MASTER_PORT"]
        iface = _interfaces_for_ip(addr)
        if not iface:
            raise ValueError(f"No interface for {addr}")
        return mscclpp.CommGroup(interfaceIpPortTrio=f"{iface}:{addr}:{port}", rank=rank, size=world)
    import torch.distributed as dist
    dist.init_process_group(backend="gloo")
    return mscclpp.CommGroup(torch_group=dist.group.WORLD)


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)

    dtype_str = os.environ.get("DTYPE", "float16")
    dtype = getattr(torch, dtype_str, torch.float16)
    accum_map = {"float32": mscclpp.DataType.float32, "float16": mscclpp.DataType.float16}
    accum_str = os.environ.get("ACCUM_DTYPE")
    accum_dtype = accum_map.get(accum_str) if accum_str else None

    comm_group = init_dist()
    cc = CustomizedComm(comm_group)

    print(f"rank {local} starting benchmarks with dtype={dtype} accum_dtype={accum_dtype}...")
    benchmark_allreduce(cc, dtype=dtype, accum_dtype=accum_dtype)
    cc.barrier(); torch.cuda.synchronize()

    benchmark_allgather(cc, dtype=dtype)
    cc.barrier(); torch.cuda.synchronize()

    cc.destroy()
    print(f"rank {local} completed successfully.")


if __name__ == "__main__":
    main()
