# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> torchrun --nnodes=1 --nproc_per_node=8  customized_comm_with_tuning.py

import os
import torch
import mscclpp.utils as mscclpp_utils
import mscclpp
import mscclpp.ext
import netifaces as ni
import ipaddress


def load_algorithms(scratch_buffer: torch.tensor, rank: int) -> mscclpp.AlgorithmCollection:
    collection_builder = mscclpp.ext.AlgorithmCollectionBuilder()
    return collection_builder.build_default_algorithms(
        scratch_buffer=scratch_buffer.data_ptr(), scratch_buffer_size=scratch_buffer.nbytes, rank=rank
    )


def interfaces_for_ip_netifaces(ip: str):
    target = ipaddress.ip_address(ip)
    for interface in ni.interfaces():
        addresses = ni.ifaddresses(interface)
        if ni.AF_INET in addresses:
            for link in addresses[ni.AF_INET]:
                if "addr" in link:
                    addr = ipaddress.ip_address(link["addr"])
                    if addr == target:
                        return interface
    return None


def to_mscclpp_reduce_op(op: torch.distributed.ReduceOp) -> mscclpp.ReduceOp:
    if op == torch.distributed.ReduceOp.SUM:
        return mscclpp.ReduceOp.SUM
    elif op == torch.distributed.ReduceOp.MIN:
        return mscclpp.ReduceOp.MIN
    else:
        raise ValueError(f"unsupported op: {op}")


class CustomizedComm:
    def __init__(self, comm: mscclpp.CommGroup, dtype=torch.float16, accum_dtype=None):
        """Initialize the customized communicator with tuning.

        Args:
            comm: The CommGroup to use.
            dtype: Data type for allreduce tensors (e.g., torch.float16, torch.float8_e4m3fn).
            accum_dtype: Accumulation data type for reduction. If None, defaults to
                         float32 for FP8 types, or same as dtype for other types.
                         Pass mscclpp.DataType.float32 for high-precision FP8 accumulation.
        """
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = comm.my_rank % comm.nranks_per_node
        self.n_ranks_per_node = comm.nranks_per_node
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        dlpack = mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(torch.float16))
        self.scratch_buffer = torch.utils.dlpack.from_dlpack(dlpack)
        self._nvls_supported = mscclpp.is_nvls_supported()
        self._barrier_tensor = torch.empty(comm.nranks, dtype=torch.float, device=torch.device("cuda"))
        algorithms = load_algorithms(scratch_buffer=self.scratch_buffer, rank=self.rank)
        self._algorithm_nvls_packet = [
            algo
            for algo in algorithms
            if algo.collective == "allreduce" and algo.name == "default_allreduce_nvls_packet"
        ][0]
        self._algorithm_rsag_zero_copy = [
            algo
            for algo in algorithms
            if algo.collective == "allreduce" and algo.name == "default_allreduce_rsag_zero_copy"
        ][0]
        self._algorithm_packet = [
            algo for algo in algorithms if algo.collective == "allreduce" and algo.name == "default_allreduce_packet"
        ][0]
        self._algorithm_fullmesh = [
            algo
            for algo in algorithms
            if algo.collective == "allreduce" and algo.name == "default_allreduce_fullmesh"
        ][0]
        if self._nvls_supported:
            self._algorithm_nvls_zero_copy = [
                algo
                for algo in algorithms
                if algo.collective == "allreduce" and algo.name == "default_allreduce_nvls_zero_copy"
            ][0]
        self._tune(n_warmup=5, n_graph_launches=10, n_ops_per_graph=100)

    def _tune(self, n_warmup, n_graph_launches, n_ops_per_graph):
        sizes = [1 << i for i in range(10, 28)]
        # Pre-fill with a platform-aware default for barrier
        if self._nvls_supported:
            self.best_configs = {1024: (self._algorithm_nvls_packet, 0, 0)}
        else:
            self.best_configs = {1024: (self._algorithm_fullmesh, 32, 512)}

        tune_tensor = mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(self.dtype))
        tune_tensor = torch.utils.dlpack.from_dlpack(tune_tensor)
        if self.dtype in (torch.float16, torch.float32, torch.bfloat16):
            tune_tensor.normal_()
        else:
            # FP8 doesn't support normal_(), fill from float16 and view as bytes
            tune_tensor.fill_(0)
        candidates_nblocks = [4, 8, 16, 24, 32, 48, 64, 128]
        candidates_nthreads = [512, 768, 1024]

        for size in sizes:
            algos = []
            if self._nvls_supported:
                algos.append(self._algorithm_nvls_zero_copy)
            if size <= 4 * 1024 * 1024:
                if self._nvls_supported:
                    algos.append(self._algorithm_nvls_packet)
                algos.append(self._algorithm_packet)
            if size >= 512 * 1024:
                algos.append(self._algorithm_rsag_zero_copy)
            # On ROCm/HIP, fullmesh is the primary large-message algorithm
            if torch.version.hip is not None:
                algos.append(self._algorithm_fullmesh)

            best_time = float("inf")
            best_config = None

            for algo in algos:
                for nb in candidates_nblocks:
                    if algo.name == "default_allreduce_nvls_packet" and nb > 16:
                        continue
                    if algo.name == "default_allreduce_packet" and nb > 56:
                        continue
                    if algo.name == "default_allreduce_fullmesh" and nb > 64:
                        continue
                    for nt in candidates_nthreads:
                        ret = self._run_algo(algo, tune_tensor, size, nb, nt)
                        torch.cuda.synchronize()
                        if ret != 0:
                            continue

                        for _ in range(n_warmup):
                            self._run_algo(algo, tune_tensor, size, nb, nt)
                        self.barrier()

                        capture_stream = torch.cuda.Stream()
                        capture_stream.wait_stream(torch.cuda.current_stream())

                        g = torch.cuda.CUDAGraph()
                        # Warmup on capture stream
                        with torch.cuda.stream(capture_stream):
                            self._run_algo(algo, tune_tensor, size, nb, nt)
                        capture_stream.synchronize()

                        with torch.cuda.graph(g, stream=capture_stream):
                            for _ in range(n_ops_per_graph):
                                self._run_algo(algo, tune_tensor, size, nb, nt)

                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record(capture_stream)
                        with torch.cuda.stream(capture_stream):
                            for _ in range(n_graph_launches):
                                g.replay()
                        end_event.record(capture_stream)
                        end_event.synchronize()

                        elapsed = start_event.elapsed_time(end_event)

                        # Synchronize timing results across all ranks to ensure consistent algorithm selection
                        # replicate n times such due to algo limitations
                        time_tensor = torch.full((self.world_size,), elapsed, dtype=torch.float64, device="cuda").to(
                            dtype=torch.float32
                        )
                        torch.cuda.current_stream().wait_stream(capture_stream)
                        # TODO: use all_reduce may cause problem if the time elapsed between different algos are too close.
                        # May change to broadcast in the future if that becomes an issue.
                        self.all_reduce(time_tensor, op=torch.distributed.ReduceOp.SUM)
                        avg_time = time_tensor[self.rank].item() / self.world_size

                        if avg_time < best_time:
                            best_time = avg_time
                            best_config = (algo, nb, nt)

            if best_config:
                self.best_configs[size] = best_config
                if self.rank == 0:
                    print(
                        f"Size {size}: Best Algo {best_config[0].name} nblocks {best_config[1]} nthreads {best_config[2]} Time {(best_time/(n_graph_launches * n_ops_per_graph))*1000:.2f} us"
                    )
        # reset the algorithms after tuning
        torch.cuda.synchronize()
        for algo in algos:
            algo.reset()

    def _run_algo(self, algo: mscclpp.Algorithm, tensor, size, nblocks, nthreads):
        return algo.execute(
            comm=self.comm.communicator,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=size,
            output_size=size,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
            op=mscclpp.ReduceOp.SUM,
            stream=torch.cuda.current_stream().cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            symmetric_memory=True,
            accum_dtype=self.accum_dtype,
        )

    def get_tuned_config(self, size):
        if size < 1024:
            target_size = 1024
        elif size > 256 * 1024 * 1024:
            target_size = 256 * 1024 * 1024
        else:
            target_size = 1 << (size - 1).bit_length()
        return self.best_configs.get(target_size)

    def all_reduce(self, tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM, stream: torch.cuda.Stream = None, symmetric_memory: bool = True):
        assert op == torch.distributed.ReduceOp.SUM
        config = self.get_tuned_config(tensor.nbytes)
        if self._nvls_supported:
            default_config = (self._algorithm_nvls_packet, 0, 0)
        else:
            default_config = (self._algorithm_fullmesh, 32, 512)
        algo, nblocks, nthreads = config if config else default_config
        ret = algo.execute(
            comm=self.comm.communicator,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=tensor.nbytes,
            output_size=tensor.nbytes,
            dtype=mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype),
            op=to_mscclpp_reduce_op(op),
            stream=stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
            symmetric_memory=symmetric_memory,
            accum_dtype=self.accum_dtype,
        )
        if ret != 0:
            print(f"Rank {self.rank}: Algo {algo.name} failed with error {ret}")

    def barrier(self):
        self.all_reduce(self._barrier_tensor, op=torch.distributed.ReduceOp.SUM, stream=torch.cuda.current_stream())

    def benchmark(self, n_warmup=10, n_graph_launches=10, n_iter_per_graph=100):
        low = 5 * 1024
        high = 80 * 1024 * 1024
        sizes = []
        curr = low
        while curr <= high:
            sizes.append(curr)
            curr *= 2

        if self.rank == 0:
            print(f"{'Size (Bytes)':<20} {'Time (us)':<20} {'AlgoBW (GB/s)':<20}")

        dtype = self.dtype
        capture_stream = torch.cuda.Stream()

        # Allocate a single large RawGpuBuffer (symmetric memory) and reuse it for all sizes.
        # Cannot allocate per-size tensors with symmetric memory.
        bench_buf = mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(dtype))
        bench_buf = torch.utils.dlpack.from_dlpack(bench_buf)
        if dtype in (torch.float16, torch.float32, torch.bfloat16):
            bench_buf.normal_()
        else:
            bench_buf.fill_(0)

        for size in sizes:
            n_elements = size // bench_buf.element_size()
            tensor = bench_buf[:n_elements]

            capture_stream.wait_stream(torch.cuda.current_stream())
            # Capture Graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=capture_stream):
                for _ in range(n_iter_per_graph):
                    self.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

            # warmup: Execute the graph once to prime the driver
            with torch.cuda.stream(capture_stream):
                for _ in range(n_warmup):
                    g.replay()
                self.barrier()
            capture_stream.synchronize()

            # Benchmark
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record(capture_stream)
            with torch.cuda.stream(capture_stream):
                for _ in range(n_graph_launches):
                    g.replay()
            end_event.record(capture_stream)
            end_event.synchronize()

            # Get elapsed time in milliseconds
            elapsed_ms = start_event.elapsed_time(end_event)
            avg_time_ms = elapsed_ms / (n_graph_launches * n_iter_per_graph)
            time_us = avg_time_ms * 1000

            alg_bw = size / (avg_time_ms * 1e-3) if avg_time_ms > 0 else 0
            if self.rank == 0:
                print(f"{size:<20} {time_us:<20.2f} {alg_bw / 1e9:<20.2f}")

    def destroy(self):
        self._algorithm_nvls_packet = None
        self._algorithm_fullmesh = None
        self._algorithm_packet = None
        self._algorithm_rsag_zero_copy = None
        if self._nvls_supported:
            self._algorithm_nvls_zero_copy = None
        self._barrier_tensor = None
        self.scratch_buffer = None
        self.comm = None


def init_dist() -> mscclpp.CommGroup:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MSCCLPP_MASTER_ADDR"]
    master_port = os.environ["MSCCLPP_MASTER_PORT"]
    interface = interfaces_for_ip_netifaces(master_addr)
    if interface is None:
        raise ValueError(f"Cannot find network interface for IP address {master_addr}")
    interfaceIpPortTrio = f"{interface}:{master_addr}:{master_port}"
    return mscclpp.CommGroup(interfaceIpPortTrio=interfaceIpPortTrio, rank=rank, size=world)


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
    }
    accum_dtype_str = os.environ.get("ACCUM_DTYPE", None)
    accum_dtype = accum_dtype_map.get(accum_dtype_str) if accum_dtype_str else None

    comm_group = init_dist()
    custom_comm = CustomizedComm(comm_group, dtype=dtype, accum_dtype=accum_dtype)
    custom_comm.benchmark(n_warmup=5, n_graph_launches=10, n_iter_per_graph=100)
    custom_comm.barrier()
    torch.cuda.synchronize()
    custom_comm.destroy()
    print(f"rank {local} All-reduce operation completed successfully.")


if __name__ == "__main__":
    main()
