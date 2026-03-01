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
    def __init__(self, comm: mscclpp.CommGroup):
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = comm.my_rank % comm.nranks_per_node
        self.n_ranks_per_node = comm.nranks_per_node
        dlpack = mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(torch.float16))
        self.scratch_buffer = torch.utils.dlpack.from_dlpack(dlpack)
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
        if mscclpp.is_nvls_supported():
            self._algorithm_nvls_zero_copy = [
                algo
                for algo in algorithms
                if algo.collective == "allreduce" and algo.name == "default_allreduce_nvls_zero_copy"
            ][0]
        self._tune(n_warmup=5, n_graph_launches=10, n_ops_per_graph=100)

    def _tune(self, n_warmup, n_graph_launches, n_ops_per_graph):
        sizes = [1 << i for i in range(10, 28)]
        # Pre-fill with defaults for barrier
        self.best_configs = {1024: (self._algorithm_nvls_packet, 0, 0)}

        tune_tensor = mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(torch.float16))
        tune_tensor = torch.utils.dlpack.from_dlpack(tune_tensor)
        tune_tensor.normal_()
        candidates_nblocks = [4, 8, 16, 24, 32, 48, 64, 128]
        candidates_nthreads = [512, 768, 1024]

        for size in sizes:
            algos = []
            if mscclpp.is_nvls_supported():
                algos.append(self._algorithm_nvls_zero_copy)
            if size <= 4 * 1024 * 1024:
                algos.append(self._algorithm_nvls_packet)
                algos.append(self._algorithm_packet)
            if size >= 512 * 1024:
                algos.append(self._algorithm_rsag_zero_copy)

            best_time = float("inf")
            best_config = None

            for algo in algos:
                for nb in candidates_nblocks:
                    if algo.name == "default_allreduce_nvls_packet" and nb > 16:
                        continue
                    if algo.name == "default_allreduce_packet" and nb > 56:
                        continue
                    for nt in candidates_nthreads:
                        if self._run_algo(algo, tune_tensor, size, nb, nt) != 0:
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
        )

    def get_tuned_config(self, size):
        if size < 1024:
            target_size = 1024
        elif size > 256 * 1024 * 1024:
            target_size = 256 * 1024 * 1024
        else:
            target_size = 1 << (size - 1).bit_length()
        return self.best_configs.get(target_size)

    def all_reduce(self, tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM, stream: torch.cuda.Stream = None):
        assert op == torch.distributed.ReduceOp.SUM
        config = self.get_tuned_config(tensor.nbytes)
        algo, nblocks, nthreads = config if config else (self._algorithm_nvls_packet, 0, 0)
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
            symmetric_memory=True,
        )
        if ret != 0:
            print(f"Rank {self.rank}: Algo {algo.name} failed with error {ret}")

    def barrier(self):
        tensor = torch.empty(self.world_size, dtype=torch.float, device=torch.device("cuda"))
        self.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, stream=torch.cuda.current_stream())

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

        dtype = torch.float16
        capture_stream = torch.cuda.Stream()

        # Allocate a single large RawGpuBuffer (symmetric memory) and reuse it for all sizes.
        # Cannot allocate per-size tensors with symmetric memory.
        bench_buf = mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(dtype))
        bench_buf = torch.utils.dlpack.from_dlpack(bench_buf)
        bench_buf.normal_()

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
        self._algorithm_nvls_nonzero_copy = None
        self._algorithm_nvls_packet = None
        self.scratch_buffer = None
        self.comm = None


def init_dist() -> CustomizedComm:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MSCCLPP_MASTER_ADDR"]
    master_port = os.environ["MSCCLPP_MASTER_PORT"]
    interface = interfaces_for_ip_netifaces(master_addr)
    if interface is None:
        raise ValueError(f"Cannot find network interface for IP address {master_addr}")
    interfaceIpPortTrio = f"{interface}:{master_addr}:{master_port}"
    mscclpp_group = mscclpp.CommGroup(interfaceIpPortTrio=interfaceIpPortTrio, rank=rank, size=world)
    return CustomizedComm(mscclpp_group)


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    comm = init_dist()
    comm.benchmark(n_warmup=5, n_graph_launches=10, n_iter_per_graph=100)
    comm.barrier()
    torch.cuda.synchronize()
    comm.destroy()
    print(f"rank {local} All-reduce operation completed successfully.")


if __name__ == "__main__":
    main()
