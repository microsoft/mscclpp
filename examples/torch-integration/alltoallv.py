# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# AllToAllV implementation for MSCCLPP
# This module provides a PyTorch-compatible alltoallv operation using MSCCLPP.
#
# Usage:
#   MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> \
#   torchrun --nnodes=1 --nproc_per_node=8 alltoallv.py
#
# For AMD GPUs:
#   MSCCLPP_MASTER_ADDR=<master_ip> MSCCLPP_MASTER_PORT=<port> \
#   GPU_MAX_HW_QUEUES=7 torchrun --nnodes=1 --nproc_per_node=8 alltoallv.py

import mscclpp
import mscclpp.utils as mscclpp_utils
import torch
import os
import netifaces as ni
import ipaddress
from typing import List, Optional

_abs_path = os.path.dirname(os.path.abspath(__file__))


def interfaces_for_ip_netifaces(ip: str):
    """Find the network interface for a given IP address."""
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


class AllToAllVComm:
    """
    AllToAllV communication class using MSCCLPP.
    
    This class provides a customized alltoallv implementation that handles
    variable element counts per rank, similar to MPI_Alltoallv or the
    batch_all_to_all_v pattern commonly used in MOE (Mixture of Experts) models.
    
    Unlike NCCL's ncclGroupStart/ncclGroupEnd approach, MSCCLPP uses explicit
    put/signal/wait operations on PortChannels for communication.
    
    Attributes:
        comm: MSCCLPP CommGroup instance
        rank: Current rank
        world_size: Total number of ranks
    """
    
    def __init__(self, comm: mscclpp.CommGroup):
        """
        Initialize AllToAllV communication.
        
        Args:
            comm: MSCCLPP CommGroup instance
        """
        self.comm = comm
        self.rank = comm.my_rank
        self.world_size = comm.nranks
        self.local_rank = comm.my_rank % comm.nranks_per_node
        self.n_ranks_per_node = comm.nranks_per_node
        self.executor = mscclpp.Executor(comm.communicator)
        
        # Compile and load the native CUDA kernel
        mscclpp_native = mscclpp.compile_native(
            name="mscclpp_alltoallv",
            file=os.path.join(_abs_path, "alltoallv_kernel.cu")
        )
        capsule = mscclpp_native.create_alltoallv_algorithm()
        self.algorithm = mscclpp.Algorithm.create_from_native_capsule(capsule)
    
    def alltoallv(
        self,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
        send_counts: torch.Tensor,
        send_displs: torch.Tensor,
        recv_counts: torch.Tensor,
        recv_displs: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None
    ):
        """
        Perform alltoallv operation with variable element counts.
        
        This function exchanges data between all ranks where each rank can send
        different amounts of data to each other rank.
        
        Args:
            send_tensor: Source tensor containing all data to be sent
            recv_tensor: Destination tensor for received data
            send_counts: Tensor of shape [world_size] with byte counts to send to each rank
            send_displs: Tensor of shape [world_size] with byte offsets in send_tensor for each rank
            recv_counts: Tensor of shape [world_size] with byte counts to receive from each rank
            recv_displs: Tensor of shape [world_size] with byte offsets in recv_tensor for each rank
            stream: Optional CUDA stream to use for the operation
            
        Note:
            - All count and displacement tensors should be on GPU and contain size_t values
            - send_counts[i] is the number of bytes to send to rank i
            - send_displs[i] is the byte offset in send_tensor for data going to rank i
            - recv_counts[i] is the number of bytes to receive from rank i
            - recv_displs[i] is the byte offset in recv_tensor for data from rank i
        """
        # Ensure counts and displacements are on GPU and have correct dtype
        assert send_counts.device.type == "cuda", "send_counts must be on GPU"
        assert send_displs.device.type == "cuda", "send_displs must be on GPU"
        assert recv_counts.device.type == "cuda", "recv_counts must be on GPU"
        assert recv_displs.device.type == "cuda", "recv_displs must be on GPU"
        
        # Prepare extras dict with device pointers for counts and displacements
        extras = {
            "sendCounts": send_counts.data_ptr(),
            "sendDispls": send_displs.data_ptr(),
            "recvCounts": recv_counts.data_ptr(),
            "recvDispls": recv_displs.data_ptr(),
        }
        
        cuda_stream = stream.cuda_stream if stream is not None else 0
        
        self.algorithm.execute(
            self.comm.communicator,
            send_tensor.data_ptr(),
            recv_tensor.data_ptr(),
            send_tensor.nbytes,
            recv_tensor.nbytes,
            mscclpp_utils.torch_dtype_to_mscclpp_dtype(send_tensor.dtype),
            stream=cuda_stream,
            extras=extras
        )
    
    def alltoallv_by_elements(
        self,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
        send_counts_elements: torch.Tensor,
        recv_counts_elements: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None
    ):
        """
        Convenience function for alltoallv with element counts (not byte counts).
        
        This function automatically computes byte counts and displacements from
        element counts, making it easier to use with typical MOE patterns.
        
        Args:
            send_tensor: Source tensor containing all data to be sent
            recv_tensor: Destination tensor for received data
            send_counts_elements: Tensor of shape [world_size] with element counts to send to each rank
            recv_counts_elements: Tensor of shape [world_size] with element counts to receive from each rank
            stream: Optional CUDA stream to use for the operation
        """
        element_size = send_tensor.element_size()
        
        # Convert element counts to byte counts
        send_counts = send_counts_elements.to(torch.int64) * element_size
        recv_counts = recv_counts_elements.to(torch.int64) * element_size
        
        # Compute displacements (exclusive prefix sum)
        send_displs = torch.zeros(self.world_size, dtype=torch.int64, device=send_tensor.device)
        recv_displs = torch.zeros(self.world_size, dtype=torch.int64, device=recv_tensor.device)
        
        if self.world_size > 1:
            send_displs[1:] = torch.cumsum(send_counts[:-1], dim=0)
            recv_displs[1:] = torch.cumsum(recv_counts[:-1], dim=0)
        
        self.alltoallv(
            send_tensor, recv_tensor,
            send_counts, send_displs,
            recv_counts, recv_displs,
            stream
        )
    
    def barrier_cpu(self):
        """CPU barrier to synchronize all ranks."""
        self.comm.barrier()


def batch_alltoallv(
    comm: AllToAllVComm,
    inputs: List[torch.Tensor],
    outputs: List[torch.Tensor],
    in_sizes: torch.Tensor,
    out_sizes: torch.Tensor,
    stream: Optional[torch.cuda.Stream] = None
):
    """
    Batch all-to-all-v operation for multiple tensors.
    
    This function replicates the pattern from MOE implementations:
    ```
    for k in range(len(inputs)):
        ncclGroupStart()
        for i in range(world_size):
            ncclSend(in_buff + in_offset, in_sizes[i], ...)
            ncclRecv(out_buff + out_offset, out_sizes[i], ...)
        ncclGroupEnd()
    ```
    
    Since MSCCLPP doesn't support ncclGroupStart/ncclGroupEnd, we implement
    this using explicit alltoallv operations for each tensor in the batch.
    
    Args:
        comm: AllToAllVComm instance
        inputs: List of input tensors to send
        outputs: List of output tensors to receive
        in_sizes: Tensor of shape [world_size] with element counts to send to each rank
        out_sizes: Tensor of shape [world_size] with element counts to receive from each rank
        stream: Optional CUDA stream
    """
    assert len(inputs) == len(outputs), "Input and output lists must have same length"
    
    # Ensure sizes are on CPU for computing displacements
    in_sizes_cpu = in_sizes.cpu().to(torch.int64)
    out_sizes_cpu = out_sizes.cpu().to(torch.int64)
    
    for k in range(len(inputs)):
        input_tensor = inputs[k]
        output_tensor = outputs[k]
        
        element_size = input_tensor.element_size()
        
        # Compute byte counts
        send_counts = (in_sizes_cpu * element_size).cuda()
        recv_counts = (out_sizes_cpu * element_size).cuda()
        
        # Compute displacements
        send_displs = torch.zeros(comm.world_size, dtype=torch.int64, device="cuda")
        recv_displs = torch.zeros(comm.world_size, dtype=torch.int64, device="cuda")
        
        if comm.world_size > 1:
            send_displs_cpu = torch.zeros(comm.world_size, dtype=torch.int64)
            recv_displs_cpu = torch.zeros(comm.world_size, dtype=torch.int64)
            send_displs_cpu[1:] = torch.cumsum(send_counts.cpu()[:-1], dim=0)
            recv_displs_cpu[1:] = torch.cumsum(recv_counts.cpu()[:-1], dim=0)
            send_displs = send_displs_cpu.cuda()
            recv_displs = recv_displs_cpu.cuda()
        
        comm.alltoallv(
            input_tensor, output_tensor,
            send_counts, send_displs,
            recv_counts, recv_displs,
            stream
        )


def main():
    """Test the alltoallv implementation."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    torch.cuda.set_device(local_rank)
    
    master_addr = os.environ["MSCCLPP_MASTER_ADDR"]
    master_port = os.environ["MSCCLPP_MASTER_PORT"]
    
    interface = interfaces_for_ip_netifaces(master_addr)
    if interface is None:
        raise ValueError(f"Cannot find network interface for IP address {master_addr}")
    
    interfaceIpPortTrio = f"{interface}:{master_addr}:{master_port}"
    mscclpp_group = mscclpp.CommGroup(interfaceIpPortTrio=interfaceIpPortTrio, rank=rank, size=world_size)
    
    # Create communicator
    comm = AllToAllVComm(mscclpp_group)
    
    # Test with variable sizes per rank
    # Each rank sends different amounts to different peers
    # For simplicity: rank i sends (i+1)*100 elements to each peer
    elements_per_peer = (rank + 1) * 100
    
    # Create send buffer with data
    total_send = elements_per_peer * world_size
    send_tensor = torch.arange(total_send, device="cuda", dtype=torch.float32) + rank * 10000
    
    # Receive buffer needs to accommodate variable amounts from each sender
    # Each sender j sends (j+1)*100 elements to us
    recv_counts_cpu = torch.tensor([(j + 1) * 100 for j in range(world_size)], dtype=torch.int64)
    total_recv = recv_counts_cpu.sum().item()
    recv_tensor = torch.zeros(total_recv, device="cuda", dtype=torch.float32)
    
    # Send counts: we send elements_per_peer to everyone
    send_counts_cpu = torch.tensor([elements_per_peer] * world_size, dtype=torch.int64)
    
    # Move to GPU
    send_counts = send_counts_cpu.cuda()
    recv_counts = recv_counts_cpu.cuda()
    
    comm.barrier_cpu()
    
    # Perform alltoallv
    comm.alltoallv_by_elements(
        send_tensor, recv_tensor,
        send_counts, recv_counts,
        stream=torch.cuda.current_stream()
    )
    
    torch.cuda.synchronize()
    comm.barrier_cpu()
    
    print(f"Rank {rank}: alltoallv completed successfully!")
    print(f"  Sent {total_send} elements, received {total_recv} elements")
    
    # Cleanup
    comm = None


if __name__ == "__main__":
    main()
