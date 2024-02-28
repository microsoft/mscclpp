import os
from mpi4py import MPI
import torch
from cupy.cuda import nccl

ROOT_RANK = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

is_group_root = rank == ROOT_RANK

world_size = comm.Get_size()

os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

device_type = "cuda"
torch.cuda.set_device(0)
device_index = 0
device = torch.device(type=device_type, index=device_index)

if is_group_root:
    id_ = nccl.get_unique_id()
else:
    id_ = None

ranks = range(world_size)
id_, ranks = comm.bcast((id_, ranks), root=0)
group = nccl.NcclCommunicator(len(ranks), id_, rank)
print(f"{rank=}, {device=}, {group=}")

M = 1024
N = 4096
K = 2048
shape_a = (M, K)
shape_b = (K, N)
shape_c = (M, N)

a = torch.ones(shape_a, device="cuda")
b = torch.ones(shape_b, device="cuda")
c = torch.mm(a, b)

print(c)

# nccl_op = nccl.NCCL_SUM
# group.allReduce(
#     sendbuf=c.data_ptr(),
#     recvbuf=c.data_ptr(),
#     count=c.nelement(),
#     datatype=nccl.NCCL_FLOAT,
#     op=nccl_op,
#     stream=torch.cuda.current_stream().cuda_stream)

# print(c)

d = torch.ones((1024 * 1024,), device="cuda")
e = torch.zeros((8 * 1024 * 1024,), device="cuda")
e[rank * 1024 * 1024 : (rank + 1) * 1024 * 1024] = d

group.allGather(
    sendbuf=d.data_ptr(),
    recvbuf=e.data_ptr(),
    count=d.nelement(),
    datatype=nccl.NCCL_FLOAT,
    stream=torch.cuda.current_stream().cuda_stream,
)

print(e)
