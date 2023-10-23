import cupy.cuda.nccl as nccl
from mpi4py import MPI

class NcclOp:
    def __init__(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Create a NCCL unique ID and communicator
        if rank == 0:
            uid = nccl.get_unique_id()
        else:
            uid = None
        uid = comm.bcast(uid, root=0)
        self.nccl_comm = nccl.NcclCommunicator(size, uid, rank)

    def make_callback(self, memory):
        def _make_callback(stream_ptr):
            self.nccl_comm.allReduce(memory.data.ptr, memory.data.ptr, memory.size, nccl.NCCL_FLOAT32, nccl.NCCL_SUM, stream_ptr)
        return _make_callback
