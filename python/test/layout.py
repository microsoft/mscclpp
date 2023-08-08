from mpi4py import MPI

class Layout:
    def __init__(self, world_comm: MPI.Comm):
        self._world_comm = world_comm

    @property
    def rank(self) -> int:
        return self._world_comm.Get_rank()

    @property
    def size(self) -> int:
        return self._world_comm.Get_size()

    def bcast(self, value, root: int):
        return self._world_comm.Bcast(value, 0)

