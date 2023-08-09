from mpi4py import MPI
from functools import wraps

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

def parametrize_layouts(*layouts):
    # here we launch processes according to the layout
    layouts_list = list(layouts)
    def decorator(func):
        def wrapper(*args, **kwargs):
            for layout in layouts_list:
                n_node = layout[0]
                comm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.rank % layout[1])
                func(comm, *args, **kwargs)
        return wrapper
    return decorator
