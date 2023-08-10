import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

from mpi4py import MPI
import pytest
import atexit
import logging

logging.basicConfig(level=logging.INFO)

def init_mpi():
    if not MPI.Is_initialized():
        MPI.Init()

# Define a function to finalize MPI
def finalize_mpi():
    if MPI.Is_initialized():
        MPI.Finalize()

# Register the function to be called on exit
atexit.register(finalize_mpi)
class Layout:
    def __init__(self, comm: MPI.Comm):
        self.comm = comm

@pytest.fixture
def layout(request):
    if (request.param is None):
        MPI.COMM_WORLD.barrier()
        pytest.skip(f"Skip for None comm {MPI.COMM_WORLD.rank}")
    yield request.param
    MPI.COMM_WORLD.barrier()

def parametrize_layouts(*tuples: tuple):
    def decorator(func):
        layouts = []
        for layout_tuple in list(tuples):
            n_gpus = layout_tuple[0] * layout_tuple[1]
            if MPI.COMM_WORLD.size < n_gpus:
                logging.warning(f"MPI.COMM_WORLD.size < {n_gpus}, skip")
                continue
            world_group = MPI.COMM_WORLD.group
            group = world_group.Incl(list(range(n_gpus)))
            comm = MPI.COMM_WORLD.Create(group)
            if comm == MPI.COMM_NULL:
                layouts.append(None)
            else:
                layouts.append(Layout(comm))
        return pytest.mark.parametrize("layout", layouts, indirect=True)(func)
    return decorator

init_mpi()