import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule

class Kernel:
    def __init__(self, args):
        # We need to compile the source code and get kernel here
        self._kernel = SourceModule().get_function(args)

    def launch_kernel(self, nblocks: int, nthreads: int, shared: int, stream: drv.Stream = None):
        self._kernel(grid=(nblocks, 1, 1), block=(nthreads, 1, 1), stream=stream, shared=shared)

class KernelBase:
    def __init__(self, file, args):
        self._defines = args

