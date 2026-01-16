# C++ Examples

## Tutorials

Check out our [tutorials](../tutorials) for step-by-step guides on using MSCCL++.

(mscclpp-test)=
## mscclpp-test

*NOTE: mscclpp-test is NOT a performance benchmark. If you want to get the latest performance numbers, please use the Python benchmark or the NCCL APIs instead.*

mscclpp-test is a set of C++ implementation examples. It requires MPI on the system, and the path should be provided via `MPI_HOME` environment variable to the CMake build system.

```bash
$ MPI_HOME=/path/to/mpi cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j allgather_test_perf allreduce_test_perf
```

For example, the following command runs the `allreduce5` algorithm with 8 GPUs starting from 3MB to 48MB messages, by doubling the message size in between. You can try different algorithms by changing the `-k 5` option to another value (e.g., `-k 3` runs `allreduce3`). Check all algorithms from the code: [allreduce_test.cu](https://github.com/microsoft/mscclpp/blob/main/test/mscclpp-test/allreduce_test.cu) and [allgather_test.cu](https://github.com/microsoft/mscclpp/blob/main/test/mscclpp-test/allgather_test.cu).

```bash
$ mpirun --bind-to numa -np 8 ./bin/allreduce_test_perf -b 3m -e 48m -G 100 -n 100 -w 20 -f 2 -k 5
```

*NOTE: a few algorithms set a condition on the total data size, such as to be a multiple of 3. If the condition is unmet, the command will throw a regarding error.*

Check the help message for more details.

```bash
$ ./bin/allreduce_test_perf --help
USAGE: allreduce_test_perf
        [-b,--minbytes <min size in bytes>]
        [-e,--maxbytes <max size in bytes>]
        [-i,--stepbytes <increment size>]
        [-f,--stepfactor <increment factor>]
        [-n,--iters <iteration count>]
        [-w,--warmup_iters <warmup iteration count>]
        [-c,--check <0/1>]
        [-T,--timeout <time in seconds>]
        [-G,--cudagraph <num graph launches>]
        [-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>]
        [-k,--kernel_num <kernel number of commnication primitive>]
        [-o, --output_file <output file name>]
        [-h,--help]
```
