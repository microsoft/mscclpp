# MSCCL++

GPU-driven computation & communication stack.

## Quick Start

### Preliminaries

- OS: tested over Ubuntu 18.04 and 20.04
- Libraries: CUDA >= 11.1.1, [gdrcopy](https://github.com/NVIDIA/gdrcopy), [libnuma](https://github.com/numactl/numactl)
- GPUs: A100 (TBU: H100)
- Azure SKUs: [ND_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series), [NDm_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/ndm-a100-v4-series) (TBD: [NC_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nc-a100-v4-series))


### Compile Library

Run `make` in the top directory. To use MPI for test code, pass `MPI_HOME` (`/usr/local/mpi` by default). For example:

```
$ MPI_HOME=/usr/local/mpi make -j
```

If you do not want to use MPI, pass `USE_MPI_FOR_TESTS=0`.

```
# Do not use MPI
$ USE_MPI_FOR_TESTS=0 make -j
```

`make` will create a header file `build/include/mscclpp.h` and a shared library `build/lib/libmscclpp.so`.

### (Optional) Tests

For verification, one can try provided sample code `bootstrap_test` or `p2p_test`. First add the MSCCL++ library path to `LD_LIBRARY_PATH`.

```
$ export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
```

Run tests using MPI:

```
$ mpirun -np 8 ./build/bin/tests/bootstrap_test 127.0.0.1:50000
$ mpirun -np 8 ./build/bin/tests/p2p_test 127.0.0.1:50000
```

If tests are compiled without MPI, pass a rank and the number of ranks as the following example. Usage of `p2p_test` is also the same as `bootstrap_test`.

```
# Terminal 1: Rank 0, #Ranks 2
$ ./build/bin/tests/bootstrap_test 127.0.0.1:50000 0 2
# Terminal 2: Rank 1, #Ranks 2
$ ./build/bin/tests/bootstrap_test 127.0.0.1:50000 1 2
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
