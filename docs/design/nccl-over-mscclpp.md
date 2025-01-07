# NCCL Over MSCCL++

(limitations)=
## Limitations

Current NCCL over MSCCL++ has a few limitations.

* We do not cover all APIs yet. See the [API Support Table](#api-support-table) for details.
* Multi-node communication is not supported yet.
* Currently, collective communication functions may not work correctly if the buffer address is differed from that of previous function calls while sharing the same base address (returned by [cuMemGetAddressRange](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b)) with the previous address. This is because the current implementation performs zero-copy communication over user buffers, and it is difficult to efficiently inform all ranks if the buffer address dynamically changes.

(api-support-table)=
## API Support Table

The table below lists all NCCL APIs (v2.21). We may cover more APIs in the future.

| API Name                 | Supported |
| :----------------------- | :-------: |
| ncclGetLastError         | X         |
| ncclGetErrorString       | O         |
| ncclGetVersion           | O         |
| ncclGetUniqueId          | O         |
| ncclCommInitRank         | O         |
| ncclCommInitAll          | X         |
| ncclCommInitRankConfig   | X         |
| ncclCommSplit            | X         |
| ncclCommFinalize         | O         |
| ncclCommDestroy          | O         |
| ncclCommAbort            | X         |
| ncclCommGetAsyncError    | O         |
| ncclCommCount            | O         |
| ncclCommCuDevice         | O         |
| ncclCommUserRank         | O         |
| ncclCommRegister         | X         |
| ncclCommDeregister       | X         |
| ncclMemAlloc             | X         |
| ncclMemFree              | X         |
| ncclAllReduce            | O         |
| ncclBroadcast            | X         |
| ncclReduce               | X         |
| ncclAllGather            | O         |
| ncclReduceScatter        | X         |
| ncclGroupStart           | O         |
| ncclGroupEnd             | O         |
| ncclSend                 | X         |
| ncclRecv                 | X         |
| ncclRedOpCreatePreMulSum | X         |
| ncclRedOpDestroy         | X         |

## Executor Support

The executor is a versatile tool designed to specify how mscclpp executes algorithms. Currently, only the allReduce operation allows for algorithm customization. The following environment variables can be managed:

- MSCCLPP_EXECUTION_PLAN_DIR: Specifies the directory where the executor will look for JSON files.

```{figure} ../figs/size_boundary_diagram.png
:name: MMSCCL++ Abstractions
:alt: MSCCL++ Abstractions
:align: center

Decision Flowchart for Message Size-Based Algorithm Execution
```

This is an example of executing the interface with the executor:
``` bash
mpirun -np 8 -x MSCCLPP_EXECUTION_PLAN_DIR=/root/azure-mscclpp/nccl/test/execution-files ./apps/nccl/test/nccl_api_test
```
