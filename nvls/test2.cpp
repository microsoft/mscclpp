#include <cuda_runtime.h>
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>
#include <nccl.h>
#include <unistd.h>
#include <cudaTypedefs.h>

#define CUCHECK(cmd) do {               \
    auto err = cmd;                     \
    if( err != 0 ) {                    \
        printf("Cuda failure %d: Line %d", err, __LINE__); \
    }                                   \
} while(false)

int main(){
  int myrank, nranks;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
 
  cudaSetDevice(myrank);
  CUresult res;


CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
 
  CUmulticastObjectProp mcProp = {};
  mcProp.numDevices = nranks;
  mcProp.size = size;
  mcProp.handleTypes = handleType;
 
  size_t minGran, gran;
  CUCHECK(cuMulticastGetGranularity(&minGran, &mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));
  CUCHECK(cuMulticastGetGranularity(&gran, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
 
  size_t mcSize = ((size+gran-1)/gran)*gran;
  mcProp.size = mcSize;
 
  //only one rank creates the multicast object
  if(!myrank) CUCHECK(cuMulticastCreate(&handle, &mcProp));
 
  int fd, peerfd;
  if(!myrank) CUCHECK(cuMemExportToShareableHandle(&fd, handle, handleType, 0 /*flags*/));
 
  //some ugly UDS business
  // Borrow ipcsocket.{c,h} from nccl code
  //in cuda 12.4 new fabric handle type is available so instead it would be possible to use MPI_Allgather for the exported handles
  // moreover it would the only way to do it on GraceHopper systems, since UDS is limited to single Unix node
 
  volatile uint32_t abortFlag = 0;
  struct ncclIpcSocket ipcSock = { 0 };
  uint64_t opId=0xdeadcafebeef;
  ncclResult_t ret = ncclSuccess;
 
  NCCLCHECK(ncclIpcSocketInit(&ipcSock, myrank, (uint64_t)opId, &abortFlag));
  MPI_Barrier(MPI_COMM_WORLD);
  if(!myrank)
    for(int p=1;p<nranks;p++) {
      NCCLCHECKGOTO(ncclIpcSocketSendFd(&ipcSock, fd, p, (uint64_t)opId), ret, error);
    } else {
      NCCLCHECKGOTO(ncclIpcSocketRecvFd(&ipcSock, &peerfd), ret, error);
  }
  error:
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));
 
  //everyone else would now have same multicast object
  if(myrank)  CUCHECK(cuMemImportFromShareableHandle(&handle, (void *)peerfd, handleType));
 
  if(myrank)
    close(peerfd);
  else
    close(fd);
  //end of ugly UDS business
 
  //everyone adds device(s), no syncs required, just need to ensure bindmem happens after all this is called
  CUCHECK(cuMulticastAddDevice(handle, mydev));
  MPI_Barrier(MPI_COMM_WORLD);
 
  CUmemGenericAllocationHandle memhandle;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = mydev;
  prop.requestedHandleTypes = handleType;
 
  //allocate physical memory (data buffer)
  CUCHECK(cuMemCreate(&memhandle, size, &prop, 0 /*flags*/));
 
  //everyone binds memory to the multicast
  CUCHECK(cuMulticastBindMem(handle, 0 /*mcOffset*/, memhandle, 0 /*memOffset*/, size, 0));
  MPI_Barrier(MPI_COMM_WORLD);
  //usual VA business: map both MC and PA to two different VA addresses
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = mydev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
 
    // Map a VA to UC space
    CUCHECK(cuMemAddressReserve(&uc_va, size, minGran, 0U, 0));
    CUCHECK(cuMemMap(uc_va, size, 0, memhandle, 0));
    // set access on UC address
    CUCHECK(cuMemSetAccess(uc_va, size, &accessDesc, 1));
 
  // Map a VA to MC space
  CUCHECK(cuMemAddressReserve(&mc_va, mcSize, minGran, 0U, 0));
  CUCHECK(cuMemMap(mc_va, mcSize, 0, handle, 0));
  // set access on MC address
  CUCHECK(cuMemSetAccess(mc_va, mcSize, &accessDesc, 1));

  MPI_Finalize();
} 
//........
 
/*
//AR kernel snippet for sm_90 only
 
#if __CUDA_ARCH__ >= 900
#define MULTIMEM_ST(val, ptr)                                                  \
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),    \
               "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)                  \
               : "memory");
//specific PTX for fp16 reduction. bf16 would be multimem.ld_reduce.global.add.v4.bf16x2 etc
#define MULTIMEM_LD(val, ptr)                                                  \
  asm("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"            \
      : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                     \
      : "l"(ptr)                                                               \
      : "memory");
#endif
 
//for allreduce we dont even need an UC pointer. just using same mc_ptr for in-place reduction
//line is assumed to be 16B 4 ints of 8 halves
const int start_elem =  threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x);
const int end_elem = max(start_elem, numlines);
__syncthreads();
  for (int line = start_elem; line < end_elem; line += loop_step0) {
    uint4 val;
    MULTIMEM_LD(val, mc_ptr + (lineoffset + line))
    MULTIMEM_ST(val, mc_ptr + (lineoffset + line))
  }
__syncthreads();
 
*/
