#include <cuda_runtime.h>
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>
// #include <nccl.h>
#include <unistd.h>
#include <cudaTypedefs.h>
#include "ipcsocket.cc"

#define CUCHECK(cmd) do {               \
    auto err = cmd;                     \
    if( err != 0 ) {                    \
        printf("Cuda failure %d: Line %d", err, __LINE__); \
    }                                   \
} while(false)

//AR kernel snippet for sm_90 only

#define MULTIMEM_ST(val, ptr)                                                  \
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),    \
               "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)                  \
               : "memory");
//specific PTX for fp16 reduction. bf16 would be multimem.ld_reduce.global.add.v4.bf16x2 etc
#define MULTIMEM_LD(val, ptr)                                                  \
  asm("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"            \
      : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                     \
      : "l"(ptr)                                                               \
      : "memory");

__global__ void testing2(float* uc_ptr){
  uc_ptr[0] = 1.0;
  printf("ptr -> %f\n", uc_ptr[0]);
}


__global__ void testing(float* mc_ptr, int numlines, int myrank, int RANKS){
  //for allreduce we dont even need an UC pointer. just using same mc_ptr for in-place reduction
  //line is assumed to be 16B 4 ints of 8 halves
  const int start_elem =  threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x);
  const int end_elem = max(start_elem, numlines);
  const int lineoffset = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int loop_step0 = (blockDim.x * gridDim.x) * 4;
  __syncthreads();
  printf("start %d, end %d step %d\n", start_elem, end_elem, loop_step0);
    for (int line = start_elem; line < end_elem; line += loop_step0) {
      uint4 val;
      MULTIMEM_LD(val, mc_ptr + (lineoffset + line));
      printf("val %f\n", *(float*)&(val.x));
      MULTIMEM_ST(val, mc_ptr + (lineoffset + line));
    }
  __syncthreads();
}

int main(){
  int myrank, nranks;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
 
  cudaSetDevice(myrank);
  CUresult res;

  size_t size = 1024*1024*512*3;
  CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
 
  CUmulticastObjectProp mcProp = {};
  mcProp.numDevices = nranks;
  mcProp.size = size;
  mcProp.handleTypes = handleType;
 
  size_t minGran, gran;
  gran = 0;
  minGran = 0;
  CUCHECK(cuMulticastGetGranularity(&minGran, &mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));
  CUCHECK(cuMulticastGetGranularity(&gran, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));

  // printf("gran = %lld, minGrad = %lld\n", gran, minGran); 
  size_t mcSize = ((size+gran-1)/gran)*gran;
  mcProp.size = mcSize;

  CUmemGenericAllocationHandle handle;
  //only one rank creates the multicast object
  if(!myrank) CUCHECK(cuMulticastCreate(&handle, &mcProp));
 
  int fd, peerfd;
  fd = 0;
  peerfd = 0;
  if(!myrank) CUCHECK(cuMemExportToShareableHandle(&fd, handle, handleType, 0 /*flags*/));
 
  //some ugly UDS business
  // Borrow ipcsocket.{c,h} from nccl code
  //in cuda 12.4 new fabric handle type is available so instead it would be possible to use MPI_Allgather for the exported handles
  // moreover it would the only way to do it on GraceHopper systems, since UDS is limited to single Unix node
 
  volatile uint32_t abortFlag = 0;
 struct ncclIpcSocket ipcSock = { 0 };
 uint64_t opId=0xdeadcafebeef;
  // ncclResult_t ret = ncclSuccess;

  ncclIpcSocketInit(&ipcSock, myrank, (uint64_t)opId, &abortFlag);
  MPI_Barrier(MPI_COMM_WORLD);
  if(!myrank) {
    for(int p=1;p<nranks;p++) {
      ncclIpcSocketSendFd(&ipcSock, fd, p, (uint64_t)opId);
    } 
  } else {
      ncclIpcSocketRecvFd(&ipcSock, &peerfd);
  }
  ncclIpcSocketClose(&ipcSock);

  printf("fd = %d peerfd = %d\n", fd, peerfd);
  // MPI_Bcast(&fd, sizeof(fd), MPI_CHAR, 0, MPI_COMM_WORLD);
  //everyone else would now have same multicast object
  if(myrank)  CUCHECK(cuMemImportFromShareableHandle(&handle, (void *)peerfd, handleType));
 
//  if(myrank)
//    close(peerfd);
//  else
    close(fd);
  //end of ugly UDS business
#if 1
  //everyone adds device(s), no syncs required, just need to ensure bindmem happens after all this is called
  int mydev = myrank;
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
  void* uc_va;
  void* mc_va;
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = mydev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
 
    // Map a VA to UC space
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)&uc_va, size, minGran, 0U, 0));
    cudaMemset(uc_va, 0, size);
    CUCHECK(cuMemMap((CUdeviceptr)uc_va, size, 0, memhandle, 0));
    // set access on UC address
    CUCHECK(cuMemSetAccess((CUdeviceptr)uc_va, size, &accessDesc, 1));
 
  // Map a VA to MC space
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)&mc_va, mcSize, minGran, 0U, 0));
  CUCHECK(cuMemMap((CUdeviceptr)mc_va, mcSize, 0, handle, 0));
  // set access on MC address
  CUCHECK(cuMemSetAccess((CUdeviceptr)mc_va, mcSize, &accessDesc, 1));
#endif
  printf("Yoooo\n");
  testing2<<<1, 1>>>((float*)mc_va);
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  testing<<<1, 1>>>((float*)mc_va, 1, myrank, nranks);
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
} 
//........
 

