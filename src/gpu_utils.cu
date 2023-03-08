#include "gpu_utils.h"

__global__ void kernelCopyFlag(void* flags, int num) {
  volatile uint64_t **pFlags = (volatile uint64_t **)flags;
  if (threadIdx.x < num) {
    *pFlags[2 * threadIdx.x + 1] = *pFlags[2 * threadIdx.x];
  }
}

cudaError_t copyFlag(uint64_t** flags, int num, cudaStream_t stream) {
  kernelCopyFlag<<<1, 32, 0, stream>>>(flags, num);
  return cudaGetLastError();
}
