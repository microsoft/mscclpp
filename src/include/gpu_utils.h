#ifndef MSCCLPP_GPU_UTILS_H_
#define MSCCLPP_GPU_UTILS_H_

#include <cuda_runtime.h>
#include <stdint.h>

cudaError_t copyFlag(uint64_t** flags, int num, cudaStream_t stream);

#endif
