// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GPU_HPP_
#define MSCCLPP_GPU_HPP_

#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaStream_t = hipStream_t;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using cudaMemcpyKind = hipMemcpyKind;
using cudaIpcMemHandle_t = hipIpcMemHandle_t;

using CUresult = hipError_t;
using CUdeviceptr = hipDeviceptr_t;
using CUmemGenericAllocationHandle = hipMemGenericAllocationHandle_t;
using CUmemAllocationProp = hipMemAllocationProp;
using CUmemAccessDesc = hipMemAccessDesc;
using CUmemAllocationHandleType = hipMemAllocationHandleType;

constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaStreamNonBlocking = hipStreamNonBlocking;
constexpr auto cudaStreamCaptureModeGlobal = hipStreamCaptureModeGlobal;
constexpr auto cudaStreamCaptureModeRelaxed = hipStreamCaptureModeRelaxed;
constexpr auto cudaHostAllocMapped = hipHostMallocMapped;
constexpr auto cudaHostAllocWriteCombined = hipHostMallocWriteCombined;
constexpr auto cudaMemcpyDefault = hipMemcpyDefault;
constexpr auto cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto cudaIpcMemLazyEnablePeerAccess = hipIpcMemLazyEnablePeerAccess;

constexpr auto CU_MEM_ALLOCATION_TYPE_PINNED = hipMemAllocationTypePinned;
constexpr auto CU_MEM_LOCATION_TYPE_DEVICE = hipMemLocationTypeDevice;
constexpr auto CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = hipMemHandleTypePosixFileDescriptor;
constexpr auto CU_MEM_ACCESS_FLAGS_PROT_READWRITE = hipMemAccessFlagsProtReadWrite;

#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS hipSuccess
#endif  // CUDA_SUCCESS

#define cudaGetErrorString(...) hipGetErrorString(__VA_ARGS__)
#define cudaGetDevice(...) hipGetDevice(__VA_ARGS__)
#define cudaGetDeviceCount(...) hipGetDeviceCount(__VA_ARGS__)
#define cudaGetDeviceProperties(...) hipGetDeviceProperties(__VA_ARGS__)
#define cudaGetLastError(...) hipGetLastError(__VA_ARGS__)
#define cudaSetDevice(...) hipSetDevice(__VA_ARGS__)
#define cudaDeviceSynchronize(...) hipDeviceSynchronize(__VA_ARGS__)
#define cudaDeviceGetPCIBusId(...) hipDeviceGetPCIBusId(__VA_ARGS__)
#define cudaHostAlloc(...) hipHostMalloc(__VA_ARGS__)
#define cudaMalloc(...) hipMalloc(__VA_ARGS__)
#define cudaFree(...) hipFree(__VA_ARGS__)
#define cudaFreeHost(...) hipHostFree(__VA_ARGS__)
#define cudaMemset(...) hipMemset(__VA_ARGS__)
#define cudaMemsetAsync(...) hipMemsetAsync(__VA_ARGS__)
#define cudaMemcpy(...) hipMemcpy(__VA_ARGS__)
#define cudaMemcpyAsync(...) hipMemcpyAsync(__VA_ARGS__)
#define cudaMemcpyToSymbol(...) hipMemcpyToSymbol(__VA_ARGS__)
#define cudaMemcpyToSymbolAsync(...) hipMemcpyToSymbolAsync(__VA_ARGS__)
#define cudaStreamCreate(...) hipStreamCreate(__VA_ARGS__)
#define cudaStreamCreateWithFlags(...) hipStreamCreateWithFlags(__VA_ARGS__)
#define cudaStreamSynchronize(...) hipStreamSynchronize(__VA_ARGS__)
#define cudaStreamBeginCapture(...) hipStreamBeginCapture(__VA_ARGS__)
#define cudaStreamEndCapture(...) hipStreamEndCapture(__VA_ARGS__)
#define cudaStreamDestroy(...) hipStreamDestroy(__VA_ARGS__)
#define cudaGraphInstantiate(...) hipGraphInstantiate(__VA_ARGS__)
#define cudaGraphLaunch(...) hipGraphLaunch(__VA_ARGS__)
#define cudaGraphDestroy(...) hipGraphDestroy(__VA_ARGS__)
#define cudaGraphExecDestroy(...) hipGraphExecDestroy(__VA_ARGS__)
#define cudaThreadExchangeStreamCaptureMode(...) hipThreadExchangeStreamCaptureMode(__VA_ARGS__)
#define cudaIpcGetMemHandle(...) hipIpcGetMemHandle(__VA_ARGS__)
#define cudaIpcOpenMemHandle(...) hipIpcOpenMemHandle(__VA_ARGS__)
#define cudaIpcCloseMemHandle(...) hipIpcCloseMemHandle(__VA_ARGS__)

#define cuGetErrorString(...) hipDrvGetErrorString(__VA_ARGS__)
#define cuMemAddressReserve(...) hipMemAddressReserve(__VA_ARGS__)
#define cuMemAddressFree(...) hipMemAddressFree(__VA_ARGS__)
#define cuMemGetAddressRange(...) hipMemGetAddressRange(__VA_ARGS__)
#define cuMemCreate(...) hipMemCreate(__VA_ARGS__)
#define cuMemRelease(...) hipMemRelease(__VA_ARGS__)
#define cuMemSetAccess(...) hipMemSetAccess(__VA_ARGS__)
#define cuMemMap(...) hipMemMap(__VA_ARGS__)
#define cuMemUnmap(...) hipMemUnmap(__VA_ARGS__)
#define cuMemRetainAllocationHandle(...) hipMemRetainAllocationHandle(__VA_ARGS__)
#define cuMemExportToShareableHandle(...) hipMemExportToShareableHandle(__VA_ARGS__)
#define cuMemImportFromShareableHandle(...) hipMemImportFromShareableHandle(__VA_ARGS__)

#else

#include <cuda.h>
#include <cuda_runtime.h>

#endif

// NVLS
#if !defined(__HIP_PLATFORM_AMD__)
#include <linux/version.h>
#if CUDART_VERSION < 12030
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#endif
// We need CUDA 12.3 above and kernel 5.6.0 above for NVLS API
#define CUDA_NVLS_API_AVAILABLE ((CUDART_VERSION >= 12030) && (LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)))
#else  // defined(__HIP_PLATFORM_AMD__)
#define CUDA_NVLS_API_AVAILABLE 0
// NVLS is not supported on AMD platform, just to avoid compilation error
#define CU_MEM_HANDLE_TYPE_FABRIC (0x8ULL)
#endif  // !defined(__HIP_PLATFORM_AMD__)

// GPU sync threads
#if defined(__HIP_PLATFORM_AMD__)
#define __syncshm() asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
#else
#define __syncshm() __syncthreads();
#endif

#endif  // MSCCLPP_GPU_HPP_
