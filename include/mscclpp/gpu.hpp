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
using cudaEvent_t = hipEvent_t;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using cudaMemcpyKind = hipMemcpyKind;
using cudaIpcMemHandle_t = hipIpcMemHandle_t;

using CUresult = hipError_t;
using CUdeviceptr = hipDeviceptr_t;

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
#define cudaEventCreate(...) hipEventCreate(__VA_ARGS__)
#define cudaEventDestroy(...) hipEventDestroy(__VA_ARGS__)
#define cudaEventRecord(...) hipEventRecord(__VA_ARGS__)
#define cudaEventElapsedTime(...) hipEventElapsedTime(__VA_ARGS__)

#define cuGetErrorString(...) hipDrvGetErrorString(__VA_ARGS__)
#define cuMemGetAddressRange(...) hipMemGetAddressRange(__VA_ARGS__)

#else

#include <cuda.h>
#include <cuda_runtime.h>

#endif

#endif  // MSCCLPP_GPU_HPP_
