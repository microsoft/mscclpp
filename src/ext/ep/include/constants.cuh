// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// Shared EP capacity, timeout, alignment, and CUDA type configuration.

#pragma once

#define NUM_MAX_FIFO_SLOTS 32768
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_CPU_TIMEOUT_SECS 100
// Kernel-side spin timeout. Default 200G cycles ≈ 100s on Hopper/Blackwell.
// Define `MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT` (e.g. -DMSCCLPP_EP_KERNEL_DEBUG_TIMEOUT)
// to use a short 10s window suitable for hang triage.
#ifndef NUM_TIMEOUT_CYCLES
#ifdef MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT
#define NUM_TIMEOUT_CYCLES 20000000000ull  // ~10s debug
#else
#define NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s
#endif
#endif
#define NUM_WAIT_NANOSECONDS 500

// Make CLion CUDA indexing work.
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900  // NOLINT(*-reserved-identifier)
#define __CUDACC_RDC__     // NOLINT(*-reserved-identifier)
__host__ __device__ __forceinline__ void host_device_printf(const char* format, ...) { asm volatile("trap;"); }
#define printf host_device_printf
#endif

// Remove Torch restrictions.
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
