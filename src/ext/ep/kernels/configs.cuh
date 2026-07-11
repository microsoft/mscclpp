// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// Kernel-side configuration. This is the MSCCL++ version of
// `DeepEP/csrc/kernels/configs.cuh` with NVSHMEM / IBGDA / mlx5dv includes
// removed so the intranode (NVLink-only) kernels can be built standalone.
// Include this file **only** from `.cu` files.

#pragma once

// Maximum number of intra-node NVLink peers per RDMA rank.
// - 8 for H100 NVL8 / HGX-style nodes (DeepEP upstream default).
// - 4 for Azure GB200 NVL72 (4 GPUs per NUMA host).
// Configurable via the CMake cache var `MSCCLPP_EP_NUM_MAX_NVL_PEERS`
// (see `src/ext/ep/CMakeLists.txt`). Default keeps DeepEP-parity at 8.
#ifndef NUM_MAX_NVL_PEERS
#define NUM_MAX_NVL_PEERS 8
#endif
#define NUM_MAX_RDMA_PEERS 20
#define NUM_MAX_FIFO_SLOTS 32768
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#ifndef NUM_PORT_CHANNELS_PER_RANK
#define NUM_PORT_CHANNELS_PER_RANK 16
#endif
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

// Phase control for low-latency kernels (internal use in kernel code)
// Public API uses low_latency::Phase enum instead
#define LOW_LATENCY_SEND_PHASE 1  // = low_latency::SEND_ONLY
#define LOW_LATENCY_RECV_PHASE 2  // = low_latency::RECV_ONLY
// (SEND_AND_RECV = 3)

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

// NVSHMEM / IBGDA / mlx5dv are only required for the RDMA internode paths and
// are not included here. The internode/low-latency kernels that need them
// will include them directly under `#ifdef MSCCLPP_EP_HAVE_NVSHMEM`.
