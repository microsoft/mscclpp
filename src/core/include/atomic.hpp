// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ATOMIC_HPP_
#define MSCCLPP_ATOMIC_HPP_

// On CUDA host-side compiles, force atomic_device.hpp's CUDA branch so host code uses
// cuda::atomic_ref (for system-scope ordering with GPU readers). On CUDA device compiles
// (MSCCLPP_DEVICE_CUDA already set by device.hpp) and on ROCm builds, include normally —
// atomic_device.hpp's branch selection works correctly without forcing.
#if defined(MSCCLPP_USE_CUDA) && !defined(MSCCLPP_DEVICE_CUDA)
#define MSCCLPP_DEVICE_CUDA
#include <mscclpp/atomic_device.hpp>
#undef MSCCLPP_DEVICE_CUDA
#else
#include <mscclpp/atomic_device.hpp>
#endif

#endif  // MSCCLPP_ATOMIC_HPP_