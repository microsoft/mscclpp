// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONNECTION_KERNEL_HPP_
#define MSCCLPP_CONNECTION_KERNEL_HPP_

#include <mscclpp/gpu.hpp>

namespace mscclpp {

cudaError_t connectionAtomicAdd(uint64_t *dst, uint64_t value, cudaStream_t stream);

}  // namespace mscclpp

#endif  // MSCCLPP_CONNECTION_KERNEL_HPP_
