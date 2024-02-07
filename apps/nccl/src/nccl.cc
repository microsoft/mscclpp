// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "nccl.h"

#include <mscclpp/core.hpp>

#define NCCL_API extern "C" __attribute__((visibility("default")))

NCCL_API ncclResult_t ncclGetVersion(int *version) {
  if (version == nullptr) return ncclInvalidArgument;
  *version = MSCCLPP_VERSION;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  if (uniqueId == nullptr) return ncclInvalidArgument;
  if (MSCCLPP_UNIQUE_ID_BYTES != NCCL_UNIQUE_ID_BYTES) return ncclInternalError;
  return ncclSuccess;
}
