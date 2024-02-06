// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "api.h"
#include "nccl.h"
#include "debug.h"

#include <mscclpp/core.hpp>

namespace mscclpp {

MSCCLPP_API ncclResult_t ncclGetVersion(int *version) {
  if (version == nullptr) {
    WARN("version is nullptr");
    return ncclInvalidArgument;
  }
  *version = MSCCLPP_VERSION;
  return ncclSuccess;
}

MSCCLPP_API ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  if (uniqueId == nullptr) {
    WARN("uniqueId is nullptr");
    return ncclInvalidArgument;
  }
  if (MSCCLPP_UNIQUE_ID_BYTES != NCCL_UNIQUE_ID_BYTES) {
    WARN("UNIQUE_ID_BYTES mismatch");
    return ncclInternalError;
  }
  
}

}  // namespace mscclpp
