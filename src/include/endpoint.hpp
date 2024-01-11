// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ENDPOINT_HPP_
#define MSCCLPP_ENDPOINT_HPP_

#include <mscclpp/core.hpp>
#include <vector>

#include "ib.hpp"

namespace mscclpp {

struct Endpoint::Impl {
  Impl(EndpointConfig config, Context::Impl& contextImpl);
  Impl(const std::vector<char>& serialization);

  Transport transport_;
  uint64_t hostHash_;

  // The following are only used for IB and are undefined for other transports.
  bool ibLocal_;
  IbQp* ibQp_;
  IbQpInfo ibQpInfo_;

  // These are only defined for multicast (NVLS) capability
  CUmulticastObjectProp mcProp_;
  CUmemGenericAllocationHandle mcHandle_;
  size_t minMcGran_;
  size_t mcGran_;
  int fileDesc_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_ENDPOINT_HPP_
