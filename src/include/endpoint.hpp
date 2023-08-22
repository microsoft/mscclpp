// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCL_ENDPOINT_HPP_
#define MSCCL_ENDPOINT_HPP_

#include <mscclpp/core.hpp>
#include <vector>

#include "ib.hpp"

namespace mscclpp {

struct Endpoint::Impl {
  Impl(Transport transport, int ibMaxCqSize, int ibMaxCqPollNum, int ibMaxSendWr, int ibMaxWrPerSend,
       Context::Impl& contextImpl);
  Impl(const std::vector<char>& serialization);

  Transport transport_;
  int rank_;
  uint64_t hostHash_;

  // The following are only used for IB and are undefined for other transports.
  bool ibLocal_;
  IbQp* ibQp_;
  IbQpInfo ibQpInfo_;
};

}  // namespace mscclpp

#endif  // MSCCL_ENDPOINT_HPP_
