// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FLAG_HPP_
#define MSCCLPP_FLAG_HPP_

#include <mscclpp/core.hpp>

namespace mscclpp {

struct Flag::Impl {
  Impl(std::shared_ptr<Connection> connection, Device device, Context& context);

  std::shared_ptr<Connection> connection_;
  Device device_;
  std::shared_ptr<uint64_t> id_;
  RegisteredMemory idMemory_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FLAG_HPP_
