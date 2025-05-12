// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_COMMUNICATOR_HPP_
#define MSCCLPP_COMMUNICATOR_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <unordered_map>
#include <vector>

#include "utils_internal.hpp"

namespace mscclpp {

class BaseRecvItem {
 public:
  virtual ~BaseRecvItem() = default;
  virtual void wait() = 0;
};

template <typename T>
class RecvItem : public BaseRecvItem {
 public:
  RecvItem(std::shared_future<T> future) : future_(future) {}

  void wait() { future_.wait(); }

 private:
  std::shared_future<T> future_;
};

struct ConnectionInfo {
  int remoteRank;
  int tag;
};

struct Communicator::Impl {
  std::shared_ptr<Bootstrap> bootstrap_;
  std::shared_ptr<Context> context_;
  std::unordered_map<const Connection*, ConnectionInfo> connectionInfos_;
  std::shared_ptr<BaseRecvItem> lastRecvItem_;

  Impl(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context);

  struct Connector;
};

}  // namespace mscclpp

#endif  // MSCCLPP_COMMUNICATOR_HPP_
