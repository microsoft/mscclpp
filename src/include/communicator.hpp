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

class RecvQueues {
 public:
  RecvQueues() = default;

  template <typename T>
  void addRecvItem(int remoteRank, int tag, std::shared_future<T> future) {
    auto& queue = queues_[std::make_pair(remoteRank, tag)];
    queue.emplace_back(std::make_shared<RecvItem<T>>(future));
  }

  size_t getNumRecvItems(int remoteRank, int tag) { return queues_[std::make_pair(remoteRank, tag)].size(); }

  void waitN(int remoteRank, int tag, size_t n) {
    auto& queue = queues_[std::make_pair(remoteRank, tag)];
    for (size_t i = 0; i < n; ++i) {
      queue[i]->wait();
    }
  }

 private:
  std::unordered_map<std::pair<int, int>, std::vector<std::shared_ptr<BaseRecvItem>>, PairHash> queues_;
};

struct ConnectionInfo {
  int remoteRank;
  int tag;
};

struct Communicator::Impl {
  std::shared_ptr<Bootstrap> bootstrap_;
  std::shared_ptr<Context> context_;
  std::unordered_map<const Connection*, ConnectionInfo> connectionInfos_;
  RecvQueues recvQueues_;

  Impl(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context);

  struct Connector;
};

}  // namespace mscclpp

#endif  // MSCCLPP_COMMUNICATOR_HPP_
