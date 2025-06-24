// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "communicator.hpp"

#include "api.h"
#include "debug.h"

namespace mscclpp {

Communicator::Impl::Impl(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context)
    : bootstrap_(bootstrap) {
  if (!context) {
    context_ = Context::create();
  } else {
    context_ = context;
  }
}

void Communicator::Impl::setLastRecvItem(int remoteRank, int tag, std::shared_ptr<BaseRecvItem> item) {
  lastRecvItems_[{remoteRank, tag}] = item;
}

std::shared_ptr<BaseRecvItem> Communicator::Impl::getLastRecvItem(int remoteRank, int tag) {
  auto it = lastRecvItems_.find({remoteRank, tag});
  if (it == lastRecvItems_.end()) {
    return nullptr;
  }
  if (it->second->isReady()) {
    lastRecvItems_.erase(it);
    return nullptr;
  }
  return it->second;
}

MSCCLPP_API_CPP Communicator::~Communicator() = default;

MSCCLPP_API_CPP Communicator::Communicator(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context)
    : pimpl_(std::make_unique<Impl>(bootstrap, context)) {}

MSCCLPP_API_CPP std::shared_ptr<Bootstrap> Communicator::bootstrap() { return pimpl_->bootstrap_; }

MSCCLPP_API_CPP std::shared_ptr<Context> Communicator::context() { return pimpl_->context_; }

MSCCLPP_API_CPP RegisteredMemory Communicator::registerMemory(void* ptr, size_t size, TransportFlags transports) {
  return context()->registerMemory(ptr, size, transports);
}

MSCCLPP_API_CPP void Communicator::sendMemory(RegisteredMemory memory, int remoteRank, int tag) {
  if (remoteRank == bootstrap()->getRank()) {
    // Sending memory to self
    auto& locRecvMemList = pimpl_->localRecvMemories_[tag];
    for (auto& locRecvMem : locRecvMemList) {
      if (!locRecvMem.isReady()) {
        // Found a local memory that is not ready, set the memory and return
        locRecvMem.set(std::move(memory));
        return;
      }
    }
    // No local memory found, create a new LocalRecvMemory and set the memory
    LocalRecvMemory locRecvMem;
    locRecvMem.set(std::move(memory));
    locRecvMemList.push_back(std::move(locRecvMem));
    return;
  }
  bootstrap()->send(memory.serialize(), remoteRank, tag);
}

MSCCLPP_API_CPP std::shared_future<RegisteredMemory> Communicator::recvMemory(int remoteRank, int tag) {
  if (remoteRank == bootstrap()->getRank()) {
    // Receiving memory from self
    auto& locRecvMemList = pimpl_->localRecvMemories_[tag];
    for (auto it = locRecvMemList.begin(); it != locRecvMemList.end(); ++it) {
      if (it->isReady()) {
        // Found a ready memory, remove it from the list and return its future
        auto future = it->reference();
        locRecvMemList.erase(it);
        return future;
      }
    }
    // No ready memory found, create a new LocalRecvMemory and return its future
    LocalRecvMemory locRecvMem;
    auto future = locRecvMem.reference();
    locRecvMemList.push_back(std::move(locRecvMem));
    return future;
  }
  auto future = std::async(std::launch::deferred,
                           [this, remoteRank, tag, lastRecvItem = pimpl_->getLastRecvItem(remoteRank, tag)]() {
                             if (lastRecvItem) {
                               // Recursive call to the previous receive items
                               lastRecvItem->wait();
                             }
                             std::vector<char> data;
                             bootstrap()->recv(data, remoteRank, tag);
                             return RegisteredMemory::deserialize(data);
                           });
  auto shared_future = std::shared_future<RegisteredMemory>(std::move(future));
  pimpl_->setLastRecvItem(remoteRank, tag, std::make_shared<RecvItem<RegisteredMemory>>(shared_future));
  return shared_future;
}

MSCCLPP_API_CPP std::shared_future<std::shared_ptr<Connection>> Communicator::connect(int remoteRank, int tag,
                                                                                      EndpointConfig localConfig) {
  auto localEndpoint = context()->createEndpoint(localConfig);

  if (remoteRank == bootstrap()->getRank()) {
    // Connection to self
    auto remoteEndpoint = context()->createEndpoint(localConfig);
    auto connection = context()->connect(localEndpoint, remoteEndpoint);
    std::promise<std::shared_ptr<Connection>> promise;
    promise.set_value(connection);
    pimpl_->connectionInfos_[connection.get()] = {remoteRank, tag};
    return std::shared_future<std::shared_ptr<Connection>>(std::move(promise.get_future()));
  }

  bootstrap()->send(localEndpoint.serialize(), remoteRank, tag);

  auto future =
      std::async(std::launch::deferred, [this, remoteRank, tag, lastRecvItem = pimpl_->getLastRecvItem(remoteRank, tag),
                                         localEndpoint = std::move(localEndpoint)]() mutable {
        if (lastRecvItem) {
          // Recursive call to the previous receive items
          lastRecvItem->wait();
        }
        std::vector<char> data;
        bootstrap()->recv(data, remoteRank, tag);
        auto remoteEndpoint = Endpoint::deserialize(data);
        auto connection = context()->connect(localEndpoint, remoteEndpoint);
        pimpl_->connectionInfos_[connection.get()] = {remoteRank, tag};
        return connection;
      });
  auto shared_future = std::shared_future<std::shared_ptr<Connection>>(std::move(future));
  pimpl_->setLastRecvItem(remoteRank, tag, std::make_shared<RecvItem<std::shared_ptr<Connection>>>(shared_future));
  return shared_future;
}

MSCCLPP_API_CPP int Communicator::remoteRankOf(const Connection& connection) {
  return pimpl_->connectionInfos_.at(&connection).remoteRank;
}

MSCCLPP_API_CPP int Communicator::tagOf(const Connection& connection) {
  return pimpl_->connectionInfos_.at(&connection).tag;
}

}  // namespace mscclpp
