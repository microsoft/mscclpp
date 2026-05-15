// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "communicator.hpp"

#include <utility>

#include "api.h"

namespace mscclpp {

namespace {

template <typename Func>
class ScopeGuard {
 public:
  explicit ScopeGuard(Func func) : func_(std::move(func)) {}
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard(ScopeGuard&&) = delete;
  ScopeGuard& operator=(ScopeGuard&&) = delete;
  ~ScopeGuard() noexcept {
    static_assert(noexcept(std::declval<Func&>()()), "ScopeGuard cleanup must be noexcept");
    func_();
  }

 private:
  Func func_;
};

template <typename T, typename Impl, typename Func>
std::shared_future<T> makeOrderedRecvFuture(Impl* impl, int remoteRank, int tag, Func func) {
  // Weak placeholder to avoid a reference cycle; updated with the real recvItem after the future is created.
  auto thisRecvItem = std::make_shared<std::weak_ptr<BaseRecvItem>>();
  auto future = std::async(
      std::launch::deferred, [impl, remoteRank, tag, thisRecvItem,
                              lastRecvItem = impl->getLastRecvItem(remoteRank, tag), func = std::move(func)]() mutable {
        [[maybe_unused]] ScopeGuard cleanup([impl, remoteRank, tag, thisRecvItem]() noexcept {
          impl->clearLastRecvItemIfMatches(remoteRank, tag, thisRecvItem->lock());
        });

        if (lastRecvItem) {
          // Recursive call to the previous receive items
          lastRecvItem->wait();
        }
        return func();
      });
  auto sharedFuture = std::shared_future<T>(std::move(future));
  auto recvItem = std::make_shared<RecvItem<T>>(sharedFuture);
  *thisRecvItem = recvItem;
  impl->setLastRecvItem(remoteRank, tag, recvItem);
  return sharedFuture;
}

}  // namespace

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

void Communicator::Impl::clearLastRecvItemIfMatches(int remoteRank, int tag,
                                                    const std::shared_ptr<BaseRecvItem>& expectedItem) {
  auto it = lastRecvItems_.find({remoteRank, tag});
  if (it != lastRecvItems_.end() && it->second == expectedItem) {
    lastRecvItems_.erase(it);
  }
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
  return makeOrderedRecvFuture<RegisteredMemory>(pimpl_.get(), remoteRank, tag, [this, remoteRank, tag]() {
    std::vector<char> data;
    bootstrap()->recv(data, remoteRank, tag);
    return RegisteredMemory::deserialize(data);
  });
}

MSCCLPP_API_CPP std::shared_future<Connection> Communicator::connect(const Endpoint& localEndpoint, int remoteRank,
                                                                     int tag) {
  if (remoteRank == bootstrap()->getRank()) {
    // Connection to self
    auto remoteEndpoint = context()->createEndpoint(localEndpoint.config());
    auto connection = context()->connect(localEndpoint, remoteEndpoint);
    std::promise<Connection> promise;
    promise.set_value(connection);
    pimpl_->connectionInfos_[connection.impl_.get()] = {remoteRank, tag};
    return std::shared_future<Connection>(promise.get_future());
  }

  bootstrap()->send(localEndpoint.serialize(), remoteRank, tag);

  return makeOrderedRecvFuture<Connection>(pimpl_.get(), remoteRank, tag,
                                           [this, remoteRank, tag, localEndpoint]() mutable {
                                             std::vector<char> data;
                                             bootstrap()->recv(data, remoteRank, tag);
                                             auto remoteEndpoint = Endpoint::deserialize(data);
                                             auto connection = context()->connect(localEndpoint, remoteEndpoint);
                                             pimpl_->connectionInfos_[connection.impl_.get()] = {remoteRank, tag};
                                             return connection;
                                           });
}

MSCCLPP_API_CPP std::shared_future<Connection> Communicator::connect(const EndpointConfig& localConfig, int remoteRank,
                                                                     int tag) {
  auto localEndpoint = context()->createEndpoint(localConfig);
  return connect(localEndpoint, remoteRank, tag);
}

MSCCLPP_API_CPP std::shared_future<Semaphore> Communicator::buildSemaphore(const Connection& connection, int remoteRank,
                                                                           int tag) {
  SemaphoreStub localStub(connection);
  bootstrap()->send(localStub.serialize(), remoteRank, tag);

  return makeOrderedRecvFuture<Semaphore>(pimpl_.get(), remoteRank, tag, [this, remoteRank, tag, localStub]() mutable {
    std::vector<char> data;
    bootstrap()->recv(data, remoteRank, tag);
    auto remoteStub = SemaphoreStub::deserialize(data);
    return Semaphore(localStub, remoteStub);
  });
}

MSCCLPP_API_CPP int Communicator::remoteRankOf(const Connection& connection) {
  return pimpl_->connectionInfos_.at(connection.impl_.get()).remoteRank;
}

MSCCLPP_API_CPP int Communicator::tagOf(const Connection& connection) {
  return pimpl_->connectionInfos_.at(connection.impl_.get()).tag;
}

}  // namespace mscclpp
