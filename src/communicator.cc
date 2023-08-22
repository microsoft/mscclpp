// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "communicator.hpp"

#include <mscclpp/core.hpp>
#include <mscclpp/cuda_utils.hpp>
#include <sstream>

#include "api.h"
#include "connection.hpp"
#include "debug.h"
#include "registered_memory.hpp"
#include "utils_internal.hpp"

namespace mscclpp {

Communicator::Impl::Impl(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context)
    : bootstrap_(bootstrap) {
  if (!context) {
    context_ = std::make_shared<Context>();
  } else {
    context_ = context;
  }

  rankToHash_.resize(bootstrap->getNranks());
  rankToHash_[bootstrap->getRank()] = context_->pimpl->hostHash_;
  bootstrap->allGather(rankToHash_.data(), sizeof(uint64_t));
}

MSCCLPP_API_CPP Communicator::~Communicator() = default;

MSCCLPP_API_CPP Communicator::Communicator(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context)
    : pimpl(std::make_unique<Impl>(bootstrap, context)) {}

MSCCLPP_API_CPP std::shared_ptr<Bootstrap> Communicator::bootstrap() { return pimpl->bootstrap_; }

MSCCLPP_API_CPP std::shared_ptr<Context> Communicator::context() { return pimpl->context_; }

MSCCLPP_API_CPP RegisteredMemory Communicator::registerMemory(void* ptr, size_t size, TransportFlags transports) {
  return context()->registerMemory(ptr, size, transports);
}

struct MemorySender : public Setuppable {
  MemorySender(RegisteredMemory memory, int remoteRank, int tag)
      : memory_(memory), remoteRank_(remoteRank), tag_(tag) {}

  void beginSetup(std::shared_ptr<Bootstrap> bootstrap) override {
    bootstrap->send(memory_.serialize(), remoteRank_, tag_);
  }

  RegisteredMemory memory_;
  int remoteRank_;
  int tag_;
};

MSCCLPP_API_CPP void Communicator::sendMemoryOnSetup(RegisteredMemory memory, int remoteRank, int tag) {
  onSetup(std::make_shared<MemorySender>(memory, remoteRank, tag));
}

struct MemoryReceiver : public Setuppable {
  MemoryReceiver(int remoteRank, int tag) : remoteRank_(remoteRank), tag_(tag) {}

  void endSetup(std::shared_ptr<Bootstrap> bootstrap) override {
    std::vector<char> data;
    bootstrap->recv(data, remoteRank_, tag_);
    memoryPromise_.set_value(RegisteredMemory::deserialize(data));
  }

  std::promise<RegisteredMemory> memoryPromise_;
  int remoteRank_;
  int tag_;
};

MSCCLPP_API_CPP NonblockingFuture<RegisteredMemory> Communicator::recvMemoryOnSetup(int remoteRank, int tag) {
  auto memoryReceiver = std::make_shared<MemoryReceiver>(remoteRank, tag);
  onSetup(memoryReceiver);
  return NonblockingFuture<RegisteredMemory>(memoryReceiver->memoryPromise_.get_future());
}

struct Connector : public Setuppable {
  Connector(Communicator& comm, int remoteRank, int tag, Transport transport, int ibMaxCqSize, int ibMaxCqPollNum,
            int ibMaxSendWr, int ibMaxWrPerSend)
      : comm_(comm),
        remoteRank_(remoteRank),
        tag_(tag),
        localEndpoint_(
            comm.context()->createEndpoint(transport, ibMaxCqSize, ibMaxCqPollNum, ibMaxSendWr, ibMaxWrPerSend)) {}

  void beginSetup(std::shared_ptr<Bootstrap> bootstrap) override {
    bootstrap->send(localEndpoint_.serialize(), remoteRank_, tag_);
  }

  void endSetup(std::shared_ptr<Bootstrap> bootstrap) override {
    std::vector<char> data;
    bootstrap->recv(data, remoteRank_, tag_);
    auto remoteEndpoint = Endpoint::deserialize(data);
    auto connection = comm_.context()->connect(localEndpoint_, remoteEndpoint);
    comm_.pimpl->connectionInfos_[connection.get()] = {remoteRank_, tag_};
    connectionPromise_.set_value(connection);
  }

  std::promise<std::shared_ptr<Connection>> connectionPromise_;
  Communicator& comm_;
  int remoteRank_;
  int tag_;
  Endpoint localEndpoint_;
};

MSCCLPP_API_CPP NonblockingFuture<std::shared_ptr<Connection>> Communicator::connectOnSetup(
    int remoteRank, int tag, Transport transport, int ibMaxCqSize, int ibMaxCqPollNum, int ibMaxSendWr,
    int ibMaxWrPerSend) {
  auto connector = std::make_shared<Connector>(*this, remoteRank, tag, transport, ibMaxCqSize, ibMaxCqPollNum,
                                               ibMaxSendWr, ibMaxWrPerSend);
  onSetup(connector);
  return NonblockingFuture<std::shared_ptr<Connection>>(connector->connectionPromise_.get_future());
}

MSCCLPP_API_CPP int Communicator::remoteRankOf(const Connection& connection) {
  return pimpl->connectionInfos_.at(&connection).remoteRank;
}

MSCCLPP_API_CPP int Communicator::tagOf(const Connection& connection) {
  return pimpl->connectionInfos_.at(&connection).tag;
}

MSCCLPP_API_CPP void Communicator::onSetup(std::shared_ptr<Setuppable> setuppable) {
  pimpl->toSetup_.push_back(setuppable);
}

MSCCLPP_API_CPP void Communicator::setup() {
  for (auto& setuppable : pimpl->toSetup_) {
    setuppable->beginSetup(pimpl->bootstrap_);
  }
  for (auto& setuppable : pimpl->toSetup_) {
    setuppable->endSetup(pimpl->bootstrap_);
  }
  pimpl->toSetup_.clear();
}

}  // namespace mscclpp
