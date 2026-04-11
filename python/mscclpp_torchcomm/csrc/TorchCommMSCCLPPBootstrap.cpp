// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "TorchCommMSCCLPPBootstrap.hpp"

#include <algorithm>
#include <comms/torchcomms/utils/StoreManager.hpp>
#include <comms/torchcomms/utils/Utils.hpp>
#include <stdexcept>

namespace torch::comms {

// Static counter ensures unique store keys when multiple communicators are
// created with the same name in the same process (e.g., separate comm groups).
int TorchCommMSCCLPPBootstrap::counter_ = 0;

// Discovers rank and world size from torchrun/torchelastic environment variables
// (RANK, WORLD_SIZE, LOCAL_RANK). query_ranksize() is a torchcomms utility.
TorchCommMSCCLPPBootstrap::TorchCommMSCCLPPBootstrap(c10::intrusive_ptr<c10d::Store> store, c10::Device device,
                                                     std::chrono::milliseconds timeout)
    : store_(std::move(store)), device_(device), timeout_(timeout) {
  auto [rank, size] = query_ranksize();
  rank_ = rank;
  size_ = size;
}

TorchCommMSCCLPPBootstrap::~TorchCommMSCCLPPBootstrap() noexcept = default;

mscclpp::UniqueId TorchCommMSCCLPPBootstrap::exchangeUniqueId(const std::string& name) {
  // Single-process: no coordination needed
  if (size_ == 1) {
    return mscclpp::TcpBootstrap::createUniqueId();
  }

  // Multi-process without a caller-supplied store: fall back to createPrefixStore
  if (!store_) {
    store_ = createPrefixStore("mscclpp", timeout_);
  }

  std::string key = "mscclpp_uniqueid_" + name + std::to_string(counter_++);

  mscclpp::UniqueId unique_id;

  if (rank_ == 0) {
    unique_id = mscclpp::TcpBootstrap::createUniqueId();
    std::vector<uint8_t> vec(unique_id.begin(), unique_id.end());
    store_->set(key, vec);
  } else {
    store_->wait({key}, timeout_);
    auto vec = store_->get(key);
    if (vec.size() != sizeof(mscclpp::UniqueId)) {
      throw std::runtime_error("[TorchCommMSCCLPPBootstrap] Invalid UniqueId size: expected " +
                               std::to_string(sizeof(mscclpp::UniqueId)) + ", got " + std::to_string(vec.size()));
    }
    std::copy(vec.begin(), vec.end(), unique_id.begin());
  }

  return unique_id;
}

std::shared_ptr<mscclpp::Communicator> TorchCommMSCCLPPBootstrap::createCommunicator(const std::string& name,
                                                                                     const CommOptions& /*options*/) {
  mscclpp::UniqueId unique_id = exchangeUniqueId(name);

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank_, size_);
  // Single-process (size==1): skip TCP initialization since there are no peers
  // to connect to. The bootstrap object is still needed by the Communicator
  // constructor, but it doesn't need an active TCP server.
  if (size_ > 1) {
    int64_t timeout_sec = std::max(int64_t{1}, std::chrono::duration_cast<std::chrono::seconds>(timeout_).count());
    bootstrap->initialize(unique_id, timeout_sec);
  }

  return std::make_shared<mscclpp::Communicator>(bootstrap);
}

}  // namespace torch::comms
