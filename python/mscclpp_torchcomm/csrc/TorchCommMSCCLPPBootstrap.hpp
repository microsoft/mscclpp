// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <c10/core/Device.h>

#include <chrono>
#include <comms/torchcomms/TorchCommOptions.hpp>
#include <memory>
#include <mscclpp/core.hpp>
#include <string>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace torch::comms {

/// Handles MSCCL++ bootstrap initialization using c10d::Store.
///
/// Follows the TorchCommNCCLBootstrap pattern:
///   - Rank 0 generates a UniqueId via TcpBootstrap::createUniqueId()
///   - Rank 0 writes raw bytes to the store
///   - All other ranks wait on the store key and read the UniqueId
///   - All ranks call TcpBootstrap::initialize(uniqueId) with the same ID
class TorchCommMSCCLPPBootstrap {
 public:
  TorchCommMSCCLPPBootstrap(c10::intrusive_ptr<c10d::Store> store, c10::Device device,
                            std::chrono::milliseconds timeout);

  ~TorchCommMSCCLPPBootstrap() noexcept;

  TorchCommMSCCLPPBootstrap(const TorchCommMSCCLPPBootstrap&) = delete;
  TorchCommMSCCLPPBootstrap& operator=(const TorchCommMSCCLPPBootstrap&) = delete;

  /// Create and initialize the MSCCL++ communicator.
  std::shared_ptr<mscclpp::Communicator> createCommunicator(const std::string& name, const CommOptions& options = {});

  int getRank() const { return rank_; }
  int getSize() const { return size_; }

 private:
  /// Exchange UniqueId via c10d::Store (rank 0 generates, others read).
  mscclpp::UniqueId exchangeUniqueId(const std::string& name);

  c10::intrusive_ptr<c10d::Store> store_;
  c10::Device device_;
  std::chrono::milliseconds timeout_;
  int rank_;
  int size_;

  static int counter_;
};

}  // namespace torch::comms
