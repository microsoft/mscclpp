// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EPOCH_HPP_
#define MSCCLPP_EPOCH_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/cuda_utils.hpp>
#include <mscclpp/poll.hpp>

namespace mscclpp {

/// @brief A base class for epochs.
///
/// An epoch is a synchronization mechanism that allows the local peer to wait for the remote peer to complete a data
/// transfer. The local peer signals the remote peer that it has completed a data transfer by incrementing the outbound
/// epoch ID. The incremented outbound epoch ID is copied to the remote peer's inbound epoch ID so that the remote peer
/// can wait for the local peer to complete a data transfer. Vice versa, the remote peer signals the local peer that it
/// has completed a data transfer by incrementing the remote peer's outbound epoch ID and copying the incremented value
/// to the local peer's inbound epoch ID.
///
/// @tparam InboundDeleter The deleter for inbound epoch IDs. This is either `std::default_delete` for host memory or
/// @ref CudaDeleter for device memory.
/// @tparam OutboundDeleter The deleter for outbound epoch IDs. This is either `std::default_delete` for host memory or
/// @ref CudaDeleter for device memory.
template <template <typename> typename InboundDeleter, template <typename> typename OutboundDeleter>
class BaseEpoch {
 protected:
  /// @brief The registered memory for the remote peer's inbound epoch ID.
  NonblockingFuture<RegisteredMemory> remoteInboundEpochIdsRegMem_;

  /// @brief The inbound epoch ID that is incremented by the remote peer and waited on by the local peer.
  ///
  /// The location of @ref localInboundEpochId_ can be either on the host or on the device.
  std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> localInboundEpochId_;

  /// @brief The expected inbound epoch ID to be incremented by the local peer and compared to the
  /// @ref localInboundEpochId_.
  ///
  /// The location of @ref expectedInboundEpochId_ can be either on the host or on the device.
  std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> expectedInboundEpochId_;

  /// @brief The outbound epoch ID that is incremented by the local peer and copied to the remote peer's @ref
  /// localInboundEpochId_.
  ///
  /// The location of @ref outboundEpochId_ can be either on the host or on the device.
  std::unique_ptr<uint64_t, OutboundDeleter<uint64_t>> outboundEpochId_;

 public:
  /// @brief Constructs a BaseEpoch.
  ///
  /// @param localInboundEpochId The inbound epoch ID
  /// @param expectedInboundEpochId The expected inbound epoch ID
  /// @param outboundEpochId The outbound epoch ID
  BaseEpoch(std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> localInboundEpochId,
            std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> expectedInboundEpochId,
            std::unique_ptr<uint64_t, OutboundDeleter<uint64_t>> outboundEpochId)
      : localInboundEpochId_(std::move(localInboundEpochId)),
        expectedInboundEpochId_(std::move(expectedInboundEpochId)),
        outboundEpochId_(std::move(outboundEpochId)) {}
};

class Host2DeviceEpoch : public BaseEpoch<CudaDeleter, std::default_delete> {
 private:
  std::shared_ptr<Connection> connection_;

 public:
  Host2DeviceEpoch(Communicator& communicator, std::shared_ptr<Connection> connection);

  // same write api as connection
  std::shared_ptr<Connection> connection();
  void signal();

  struct DeviceHandle {
#ifdef __CUDACC__
    __forceinline__ __device__ void wait() {
      (*expectedInboundEpochId) += 1;
      POLL_MAYBE_JAILBREAK(*(volatile uint64_t*)(inboundEpochId) < (*expectedInboundEpochId), 1000000);
    }
#endif  // __CUDACC__

    uint64_t* inboundEpochId;
    uint64_t* expectedInboundEpochId;
  };

  DeviceHandle deviceHandle();
};

class Host2HostEpoch : public BaseEpoch<std::default_delete, std::default_delete> {
 public:
  Host2HostEpoch(Communicator& communicator, std::shared_ptr<Connection> connection);

  std::shared_ptr<Connection> connection();
  void signal();
  void wait();

 private:
  std::shared_ptr<Connection> connection_;
};

class SmDevice2DeviceEpoch : public BaseEpoch<CudaDeleter, CudaDeleter> {
 public:
  SmDevice2DeviceEpoch(Communicator& communicator, std::shared_ptr<Connection> connection);
  SmDevice2DeviceEpoch() = default;

  struct DeviceHandle {
#ifdef __CUDACC__
    __forceinline__ __device__ void wait() {
      (*expectedInboundEpochId) += 1;
      POLL_MAYBE_JAILBREAK(*inboundEpochId < (*expectedInboundEpochId), 1000000);
    }

    __forceinline__ __device__ void signal() {
      // This fence ensures that preceding writes are visible on the peer GPU before the incremented `outboundEpochId`
      // is visible.
      __threadfence_system();
      epochIncrement();
      *remoteInboundEpochId = epochGetLocal();
    }

    __forceinline__ __device__ void signalPacket() {
      epochIncrement();
      *remoteInboundEpochId = epochGetLocal();
    }

    __forceinline__ __device__ void epochIncrement() { *outboundEpochId += 1; }

    __forceinline__ __device__ uint64_t epochGetLocal() const { return *outboundEpochId; }
#endif  // __CUDACC__

    volatile uint64_t* inboundEpochId;
    uint64_t* outboundEpochId;
    volatile uint64_t* remoteInboundEpochId;
    uint64_t* expectedInboundEpochId;
  };

  DeviceHandle deviceHandle() const;
  bool isRemoteInboundEpochIdSet_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_EPOCH_HPP_
