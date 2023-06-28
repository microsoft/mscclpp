// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SEMAPHORE_HPP_
#define MSCCLPP_SEMAPHORE_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/cuda_utils.hpp>
#include <mscclpp/poll.hpp>

namespace mscclpp {

/// @brief A base class for semaphores.
///
/// An semaphore is a synchronization mechanism that allows the local peer to wait for the remote peer to complete a
/// data transfer. The local peer signals the remote peer that it has completed a data transfer by incrementing the
/// outbound semaphore ID. The incremented outbound semaphore ID is copied to the remote peer's inbound semaphore ID so
/// that the remote peer can wait for the local peer to complete a data transfer. Vice versa, the remote peer signals
/// the local peer that it has completed a data transfer by incrementing the remote peer's outbound semaphore ID and
/// copying the incremented value to the local peer's inbound semaphore ID.
///
/// @tparam InboundDeleter The deleter for inbound semaphore IDs. This is either `std::default_delete` for host memory
/// or
/// @ref CudaDeleter for device memory.
/// @tparam OutboundDeleter The deleter for outbound semaphore IDs. This is either `std::default_delete` for host memory
/// or
/// @ref CudaDeleter for device memory.
template <template <typename> typename InboundDeleter, template <typename> typename OutboundDeleter>
class BaseSemaphore {
 protected:
  /// @brief The registered memory for the remote peer's inbound semaphore ID.
  NonblockingFuture<RegisteredMemory> remoteInboundSemaphoreIdsRegMem_;

  /// @brief The inbound semaphore ID that is incremented by the remote peer and waited on by the local peer.
  ///
  /// The location of @ref localInboundSemaphore_ can be either on the host or on the device.
  std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> localInboundSemaphore_;

  /// @brief The expected inbound semaphore ID to be incremented by the local peer and compared to the
  /// @ref localInboundSemaphore_.
  ///
  /// The location of @ref expectedInboundSemaphore_ can be either on the host or on the device.
  std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> expectedInboundSemaphore_;

  /// @brief The outbound semaphore ID that is incremented by the local peer and copied to the remote peer's @ref
  /// localInboundSemaphore_.
  ///
  /// The location of @ref outboundSemaphore_ can be either on the host or on the device.
  std::unique_ptr<uint64_t, OutboundDeleter<uint64_t>> outboundSemaphore_;

 public:
  /// @brief Constructs a BaseSemaphore.
  ///
  /// @param localInboundSemaphoreId The inbound semaphore ID
  /// @param expectedInboundSemaphoreId The expected inbound semaphore ID
  /// @param outboundSemaphoreId The outbound semaphore ID
  BaseSemaphore(std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> localInboundSemaphoreId,
                std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> expectedInboundSemaphoreId,
                std::unique_ptr<uint64_t, OutboundDeleter<uint64_t>> outboundSemaphoreId)
      : localInboundSemaphore_(std::move(localInboundSemaphoreId)),
        expectedInboundSemaphore_(std::move(expectedInboundSemaphoreId)),
        outboundSemaphore_(std::move(outboundSemaphoreId)) {}
};

class Host2DeviceSemaphore : public BaseSemaphore<CudaDeleter, std::default_delete> {
 private:
  std::shared_ptr<Connection> connection_;

 public:
  Host2DeviceSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  // same write api as connection
  std::shared_ptr<Connection> connection();
  void signal();

  struct DeviceHandle {
#ifdef __CUDACC__
    __forceinline__ __device__ void wait() {
      (*expectedInboundSemaphoreId) += 1;
      POLL_MAYBE_JAILBREAK(*(volatile uint64_t*)(inboundSemaphoreId) < (*expectedInboundSemaphoreId), 1000000);
    }
#endif  // __CUDACC__

    uint64_t* inboundSemaphoreId;
    uint64_t* expectedInboundSemaphoreId;
  };

  DeviceHandle deviceHandle();
};

class Host2HostSemaphore : public BaseSemaphore<std::default_delete, std::default_delete> {
 public:
  Host2HostSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  std::shared_ptr<Connection> connection();
  void signal();
  void wait();

 private:
  std::shared_ptr<Connection> connection_;
};

class SmDevice2DeviceSemaphore : public BaseSemaphore<CudaDeleter, CudaDeleter> {
 public:
  SmDevice2DeviceSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);
  SmDevice2DeviceSemaphore() = default;

  struct DeviceHandle {
#ifdef __CUDACC__
    __forceinline__ __device__ void wait() {
      (*expectedInboundSemaphoreId) += 1;
      POLL_MAYBE_JAILBREAK(*inboundSemaphoreId < (*expectedInboundSemaphoreId), 1000000);
    }

    __forceinline__ __device__ void signal() {
      // This fence ensures that preceding writes are visible on the peer GPU before the incremented
      // `outboundSemaphoreId` is visible.
      __threadfence_system();
      semaphoreIncrement();
      *remoteInboundSemaphoreId = semaphoreGetLocal();
    }

    __forceinline__ __device__ void signalPacket() {
      semaphoreIncrement();
      *remoteInboundSemaphoreId = semaphoreGetLocal();
    }

    __forceinline__ __device__ void semaphoreIncrement() { *outboundSemaphoreId += 1; }

    __forceinline__ __device__ uint64_t semaphoreGetLocal() const { return *outboundSemaphoreId; }
#endif  // __CUDACC__

    volatile uint64_t* inboundSemaphoreId;
    uint64_t* outboundSemaphoreId;
    volatile uint64_t* remoteInboundSemaphoreId;
    uint64_t* expectedInboundSemaphoreId;
  };

  DeviceHandle deviceHandle() const;
  bool isRemoteInboundSemaphoreIdSet_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_HPP_
