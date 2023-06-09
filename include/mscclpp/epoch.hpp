#ifndef MSCCLPP_EPOCH_HPP_
#define MSCCLPP_EPOCH_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/cuda_utils.hpp>
#include <mscclpp/poll.hpp>

namespace mscclpp {

// struct alignas(16) EpochIds {
//   uint64_t outbound;
//   uint64_t inboundReplica;
// };

template <template <typename> typename Deleter>
class BaseEpoch {
 private:
  std::shared_ptr<Connection> connection_;
  // RegisteredMemory localInboundEpochIdsRegMem_;

 protected:
  NonblockingFuture<RegisteredMemory> remoteInboundEpochIdsRegMem_;
  uint64_t outBoundEpochId_; // always on the host
  std::unique_ptr<uint64_t, Deleter<uint64_t>> inboundEpochId_; // could be device or host
  std::unique_ptr<uint64_t, Deleter<uint64_t>> expectedInboundEpochId_; // could be device or host

 public:
  BaseEpoch(std::shared_ptr<Connection> connection, std::unique_ptr<uint64_t, Deleter<uint64_t>> inboundEpochId,
            std::unique_ptr<uint64_t, Deleter<uint64_t>> expectedInboundEpochId)
      : connection_(connection),
        outBoundEpochId_(0),
        inboundEpochId_(std::move(inboundEpochId)),
        expectedInboundEpochId_(std::move(expectedInboundEpochId)) {}

  void setup(Communicator& communicator) {
    auto localInboundEpochIdsRegMem = communicator.registerMemory(inboundEpochId_.get(), sizeof(uint64_t), connection_->transport());
    communicator.sendMemoryOnSetup(localInboundEpochIdsRegMem, connection_->remoteRank(), connection_->tag());
    remoteInboundEpochIdsRegMem_ = communicator.recvMemoryOnSetup(connection_->remoteRank(), connection_->tag());
  }

  void signal() {
    connection_->updateAndSync(remoteEpochIdsRegMem_.get(), 0, &outBoundEpochId_, outBoundEpochId_ + 1);
  }
};

class DeviceEpoch : BaseEpoch<CudaDeleter> {
 public:
  DeviceEpoch(Communicator& communicator, std::shared_ptr<Connection> connection);
  // void signal();

  struct DeviceHandle {
#ifdef __CUDACC__
    __forceinline__ __device__ void wait() {
      (*expectedInboundEpochId) += 1;
      while (*(volatile uint64_t*)&(inboundEpochId) < (*expectedInboundEpochId)){
      }
    }
#endif  // __CUDACC__

    uint64_t* inboundEpochId;
    // EpochIds* remoteEpochIds;
    uint64_t* expectedInboundEpochId;
  };

  DeviceHandle deviceHandle();
};

class HostEpoch : public BaseEpoch<std::default_delete> {
 public:
  HostEpoch(Communicator& communicator, std::shared_ptr<Connection> connection);

  // void incrementAndSignal();
  void wait();
};

class DirectEpoch {
  NonblockingFuture<RegisteredMemory> remoteInboundEpochIdsRegMem_;
  std::unique_ptr<uint64_t, CudaDeleter<uint64_t>> localInboundEpochId_;
  std::unique_ptr<uint64_t, CudaDeleter<uint64_t>> expectedInboundEpochId_;
  std::unique_ptr<uint64_t, CudaDeleter<uint64_t>> outboundEpochId_;

public:
  DirectEpoch(Communicator& communicator, std::shared_ptr<Connection> connection);
  struct DeviceHandle {
#ifdef __CUDACC__
    __forceinline__ __device__ void wait() {
      (*expectedInboundEpochId) += 1;
      while (*inboundEpochId < (*expectedInboundEpochId)){
      }
    }

    __forceinline__ __device__ void signalDirect() {
      // This fence ensures that the writes from a preceding putDirect() are visible on the peer GPU before the
      // incremented epoch id is visible.
      __threadfence_system();
      *outboundEpochId += 1;
      *remoteInboundEpochId = *outboundEpochId;
    }
#endif  // __CUDACC__
    volatile uint64_t* inboundEpochId;
    uint64_t* outboundEpochId;
    volatile uint64_t* remoteInboundEpochId;
    uint64_t* expectedInboundEpochId;
  };

  DeviceHandle deviceHandle();
};

}  // namespace mscclpp

#endif  // MSCCLPP_EPOCH_HPP_
