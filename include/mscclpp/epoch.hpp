#ifndef MSCCLPP_EPOCH_HPP_
#define MSCCLPP_EPOCH_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/cuda_utils.hpp>

namespace mscclpp {

struct alignas(16) EpochIds {
  uint64_t outbound;
  uint64_t inboundReplica;
};

template <template <typename> typename Deleter>
class BaseEpoch {
 private:
  std::shared_ptr<Connection> connection_;
  RegisteredMemory localEpochIdsRegMem_;

 protected:
  NonblockingFuture<RegisteredMemory> remoteEpochIdsRegMem_;
  std::unique_ptr<EpochIds, Deleter<EpochIds>> epochIds_;
  std::unique_ptr<uint64_t, Deleter<uint64_t>> expectedInboundEpochId_;

 public:
  BaseEpoch(std::shared_ptr<Connection> connection, std::unique_ptr<EpochIds, Deleter<EpochIds>> epochIds,
            std::unique_ptr<uint64_t, Deleter<uint64_t>> expectedInboundEpochId)
      : connection_(connection),
        epochIds_(std::move(epochIds)),
        expectedInboundEpochId_(std::move(expectedInboundEpochId)) {}

  void setup(Communicator& communicator) {
    localEpochIdsRegMem_ = communicator.registerMemory(epochIds_.get(), sizeof(epochIds_), connection_->transport());
    communicator.sendMemoryOnSetup(localEpochIdsRegMem_, connection_->remoteRank(), connection_->tag());
    remoteEpochIdsRegMem_ = communicator.recvMemoryOnSetup(connection_->remoteRank(), connection_->tag());
  }

  void signal() {
    connection_->write(remoteEpochIdsRegMem_.get(), offsetof(EpochIds, inboundReplica), localEpochIdsRegMem_,
                       offsetof(EpochIds, outbound), sizeof(epochIds_));
  }
};

class DeviceEpoch : BaseEpoch<CudaDeleter> {
 public:
  DeviceEpoch(Communicator& communicator, std::shared_ptr<Connection> connection);
  void signal();

  struct DeviceHandle {
#ifdef __CUDACC__
    __forceinline__ __device__ void wait() {
      (*expectedInboundEpochId) += 1;
      while (*(volatile uint64_t*)&(epochIds->inboundReplica) < (*expectedInboundEpochId)){
        printf("waiting for epoch %lu vs %lu\n", *expectedInboundEpochId, *(volatile uint64_t*)&(epochIds->inboundReplica));
      }
    }

    __forceinline__ __device__ void epochIncrement() { *(volatile uint64_t*)&(epochIds->outbound) += 1; }

    __forceinline__ __device__ void signalDirect() {
      // This fence ensures that the writes from a preceding putDirect() are visible on the peer GPU before the
      // incremented epoch id is visible.
      __threadfence_system();
      epochIncrement();
      *(volatile uint64_t*)&(remoteEpochIds->inboundReplica) = epochIds->outbound;
    }
#endif  // __CUDACC__

    EpochIds* epochIds;
    EpochIds* remoteEpochIds;
    uint64_t* expectedInboundEpochId;
  };

  DeviceHandle deviceHandle();
};

class HostEpoch : BaseEpoch<std::default_delete> {
 public:
  HostEpoch(Communicator& communicator, std::shared_ptr<Connection> connection);

  void incrementAndSignal();
  void wait();
};

}  // namespace mscclpp

#endif  // MSCCLPP_EPOCH_HPP_
