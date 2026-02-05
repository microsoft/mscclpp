// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// AllToAllV implementation for MSCCLPP
// This kernel handles variable element counts per rank for alltoallv operations.
// Unlike NCCL's ncclGroupStart/ncclGroupEnd approach, mscclpp uses explicit
// put/signal/wait operations on PortChannels.

#include <Python.h>
#include <pybind11/pybind11.h>

#include <mscclpp/algorithm.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/concurrency_device.hpp>

namespace py = pybind11;

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

// Device syncer for synchronization across blocks
__device__ mscclpp::DeviceSyncer alltoallvDeviceSyncer;

/**
 * AllToAllV kernel implementation
 *
 * This kernel performs an all-to-all exchange with variable-length data per rank.
 * Each rank sends sendCounts[i] elements to rank i at sendDispls[i] offset,
 * and receives recvCounts[i] elements from rank i at recvDispls[i] offset.
 *
 * Since mscclpp doesn't support ncclGroupStart/ncclGroupEnd, we implement
 * the exchange using explicit put/signal/wait operations on PortChannels.
 * The communication pattern uses a ring-based approach to avoid deadlocks.
 *
 * @param portChannels Array of PortChannel handles for each peer (worldSize-1 channels)
 * @param rank Current rank
 * @param worldSize Total number of ranks
 * @param sendBuff Source buffer containing data to send
 * @param recvBuff Destination buffer for received data
 * @param sendCounts Array of send counts for each rank (in bytes)
 * @param sendDispls Array of send displacements for each rank (in bytes)
 * @param recvCounts Array of receive counts for each rank (in bytes)
 * @param recvDispls Array of receive displacements for each rank (in bytes)
 */
__global__ void __launch_bounds__(1024)
    alltoallv_kernel(mscclpp::DeviceHandle<mscclpp::PortChannel>* portChannels,
                     int rank,
                     int worldSize,
                     const void* sendBuff,
                     void* recvBuff,
                     const size_t* sendCounts,
                     const size_t* sendDispls,
                     const size_t* recvCounts,
                     const size_t* recvDispls) {
  // First, copy local data (rank's own portion) from send to recv buffer
  // This doesn't require any communication
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (sendCounts[rank] > 0) {
      // Local copy: sendBuff[sendDispls[rank]] -> recvBuff[recvDispls[rank]]
      const char* src = (const char*)sendBuff + sendDispls[rank];
      char* dst = (char*)recvBuff + recvDispls[rank];
      memcpy(dst, src, sendCounts[rank]);
    }
  }
  __syncthreads();

  // Ring-based exchange pattern to avoid deadlocks
  // In each step i, rank sends to (rank + i) % worldSize and receives from (rank - i + worldSize) % worldSize
  for (int step = 1; step < worldSize; step++) {
    int sendPeer = (rank + step) % worldSize;
    int recvPeer = (rank - step + worldSize) % worldSize;

    // Get channel indices (portChannels excludes self, so adjust index)
    int sendChanIdx = sendPeer < rank ? sendPeer : sendPeer - 1;
    int recvChanIdx = recvPeer < rank ? recvPeer : recvPeer - 1;

    // Each warp handles one peer
    int wid = threadIdx.x / WARP_SIZE;
    int lid = threadIdx.x % WARP_SIZE;

    // Send data to sendPeer if there's data to send
    if (wid == 0 && lid == 0) {
      if (sendCounts[sendPeer] > 0) {
        // putWithSignal: copy data and signal completion
        // src offset: sendDispls[sendPeer] in our sendBuff
        // dst offset: recvDispls[rank] in peer's recvBuff (where our data should go)
        portChannels[sendChanIdx].putWithSignal(
            recvDispls[rank],           // dst offset in peer's recv buffer (where we write)
            sendDispls[sendPeer],       // src offset in our send buffer
            sendCounts[sendPeer]        // size in bytes
        );
      }
    }

    // Sync all threads before flushing
    alltoallvDeviceSyncer.sync(gridDim.x);

    // Flush to ensure data is sent
    if (wid == 0 && lid == 0) {
      if (sendCounts[sendPeer] > 0) {
        portChannels[sendChanIdx].flush();
      }
    }

    // Wait for data from recvPeer if we're expecting data
    if (wid == 0 && lid == 0) {
      if (recvCounts[recvPeer] > 0) {
        portChannels[recvChanIdx].wait();
      }
    }

    // Sync all threads before next step
    alltoallvDeviceSyncer.sync(gridDim.x);
  }
}

/**
 * Simplified AllToAllV kernel for single-block execution
 *
 * This version is optimized for cases where all communication can be
 * handled within a single thread block.
 */
__global__ void __launch_bounds__(1024)
    alltoallv_simple_kernel(mscclpp::DeviceHandle<mscclpp::PortChannel>* portChannels,
                            int rank,
                            int worldSize,
                            const void* sendBuff,
                            void* recvBuff,
                            const size_t* sendCounts,
                            const size_t* sendDispls,
                            const size_t* recvCounts,
                            const size_t* recvDispls) {
  int tid = threadIdx.x;
  int nPeers = worldSize - 1;

  // Step 1: Copy local data
  if (tid == 0 && sendCounts[rank] > 0) {
    const char* src = (const char*)sendBuff + sendDispls[rank];
    char* dst = (char*)recvBuff + recvDispls[rank];
    memcpy(dst, src, sendCounts[rank]);
  }
  __syncthreads();

  // Step 2: Each warp handles one peer for sending
  // We have worldSize-1 peers, assign one warp per peer
  int warpId = tid / WARP_SIZE;
  int laneId = tid % WARP_SIZE;

  if (warpId < nPeers && laneId == 0) {
    // Determine which peer this warp handles
    int peer = warpId < rank ? warpId : warpId + 1;
    int chanIdx = warpId;

    if (sendCounts[peer] > 0) {
      portChannels[chanIdx].putWithSignal(
          recvDispls[rank],       // dst offset in peer's buffer
          sendDispls[peer],       // src offset in our buffer
          sendCounts[peer]        // size
      );
    }
  }
  __syncthreads();

  // Step 3: Flush all pending operations
  if (warpId < nPeers && laneId == 0) {
    int peer = warpId < rank ? warpId : warpId + 1;
    if (sendCounts[peer] > 0) {
      portChannels[warpId].flush();
    }
  }
  __syncthreads();

  // Step 4: Wait for all incoming data
  if (warpId < nPeers && laneId == 0) {
    int peer = warpId < rank ? warpId : warpId + 1;
    if (recvCounts[peer] > 0) {
      portChannels[warpId].wait();
    }
  }
  __syncthreads();
}

// Context to hold all necessary state for alltoallv execution
struct AllToAllVContext {
  int rank;
  int worldSize;
  int nRanksPerNode;

  std::vector<mscclpp::RegisteredMemory> registeredMemories;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::PortChannel>> portChannelDeviceHandles;

  // Device memory for counts and displacements
  size_t* d_sendCounts;
  size_t* d_sendDispls;
  size_t* d_recvCounts;
  size_t* d_recvDispls;
};

class AllToAllVAlgoBuilder : public mscclpp::AlgorithmBuilder {
 public:
  AllToAllVAlgoBuilder() = default;
  ~AllToAllVAlgoBuilder() {
    if (proxyService_) {
      proxyService_->stopProxy();
    }
  }

  std::shared_ptr<mscclpp::Algorithm> build() override {
    auto self = std::make_shared<AllToAllVAlgoBuilder>();
    std::shared_ptr<mscclpp::Algorithm> alltoallvAlgo = std::make_shared<mscclpp::NativeAlgorithm>(
        "alltoallv", "alltoallv",
        // Initialize function
        [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
        // Kernel execution function
        [self](const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize, size_t outputSize,
               mscclpp::DataType dtype, [[maybe_unused]] mscclpp::ReduceOp op, cudaStream_t stream, int nBlocks,
               int nThreadsPerBlock, const std::unordered_map<std::string, uintptr_t>& extras) {
          return self->alltoallvKernelFunc(ctx, input, output, inputSize, outputSize, dtype, stream, extras);
        },
        // Context initialization function
        [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
               size_t outputSize,
               mscclpp::DataType dtype) { return self->initAlltoallvContext(comm, input, output, inputSize, outputSize, dtype); },
        // Context key generation function
        [self](const void* input, void* output, size_t inputSize, size_t outputSize, mscclpp::DataType dtype) {
          return self->generateAlltoallvContextKey(input, output, inputSize, outputSize, dtype);
        });
    return alltoallvAlgo;
  }

 private:
  std::vector<mscclpp::Connection> conns_;
  std::shared_ptr<mscclpp::ProxyService> proxyService_;
  int worldSize_;

  void initialize(std::shared_ptr<mscclpp::Communicator> comm) {
    std::vector<std::shared_future<mscclpp::Connection>> connectionFutures;
    worldSize_ = comm->bootstrap()->getNranks();
    for (int i = 0; i < worldSize_; i++) {
      if (i == comm->bootstrap()->getRank()) continue;
      connectionFutures.push_back(comm->connect(mscclpp::Transport::CudaIpc, i));
    }
    std::vector<mscclpp::Connection> connections;
    std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
                   [](const auto& future) { return future.get(); });
    this->conns_ = std::move(connections);
    proxyService_ = std::make_shared<mscclpp::ProxyService>();
    proxyService_->startProxy(true);
  }

  mscclpp::CommResult alltoallvKernelFunc(const std::shared_ptr<void> ctx, const void* input, void* output,
                                          size_t inputSize, size_t outputSize,
                                          [[maybe_unused]] mscclpp::DataType dtype,
                                          cudaStream_t stream,
                                          const std::unordered_map<std::string, uintptr_t>& extras) {
    auto algoCtx = std::static_pointer_cast<AllToAllVContext>(ctx);
    int rank = algoCtx->rank;
    int worldSize = algoCtx->worldSize;

    // Extract send/recv counts and displacements from extras
    // The caller should pass these as device pointers via extras map
    auto it_sendCounts = extras.find("sendCounts");
    auto it_sendDispls = extras.find("sendDispls");
    auto it_recvCounts = extras.find("recvCounts");
    auto it_recvDispls = extras.find("recvDispls");

    if (it_sendCounts == extras.end() || it_sendDispls == extras.end() ||
        it_recvCounts == extras.end() || it_recvDispls == extras.end()) {
      return mscclpp::CommResult::CommInternalError;
    }

    const size_t* d_sendCounts = reinterpret_cast<const size_t*>(it_sendCounts->second);
    const size_t* d_sendDispls = reinterpret_cast<const size_t*>(it_sendDispls->second);
    const size_t* d_recvCounts = reinterpret_cast<const size_t*>(it_recvCounts->second);
    const size_t* d_recvDispls = reinterpret_cast<const size_t*>(it_recvDispls->second);

    // Reset device syncer
    mscclpp::DeviceSyncer syncer = {};
    cudaMemcpyToSymbolAsync(alltoallvDeviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer), 0,
                            cudaMemcpyHostToDevice, stream);

    // Use simple kernel for small world sizes, multi-block for larger
    if (worldSize <= 16) {
      int nThreads = (worldSize - 1) * WARP_SIZE;
      if (nThreads < 32) nThreads = 32;
      if (nThreads > 1024) nThreads = 1024;

      alltoallv_simple_kernel<<<1, nThreads, 0, stream>>>(
          algoCtx->portChannelDeviceHandles.get(),
          rank, worldSize,
          input, output,
          d_sendCounts, d_sendDispls,
          d_recvCounts, d_recvDispls);
    } else {
      alltoallv_kernel<<<1, 1024, 0, stream>>>(
          algoCtx->portChannelDeviceHandles.get(),
          rank, worldSize,
          input, output,
          d_sendCounts, d_sendDispls,
          d_recvCounts, d_recvDispls);
    }

    if (cudaGetLastError() == cudaSuccess) {
      return mscclpp::CommResult::CommSuccess;
    }
    return mscclpp::CommResult::CommInternalError;
  }

  std::shared_ptr<void> initAlltoallvContext(std::shared_ptr<mscclpp::Communicator> comm, const void* input,
                                             void* output, size_t inputSize, size_t outputSize,
                                             mscclpp::DataType dtype) {
    auto ctx = std::make_shared<AllToAllVContext>();
    ctx->rank = comm->bootstrap()->getRank();
    ctx->worldSize = comm->bootstrap()->getNranks();
    ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

    // Register memories for input and output buffers
    mscclpp::RegisteredMemory inputBufRegMem =
        comm->registerMemory((void*)input, inputSize, mscclpp::Transport::CudaIpc);
    mscclpp::RegisteredMemory outputBufRegMem =
        comm->registerMemory(output, outputSize, mscclpp::Transport::CudaIpc);

    // Exchange output buffer registration with all peers
    std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteRegMemories;
    for (int i = 0; i < ctx->worldSize; i++) {
      if (i == ctx->rank) continue;
      comm->sendMemory(outputBufRegMem, i, 0);
      remoteRegMemories.push_back(comm->recvMemory(i, 0));
    }

    // Setup port channels for each peer
    std::vector<mscclpp::DeviceHandle<mscclpp::PortChannel>> portChannels;
    mscclpp::MemoryId inputMemoryId = this->proxyService_->addMemory(inputBufRegMem);

    for (size_t i = 0; i < this->conns_.size(); i++) {
      auto remoteMemory = remoteRegMemories[i].get();
      mscclpp::MemoryId remoteMemoryId = this->proxyService_->addMemory(remoteMemory);
      portChannels.push_back(mscclpp::deviceHandle(this->proxyService_->portChannel(
          this->proxyService_->buildAndAddSemaphore(*comm, this->conns_[i]), remoteMemoryId, inputMemoryId)));
    }

    // Allocate and copy port channels to device
    ctx->portChannelDeviceHandles =
        mscclpp::detail::gpuCallocShared<mscclpp::DeviceHandle<mscclpp::PortChannel>>(portChannels.size());
    mscclpp::gpuMemcpy(ctx->portChannelDeviceHandles.get(), portChannels.data(), portChannels.size(),
                       cudaMemcpyHostToDevice);

    // Keep registered memory references to prevent deallocation
    std::transform(remoteRegMemories.begin(), remoteRegMemories.end(), std::back_inserter(ctx->registeredMemories),
                   [](const auto& fut) { return fut.get(); });
    ctx->registeredMemories.push_back(inputBufRegMem);
    ctx->registeredMemories.push_back(outputBufRegMem);

    return ctx;
  }

  mscclpp::AlgorithmCtxKey generateAlltoallvContextKey(const void* input, void* output, size_t inputSize,
                                                       size_t outputSize, mscclpp::DataType dtype) {
    return {(void*)input, output, inputSize, outputSize, 0};
  }
};

std::shared_ptr<mscclpp::Algorithm> createAlltoallvAlgorithm() {
  auto alltoallvAlgoBuilder = std::make_shared<AllToAllVAlgoBuilder>();
  return alltoallvAlgoBuilder->build();
}

void deletePtr(PyObject* capsule) {
  const char* name = PyCapsule_GetName(capsule);
  void* p = PyCapsule_GetPointer(capsule, name);
  if (p == nullptr) {
    PyErr_WriteUnraisable(capsule);
    return;
  }
  auto* ptr = static_cast<std::shared_ptr<mscclpp::Algorithm>*>(p);
  delete ptr;
}

PyObject* getCapsule(std::shared_ptr<mscclpp::Algorithm> algo) {
  auto* ptrCopy = new std::shared_ptr<mscclpp::Algorithm>(algo);
  PyObject* capsule = PyCapsule_New(ptrCopy, mscclpp::ALGORITHM_NATIVE_CAPSULE_NAME, deletePtr);
  if (capsule == nullptr) {
    delete ptrCopy;
    throw pybind11::error_already_set();
  }
  return capsule;
}

PYBIND11_MODULE(mscclpp_alltoallv, m) {
  m.doc() = "AllToAllV implementation for MSCCLPP - handles variable element counts per rank";
  m.def(
      "create_alltoallv_algorithm",
      []() { return py::reinterpret_steal<py::capsule>(getCapsule(createAlltoallvAlgorithm())); },
      "Create an alltoallv algorithm and return it as a PyCapsule usable by MSCCL++ Python bindings");
}
