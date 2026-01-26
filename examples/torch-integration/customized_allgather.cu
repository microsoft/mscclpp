// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <Python.h>
#include <mscclpp/ext/nccl/nccl.h>
#include <pybind11/pybind11.h>

#include <mscclpp/algorithm.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>

namespace py = pybind11;

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

__global__ void __launch_bounds__(1024)
    allgather(mscclpp::DeviceHandle<mscclpp::PortChannel>* portChannels, int rank, size_t nbytesPerGPU) {
  int warpId = threadIdx.x / WARP_SIZE;

  // Each warp is responsible for one of the remote ranks
  mscclpp::DeviceHandle<mscclpp::PortChannel> portChan = portChannels[warpId];

  // this allgather is really simple and implemented as an alltoall

  // this thread's role is a sender role
  // put your data asynchronously
  if (threadIdx.x % WARP_SIZE == 0) {
    portChan.putWithSignal(rank * nbytesPerGPU, 0, nbytesPerGPU);
  }
  // make sure everyone is put their data before some thread randomly blocks everyone else in signal
  __syncthreads();
  // push with flag and sync to make sure the data is received
  if (threadIdx.x % WARP_SIZE == 0) {
    portChan.flush();
  }

  // this thread's role is a receiver role. wait on the semaphore to make sure the data is ready
  if (threadIdx.x % WARP_SIZE == 0) {
    portChan.wait();
  }
}

struct Context {
  int rank;
  int workSize;
  int nRanksPerNode;

  std::vector<mscclpp::RegisteredMemory> registeredMemories;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::PortChannel>> portChannelDeviceHandles;
};

class AllgatherAlgoBuilder : public mscclpp::AlgorithmBuilder {
 public:
  AllgatherAlgoBuilder() = default;
  ~AllgatherAlgoBuilder() {
    if (proxyService_) {
      proxyService_->stopProxy();
    }
  }

  std::shared_ptr<mscclpp::Algorithm> build() override {
    auto self = std::make_shared<AllgatherAlgoBuilder>();
    std::shared_ptr<mscclpp::Algorithm> allgatherAlgo = std::make_shared<mscclpp::NativeAlgorithm>(
        "allgather", "allgather", [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
        [self](const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize, size_t outputSize,
               mscclpp::DataType dtype, [[maybe_unused]] mscclpp::ReduceOp op, cudaStream_t stream, int nBlocks,
               int nThreadsPerBlock, const std::unordered_map<std::string, uintptr_t>& extras) {
          return self->allgatherKernelFunc(ctx, input, output, inputSize, dtype, stream);
        },
        [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
               size_t outputSize,
               mscclpp::DataType dtype) { return self->initAllgatherContext(comm, input, output, inputSize, dtype); },
        [self](const void* input, void* output, size_t inputSize, size_t outputSize, mscclpp::DataType dtype) {
          return self->generateAllgatherContextKey(input, output, inputSize, outputSize, dtype);
        });
    return allgatherAlgo;
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

  mscclpp::CommResult allgatherKernelFunc(const std::shared_ptr<void> ctx, const void* input, void* output,
                                          size_t inputBytes, [[maybe_unused]] mscclpp::DataType dtype,
                                          cudaStream_t stream) {
    auto algoCtx = std::static_pointer_cast<Context>(ctx);
    int rank = algoCtx->rank;
    int worldSize = algoCtx->workSize;

    int nThreadsPerBlock = (worldSize - 1) * WARP_SIZE;
    allgather<<<1, nThreadsPerBlock, 0, stream>>>(algoCtx->portChannelDeviceHandles.get(), rank, inputBytes);
    if (cudaGetLastError() == cudaSuccess) {
      return mscclpp::CommResult::CommSuccess;
    }
    return mscclpp::CommResult::CommInternalError;
  }

  std::shared_ptr<void> initAllgatherContext(std::shared_ptr<mscclpp::Communicator> comm, const void* input,
                                             void* output, size_t inputBytes, mscclpp::DataType dtype) {
    auto ctx = std::make_shared<Context>();
    ctx->rank = comm->bootstrap()->getRank();
    ctx->workSize = comm->bootstrap()->getNranks();
    ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

    // register memories
    mscclpp::RegisteredMemory inputBufRegMem =
        comm->registerMemory((void*)input, inputBytes, mscclpp::Transport::CudaIpc);
    mscclpp::RegisteredMemory outputBufRegMem =
        comm->registerMemory(output, inputBytes * ctx->workSize, mscclpp::Transport::CudaIpc);
    std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteRegMemories;
    for (int i = 0; i < ctx->workSize; i++) {
      if (i == ctx->rank) continue;
      comm->sendMemory(outputBufRegMem, i, 0);
      remoteRegMemories.push_back(comm->recvMemory(i, 0));
    }

    // setup channels
    std::vector<mscclpp::DeviceHandle<mscclpp::PortChannel>> portChannels;
    mscclpp::MemoryId inputMemoryId = this->proxyService_->addMemory(inputBufRegMem);
    for (int i = 0; i < this->conns_.size(); i++) {
      auto remoteMemory = remoteRegMemories[i].get();
      mscclpp::MemoryId remoteMemoryId = this->proxyService_->addMemory(remoteMemory);
      portChannels.push_back(mscclpp::deviceHandle(this->proxyService_->portChannel(
          this->proxyService_->buildAndAddSemaphore(*comm, this->conns_[i]), remoteMemoryId, inputMemoryId)));
    }
    ctx->portChannelDeviceHandles =
        mscclpp::detail::gpuCallocShared<mscclpp::DeviceHandle<mscclpp::PortChannel>>(portChannels.size());
    mscclpp::gpuMemcpy(ctx->portChannelDeviceHandles.get(), portChannels.data(), portChannels.size(),
                       cudaMemcpyHostToDevice);

    // keep registered memory references
    std::transform(remoteRegMemories.begin(), remoteRegMemories.end(), std::back_inserter(ctx->registeredMemories),
                   [](const auto& fut) { return fut.get(); });
    ctx->registeredMemories.push_back(inputBufRegMem);
    ctx->registeredMemories.push_back(outputBufRegMem);

    return ctx;
  }

  mscclpp::AlgorithmCtxKey generateAllgatherContextKey(const void* input, void* output, size_t inputSize,
                                                       size_t outputSize, mscclpp::DataType dtype) {
    return {(void*)input, output, inputSize, outputSize, 0};
  }
};

std::shared_ptr<mscclpp::Algorithm> createAllgatherAlgorithm() {
  auto allgatherAlgoBuilder = std::make_shared<AllgatherAlgoBuilder>();
  return allgatherAlgoBuilder->build();
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

PYBIND11_MODULE(mscclpp_native, m) {
  m.doc() = "A simple C++ extension for mscclpp customized algorithm";
  m.def(
      "create_allgather_algorithm",
      []() { return py::reinterpret_steal<py::capsule>(getCapsule(createAllgatherAlgorithm())); },
      "Create an allgather algorithm and return it as a PyCapsule usable by MSCCL++ Python bindings");
}