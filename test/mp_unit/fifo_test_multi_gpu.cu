// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gtest/gtest.h>

#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>

#include "mp_unit_tests.hpp"

#define MAGIC_CONST 777

// Constants for timeout and trigger calculation
constexpr uint64_t TIMEOUT_SPINS = 1000000;
constexpr int MIN_TRIGGERS = 1000;
constexpr int MIN_WARMUP_TRIGGERS = 100;
constexpr int TRIGGERS_PER_FIFO_SIZE = 10;
constexpr int WARMUP_TRIGGERS_PER_FIFO_SIZE = 2;

void FifoMultiGPUTest::SetUp() {
  // Need at least two ranks within a node
  if (gEnv->nRanksPerNode < 2) {
    GTEST_SKIP();
  }
  // Use only two ranks
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
}

void FifoMultiGPUTest::TearDown() { CommunicatorTestBase::TearDown(); }

void FifoMultiGPUTest::setupMeshConnections(std::vector<mscclpp::PortChannel>& portChannels, bool useIPC,
                                                   bool useIb, bool useEthernet, void* sendBuff, size_t sendBuffBytes,
                                                   void* recvBuff, size_t recvBuffBytes) {
  const int rank = communicator->bootstrap()->getRank();
  const int worldSize = communicator->bootstrap()->getNranks();
  const bool isInPlace = (recvBuff == nullptr);
  mscclpp::TransportFlags transport;

  if (useIPC) transport |= mscclpp::Transport::CudaIpc;
  if (useIb) transport |= ibTransport;
  if (useEthernet) transport |= mscclpp::Transport::Ethernet;

  mscclpp::RegisteredMemory sendBufRegMem = communicator->registerMemory(sendBuff, sendBuffBytes, transport);
  mscclpp::RegisteredMemory recvBufRegMem;
  if (!isInPlace) {
    recvBufRegMem = communicator->registerMemory(recvBuff, recvBuffBytes, transport);
  }

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    if ((rankToNode(r) == rankToNode(gEnv->rank)) && useIPC) {
      this->connectionFutures[r] = communicator->connect(r, 0, mscclpp::Transport::CudaIpc);
    } else if (useIb) {
      this->connectionFutures[r] = communicator->connect(r, 0, ibTransport);
    } else if (useEthernet) {
      this->connectionFutures[r] = communicator->connect(r, 0, mscclpp::Transport::Ethernet);
    }

    if (isInPlace) {
      communicator->sendMemory(sendBufRegMem, r, 0);
    } else {
      communicator->sendMemory(recvBufRegMem, r, 0);
    }
    remoteMemFutures[r] = communicator->recvMemory(r, 0);
  }

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    mscclpp::SemaphoreId cid = proxyService->buildAndAddSemaphore(*communicator, this->connectionFutures[r].get());

    portChannels.emplace_back(proxyService->portChannel(cid, proxyService->addMemory(this->remoteMemFutures[r].get()),
                                                        proxyService->addMemory(sendBufRegMem)));
  }
}

__constant__ mscclpp::PortChannelDeviceHandle gPortChannel;

__constant__ mscclpp::FifoDeviceHandle gFifoDeviceHandle;

__global__ void kernelFifoPushAndSignal(mscclpp::PortChannelDeviceHandle portHandle, size_t numTriggers, mscclpp::SemaphoreId semaphoreId) {
  mscclpp::FifoDeviceHandle& fifo = gFifoDeviceHandle;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  mscclpp::ProxyTrigger trigger;
  for (size_t i = 1; i <= numTriggers; ++i) {
    trigger.fst = semaphoreId;
    trigger.snd = tid ^ i;
    fifo.push(trigger);
  }
  __syncthreads();

  if (tid == 0) {
      portHandle.signal();
  }
}

__global__ void kernelWaitAndCheck(mscclpp::PortChannelDeviceHandle portHandle, uint64_t* localFlag, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Wait for signal from GPU0
    portHandle.wait();
    __syncthreads();

    // Check if flag was updated
    if (tid == 0) {
        if (*localFlag != 1ULL) {
            *result = 1; // Error
        }
    }
}

static void setupCuda(int& cudaDevice, int& numaNode) {
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  numaNode = mscclpp::getDeviceNumaNode(cudaDevice);
  mscclpp::numaBind(numaNode);
}

// Helper function to consume triggers from FIFO
static bool consumeTriggers(std::unique_ptr<mscclpp::Fifo>& hostFifo, int numTriggers, int parallel, int flushPeriod,
    std::shared_ptr<mscclpp::Connection> connection,
    mscclpp::RegisteredMemory remoteFlagRegMem) {
  int totalTriggers = numTriggers * parallel;
  std::unordered_map<int, int> triggerCounts;
  for (int i = 0; i < totalTriggers; ++i) {
    mscclpp::ProxyTrigger trigger;
    uint64_t spin = 0;
    do {
      trigger = hostFifo->poll();
      if (spin++ > TIMEOUT_SPINS) {
        return false;
      }
      // printf("get fst is %lu, snd is %lu idx is %d\n", trigger.fst, trigger.snd, i);
    } while (trigger.fst == 0 || trigger.snd == 0);

    // Process trigger (see src/proxy.cc)
    trigger.snd ^= ((uint64_t)1 << (uint64_t)63);
    trigger.snd = trigger.snd ^ trigger.fst;
    assert(triggerCounts[trigger.snd] + 1 == trigger.fst);
    triggerCounts[trigger.snd]++;
    hostFifo->pop();

    // Host-side atomicAdd for each trigger
    connection->atomicAdd(remoteFlagRegMem, 0, 1);

    // Flush periodically
    if (((i + 1) % flushPeriod) == 0) {
      hostFifo->flushTail();
    }
  }
  return true;
}

void FifoMultiGPUTest::testFifo(FifoTestParams params) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::PortChannel> portChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(nElem).memory();
  setupMeshConnections(portChannels, params.useIPC, params.useIB, params.useEthernet, buff.get(), nElem * sizeof(int));

  std::vector<DeviceHandle<mscclpp::PortChannel>> portChannelHandles;
  for (auto& ch : portChannels) portChannelHandles.push_back(ch.deviceHandle());

  ASSERT_EQ(portChannels.size(), 1);

  // Allocate semaphore flags on both GPUs
  cudaSetDevice(0);
  uint64_t* gpu0_semaphore_flag;
  cudaMalloc(&gpu0_semaphore_flag, sizeof(uint64_t));
  cudaMemset(gpu0_semaphore_flag, 0, sizeof(uint64_t));

  cudaSetDevice(1);
  uint64_t* gpu1_semaphore_flag;
  cudaMalloc(&gpu1_semaphore_flag, sizeof(uint64_t));
  cudaMemset(gpu1_semaphore_flag, 0, sizeof(uint64_t));

  // Register semaphore flags as remote memory
  auto gpu0_flag_regmem = communicator->registerMemory(gpu0_semaphore_flag, sizeof(uint64_t), ibTransport);
  auto gpu1_flag_regmem = communicator->registerMemory(gpu1_semaphore_flag, sizeof(uint64_t), ibTransport);

  // On host, after connections are established
  auto semaphoreId = proxyService->buildAndAddSemaphore(*communicator, connectionFutures[1].get());
  auto portChannel = proxyService->portChannel(semaphoreId, proxyService->addMemory(gpu1_flag_regmem), proxyService->addMemory(gpu0_flag_regmem));
  auto portChannelHandle = portChannel.deviceHandle();
  cudaMemcpyToSymbol(gPortChannel, &portChannelHandle, sizeof(portChannelHandle), 0, cudaMemcpyHostToDevice);

  int* d_result;
  cudaMalloc(&d_result, sizeof(int));
  cudaMemset(d_result, 0, sizeof(int));

  proxyService->startProxy();

  int cudaDevice, numaNode;
  setupCuda(cudaDevice, numaNode);

  // Allocate FIFO on device 0
  cudaSetDevice(0);
  int fifoSize = 1024 * 1024; // 1M elements
  auto hostFifo = std::make_unique<mscclpp::Fifo>(fifoSize);

  // On host, after reading trigger from FIFO
  // Calculate triggers based on FIFO size
  const int numTriggers = std::max(MIN_TRIGGERS, static_cast<int>(hostFifo->size() * TRIGGERS_PER_FIFO_SIZE));
  const int warmupTriggers =
      std::max(MIN_WARMUP_TRIGGERS, static_cast<int>(hostFifo->size() * WARMUP_TRIGGERS_PER_FIFO_SIZE));

  // Launch on GPU0
  cudaSetDevice(0);
  kernelFifoPushAndSignal<<<1, 32>>>(gPortChannel, numTriggers, semaphoreId);

  // Launch on GPU1
  cudaSetDevice(1);
  kernelWaitAndCheck<<<1, 32>>>(gPortChannel, gpu1_semaphore_flag, d_result);

  int numParallel = 8;
  int flushPeriod = 64;

  auto remoteFlagRegMem = remoteMemFutures[1].get();
  auto connection = connectionFutures[1].get();
  if (!consumeTriggers(hostFifo, numTriggers, numParallel, flushPeriod, connection, remoteFlagRegMem)) {
      // handle error
  }
  hostFifo->flushTail(true);

  proxyService->stopProxy();

  // Synchronize and check
  cudaSetDevice(0);
  cudaDeviceSynchronize();
  cudaSetDevice(1);
  cudaDeviceSynchronize();

  int h_result = 0;
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  ASSERT_EQ(h_result, 0);
}

TEST_F(FifoMultiGPUTest, Fifo) {
  testFifo(FifoTestParams{.useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false});
}
