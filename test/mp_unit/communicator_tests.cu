// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mpi.h>

#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/semaphore.hpp>

#include "mp_unit_tests.hpp"

void CommunicatorTestBase::SetUp() {
  MultiProcessTest::SetUp();

  if (numRanksToUse == -1) {
    numRanksToUse = gEnv->worldSize;
  }
  ASSERT_LE(numRanksToUse, gEnv->worldSize);

  ibTransport = ibIdToTransport(rankToLocalRank(gEnv->rank));
  MSCCLPP_CUDATHROW(cudaSetDevice(rankToLocalRank(gEnv->rank)));

  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  mscclpp::UniqueId id;
  if (gEnv->rank < numRanksToUse) {
    bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, numRanksToUse);
    if (gEnv->rank == 0) id = bootstrap->createUniqueId();
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (gEnv->rank >= numRanksToUse) {
    return;
  }
  bootstrap->initialize(id);
  communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
}

void CommunicatorTestBase::TearDown() {
  connections.clear();
  communicator.reset();
  registeredMemories.clear();
  MultiProcessTest::TearDown();
}

void CommunicatorTestBase::setNumRanksToUse(int num) { numRanksToUse = num; }

void CommunicatorTestBase::connectMesh(bool useIpc, bool useIb, bool useEthernet) {
  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures(numRanksToUse);
  std::vector<std::shared_future<mscclpp::Connection>> cpuConnectionFutures(numRanksToUse);
  for (int i = 0; i < numRanksToUse; i++) {
    if (i != gEnv->rank) {
      if ((rankToNode(i) == rankToNode(gEnv->rank)) && useIpc) {
        connectionFutures[i] = communicator->connect(mscclpp::Transport::CudaIpc, i);
      } else if (useIb) {
        connectionFutures[i] = communicator->connect(ibTransport, i);
        cpuConnectionFutures[i] = communicator->connect({ibTransport, mscclpp::DeviceType::CPU}, i);
      } else if (useEthernet) {
        connectionFutures[i] = communicator->connect(mscclpp::Transport::Ethernet, i);
        cpuConnectionFutures[i] = communicator->connect({mscclpp::Transport::Ethernet, mscclpp::DeviceType::CPU}, i);
      }
    }
  }
  for (int i = 0; i < numRanksToUse; i++) {
    if (i != gEnv->rank) {
      connections[i] = connectionFutures[i].get();
      if (cpuConnectionFutures[i].valid()) {
        cpuConnections[i] = cpuConnectionFutures[i].get();
      }
    }
  }
}

// Register a local memory and receive corresponding remote memories
void CommunicatorTestBase::registerMemoryPairs(void* buff, size_t buffSize, mscclpp::TransportFlags transport, int tag,
                                               const std::vector<int>& remoteRanks,
                                               mscclpp::RegisteredMemory& localMemory,
                                               std::unordered_map<int, mscclpp::RegisteredMemory>& remoteMemories) {
  localMemory = communicator->registerMemory(buff, buffSize, transport);
  std::unordered_map<int, std::shared_future<mscclpp::RegisteredMemory>> futureRemoteMemories;
  for (int remoteRank : remoteRanks) {
    if (remoteRank != communicator->bootstrap()->getRank()) {
      communicator->sendMemory(localMemory, remoteRank, tag);
      futureRemoteMemories[remoteRank] = communicator->recvMemory(remoteRank, tag);
    }
  }
  for (int remoteRank : remoteRanks) {
    if (remoteRank != communicator->bootstrap()->getRank()) {
      remoteMemories[remoteRank] = futureRemoteMemories[remoteRank].get();
    }
  }
}

// Register a local memory an receive one corresponding remote memory
void CommunicatorTestBase::registerMemoryPair(void* buff, size_t buffSize, mscclpp::TransportFlags transport, int tag,
                                              int remoteRank, mscclpp::RegisteredMemory& localMemory,
                                              mscclpp::RegisteredMemory& remoteMemory) {
  std::vector<int> remoteRanks = {remoteRank};
  std::unordered_map<int, mscclpp::RegisteredMemory> remoteMemories;
  registerMemoryPairs(buff, buffSize, transport, tag, remoteRanks, localMemory, remoteMemories);
  remoteMemory = remoteMemories[remoteRank];
}

void CommunicatorTest::SetUp() {
  CommunicatorTestBase::SetUp();

  ASSERT_EQ((deviceBufferSize / sizeof(int)) % gEnv->worldSize, 0);

#if defined(USE_IBVERBS)
  connectMesh(true, true, false);
#else
  connectMesh(true, false, false);
#endif

  devicePtr.resize(numBuffers);
  localMemory.resize(numBuffers);
  remoteMemory.resize(numBuffers);

  std::vector<int> remoteRanks;
  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank) {
      remoteRanks.push_back(i);
    }
  }

#if defined(USE_IBVERBS)
  auto transport = mscclpp::Transport::CudaIpc | ibTransport;
#else
  auto transport = mscclpp::Transport::CudaIpc;
#endif

  for (size_t n = 0; n < numBuffers; n++) {
    devicePtr[n] = mscclpp::detail::gpuCallocShared<int>(deviceBufferSize / sizeof(int));
    registerMemoryPairs(devicePtr[n].get(), deviceBufferSize, transport, 0, remoteRanks, localMemory[n],
                        remoteMemory[n]);
  }
}

void CommunicatorTest::TearDown() {
  remoteMemory.clear();
  localMemory.clear();
  devicePtr.clear();
  CommunicatorTestBase::TearDown();
}

void CommunicatorTest::deviceBufferInit() {
  size_t dataCount = deviceBufferSize / sizeof(int);
  for (int n = 0; n < (int)devicePtr.size(); n++) {
    std::vector<int> hostBuffer(dataCount, 0);
    for (size_t i = 0; i < dataCount; i++) {
      hostBuffer[i] = gEnv->rank + n * gEnv->worldSize;
    }
    mscclpp::gpuMemcpy<int>(devicePtr[n].get(), hostBuffer.data(), dataCount, cudaMemcpyHostToDevice);
  }
}

void CommunicatorTest::writeToRemote(int dataCountPerRank) {
  for (size_t n = 0; n < numBuffers; n++) {
    for (int i = 0; i < gEnv->worldSize; i++) {
      if (i != gEnv->rank) {
        auto& conn = connections.at(i);
        auto& peerMemory = remoteMemory[n].at(i);
        conn.write(peerMemory, gEnv->rank * dataCountPerRank * sizeof(int), localMemory[n],
                   gEnv->rank * dataCountPerRank * sizeof(int), dataCountPerRank * sizeof(int));
        conn.flush();
      }
    }
  }
}

bool CommunicatorTest::testWriteCorrectness(bool skipLocal) {
  size_t dataCount = deviceBufferSize / sizeof(int);
  for (int n = 0; n < (int)devicePtr.size(); n++) {
    std::vector<int> hostBuffer(dataCount, 0);
    mscclpp::gpuMemcpy<int>(hostBuffer.data(), devicePtr[n].get(), dataCount, cudaMemcpyDeviceToHost);
    for (int i = 0; i < gEnv->worldSize; i++) {
      if (((i / gEnv->nRanksPerNode) == (gEnv->rank / gEnv->nRanksPerNode)) && skipLocal) {
        continue;
      }
      for (size_t j = i * dataCount / gEnv->worldSize; j < (i + 1) * dataCount / gEnv->worldSize; j++) {
        if (hostBuffer[j] != i + n * gEnv->worldSize) {
          return false;
        }
      }
    }
  }
  return true;
}

TEST(CommunicatorTest, BasicWrite) {
  if (gEnv->rank >= numRanksToUse) return;

  deviceBufferInit();
  communicator->bootstrap()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);
  communicator->bootstrap()->barrier();

  // polling until it becomes ready
  bool ready = false;
  int niter = 0;
  do {
    ready = testWriteCorrectness();
    niter++;
    if (niter == 10000) {
      FAIL() << "Polling is stuck.";
    }
  } while (!ready);
  communicator->bootstrap()->barrier();

  // Verify the nonzero byte-offset update only on connections that actually use CudaIpc.
  // The capacity decision is uniform; when it fits, all ranks enter every barrier and IB edges skip work.
  constexpr uint64_t offsetStrideBytes = 128;
  constexpr uint64_t guardValue = 0xBAD00BADDEADBEEFULL;
  constexpr uint64_t updateValueBase = 0xD15EA5E000000000ULL;
  // The stride separates sender slots; the former byte-plus-word offset lands at 9 * offset.
  auto offsetForSender = [offsetStrideBytes](int sender) {
    return sizeof(uint64_t) + static_cast<uint64_t>(sender) * offsetStrideBytes;
  };
  const uint64_t largestFormerWrongEnd = 9 * offsetForSender(gEnv->worldSize - 1) + sizeof(uint64_t);
  // All ranks see the same world size, so an undersized fixture skips this extra phase uniformly.
  if (largestFormerWrongEnd > static_cast<uint64_t>(deviceBufferSize)) return;

  auto* localWords = reinterpret_cast<uint64_t*>(devicePtr[0].get());

  for (auto& [peer, conn] : connections) {
    if (conn.transport() != mscclpp::Transport::CudaIpc) continue;
    const uint64_t offset = offsetForSender(peer);
    const uint64_t formerWrongOffset = 9 * offset;
    mscclpp::gpuMemcpy<uint64_t>(localWords + offset / sizeof(uint64_t), &guardValue, 1, cudaMemcpyHostToDevice);
    mscclpp::gpuMemcpy<uint64_t>(localWords + formerWrongOffset / sizeof(uint64_t), &guardValue, 1,
                                 cudaMemcpyHostToDevice);
  }
  communicator->bootstrap()->barrier();

  for (auto& [peer, conn] : connections) {
    if (conn.transport() != mscclpp::Transport::CudaIpc) continue;
    const uint64_t offset = offsetForSender(gEnv->rank);
    uint64_t source = 0;
    conn.updateAndSync(remoteMemory[0].at(peer), offset, &source, updateValueBase + gEnv->rank);
    conn.flush();  // Keep the host source word alive until the asynchronous copy completes.
  }
  communicator->bootstrap()->barrier();

  for (const auto& [peer, conn] : connections) {
    if (conn.transport() != mscclpp::Transport::CudaIpc) continue;
    const uint64_t offset = offsetForSender(peer);
    const uint64_t formerWrongOffset = 9 * offset;
    uint64_t observed = 0;
    uint64_t formerWrong = 0;
    mscclpp::gpuMemcpy<uint64_t>(&observed, localWords + offset / sizeof(uint64_t), 1, cudaMemcpyDeviceToHost);
    mscclpp::gpuMemcpy<uint64_t>(&formerWrong, localWords + formerWrongOffset / sizeof(uint64_t), 1,
                                 cudaMemcpyDeviceToHost);
    EXPECT_EQ(observed, updateValueBase + static_cast<uint64_t>(peer));
    EXPECT_EQ(formerWrong, guardValue);
  }
  communicator->bootstrap()->barrier();
}

__global__ void kernelWaitSemaphores(mscclpp::Host2DeviceSemaphore::DeviceHandle* deviceSemaphores, int rank,
                                     int worldSize) {
  int tid = threadIdx.x;
  if (tid != rank && tid < worldSize) {
    deviceSemaphores[tid].wait();
  }
}

TEST(CommunicatorTest, WriteWithDeviceSemaphores) {
  if (gEnv->rank >= numRanksToUse) return;

  std::unordered_map<int, std::shared_ptr<mscclpp::Host2DeviceSemaphore>> semaphores;
  for (auto entry : connections) {
    auto& conn = entry.second;
    semaphores.insert({entry.first, std::make_shared<mscclpp::Host2DeviceSemaphore>(*communicator.get(), conn)});
  }
  communicator->bootstrap()->barrier();

  deviceBufferInit();
  communicator->bootstrap()->barrier();

  auto deviceSemaphoreHandles =
      mscclpp::detail::gpuCallocShared<mscclpp::Host2DeviceSemaphore::DeviceHandle>(gEnv->worldSize);
  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank) {
      mscclpp::Host2DeviceSemaphore::DeviceHandle deviceHandle = semaphores[i]->deviceHandle();
      mscclpp::gpuMemcpy<mscclpp::Host2DeviceSemaphore::DeviceHandle>(deviceSemaphoreHandles.get() + i, &deviceHandle,
                                                                      1, cudaMemcpyHostToDevice);
    }
  }
  communicator->bootstrap()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank) {
      semaphores[i]->signal();
    }
  }

  kernelWaitSemaphores<<<1, gEnv->worldSize>>>(deviceSemaphoreHandles.get(), gEnv->rank, gEnv->worldSize);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  ASSERT_TRUE(testWriteCorrectness());
  communicator->bootstrap()->barrier();
}

TEST(CommunicatorTest, WriteWithHostSemaphores) {
  if (gEnv->rank >= numRanksToUse) return;

  std::unordered_map<int, std::shared_ptr<mscclpp::Host2HostSemaphore>> semaphores;
  // Iterate over cpuConnections, which is guaranteed by design to only contain CPU connections.
  for (auto entry : cpuConnections) {
    auto& conn = entry.second;
    // Host2HostSemaphore cannot be used with CudaIpc transport
    if (conn.transport() == mscclpp::Transport::CudaIpc) continue;
    semaphores.insert({entry.first, std::make_shared<mscclpp::Host2HostSemaphore>(*communicator.get(), conn)});
  }
  communicator->bootstrap()->barrier();

  deviceBufferInit();
  communicator->bootstrap()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i].transport() != mscclpp::Transport::CudaIpc) {
      semaphores[i]->signal();
    }
  }

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i].transport() != mscclpp::Transport::CudaIpc) {
      semaphores[i]->wait();
    }
  }

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i].transport() != mscclpp::Transport::CudaIpc) {
      semaphores[i]->signal();
    }
  }

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i].transport() != mscclpp::Transport::CudaIpc) {
      semaphores[i]->wait();
    }
  }

  ASSERT_TRUE(testWriteCorrectness());
  communicator->bootstrap()->barrier();
}