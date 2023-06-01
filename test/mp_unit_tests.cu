#include <gtest/gtest.h>
#include <mpi.h>

#include <iostream>
#include <mscclpp/channel.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/cuda_utils.hpp>
#include <mscclpp/epoch.hpp>
#include <mscclpp/utils.hpp>
#include <sstream>

#include "ib.hpp"
#include "infiniband/verbs.h"

static const char gDefaultIpPort[] = "127.0.0.1:50053";

class MultiProcessTestEnv : public ::testing::Environment {
 public:
  MultiProcessTestEnv(int argc, const char** argv) : argc(argc), argv(argv) {}

  // Override this to define how to set up the environment.
  void SetUp() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    // get the local number of nodes with MPI
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    int shmrank;
    MPI_Comm_size(shmcomm, &shmrank);
    nRanksPerNode = shmrank;
    MPI_Comm_free(&shmcomm);

    // parse the command line arguments
    args = parseArgs(argc, argv);
  }

  // Override this to define how to tear down the environment.
  void TearDown() { MPI_Finalize(); }

  static std::unordered_map<std::string, std::string> parseArgs(int argc, const char* argv[]) {
    auto printUsage = [](const char* prog) {
      std::stringstream ss;
      ss << "Usage: " << prog << " [-ip_port IP:PORT]\n";
      std::cout << ss.str();
    };

    std::unordered_map<std::string, std::string> options;

    // Default values
    options["ip_port"] = gDefaultIpPort;

    // Parse the command line arguments
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "-ip_port") {
        if (i + 1 < argc) {
          options["ip_port"] = argv[++i];
        } else {
          throw std::invalid_argument("Error: -ip_port option requires an argument.\n");
        }
      } else if (arg == "-help" || arg == "-h") {
        printUsage(argv[0]);
        exit(0);
      } else {
        throw std::invalid_argument("Error: Unknown option " + std::string(argv[i]) + "\n");
      }
    }
    return options;
  }

  const int argc;
  const char** argv;
  int rank;
  int worldSize;
  int nRanksPerNode;
  std::unordered_map<std::string, std::string> args;
};

MultiProcessTestEnv* gEnv = nullptr;

class MultiProcessTest : public ::testing::Test {};

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gEnv = new MultiProcessTestEnv(argc, (const char**)argv);
  ::testing::AddGlobalTestEnvironment(gEnv);
  return RUN_ALL_TESTS();
}

TEST_F(MultiProcessTest, Prelim) {
  // Test to make sure the MPI environment is set up correctly
  ASSERT_GE(gEnv->worldSize, 2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bootstrap tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class BootstrapTest : public MultiProcessTest {};

void bootstrapTestAllGather(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap) {
  std::vector<int> tmp(bootstrap->getNranks(), 0);
  tmp[bootstrap->getRank()] = bootstrap->getRank() + 1;
  bootstrap->allGather(tmp.data(), sizeof(int));
  for (int i = 0; i < bootstrap->getNranks(); ++i) {
    EXPECT_EQ(tmp[i], i + 1);
  }
}

void bootstrapTestBarrier(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap) { bootstrap->barrier(); }

void bootstrapTestSendRecv(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap) {
  for (int i = 0; i < bootstrap->getNranks(); i++) {
    if (bootstrap->getRank() == i) continue;
    int msg1 = (bootstrap->getRank() + 1) * 3;
    int msg2 = (bootstrap->getRank() + 1) * 3 + 1;
    int msg3 = (bootstrap->getRank() + 1) * 3 + 2;
    bootstrap->send(&msg1, sizeof(int), i, 0);
    bootstrap->send(&msg2, sizeof(int), i, 1);
    bootstrap->send(&msg3, sizeof(int), i, 2);
  }

  for (int i = 0; i < bootstrap->getNranks(); i++) {
    if (bootstrap->getRank() == i) continue;
    int msg1 = 0;
    int msg2 = 0;
    int msg3 = 0;
    // recv them in the opposite order to check correctness
    bootstrap->recv(&msg2, sizeof(int), i, 1);
    bootstrap->recv(&msg3, sizeof(int), i, 2);
    bootstrap->recv(&msg1, sizeof(int), i, 0);
    EXPECT_EQ(msg1, (i + 1) * 3);
    EXPECT_EQ(msg2, (i + 1) * 3 + 1);
    EXPECT_EQ(msg3, (i + 1) * 3 + 2);
  }
}

void bootstrapTestAll(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap) {
  bootstrapTestAllGather(bootstrap);
  bootstrapTestBarrier(bootstrap);
  bootstrapTestSendRecv(bootstrap);
}

TEST_F(BootstrapTest, WithId) {
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, gEnv->worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  bootstrapTestAll(bootstrap);
}

TEST_F(BootstrapTest, WithIpPortPair) {
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, gEnv->worldSize);
  bootstrap->initialize(gEnv->args["ip_port"]);
  bootstrapTestAll(bootstrap);
}

class MPIBootstrap : public mscclpp::BaseBootstrap {
 public:
  MPIBootstrap() : BaseBootstrap() {}
  int getRank() override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }
  int getNranks() override {
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    return worldSize;
  }
  void allGather(void* sendbuf, int size) override {
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, sendbuf, size, MPI_BYTE, MPI_COMM_WORLD);
  }
  void barrier() override { MPI_Barrier(MPI_COMM_WORLD); }
  void send(void* sendbuf, int size, int dest, int tag) override {
    MPI_Send(sendbuf, size, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
  }
  void recv(void* recvbuf, int size, int source, int tag) override {
    MPI_Recv(recvbuf, size, MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
};

TEST_F(BootstrapTest, MPIBootstrap) {
  auto bootstrap = std::make_shared<MPIBootstrap>();
  bootstrapTestAll(bootstrap);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// InfiniBand tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static mscclpp::Transport ibIdToTransport(int id) {
  mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                              mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                              mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  return IBs[id];
}

class IbTest : public MultiProcessTest {
 protected:
  void SetUp() override {
    MSCCLPP_CUDATHROW(cudaGetDeviceCount(&cudaDevNum));
    cudaDevId = (gEnv->rank % gEnv->nRanksPerNode) % cudaDevNum;
    MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevId));

    int ibDevId = (gEnv->rank % gEnv->nRanksPerNode) / mscclpp::getIBDeviceCount();
    ibDevName = mscclpp::getIBDeviceName(ibIdToTransport(ibDevId));
  }

  int cudaDevNum;
  int cudaDevId;
  std::string ibDevName;
};

TEST_F(IbTest, SimpleSendRecv) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  const int maxIter = 100000;
  const int nelem = 1;
  auto data = mscclpp::allocUniqueCuda<int>(nelem);

  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, 2);

  mscclpp::UniqueId id;
  if (gEnv->rank == 0) {
    id = bootstrap->createUniqueId();
    MPI_Send(&id, sizeof(id), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(&id, sizeof(id), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  bootstrap->initialize(id);

  mscclpp::IbCtx ctx(ibDevName);
  mscclpp::IbQp* qp = ctx.createQp();
  const mscclpp::IbMr* mr = ctx.registerMr(data.get(), sizeof(int) * nelem);

  std::array<mscclpp::IbQpInfo, 2> qpInfo;
  qpInfo[gEnv->rank] = qp->getInfo();

  std::array<mscclpp::IbMrInfo, 2> mrInfo;
  mrInfo[gEnv->rank] = mr->getInfo();

  bootstrap->allGather(qpInfo.data(), sizeof(mscclpp::IbQpInfo));
  bootstrap->allGather(mrInfo.data(), sizeof(mscclpp::IbMrInfo));

  for (int i = 0; i < bootstrap->getNranks(); ++i) {
    if (i == gEnv->rank) continue;
    qp->rtr(qpInfo[i]);
    qp->rts();
    break;
  }
  bootstrap->barrier();

  if (gEnv->rank == 1) {
    mscclpp::Timer timer;
    for (int iter = 0; iter < maxIter; ++iter) {
      qp->stageSend(mr, mrInfo[0], sizeof(int) * nelem, 0, 0, 0, true);
      qp->postSend();
      bool waiting = true;
      int spin = 0;
      while (waiting) {
        int wcNum = qp->pollCq();
        ASSERT_GE(wcNum, 0);
        for (int i = 0; i < wcNum; ++i) {
          const ibv_wc* wc = qp->getWc(i);
          EXPECT_EQ(wc->status, IBV_WC_SUCCESS);
          waiting = false;
          break;
        }
        if (spin++ > 1000000) {
          FAIL() << "Polling is stuck.";
        }
      }
    }
    float us = (float)timer.elapsed();
    std::cout << "IbTest.SimpleSendRecv: " << us / maxIter << " us/iter" << std::endl;
  }
  bootstrap->barrier();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Communicator tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CommunicatorTestBase : public MultiProcessTest {
 protected:
  void SetUp() override {
    MultiProcessTest::SetUp();

    if (numRanksToUse == -1) {
      numRanksToUse = gEnv->worldSize;
    }
    ASSERT_LE(numRanksToUse, gEnv->worldSize);

    std::shared_ptr<mscclpp::Bootstrap> bootstrap;
    mscclpp::UniqueId id;
    if (gEnv->rank < numRanksToUse) {
      bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, numRanksToUse);
      if (gEnv->rank == 0) id = bootstrap->createUniqueId();
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (gEnv->rank >= numRanksToUse) {
      return;
    }
    bootstrap->initialize(id);
    communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
    ibTransport = ibIdToTransport(rankToLocalRank(gEnv->rank));
  }

  void TearDown() override {
    connections.clear();
    communicator.reset();
    MultiProcessTest::TearDown();
  }

  void setNumRanksToUse(int num) { numRanksToUse = num; }

  int rankToLocalRank(int rank) const { return rank % gEnv->nRanksPerNode; }

  int rankToNode(int rank) const { return rank / gEnv->nRanksPerNode; }

  void connectMesh(bool useIbOnly = false) {
    for (int i = 0; i < numRanksToUse; i++) {
      if (i != gEnv->rank) {
        if ((rankToNode(i) == rankToNode(gEnv->rank)) && !useIbOnly) {
          connections[i] = communicator->connectOnSetup(i, 0, mscclpp::Transport::CudaIpc);
        } else {
          connections[i] = communicator->connectOnSetup(i, 0, ibTransport);
        }
      }
    }
    communicator->setup();
  }

  // Register a local memory and receive corresponding remote memories
  void registerMemoryPairs(void* buff, size_t buffSize, mscclpp::TransportFlags transport, int tag,
                           const std::vector<int>& remoteRanks, mscclpp::RegisteredMemory& localMemory,
                           std::unordered_map<int, mscclpp::RegisteredMemory>& remoteMemories) {
    localMemory = communicator->registerMemory(buff, buffSize, transport);
    std::unordered_map<int, mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> futureRemoteMemories;
    for (int remoteRank : remoteRanks) {
      if (remoteRank != communicator->bootstrapper()->getRank()) {
        communicator->sendMemoryOnSetup(localMemory, remoteRank, tag);
        futureRemoteMemories[remoteRank] = communicator->recvMemoryOnSetup(remoteRank, tag);
      }
    }
    communicator->setup();
    for (int remoteRank : remoteRanks) {
      if (remoteRank != communicator->bootstrapper()->getRank()) {
        remoteMemories[remoteRank] = futureRemoteMemories[remoteRank].get();
      }
    }
  }

  // Register a local memory an receive one corresponding remote memory
  void registerMemoryPair(void* buff, size_t buffSize, mscclpp::TransportFlags transport, int tag, int remoteRank,
                          mscclpp::RegisteredMemory& localMemory, mscclpp::RegisteredMemory& remoteMemory) {
    std::vector<int> remoteRanks = {remoteRank};
    std::unordered_map<int, mscclpp::RegisteredMemory> remoteMemories;
    registerMemoryPairs(buff, buffSize, transport, tag, remoteRanks, localMemory, remoteMemories);
    remoteMemory = remoteMemories[remoteRank];
  }

  int numRanksToUse = -1;
  std::shared_ptr<mscclpp::Communicator> communicator;
  mscclpp::Transport ibTransport;
  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> connections;
};

class CommunicatorTest : public CommunicatorTestBase {
 protected:
  void SetUp() override {
    CommunicatorTestBase::SetUp();

    ASSERT_EQ((deviceBufferSize / sizeof(int)) % gEnv->worldSize, 0);

    connectMesh();

    devicePtr.resize(numBuffers);
    localMemory.resize(numBuffers);
    remoteMemory.resize(numBuffers);

    std::vector<int> remoteRanks;
    for (int i = 0; i < gEnv->worldSize; i++) {
      if (i != gEnv->rank) {
        remoteRanks.push_back(i);
      }
    }

    for (int n = 0; n < numBuffers; n++) {
      devicePtr[n] = mscclpp::allocSharedCuda<int>(deviceBufferSize / sizeof(int));
      registerMemoryPairs(devicePtr[n].get(), deviceBufferSize, mscclpp::Transport::CudaIpc | ibTransport, 0,
                          remoteRanks, localMemory[n], remoteMemory[n]);
    }
  }

  void TearDown() override {
    remoteMemory.clear();
    localMemory.clear();
    devicePtr.clear();
    CommunicatorTestBase::TearDown();
  }

  void deviceBufferInit() {
    size_t dataCount = deviceBufferSize / sizeof(int);
    for (int n = 0; n < (int)devicePtr.size(); n++) {
      std::vector<int> hostBuffer(dataCount, 0);
      for (int i = 0; i < dataCount; i++) {
        hostBuffer[i] = gEnv->rank + n * gEnv->worldSize;
      }
      mscclpp::memcpyCuda<int>(devicePtr[n].get(), hostBuffer.data(), dataCount, cudaMemcpyHostToDevice);
    }
  }

  void writeToRemote(int dataCountPerRank) {
    for (int n = 0; n < numBuffers; n++) {
      for (int i = 0; i < gEnv->worldSize; i++) {
        if (i != gEnv->rank) {
          auto& conn = connections.at(i);
          auto& peerMemory = remoteMemory[n].at(i);
          conn->write(peerMemory, gEnv->rank * dataCountPerRank * sizeof(int), localMemory[n],
                      gEnv->rank * dataCountPerRank * sizeof(int), dataCountPerRank * sizeof(int));
          conn->flush();
        }
      }
    }
  }

  bool testWriteCorrectness(bool skipLocal = false) {
    size_t dataCount = deviceBufferSize / sizeof(int);
    for (int n = 0; n < (int)devicePtr.size(); n++) {
      std::vector<int> hostBuffer(dataCount, 0);
      mscclpp::memcpyCuda<int>(hostBuffer.data(), devicePtr[n].get(), dataCount, cudaMemcpyDeviceToHost);
      for (int i = 0; i < gEnv->worldSize; i++) {
        if (((i / gEnv->nRanksPerNode) == (gEnv->rank / gEnv->nRanksPerNode)) && skipLocal) {
          continue;
        }
        for (int j = i * dataCount / gEnv->worldSize; j < (i + 1) * dataCount / gEnv->worldSize; j++) {
          if (hostBuffer[j] != i + n * gEnv->worldSize) {
            return false;
          }
        }
      }
    }
    return true;
  }

  const size_t numBuffers = 10;
  const int deviceBufferSize = 1024 * 1024;
  std::vector<std::shared_ptr<int>> devicePtr;
  std::vector<mscclpp::RegisteredMemory> localMemory;
  std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>> remoteMemory;
};

TEST_F(CommunicatorTest, BasicWrite) {
  if (gEnv->rank >= numRanksToUse) return;

  deviceBufferInit();
  communicator->bootstrapper()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);
  communicator->bootstrapper()->barrier();

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
  communicator->bootstrapper()->barrier();
}

__global__ void kernelIncEpochs(mscclpp::DeviceEpoch::DeviceHandle* deviceEpochs, int rank, int worldSize) {
  int tid = threadIdx.x;
  if (tid != rank && tid < worldSize) {
    deviceEpochs[tid].epochIncrement();
  }
}

__global__ void kernelWaitEpochs(mscclpp::DeviceEpoch::DeviceHandle* deviceEpochs, int rank, int worldSize) {
  int tid = threadIdx.x;
  if (tid != rank && tid < worldSize) {
    deviceEpochs[tid].wait();
  }
}

TEST_F(CommunicatorTest, WriteWithDeviceEpochs) {
  if (gEnv->rank >= numRanksToUse) return;

  std::unordered_map<int, std::shared_ptr<mscclpp::DeviceEpoch>> epochs;
  for (auto entry : connections) {
    auto& conn = entry.second;
    epochs.insert({entry.first, std::make_shared<mscclpp::DeviceEpoch>(*communicator.get(), conn)});
  }
  communicator->setup();
  communicator->bootstrapper()->barrier();

  deviceBufferInit();
  communicator->bootstrapper()->barrier();

  auto deviceEpochHandles = mscclpp::allocSharedCuda<mscclpp::DeviceEpoch::DeviceHandle>(gEnv->worldSize);
  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank) {
      mscclpp::DeviceEpoch::DeviceHandle deviceHandle = epochs[i]->deviceHandle();
      mscclpp::memcpyCuda<mscclpp::DeviceEpoch::DeviceHandle>(deviceEpochHandles.get() + i, &deviceHandle, 1,
                                                              cudaMemcpyHostToDevice);
    }
  }
  communicator->bootstrapper()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);

  kernelIncEpochs<<<1, gEnv->worldSize>>>(deviceEpochHandles.get(), gEnv->rank, gEnv->worldSize);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank) {
      epochs[i]->signal();
    }
  }

  kernelWaitEpochs<<<1, gEnv->worldSize>>>(deviceEpochHandles.get(), gEnv->rank, gEnv->worldSize);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  ASSERT_TRUE(testWriteCorrectness());
  communicator->bootstrapper()->barrier();
}

TEST_F(CommunicatorTest, WriteWithHostEpochs) {
  if (gEnv->rank >= numRanksToUse) return;

  std::unordered_map<int, std::shared_ptr<mscclpp::HostEpoch>> epochs;
  for (auto entry : connections) {
    auto& conn = entry.second;
    // HostEpoch cannot be used with CudaIpc transport
    if (conn->transport() == mscclpp::Transport::CudaIpc) continue;
    epochs.insert({entry.first, std::make_shared<mscclpp::HostEpoch>(*communicator.get(), conn)});
  }
  communicator->setup();
  communicator->bootstrapper()->barrier();

  deviceBufferInit();
  communicator->bootstrapper()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      epochs[i]->incrementAndSignal();
    }
  }

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      epochs[i]->wait();
    }
  }

  ASSERT_TRUE(testWriteCorrectness());
  communicator->bootstrapper()->barrier();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Channel tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ChannelOneToOneTest : public CommunicatorTestBase {
 protected:
  void SetUp() override {
    // Use only two ranks
    setNumRanksToUse(2);
    CommunicatorTestBase::SetUp();
    channelService = std::make_shared<mscclpp::channel::DeviceChannelService>(*communicator.get());
  }

  void TearDown() override { CommunicatorTestBase::TearDown(); }

  void setupMeshConnections(std::vector<mscclpp::channel::SimpleDeviceChannel>& devChannels, bool useIbOnly,
                            void* sendBuff, size_t sendBuffBytes, void* recvBuff = nullptr, size_t recvBuffBytes = 0) {
    const int rank = communicator->bootstrapper()->getRank();
    const int worldSize = communicator->bootstrapper()->getNranks();
    const bool isInPlace = (recvBuff == nullptr);
    mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc | ibTransport;

    connectMesh(useIbOnly);

    for (int r = 0; r < worldSize; r++) {
      if (r == rank) {
        continue;
      }
      mscclpp::RegisteredMemory sendMemory;
      mscclpp::RegisteredMemory remoteMemory;
      void* tmpBuff = nullptr;

      if (isInPlace) {
        registerMemoryPair(sendBuff, sendBuffBytes, transport, 0, r, sendMemory, remoteMemory);
      } else {
        sendMemory = communicator->registerMemory(recvBuff, recvBuffBytes, transport);
        mscclpp::RegisteredMemory recvMemory;
        registerMemoryPair(recvBuff, recvBuffBytes, transport, 0, r, recvMemory, remoteMemory);
        tmpBuff = recvMemory.data();
      }
      // TODO: enable this when we support out-of-place
      // devChannels.emplace_back(channelService->deviceChannel(channelService->addChannel(connections[r])),
      //                          channelService->addMemory(remoteMemory), channelService->addMemory(sendMemory),
      //                          remoteMemory.data(), sendMemory.data(), tmpBuff);
      devChannels.emplace_back(channelService->deviceChannel(channelService->addChannel(connections[r])),
                               channelService->addMemory(remoteMemory), channelService->addMemory(sendMemory),
                               remoteMemory.data(), sendMemory.data());
    }
  }

  std::shared_ptr<mscclpp::channel::DeviceChannelService> channelService;
};

__constant__ mscclpp::channel::SimpleDeviceChannel gChannelOneToOneTestConstDevChans;

__global__ void kernelPingPong(int rank, int nElem) {
  mscclpp::channel::SimpleDeviceChannel& devChan = gChannelOneToOneTestConstDevChans;
  volatile int* sendBuff = (volatile int*)devChan.srcPtr_;
  int nTries = 1000;
  int flusher = 0;
  int rank1Offset = 10000000;
  for (int i = 0; i < nTries; i++) {
    if (rank == 0) {
      if (i > 0) {
        if (threadIdx.x == 0) devChan.wait();
        __syncthreads();
        for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
          if (sendBuff[j] != rank1Offset + i - 1 + j) {
            printf("rank 0 ERROR: sendBuff[%d] = %d, expected %d\n", j, sendBuff[j], 100000 + i - 1 + j);
          }
        }
      }
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        sendBuff[j] = i + j;
      }
      __syncthreads();
      // __threadfence_system(); // not necessary if we make sendBuff volatile
      if (threadIdx.x == 0) devChan.putWithSignal(0, nElem * sizeof(int));
    }
    if (rank == 1) {
      if (threadIdx.x == 0) devChan.wait();
      __syncthreads();
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        if (sendBuff[j] != i + j) {
          printf("rank 1 ERROR: sendBuff[%d] = %d, expected %d\n", j, sendBuff[j], i + j);
        }
      }
      if (i < nTries - 1) {
        for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
          sendBuff[j] = rank1Offset + i + j;
        }
        __syncthreads();
        // __threadfence_system(); // not necessary if we make sendBuff volatile
        if (threadIdx.x == 0) devChan.putWithSignal(0, nElem * sizeof(int));
      }
    }
    flusher++;
    if (flusher == 100) {
      devChan.flush();
      flusher = 0;
    }
  }
}

TEST_F(ChannelOneToOneTest, PingPongIb) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 64 * 1024 * 1024;

  std::vector<mscclpp::channel::SimpleDeviceChannel> devChannels;
  std::shared_ptr<int> buff = mscclpp::allocSharedCuda<int>(nElem);
  setupMeshConnections(devChannels, true, buff.get(), nElem * sizeof(int));

  ASSERT_EQ(devChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(&gChannelOneToOneTestConstDevChans, devChannels.data(),
                                       sizeof(mscclpp::channel::SimpleDeviceChannel)));

  channelService->startProxy();

  kernelPingPong<<<24, 1024>>>(gEnv->rank, nElem);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  channelService->stopProxy();
}
