#include <gtest/gtest.h>
#include <mpi.h>

#include <iostream>
#include <mscclpp/channel.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/cuda_utils.hpp>
#include <mscclpp/epoch.hpp>
#include <mscclpp/utils.hpp>
#include <sstream>

#include "config.hpp"
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

class MultiProcessTest : public ::testing::Test {
 protected:
  void TearDown() override {
    // Wait for all ranks to finish the previous test
    MPI_Barrier(MPI_COMM_WORLD);
  }
};

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

class BootstrapTest : public MultiProcessTest {
 protected:
  // Each test case should finish within 3 seconds.
  mscclpp::Timer bootstrapTestTimer{3};
};

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

TEST_F(BootstrapTest, ResumeWithId) {
  for (int i = 0; i < 5; ++i) {
    auto bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, gEnv->worldSize);
    mscclpp::UniqueId id;
    if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    bootstrap->initialize(id);
  }
}

TEST_F(BootstrapTest, ResumeWithIpPortPair) {
  // TODO: enable when the bug is fixed. bootstrap hangs and even timer doesn't work
#if 0
  for (int i = 0; i < 5; ++i) {
    auto bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, gEnv->worldSize);
    bootstrap->initialize(gEnv->args["ip_port"]);
  }
#else
  // TODO: remove when the bug is fixed.
  FAIL();
#endif
}

TEST_F(BootstrapTest, ExitBeforeConnect) {
  // TODO: enable when the bug is fixed. bootstrap rootThread_ does not exit gracefully
#if 0
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, gEnv->worldSize);
  mscclpp::UniqueId id = bootstrap->createUniqueId();
#else
  // TODO: remove when the bug is fixed.
  FAIL();
#endif
}

TEST_F(BootstrapTest, TimeoutWithId) {
  // TODO: enable when BootstrapTest.ExitBeforeConnect passes.
#if 0
  // Set bootstrap timeout to 1 second
  mscclpp::Config* cfg = mscclpp::Config::getInstance();
  cfg->setBootstrapConnectionTimeoutConfig(1);

  // All ranks initialize a bootstrap with their own id (will hang)
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, gEnv->worldSize);
  mscclpp::UniqueId id = bootstrap->createUniqueId();

  ASSERT_THROW(bootstrap->initialize(id), mscclpp::Error);

  // Timeout should be less than 3 seconds
  ASSERT_LT(timer.elapsed(), 3000000);
#else
  // TODO: remove when BootstrapTest.ExitBeforeConnect passes.
  FAIL();
#endif
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

class IbTestBase : public MultiProcessTest {
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

class IbPeerToPeerTest : public IbTestBase {
 protected:
  void SetUp() override {
    IbTestBase::SetUp();

    mscclpp::UniqueId id;

    if (gEnv->rank < 2) {
      // This test needs only two ranks
      bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, 2);
      if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (gEnv->rank >= 2) {
      // This test needs only two ranks
      return;
    }

    bootstrap->initialize(id);

    ibCtx = std::make_shared<mscclpp::IbCtx>(ibDevName);
    qp = ibCtx->createQp();

    qpInfo[gEnv->rank] = qp->getInfo();
    bootstrap->allGather(qpInfo.data(), sizeof(mscclpp::IbQpInfo));
  }

  void registerBufferAndConnect(void* buf, size_t size) {
    bufSize = size;
    mr = ibCtx->registerMr(buf, size);
    mrInfo[gEnv->rank] = mr->getInfo();
    bootstrap->allGather(mrInfo.data(), sizeof(mscclpp::IbMrInfo));

    for (int i = 0; i < bootstrap->getNranks(); ++i) {
      if (i == gEnv->rank) continue;
      qp->rtr(qpInfo[i]);
      qp->rts();
      break;
    }
    bootstrap->barrier();
  }

  void stageSend(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled) {
    const mscclpp::IbMrInfo& remoteMrInfo = mrInfo[(gEnv->rank == 1) ? 0 : 1];
    qp->stageSend(mr, remoteMrInfo, size, wrId, srcOffset, dstOffset, signaled);
  }

  void stageAtomicAdd(uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, uint64_t addVal) {
    const mscclpp::IbMrInfo& remoteMrInfo = mrInfo[(gEnv->rank == 1) ? 0 : 1];
    qp->stageAtomicAdd(mr, remoteMrInfo, wrId, dstOffset, addVal);
  }

  void stageSendWithImm(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled,
                        unsigned int immData) {
    const mscclpp::IbMrInfo& remoteMrInfo = mrInfo[(gEnv->rank == 1) ? 0 : 1];
    qp->stageSendWithImm(mr, remoteMrInfo, size, wrId, srcOffset, dstOffset, signaled, immData);
  }

  std::shared_ptr<mscclpp::Bootstrap> bootstrap;
  std::shared_ptr<mscclpp::IbCtx> ibCtx;
  mscclpp::IbQp* qp;
  const mscclpp::IbMr* mr;
  size_t bufSize;

  std::array<mscclpp::IbQpInfo, 2> qpInfo;
  std::array<mscclpp::IbMrInfo, 2> mrInfo;
};

TEST_F(IbPeerToPeerTest, SimpleSendRecv) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  mscclpp::Timer timeout(3);

  const int maxIter = 100000;
  const int nelem = 1;
  auto data = mscclpp::allocUniqueCuda<int>(nelem);

  registerBufferAndConnect(data.get(), sizeof(int) * nelem);

  if (gEnv->rank == 1) {
    mscclpp::Timer timer;
    for (int iter = 0; iter < maxIter; ++iter) {
      stageSend(sizeof(int) * nelem, 0, 0, 0, true);
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
    std::cout << "IbPeerToPeerTest.SimpleSendRecv: " << us / maxIter << " us/iter" << std::endl;
  }
  bootstrap->barrier();
}

__global__ void kernelMemoryConsistency(uint64_t* data, volatile uint64_t* curIter, volatile int* result,
                                        uint64_t nelem, uint64_t maxIter) {
  if (blockIdx.x != 0) return;

  constexpr int FlagWrong = 1;
  constexpr int FlagAbort = 2;

  volatile uint64_t* ptr = data;
  for (uint64_t iter = 1; iter < maxIter + 1; ++iter) {
    int err = 0;

    if (threadIdx.x == 0) {
      *curIter = iter;

      // Wait for the first element arrival (expect equal to iter). Expect that the first element is delivered in
      // a special way that guarantees all other elements are completely delivered.
      uint64_t spin = 0;
      while (ptr[0] != iter) {
        if (spin++ == 1000000) {
          // Assume the program is stuck. Set the abort flag and escape the loop.
          *result |= FlagAbort;
          err = 1;
          break;
        }
      }
    }
    __syncthreads();

    // Check results (expect equal to iter) in backward that is more likely to see the wrong result.
    for (size_t i = nelem - 1 + threadIdx.x; i >= blockDim.x; i -= blockDim.x) {
      if (data[i - blockDim.x] != iter) {
#if 1
        *result |= FlagWrong;
        err = 1;
        break;
#else
        // For debugging purposes: try waiting for the correct result.
        uint64_t spin = 0;
        while (ptr[i - blockDim.x] != iter) {
          if (spin++ == 1000000) {
            *result |= FlagAbort;
            err = 1;
            break;
          }
        }
        if (spin >= 1000000) {
          break;
        }
#endif
      }
    }
    __threadfence();
    __syncthreads();

    // Shuffle err
    for (int i = 16; i > 0; i /= 2) {
      err += __shfl_xor_sync(0xffffffff, err, i);
    }

    if (err > 0) {
      // Exit if any error is detected.
      return;
    }
  }
  if (threadIdx.x == 0) {
    *curIter = maxIter + 1;
  }
}

TEST_F(IbPeerToPeerTest, MemoryConsistency) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  const uint64_t signalPeriod = 1024;
  const uint64_t maxIter = 10000;
  const uint64_t nelem = 65536 + 1;
  auto data = mscclpp::allocUniqueCuda<uint64_t>(nelem);

  registerBufferAndConnect(data.get(), sizeof(uint64_t) * nelem);

  uint64_t res = 0;
  uint64_t iter = 0;

  if (gEnv->rank == 0) {
    // Receiver
    auto curIter = mscclpp::makeUniqueCudaHost<uint64_t>(0);
    auto result = mscclpp::makeUniqueCudaHost<int>(0);

    volatile uint64_t* ptrCurIter = (volatile uint64_t*)curIter.get();
    volatile int* ptrResult = (volatile int*)result.get();

    ASSERT_EQ(*ptrCurIter, 0);
    ASSERT_EQ(*ptrResult, 0);

    kernelMemoryConsistency<<<1, 1024>>>(data.get(), ptrCurIter, ptrResult, nelem, maxIter);
    MSCCLPP_CUDATHROW(cudaGetLastError());

    for (iter = 1; iter < maxIter + 1; ++iter) {
      mscclpp::Timer timeout(5);

      while (*ptrCurIter != iter + 1) {
        res = *ptrResult;
        if (res != 0) break;
      }

      // Send the result to the sender
      res = *ptrResult;
      uint64_t tmp[2];
      tmp[0] = res;
      bootstrap->allGather(tmp, sizeof(uint64_t));

      if (res != 0) break;
    }

    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  } else if (gEnv->rank == 1) {
    // Sender
    std::vector<uint64_t> hostBuffer(nelem, 0);

    for (iter = 1; iter < maxIter + 1; ++iter) {
      mscclpp::Timer timeout(5);

      // Set data
      for (uint64_t i = 0; i < nelem; i++) {
        hostBuffer[i] = iter;
      }
      mscclpp::memcpyCuda<uint64_t>(data.get(), hostBuffer.data(), nelem, cudaMemcpyHostToDevice);

      // Need to signal from time to time to empty the IB send queue
      bool signaled = (iter % signalPeriod == 0);

      // Send from the second element to the last
      stageSend(sizeof(uint64_t) * (nelem - 1), 0, sizeof(uint64_t), sizeof(uint64_t), signaled);
      qp->postSend();

#if 0
      // Send the first element using a normal send. This should occasionally see the wrong result.
      stageSend(sizeof(uint64_t), 0, 0, 0, false);
      qp->postSend();
#else
      // For reference: send the first element using AtomicAdd. This should see the correct result.
      stageAtomicAdd(0, 0, 0, 1);
      qp->postSend();
#endif

      if (signaled) {
        int wcNum = qp->pollCq();
        while (wcNum == 0) {
          wcNum = qp->pollCq();
        }
        ASSERT_EQ(wcNum, 1);
        const ibv_wc* wc = qp->getWc(0);
        ASSERT_EQ(wc->status, IBV_WC_SUCCESS);
      }

      // Get the result from the receiver
      uint64_t tmp[2];
      bootstrap->allGather(tmp, sizeof(uint64_t));
      res = tmp[0];

      if (res != 0) break;
    }
  }

  if (res & 2) {
    FAIL() << "The receiver is stuck at iteration " << iter << ".";
  } else if (res != 0 && res != 1) {
    FAIL() << "Unknown error is detected at iteration " << iter << ". res =" << res;
  }

  EXPECT_EQ(res, 0);
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
      epochs[i]->signal();
    }
  }

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      epochs[i]->wait();
    }
  }

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      epochs[i]->signal();
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
// DeviceChannel tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DeviceChannelOneToOneTest : public CommunicatorTestBase {
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

      if (isInPlace) {
        registerMemoryPair(sendBuff, sendBuffBytes, transport, 0, r, sendMemory, remoteMemory);
      } else {
        sendMemory = communicator->registerMemory(recvBuff, recvBuffBytes, transport);
        mscclpp::RegisteredMemory recvMemory;
        registerMemoryPair(recvBuff, recvBuffBytes, transport, 0, r, recvMemory, remoteMemory);
      }

      mscclpp::channel::ChannelId cid = channelService->addChannel(connections[r]);
      communicator->setup();

      devChannels.emplace_back(channelService->deviceChannel(cid), channelService->addMemory(remoteMemory),
                               channelService->addMemory(sendMemory));
    }
  }

  std::shared_ptr<mscclpp::channel::DeviceChannelService> channelService;
};

__constant__ mscclpp::channel::SimpleDeviceChannel gChannelOneToOneTestConstDevChans;

__global__ void kernelDevicePingPong(int* buff, int rank, int nElem, int* ret) {
  mscclpp::channel::SimpleDeviceChannel& devChan = gChannelOneToOneTestConstDevChans;
  volatile int* sendBuff = (volatile int*)buff;
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
            // printf("rank 0 ERROR: sendBuff[%d] = %d, expected %d\n", j, sendBuff[j], rank1Offset + i - 1 + j);
            *ret = 1;
            break;
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
          // printf("rank 1 ERROR: sendBuff[%d] = %d, expected %d\n", j, sendBuff[j], i + j);
          *ret = 1;
          break;
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

TEST_F(DeviceChannelOneToOneTest, PingPongIb) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::channel::SimpleDeviceChannel> devChannels;
  std::shared_ptr<int> buff = mscclpp::allocSharedCuda<int>(nElem);
  setupMeshConnections(devChannels, true, buff.get(), nElem * sizeof(int));

  ASSERT_EQ(devChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstDevChans, devChannels.data(),
                                       sizeof(mscclpp::channel::SimpleDeviceChannel)));

  channelService->startProxy();

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  kernelDevicePingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelDevicePingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelDevicePingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelDevicePingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  channelService->stopProxy();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DirectChannel tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DirectChannelOneToOneTest : public CommunicatorTestBase {
 protected:
  void SetUp() override {
    // Use only two ranks
    setNumRanksToUse(2);
    CommunicatorTestBase::SetUp();
  }

  void TearDown() override { CommunicatorTestBase::TearDown(); }

  void setupMeshConnections(std::vector<mscclpp::channel::DirectChannel>& dirChannels, void* inputBuff,
                            size_t inputBuffBytes, void* outputBuff = nullptr, size_t outputBuffBytes = 0) {
    const int rank = communicator->bootstrapper()->getRank();
    const int worldSize = communicator->bootstrapper()->getNranks();
    const bool isInPlace = (outputBuff == nullptr);
    mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc | ibTransport;

    mscclpp::RegisteredMemory inputBufRegMem = communicator->registerMemory(inputBuff, inputBuffBytes, transport);
    mscclpp::RegisteredMemory outputBufRegMem;
    if (!isInPlace) {
      outputBufRegMem = communicator->registerMemory(outputBuff, outputBuffBytes, transport);
    }

    for (int r = 0; r < worldSize; r++) {
      if (r == rank) {
        continue;
      }
      std::shared_ptr<mscclpp::Connection> conn;
      if (rankToNode(r) == rankToNode(gEnv->rank)) {
        conn = communicator->connectOnSetup(r, 0, mscclpp::Transport::CudaIpc);
      } else {
        conn = communicator->connectOnSetup(r, 0, ibTransport);
      }
      connections[r] = conn;

      if (isInPlace) {
        communicator->sendMemoryOnSetup(inputBufRegMem, r, 0);
      } else {
        communicator->sendMemoryOnSetup(outputBufRegMem, r, 0);
      }
      auto remoteMemory = communicator->recvMemoryOnSetup(r, 0);

      communicator->setup();

      directEpochs[r] = std::make_shared<mscclpp::DirectEpoch>(*communicator, conn);

      communicator->setup();

      dirChannels.emplace_back(directEpochs[r]->deviceHandle(), remoteMemory.get(), inputBufRegMem.data(),
                               (isInPlace ? nullptr : outputBufRegMem.data()));
    }
  }

  std::unordered_map<int, std::shared_ptr<mscclpp::DirectEpoch>> directEpochs;
};

__constant__ mscclpp::channel::DirectChannel gChannelOneToOneTestConstDirChans;

__global__ void kernelDirectPingPong(int* buff, int rank, int nElem, int* ret) {
  mscclpp::channel::DirectChannel& dirChan = gChannelOneToOneTestConstDirChans;
  volatile int* sendBuff = (volatile int*)buff;
  int nTries = 1000;
  int rank1Offset = 10000000;
  for (int i = 0; i < nTries; i++) {
    if (rank == 0) {
      if (i > 0) {
        if (threadIdx.x == 0) dirChan.wait();
        __syncthreads();
        for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
          if (sendBuff[j] != rank1Offset + i - 1 + j) {
            // printf("rank 0 ERROR: sendBuff[%d] = %d, expected %d\n", j, sendBuff[j], rank1Offset + i - 1 + j);
            *ret = 1;
            break;
          }
        }
      }
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        sendBuff[j] = i + j;
      }
      __syncthreads();
      dirChan.put(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x);
      if (threadIdx.x == 0) dirChan.signal();
    }
    if (rank == 1) {
      if (threadIdx.x == 0) dirChan.wait();
      __syncthreads();
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        if (sendBuff[j] != i + j) {
          // printf("rank 1 ERROR: sendBuff[%d] = %d, expected %d\n", j, sendBuff[j], i + j);
          *ret = 1;
          break;
        }
      }
      if (i < nTries - 1) {
        for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
          sendBuff[j] = rank1Offset + i + j;
        }
        __syncthreads();
        dirChan.put(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x);
        if (threadIdx.x == 0) dirChan.signal();
      }
    }
  }
}

TEST_F(DirectChannelOneToOneTest, PingPong) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::channel::DirectChannel> dirChannels;
  std::shared_ptr<int> buff = mscclpp::allocSharedCuda<int>(nElem);
  setupMeshConnections(dirChannels, buff.get(), nElem * sizeof(int));

  ASSERT_EQ(dirChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstDirChans, dirChannels.data(),
                                       sizeof(mscclpp::channel::DirectChannel)));

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  kernelDirectPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelDirectPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelDirectPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelDirectPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
}

__global__ void kernelDirectPacketPingPong(int* buff, int rank, int nElem, int* ret) {
  mscclpp::channel::DirectChannel& dirChan = gChannelOneToOneTestConstDirChans;
  volatile int* sendBuff = (volatile int*)buff;
  int nTries = 1000;
  int rank1Offset = 10000000;
  for (int i = 0; i < nTries; i++) {
    uint64_t flag = (uint64_t)i + 1;
    if (rank == 0) {
      if (i > 0) {
        dirChan.getPacket(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
        // If each thread reads 8 bytes at once, we don't need a barrier after getPacket().
        for (int j = threadIdx.x; j < nElem / 2; j += blockDim.x) {
          if (sendBuff[2 * j] != rank1Offset + i - 1 + 2 * j) {
            // printf("rank 0 ERROR: sendBuff[%d] = %d, expected %d. Skipping following errors\n",
            //        2 * j, sendBuff[2 * j], rank1Offset + i - 1 + 2 * j);
            *ret = 1;
            break;
          }
          if (sendBuff[2 * j + 1] != rank1Offset + i - 1 + 2 * j + 1) {
            // printf("rank 0 ERROR: sendBuff[%d] = %d, expected %d. Skipping following errors\n",
            //        2 * j + 1, sendBuff[2 * j + 1], rank1Offset + i - 1 + 2 * j + 1);
            *ret = 1;
            break;
          }
        }
      }
      // If each thread writes 8 bytes at once, we don't need a barrier before putPacket().
      for (int j = threadIdx.x; j < nElem / 2; j += blockDim.x) {
        sendBuff[2 * j] = i + 2 * j;
        sendBuff[2 * j + 1] = i + 2 * j + 1;
      }
      dirChan.putPacket(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
    }
    if (rank == 1) {
      dirChan.getPacket(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
      // If each thread reads 8 bytes at once, we don't need a barrier after getPacket().
      for (int j = threadIdx.x; j < nElem / 2; j += blockDim.x) {
        if (sendBuff[2 * j] != i + 2 * j) {
          // printf("rank 1 ERROR: sendBuff[%d] = %d, expected %d. Skipping following errors\n",
          //        2 * j, sendBuff[2 * j], i + 2 * j);
          *ret = 1;
          break;
        }
        if (sendBuff[2 * j + 1] != i + 2 * j + 1) {
          // printf("rank 1 ERROR: sendBuff[%d] = %d, expected %d. Skipping following errors\n",
          //        2 * j + 1, sendBuff[2 * j + 1], i + 2 * j + 1);
          *ret = 1;
          break;
        }
      }
      if (i < nTries - 1) {
        // If each thread writes 8 bytes at once, we don't need a barrier before putPacket().
        for (int j = threadIdx.x; j < nElem / 2; j += blockDim.x) {
          sendBuff[2 * j] = rank1Offset + i + 2 * j;
          sendBuff[2 * j + 1] = rank1Offset + i + 2 * j + 1;
        }
        dirChan.putPacket(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
      }
    }
  }
}

TEST_F(DirectChannelOneToOneTest, PacketPingPong) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::channel::DirectChannel> dirChannels;
  std::shared_ptr<int> buff = mscclpp::allocSharedCuda<int>(nElem);
  std::shared_ptr<int> intermBuff = mscclpp::allocSharedCuda<int>(nElem * 2);
  setupMeshConnections(dirChannels, buff.get(), nElem * sizeof(int), intermBuff.get(), nElem * 2 * sizeof(int));

  ASSERT_EQ(dirChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstDirChans, dirChannels.data(),
                                       sizeof(mscclpp::channel::DirectChannel)));

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  // The least nelem is 2 for packet ping pong
  // kernelDirectPacketPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 2, ret.get());
  // MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  // EXPECT_EQ(*ret, 0);
  // *ret = 0;

  // kernelDirectPacketPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, ret.get());
  // MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  // EXPECT_EQ(*ret, 0);
  // *ret = 0;

  // kernelDirectPacketPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, ret.get());
  // MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  // EXPECT_EQ(*ret, 0);
  // *ret = 0;

  // kernelDirectPacketPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get());
  // MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  // EXPECT_EQ(*ret, 0);
}
