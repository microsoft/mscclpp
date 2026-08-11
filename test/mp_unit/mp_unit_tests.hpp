// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_MP_UNIT_TESTS_HPP_
#define MSCCLPP_MP_UNIT_TESTS_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/executor.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/utils.hpp>

#include "../framework.hpp"
#include "ib.hpp"
#include "utils_internal.hpp"

// Skip the current test if IBVerbs is not available in this build
#if defined(USE_IBVERBS)
#define REQUIRE_IBVERBS
#else
#define REQUIRE_IBVERBS SKIP_TEST() << "This test requires IBVerbs that the current build does not support."
#endif

class MultiProcessTestEnv : public ::mscclpp::test::Environment {
 public:
  MultiProcessTestEnv(int argc, const char** argv);

  void SetUp();
  void TearDown();

  const int argc;
  const char** argv;
  int rank;
  int worldSize;
  int nRanksPerNode;
  std::unordered_map<std::string, std::string> args;
};

extern MultiProcessTestEnv* gEnv;

mscclpp::Transport ibIdToTransport(int id);
int rankToLocalRank(int rank);
int rankToNode(int rank);

class MultiProcessTest : public ::mscclpp::test::TestCase {
 protected:
  void TearDown() override;
};

class BootstrapTest : public MultiProcessTest {
 protected:
  void bootstrapTestAllGather(std::shared_ptr<mscclpp::Bootstrap> bootstrap);

  void bootstrapTestBarrier(std::shared_ptr<mscclpp::Bootstrap> bootstrap);

  void bootstrapTestSendRecv(std::shared_ptr<mscclpp::Bootstrap> bootstrap);

  void bootstrapTestIpcDomain(std::shared_ptr<mscclpp::Bootstrap> bootstrap);

  void bootstrapTestAll(std::shared_ptr<mscclpp::Bootstrap> bootstrap);

  // Each test case should finish within 30 seconds.
  mscclpp::Timer bootstrapTestTimer{30};
};

class IbTestBase : public MultiProcessTest {
 protected:
  void SetUp() override;

  int cudaDevNum;
  int cudaDevId;
  std::string ibDevName;
};

class IbPeerToPeerTest : public IbTestBase {
 protected:
  void SetUp() override;

  void registerBufferAndConnect(void* buf, size_t size);

  void stageSendWrite(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled);

  void stageSendAtomicAdd(uint64_t wrId, uint64_t dstOffset, uint64_t addVal, bool signaled);

  void stageSendWriteWithImm(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled,
                             unsigned int immData);

  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  std::shared_ptr<mscclpp::IbCtx> ibCtx;
  std::shared_ptr<mscclpp::IbQp> qp;
  std::shared_ptr<const mscclpp::IbMr> mr;
  size_t bufSize;

  std::array<mscclpp::IbQpInfo, 2> qpInfo;
  std::array<mscclpp::IbMrInfo, 2> mrInfo;
};

class CommunicatorTestBase : public MultiProcessTest {
 protected:
  void SetUp() override;
  void TearDown() override;

  void setNumRanksToUse(int num);
  void connectMesh(bool useIpc = true, bool useIb = true, bool useEthernet = false);

  // Register a local memory and receive corresponding remote memories
  void registerMemoryPairs(void* buff, size_t buffSize, mscclpp::TransportFlags transport, int tag,
                           const std::vector<int>& remoteRanks, mscclpp::RegisteredMemory& localMemory,
                           std::unordered_map<int, mscclpp::RegisteredMemory>& remoteMemories);
  // Register a local memory an receive one corresponding remote memory
  void registerMemoryPair(void* buff, size_t buffSize, mscclpp::TransportFlags transport, int tag, int remoteRank,
                          mscclpp::RegisteredMemory& localMemory, mscclpp::RegisteredMemory& remoteMemory);

  int numRanksToUse = -1;
  std::shared_ptr<mscclpp::Communicator> communicator;
  mscclpp::Transport ibTransport;
  std::vector<mscclpp::RegisteredMemory> registeredMemories;
  std::unordered_map<int, mscclpp::Connection> connections;
  std::unordered_map<int, mscclpp::Connection> cpuConnections;
};

class CommunicatorTest : public CommunicatorTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;

  void deviceBufferInit();
  void writeToRemote(int dataCountPerRank);
  bool testWriteCorrectness(bool skipLocal = false);

  const size_t numBuffers = 10;
  const int deviceBufferSize = 1024 * 1024;
  std::vector<std::shared_ptr<int>> devicePtr;
  std::vector<mscclpp::RegisteredMemory> localMemory;
  std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>> remoteMemory;
};

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;

using IbMode = mscclpp::EndpointConfig::Ib::Mode;

class PortChannelOneToOneTest : public CommunicatorTestBase {
 protected:
  struct PingPongTestParams {
    bool useIPC;
    bool useIB;
    bool useEthernet;
    bool waitWithPoll;
    IbMode ibMode;
  };

  void SetUp() override;
  void TearDown() override;

  void setupMeshConnections(std::vector<mscclpp::PortChannel>& portChannels, bool useIPC, bool useIb, bool useEthernet,
                            void* sendBuff, size_t sendBuffBytes, void* recvBuff = nullptr, size_t recvBuffBytes = 0,
                            IbMode ibMode = IbMode::Default);
  void testPingPong(PingPongTestParams params);
  void testPingPongPerf(PingPongTestParams params);
  void testPacketPingPong(bool useIbOnly, IbMode ibMode = IbMode::Default);
  void testPacketPingPongPerf(bool useIbOnly, IbMode ibMode = IbMode::Default);
  void testAtomicAdd(bool useIPC, bool useIb, bool useEthernet, IbMode ibMode = IbMode::Default);
  void testBandwidth(PingPongTestParams params);
  void setupMultiQpChannels(int numQps, size_t elemsPerChan, IbMode ibMode, int tagBase,
                            std::vector<std::shared_ptr<int>>& sendBuffs,
                            std::vector<mscclpp::RegisteredMemory>& localMems,
                            std::vector<mscclpp::RegisteredMemory>& remoteMems,
                            std::vector<mscclpp::PortChannel>& portChannels);
  void testMultiQpBandwidth(IbMode ibMode, int numQps);
  void testMultiQpFlushStress(IbMode ibMode, int numQps);
  void testSameChanConcurrentFlush(IbMode ibMode);

  std::shared_ptr<mscclpp::ProxyService> proxyService;
};

class MemoryChannelOneToOneTest : public CommunicatorTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;

  void setupMeshConnections(std::vector<mscclpp::MemoryChannel>& memoryChannels, void* inputBuff, size_t inputBuffBytes,
                            void* outputBuff = nullptr, size_t outputBuffBytes = 0);
  using PacketPingPongKernelWrapper = std::function<void(int*, int, int, int*, int)>;
  void packetPingPongTest(const std::string testName, PacketPingPongKernelWrapper kernelWrapper);

  std::unordered_map<int, std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memorySemaphores;
};

/// Fixture for the bulk-copy usage patterns, shaped after the expert-parallel dispatch and combine
/// kernels: a full mesh where every rank holds a receive pool addressable by every peer, channels
/// used only for synchronization, and the peer pool pointers handed to the kernel as a plain array.
class BulkPatternTest : public CommunicatorTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;

  /// Register @p pool on every rank, exchange it around a full mesh, and build one synchronization
  /// channel per peer. Fills peerPools and syncHandles.
  void setupPeerPools(void* pool, size_t poolBytes);

  int worldSize = 0;
  int rank = 0;
  /// Receive pool base address of each rank, indexed by rank. The local entry is the local pool.
  /// This mirrors the expert-parallel kernels, which address peers by raw pointer and never route
  /// bulk data through a channel.
  std::vector<void*> peerPools;
  /// One channel per peer, used only for signal/wait.
  std::vector<mscclpp::BaseMemoryChannel> syncChannels;
  std::vector<DeviceHandle<mscclpp::BaseMemoryChannel>> syncHandles;
  std::vector<mscclpp::RegisteredMemory> remotePoolMemories;
};

class SemaphorePerfTest : public CommunicatorTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;
};

class SwitchChannelTest : public CommunicatorTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;
};

class SwitchChannelPerfTest : public CommunicatorTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;
};

class ExecutorTest : public MultiProcessTest {
 protected:
  void SetUp() override;
  void TearDown() override;

  std::shared_ptr<mscclpp::Executor> executor;
  std::string npkitDumpDir;
};
#endif  // MSCCLPP_MP_UNIT_TESTS_HPP_
