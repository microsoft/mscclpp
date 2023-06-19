#ifndef MSCCLPP_MP_UNIT_TESTS_HPP_
#define MSCCLPP_MP_UNIT_TESTS_HPP_

#include <gtest/gtest.h>

#include <mscclpp/channel.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/utils.hpp>

#include "ib.hpp"

class MultiProcessTestEnv : public ::testing::Environment {
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

class MultiProcessTest : public ::testing::Test {
 protected:
  void TearDown() override;
};

class BootstrapTest : public MultiProcessTest {
 protected:
  void bootstrapTestAllGather(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap);

  void bootstrapTestBarrier(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap);

  void bootstrapTestSendRecv(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap);

  void bootstrapTestAll(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap);

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

  void stageSend(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled);

  void stageAtomicAdd(uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, uint64_t addVal);

  void stageSendWithImm(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled,
                        unsigned int immData);

  std::shared_ptr<mscclpp::Bootstrap> bootstrap;
  std::shared_ptr<mscclpp::IbCtx> ibCtx;
  mscclpp::IbQp* qp;
  const mscclpp::IbMr* mr;
  size_t bufSize;

  std::array<mscclpp::IbQpInfo, 2> qpInfo;
  std::array<mscclpp::IbMrInfo, 2> mrInfo;
};

class CommunicatorTestBase : public MultiProcessTest {
 protected:
  void SetUp() override;
  void TearDown() override;

  void setNumRanksToUse(int num);
  void connectMesh(bool useIbOnly = false);

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
  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> connections;
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

class DeviceChannelOneToOneTest : public CommunicatorTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;

  void setupMeshConnections(std::vector<mscclpp::channel::SimpleDeviceChannel>& devChannels, bool useIbOnly,
                            void* sendBuff, size_t sendBuffBytes, void* recvBuff = nullptr, size_t recvBuffBytes = 0);

  std::shared_ptr<mscclpp::channel::DeviceChannelService> channelService;
};

class SmChannelOneToOneTest : public CommunicatorTestBase {
 protected:
  void SetUp() override;
  void TearDown() override;

  void setupMeshConnections(std::vector<mscclpp::channel::SmChannel>& smChannels, void* inputBuff,
                            size_t inputBuffBytes, void* outputBuff = nullptr, size_t outputBuffBytes = 0);

  std::unordered_map<int, std::shared_ptr<mscclpp::SmEpoch>> smEpochs;
};

#endif  // MSCCLPP_MP_UNIT_TESTS_HPP_
