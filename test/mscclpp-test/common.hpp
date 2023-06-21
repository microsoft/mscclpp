// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_TESTS_COMMON_H_
#define MSCCLPP_TESTS_COMMON_H_

#include <mscclpp/channel.hpp>
#include <mscclpp/core.hpp>

#define CUDATHROW(cmd)                                                                                                \
  do {                                                                                                                \
    cudaError_t err = cmd;                                                                                            \
    if (err != cudaSuccess) {                                                                                         \
      std::string msg = std::string("Test CUDA failure: ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                        " '" + cudaGetErrorString(err) + "'";                                                         \
      throw std::runtime_error(msg);                                                                                  \
    }                                                                                                                 \
  } while (0)

int getDeviceNumaNode(int cudaDev);
void numaBind(int node);

struct TestArgs {
  size_t minBytes;
  size_t maxBytes;
  size_t stepBytes;
  size_t stepFactor;

  int totalRanks;
  int rank;
  int gpuNum;
  int localRank;
  int nRanksPerNode;
  int kernelNum;
  int reportErrors;
};

class BaseTestColl {
 public:
  BaseTestColl() {}
  virtual ~BaseTestColl() {}
  virtual void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) = 0;
  virtual void runColl(const TestArgs& args, cudaStream_t stream) = 0;
  virtual void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) = 0;

  void setupCollTest(const TestArgs& args, size_t size);
  void setChanService(std::shared_ptr<mscclpp::channel::BaseChannelService> chanService) { chanService_ = chanService; }
  size_t getSendBytes() { return sendCount_ * typeSize_; }
  size_t getRecvBytes() { return recvCount_ * typeSize_; }
  size_t getExpectedBytes() { return expectedCount_ * typeSize_; }
  size_t getParamBytes() { return paramCount_ * typeSize_; }

 protected:
  size_t sendCount_;
  size_t recvCount_;
  size_t expectedCount_;
  size_t paramCount_;
  int typeSize_;
  int worldSize_;
  int kernelNum_;

  std::shared_ptr<mscclpp::channel::BaseChannelService> chanService_;

 private:
  virtual void setupCollTest(size_t size) = 0;
};

class BaseTestEngine {
 public:
  BaseTestEngine(const TestArgs& args, const std::string& name);
  virtual ~BaseTestEngine();
  virtual void allocateBuffer() = 0;

  int getTestErrors() { return error_; }
  void setupTest();
  void bootstrap();
  void runTest();
  void barrier();
  size_t checkData();

  virtual std::vector<void*> getSendBuff() = 0;
  virtual void* getRecvBuff() = 0;
  virtual void* getScratchBuff() = 0;

 private:
  virtual void setupConnections() = 0;
  virtual std::shared_ptr<mscclpp::channel::BaseChannelService> createChannelService();
  virtual void* getExpectedBuff() = 0;

  double benchTime();

  void setupMeshConnectionsInternal(
      std::vector<std::shared_ptr<mscclpp::Connection>>& connections, mscclpp::RegisteredMemory& inputBufRegMem,
      mscclpp::RegisteredMemory& outputBufRegMem,
      std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>>& remoteRegMemories, void* inputBuff,
      size_t inputBuffBytes, void* outputBuff, size_t outputBuffBytes);

 protected:
  using SetupChannelFunc = std::function<void(std::vector<std::shared_ptr<mscclpp::Connection>>,
                                              std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>>&,
                                              const mscclpp::RegisteredMemory&)>;
  void setupMeshConnections(std::vector<mscclpp::channel::SimpleDeviceChannel>& devChannels, void* inputBuff,
                            size_t inputBuffBytes, void* outputBuff = nullptr, size_t outputBuffBytes = 0,
                            SetupChannelFunc setupChannel = nullptr);
  void setupMeshConnections(std::vector<mscclpp::channel::DirectChannel>& dirChannels, void* inputBuff,
                            size_t inputBuffBytes, void* outputBuff, size_t outputBuffBytes = 0);

  const TestArgs args_;
  const std::string name_;
  bool inPlace_;
  std::shared_ptr<BaseTestColl> coll_;
  std::shared_ptr<mscclpp::Communicator> comm_;
  std::shared_ptr<mscclpp::channel::BaseChannelService> chanService_;
  cudaStream_t stream_;
  int error_;
};

extern std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args);
extern std::shared_ptr<BaseTestColl> getTestColl();
extern mscclpp::Transport IBs[];

#endif  // MSCCLPP_TESTS_COMMON_H_
