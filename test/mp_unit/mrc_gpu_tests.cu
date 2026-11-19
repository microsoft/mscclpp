// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <chrono>
#include <limits>
#include <mscclpp/gpu_utils.hpp>
#include <thread>

#include "atomic.hpp"
#include "gdr.hpp"
#include "mp_unit_tests.hpp"

namespace {

constexpr unsigned int mrcVisibilityRounds = 64;
constexpr unsigned int mrcPayloadWords = 1024;
constexpr uint64_t mrcAbortToken = std::numeric_limits<uint64_t>::max();
constexpr uint64_t mrcKernelTimeoutNs = 20000000000ULL;

struct MrcGpuResult {
  uint64_t ready;
  uint64_t completed;
  uint64_t failure;
  uint64_t badIndex;
  uint64_t expected;
  uint64_t observed;
  uint64_t warmupChecksum;
};

__host__ __device__ uint32_t mrcPayloadValue(unsigned int round, unsigned int index, int rank) {
  return round * 0x9e3779b9U ^ index * 0x85ebca6bU ^ (rank + 1) * 0x01010101U;
}

__device__ uint64_t mrcGlobalTimer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__global__ void mrcObservePayload(const uint32_t* payload, uint64_t* token, MrcGpuResult* result, int peerRank) {
  uint64_t start = mrcGlobalTimer();
  uint64_t checksum = 0;
  for (unsigned int i = 0; i < mrcPayloadWords; ++i) checksum += payload[i];
  result->warmupChecksum = checksum;
  mscclpp::atomicStore(&result->ready, uint64_t{1}, mscclpp::memoryOrderRelease);
  for (unsigned int round = 1; round <= mrcVisibilityRounds; ++round) {
    for (;;) {
      uint64_t observed = mscclpp::atomicLoad(token, mscclpp::memoryOrderAcquire);
      if (observed == round) break;
      if (observed == mrcAbortToken || observed > round || mrcGlobalTimer() - start > mrcKernelTimeoutNs) {
        result->expected = round;
        result->observed = observed;
        mscclpp::atomicStore(&result->failure, uint64_t{1}, mscclpp::memoryOrderRelease);
        return;
      }
      __nanosleep(64);
    }
    for (unsigned int i = 0; i < mrcPayloadWords; ++i) {
      uint32_t expected = mrcPayloadValue(round, i, peerRank);
      uint32_t observed = payload[i];
      if (observed != expected) {
        result->badIndex = i;
        result->expected = expected;
        result->observed = observed;
        mscclpp::atomicStore(&result->failure, uint64_t{2}, mscclpp::memoryOrderRelease);
        return;
      }
    }
    mscclpp::atomicStore(&result->completed, static_cast<uint64_t>(round), mscclpp::memoryOrderRelease);
  }
}

}  // namespace

class MrcGpuVisibilityTest : public MrcTestBase {
 public:
  ~MrcGpuVisibilityTest() override { stopKernel(); }

 protected:
  void SetUp() override {
    MrcTestBase::SetUp();
    runOnBothRanks("GPU/native resources", [&] {
      int count = 0;
      MSCCLPP_CUDATHROW(cudaGetDeviceCount(&count));
      ASSERT_GT(count, 0);
      int deviceId = rankToLocalRank(gEnv->rank) % count;
      MSCCLPP_CUDATHROW(cudaSetDevice(deviceId));
      if (!mscclpp::gdrEnabled()) FAIL() << mscclpp::gdrStatusMessage();
      context = std::make_unique<mscclpp::MrcCtx>(mrcDevice);
      source = mscclpp::detail::gpuCallocShared<uint32_t>(mrcPayloadWords * mrcVisibilityRounds);
      destination = mscclpp::detail::gpuCallocShared<uint32_t>(mrcPayloadWords);
      token = mscclpp::detail::gpuCallocShared<uint64_t>();
      tokenMap = std::make_unique<mscclpp::GdrMap>(token, deviceId);
      ASSERT_TRUE(tokenMap->valid());
      // GPU acknowledgements go to host memory, avoiding a GPU BAR readback after token publication.
      result = mscclpp::detail::gpuCallocHostShared<MrcGpuResult>();
      MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(reinterpret_cast<void**>(&deviceResult), result.get(), 0));
      std::vector<uint32_t> initial(mrcPayloadWords * mrcVisibilityRounds);
      for (unsigned int round = 1; round <= mrcVisibilityRounds; ++round) {
        for (unsigned int i = 0; i < mrcPayloadWords; ++i) {
          initial[(round - 1) * mrcPayloadWords + i] = mrcPayloadValue(round, i, gEnv->rank);
        }
      }
      MSCCLPP_CUDATHROW(
          cudaMemcpy(source.get(), initial.data(), initial.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
      MSCCLPP_CUDATHROW(cudaMemset(destination.get(), 0x5a, mrcPayloadWords * sizeof(uint32_t)));
      sourceMr = context->registerMr(source.get(), initial.size() * sizeof(uint32_t));
      destinationMr = context->registerMr(destination.get(), mrcPayloadWords * sizeof(uint32_t));
      qp = context->createQp(1, std::stoi(gEnv->args["ib_gid_index"]), 0, queueDepth, queueDepth, queueDepth);
      MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
      stream.set(cudaStreamNonBlocking);
    });
    remoteMr = connectPeer(*qp, *destinationMr);
  }

  void TearDown() override {
    runOnBothRanks("GPU kernel shutdown", [&] { MSCCLPP_CUDATHROW(stopKernel()); });
    MultiProcessTest::TearDown();
  }

  cudaError_t stopKernel() {
    if (!kernelStarted) return cudaSuccess;
    mscclpp::atomicStore(tokenMap->hostPtr(), mrcAbortToken, mscclpp::memoryOrderRelease);
    kernelStarted = false;
    return cudaStreamSynchronize(stream);
  }

  void checkGpu() {
    uint64_t failure = mscclpp::atomicLoad(&result->failure, mscclpp::memoryOrderAcquire);
    if (failure != 0) {
      FAIL() << "GPU visibility failure=" << failure << " completed=" << result->completed
             << " index=" << result->badIndex << " expected=" << result->expected << " observed=" << result->observed;
    }
  }

  void testVisibility(unsigned int writes) {
    runOnBothRanks("persistent GPU receiver readiness", [&] {
      mrcObservePayload<<<1, 1, 0, stream>>>(destination.get(), token.get(), deviceResult, 1 - gEnv->rank);
      kernelStarted = true;
      MSCCLPP_CUDATHROW(cudaGetLastError());
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
      while (mscclpp::atomicLoad(&result->ready, mscclpp::memoryOrderAcquire) == 0) {
        checkGpu();
        if (std::chrono::steady_clock::now() >= deadline) FAIL() << "GPU receiver readiness timeout";
        std::this_thread::yield();
      }
    });
    for (unsigned int round = 1; round <= mrcVisibilityRounds; ++round) {
      runOnBothRanks("receive credit acknowledgement", [&] {
        qp->stageRecv(round);
        qp->postRecv();
      });
      runOnBothRanks("GPU payload before forwarded token", [&] {
        constexpr unsigned int payloadBytes = mrcPayloadWords * sizeof(uint32_t);
        uint64_t sourceOffset = (round - 1) * payloadBytes;
        if (writes == 0) {
          qp->stageSendWriteWithImm(sourceMr.get(), remoteMr, payloadBytes, round, sourceOffset, 0, true, round);
        } else {
          unsigned int chunkBytes = payloadBytes / writes;
          for (unsigned int i = 0; i < writes; ++i) {
            qp->stageSendWrite(sourceMr.get(), remoteMr, chunkBytes, round * 8 + i, sourceOffset + i * chunkBytes,
                               i * chunkBytes, true);
          }
          qp->stageSendWriteWithImm(nullptr, remoteMr, 0, round * 8 + writes, 0, 0, true, round);
        }
        qp->postSend();
        int received = 0;
        int sent = 0;
        int expectedSends = writes + 1;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
        while (received != 1 || sent != expectedSends ||
               mscclpp::atomicLoad(&result->completed, mscclpp::memoryOrderAcquire) < round) {
          int count = qp->pollRecvCq();
          ASSERT_GE(count, 0);
          for (int i = 0; i < count; ++i) {
            if (qp->getRecvWcStatus(i) != 0) FAIL() << qp->getRecvWcStatusString(i);
            ++received;
            ASSERT_EQ(received, 1);
            ASSERT_EQ(qp->getRecvWcImmData(i), round);
            // No CUDA operation, payload copyback or GPU BAR read intervenes in this handoff.
            mscclpp::atomicStore(tokenMap->hostPtr(), static_cast<uint64_t>(round), mscclpp::memoryOrderRelease);
          }
          count = qp->pollSendCq();
          ASSERT_GE(count, 0);
          for (int i = 0; i < count; ++i) {
            if (qp->getSendWcStatus(i) != 0) FAIL() << qp->getSendWcStatusString(i);
          }
          sent += count;
          ASSERT_LE(sent, expectedSends);
          checkGpu();
          if (std::chrono::steady_clock::now() >= deadline) FAIL() << "Native/GPU completion timeout";
          std::this_thread::yield();
        }
        checkGpu();
        ASSERT_EQ(qp->getNumSendCqItems(), 0);
      });
    }
    runOnBothRanks("persistent GPU receiver completion", [&] {
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      kernelStarted = false;
      checkGpu();
      ASSERT_EQ(mscclpp::atomicLoad(&result->completed, mscclpp::memoryOrderAcquire), mrcVisibilityRounds);
    });
  }

  std::unique_ptr<mscclpp::MrcCtx> context;
  std::shared_ptr<uint32_t> source;
  std::shared_ptr<uint32_t> destination;
  std::shared_ptr<uint64_t> token;
  std::shared_ptr<MrcGpuResult> result;
  std::unique_ptr<mscclpp::GdrMap> tokenMap;
  std::unique_ptr<const mscclpp::MrcMr> sourceMr;
  std::unique_ptr<const mscclpp::MrcMr> destinationMr;
  std::shared_ptr<mscclpp::MrcQp> qp;
  mscclpp::MrcBuffer remoteMr;
  MrcGpuResult* deviceResult = nullptr;
  mscclpp::CudaStreamWithFlags stream;
  bool kernelStarted = false;
};

TEST(MrcGpuVisibilityTest, WriteWithImmediatePayload) { testVisibility(0); }

TEST(MrcGpuVisibilityTest, WriteThenZeroByteNotification) { testVisibility(1); }

TEST(MrcGpuVisibilityTest, NonoverlappingWritesThenZeroByteNotification) { testVisibility(4); }
