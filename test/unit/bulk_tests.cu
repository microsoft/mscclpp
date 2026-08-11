// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <vector>

#include "../framework.hpp"

#if defined(MSCCLPP_DEVICE_CUDA)

constexpr uint32_t kTile = 4096;
constexpr uint32_t kElems = kTile / sizeof(int);

// Load one tile into shared memory, then copy it out.
__global__ void kernelBulkLoad([[maybe_unused]] const int* src, [[maybe_unused]] int* dst) {
#if MSCCLPP_BULK_AVAILABLE
  __shared__ alignas(128) int tile[kElems];
  __shared__ mscclpp::BulkBarrier barrier;

  uint32_t phase = 0;
  if (threadIdx.x == 0) {
    barrier.init();
    barrier.arriveAndExpect(kTile);
    mscclpp::bulkLoad(tile, src, kTile, barrier);
    barrier.wait(phase);
  }
  __syncthreads();
  mscclpp::bulkFence();

  for (uint32_t i = threadIdx.x; i < kElems; i += blockDim.x) dst[i] = tile[i];

  __syncthreads();
  if (threadIdx.x == 0) barrier.invalidate();
#endif
}

// Gather NumSrc tiles into one barrier, then reduce them.
template <int NumSrc>
__global__ void kernelBulkGather([[maybe_unused]] const int* src, [[maybe_unused]] int* dst) {
#if MSCCLPP_BULK_AVAILABLE
  __shared__ alignas(128) int tiles[NumSrc][kElems];
  __shared__ mscclpp::BulkBarrier barrier;

  uint32_t phase = 0;
  if (threadIdx.x == 0) {
    barrier.init();
    barrier.arriveAndExpect(kTile * NumSrc);  // whole batch total, before any arrival completes
    for (int s = 0; s < NumSrc; ++s) mscclpp::bulkLoad(tiles[s], src + s * kElems, kTile, barrier);
    barrier.wait(phase);
  }
  __syncthreads();
  mscclpp::bulkFence();

  for (uint32_t i = threadIdx.x; i < kElems; i += blockDim.x) {
    int acc = 0;
    for (int s = 0; s < NumSrc; ++s) acc += tiles[s][i];
    dst[i] = acc;
  }

  __syncthreads();
  if (threadIdx.x == 0) barrier.invalidate();
#endif
}

// Reuse one barrier across NumChunks phases, staging each chunk in and storing it back out.
template <int NumChunks>
__global__ void kernelBulkRoundTrip([[maybe_unused]] const int* src, [[maybe_unused]] int* dst) {
#if MSCCLPP_BULK_AVAILABLE
  __shared__ alignas(128) int tile[kElems];
  __shared__ mscclpp::BulkBarrier barrier;

  if (threadIdx.x == 0) barrier.init();
  __syncthreads();

  uint32_t phase = 0;  // one barrier, no re-initialization; wait() advances the phase
  for (int c = 0; c < NumChunks; ++c) {
    if (threadIdx.x == 0) {
      barrier.arriveAndExpect(kTile);
      mscclpp::bulkLoad(tile, src + c * kElems, kTile, barrier);
      barrier.wait(phase);
    }
    __syncthreads();
    mscclpp::bulkFence();

    for (uint32_t i = threadIdx.x; i < kElems; i += blockDim.x) tile[i] += 1;

    __syncthreads();
    mscclpp::bulkFence();
    if (threadIdx.x == 0) {
      mscclpp::bulkStore(dst + c * kElems, tile, kTile);
      mscclpp::bulkStoreCommit();
      mscclpp::bulkStoreWaitSource();  // tile refillable; the store may still be in flight
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    mscclpp::bulkStoreWait();  // every store has landed
    barrier.invalidate();
  }
#endif
}

// Accumulate a staged tile into a seeded destination with the copy engine.
template <typename T>
__global__ void kernelBulkReduce([[maybe_unused]] T* dst, [[maybe_unused]] T addend, [[maybe_unused]] uint32_t count) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t raw[];
  T* tile = reinterpret_cast<T*>(raw);
  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) tile[i] = addend;
  __syncthreads();
  mscclpp::bulkFence();
  if (threadIdx.x == 0) {
    mscclpp::bulkReduceStore<T>(dst, tile, count * sizeof(T));
    mscclpp::bulkStoreCommit();
    mscclpp::bulkStoreWait();
  }
#endif
}

class BulkTestData {
 public:
  BulkTestData(int numElems)
      : src_(mscclpp::GpuBuffer<int>(numElems).memory()),
        dst_(mscclpp::GpuBuffer<int>(numElems).memory()),
        numElems_(numElems) {
    std::vector<int> host(numElems);
    for (int i = 0; i < numElems; ++i) host[i] = i + 1;
    MSCCLPP_CUDATHROW(cudaMemcpy(src_.get(), host.data(), numElems * sizeof(int), cudaMemcpyHostToDevice));
    MSCCLPP_CUDATHROW(cudaMemset(dst_.get(), 0, numElems * sizeof(int)));
  }

  std::vector<int> result() {
    std::vector<int> out(numElems_);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    MSCCLPP_CUDATHROW(cudaMemcpy(out.data(), dst_.get(), out.size() * sizeof(int), cudaMemcpyDeviceToHost));
    return out;
  }

  int* src() { return src_.get(); }
  int* dst() { return dst_.get(); }

 private:
  std::shared_ptr<int> src_;
  std::shared_ptr<int> dst_;
  int numElems_;
};

TEST(BulkTest, Load) {
  if (!mscclpp::isBulkSupported()) {
    SKIP_TEST() << "Bulk asynchronous copy requires compute capability 9.0 or higher.";
    return;
  }
  BulkTestData f(kElems);
  kernelBulkLoad<<<1, 256>>>(f.src(), f.dst());
  std::vector<int> out = f.result();
  for (uint32_t i = 0; i < kElems; ++i) EXPECT_EQ(out[i], (int)i + 1);
}

TEST(BulkTest, Gather) {
  if (!mscclpp::isBulkSupported()) {
    SKIP_TEST() << "Bulk asynchronous copy requires compute capability 9.0 or higher.";
    return;
  }
  constexpr int kNumSrc = 4;
  BulkTestData f(kElems * kNumSrc);
  kernelBulkGather<kNumSrc><<<1, 256>>>(f.src(), f.dst());
  std::vector<int> out = f.result();
  for (uint32_t i = 0; i < kElems; ++i) {
    int expected = 0;
    for (int s = 0; s < kNumSrc; ++s) expected += s * kElems + i + 1;
    EXPECT_EQ(out[i], expected);
  }
}

TEST(BulkTest, RoundTrip) {
  if (!mscclpp::isBulkSupported()) {
    SKIP_TEST() << "Bulk asynchronous copy requires compute capability 9.0 or higher.";
    return;
  }
  constexpr int kNumChunks = 8;
  BulkTestData f(kElems * kNumChunks);
  kernelBulkRoundTrip<kNumChunks><<<1, 256>>>(f.src(), f.dst());
  std::vector<int> out = f.result();
  for (uint32_t i = 0; i < kElems * kNumChunks; ++i) EXPECT_EQ(out[i], (int)i + 2);
}

// Each reduction type accumulates into a destination seeded with a known base, so the test
// distinguishes an accumulate from an overwrite.
template <typename T>
static void reduceStoreTest(float base, float addend) {
  constexpr uint32_t kCount = 1024;
  std::shared_ptr<T> dst = mscclpp::GpuBuffer<T>(kCount).memory();
  std::vector<T> host(kCount, static_cast<T>(base));
  MSCCLPP_CUDATHROW(cudaMemcpy(dst.get(), host.data(), kCount * sizeof(T), cudaMemcpyHostToDevice));

  kernelBulkReduce<T><<<1, 256, kCount * sizeof(T)>>>(dst.get(), static_cast<T>(addend), kCount);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  std::vector<T> out(kCount);
  MSCCLPP_CUDATHROW(cudaMemcpy(out.data(), dst.get(), kCount * sizeof(T), cudaMemcpyDeviceToHost));
  for (uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(static_cast<float>(out[i]), base + addend);
  }
}

TEST(BulkTest, ReduceStoreFloat) {
  if (!mscclpp::isBulkSupported()) {
    SKIP_TEST() << "Bulk asynchronous copy requires compute capability 9.0 or higher.";
    return;
  }
  reduceStoreTest<float>(10.0f, 1.5f);
}

TEST(BulkTest, ReduceStoreBf16) {
  if (!mscclpp::isBulkSupported()) {
    SKIP_TEST() << "Bulk asynchronous copy requires compute capability 9.0 or higher.";
    return;
  }
  reduceStoreTest<__nv_bfloat16>(10.0f, 2.0f);
}

TEST(BulkTest, ReduceStoreUint32) {
  if (!mscclpp::isBulkSupported()) {
    SKIP_TEST() << "Bulk asynchronous copy requires compute capability 9.0 or higher.";
    return;
  }
  reduceStoreTest<uint32_t>(10.0f, 3.0f);
}

#endif  // defined(MSCCLPP_DEVICE_CUDA)
