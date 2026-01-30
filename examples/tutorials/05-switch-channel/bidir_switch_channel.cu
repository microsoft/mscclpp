// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mscclpp/switch_channel.hpp>
#include <mscclpp/switch_channel_device.hpp>

// -------------------------
// Minimal error helpers
// -------------------------
#define CUDA_THROW(call)                                                                       \
  do {                                                                                         \
    cudaError_t _e = (call);                                                                   \
    if (_e != cudaSuccess) {                                                                   \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e) +          \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));           \
    }                                                                                          \
  } while (0)

static int getenv_int_any(const std::vector<const char*>& keys, int default_val = -1) {
  for (auto* k : keys) {
    const char* v = std::getenv(k);
    if (v && *v) return std::atoi(v);
  }
  return default_val;
}

static int parse_arg_int(int argc, char** argv, const char* key, int default_val = -1) {
  // Supports: --rank=3  or  --rank 3
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], key, std::strlen(key)) == 0) {
      const char* s = argv[i] + std::strlen(key);
      if (*s == '=') return std::atoi(s + 1);
      if (*s == '\0' && i + 1 < argc) return std::atoi(argv[i + 1]);
    }
  }
  return default_val;
}

static void usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [--rank N] [--world-size N] [--local-rank N] [--ranks-per-node N]\n"
      << "Typically run under mpirun/srun, in which case rank info is taken from env.\n";
}

// -------------------------
// Kernel
// -------------------------
__constant__ mscclpp::SwitchChannelDeviceHandle gConstSwitchChan;

__global__ void kernelSwitchReduce() {
#if (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
  auto val = gConstSwitchChan.reduce<mscclpp::f32x1>(0);
  gConstSwitchChan.broadcast(0, val);
#endif
}

// -------------------------
// Main
// -------------------------
int main(int argc, char** argv) {
  try {

     MPI_Init(&argc, &argv);
      int rank = 0, world_size = 1;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &world_size);

     printf("mpi initialized\n");

    if (!mscclpp::isNvlsSupported()) {
      if (rank == 0) std::cerr << "Skipping: NVLS not supported on this system.\n";
      return 0;
    }

    // The original test uses only 2 ranks.
    const int numRanksToUse = 2;
    if (world_size < numRanksToUse) {
      if (rank == 0) std::cerr << "Skipping: need world_size >= " << numRanksToUse << "\n";
      return 0;
    }
    if (rank >= numRanksToUse) {
      // Mirror `if (gEnv->rank >= numRanksToUse) return;`
      return 0;
    }

    int local_rank=rank;
    // 3) Select device based on local_rank (common practice in multi-proc GPU runs).
    int device_count = 0;
    CUDA_THROW(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) throw std::runtime_error("No CUDA devices visible.");
    int dev = local_rank % device_count;
    CUDA_THROW(cudaSetDevice(dev));

    printf("bootstrap\n");
    // 4) Bootstrap + Communicator.
    // MSCCL++ docs show TcpBootstrap(rank, world_size) as the standard host-side setup. :contentReference[oaicite:1]{index=1}
    std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
    mscclpp::UniqueId id;
    bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
    if (rank == 0) id = bootstrap->createUniqueId();
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    bootstrap->initialize(id);
    std::shared_ptr<mscclpp::Communicator> communicator = std::make_shared<mscclpp::Communicator>(bootstrap);



    //auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
    //auto communicator = std::make_shared<mscclpp::Communicator>(bootstrap);

     printf("bootstrap done. allocate GPU buffer\n");
    // 5) Build ranks list [0,1] and allocate GPU buffer.
    std::vector<int> ranks;
    ranks.reserve(numRanksToUse);
    for (int i = 0; i < numRanksToUse; i++) ranks.push_back(i);

    auto buffer = mscclpp::GpuBuffer<float>(1024);
    float data = static_cast<float>(rank) + 1.0f;
    CUDA_THROW(cudaMemcpy(buffer.data(), &data, sizeof(data), cudaMemcpyHostToDevice));

    printf("start nvls collective\n");
    // 6) NVLS collective connect + bind memory to SwitchChannel.
    auto nvlsConnection = mscclpp::connectNvlsCollective(communicator, ranks, 1024);

    printf("start nvls collective2\n");
    auto switchChannel = nvlsConnection->bindAllocatedMemory(CUdeviceptr(buffer.data()), 1024);

    printf("start nvls collective3\n");
    auto deviceHandle = switchChannel.deviceHandle();

    printf("start nvls collective4\n");
    CUDA_THROW(cudaMemcpyToSymbol(gConstSwitchChan, &deviceHandle, sizeof(deviceHandle)));
    CUDA_THROW(cudaDeviceSynchronize());

    printf("nvls collective done\n");

    // 7) Barrier, run kernel on rank 0, barrier.
    communicator->bootstrap()->barrier();

    if (rank == 0) {
      kernelSwitchReduce<<<1, 1>>>();
      CUDA_THROW(cudaGetLastError());
      CUDA_THROW(cudaDeviceSynchronize());
    }

    communicator->bootstrap()->barrier();

    // 8) Validate.
    float result = 0.0f;
    CUDA_THROW(cudaMemcpy(&result, buffer.data(), sizeof(result), cudaMemcpyDeviceToHost));

    float expected = 0.0f;
    for (int i = 0; i < numRanksToUse; i++) expected += static_cast<float>(i) + 1.0f;

    if (result != expected) {
      std::cerr << "FAIL: Expected " << expected << " but got " << result
                << " for rank " << rank << "\n";
      return 1;
    }

    if (rank == 0) std::cout << "PASS: result=" << result << " expected=" << expected << "\n";
    MPI_Finalize();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << "\n";
    return 3;
  }
}

