// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sys/wait.h>
#include <unistd.h>

#include <functional>
#include <iostream>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/switch_channel_device.hpp>
#include <sstream>

#define PORT_NUMBER "50505"

template <typename... Args>
void log(Args &&...args) {
  std::stringstream ss;
  (ss << ... << args);
  ss << std::endl;
  std::cout << ss.str();
}

int spawn_process(std::function<void()> func) {
  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    // Child process
    func();
    exit(0);
  }
  return pid;
}

int wait_process(int pid) {
  int status;
  if (waitpid(pid, &status, 0) < 0) {
    return -1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return -1;
}

__constant__ mscclpp::SwitchChannelDeviceHandle gConstSwitchChan;

__device__ mscclpp::DeviceSyncer devSyncer;

__global__ void kernelSwitchReduce(int rank, int numElements) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // rank0: 0-(numElements/2)-1
  // rank1: (numElements/2)-(numElements-1)
  int min = rank * (numElements / 2);
  int max = (rank + 1) * (numElements / 2);

  for (int i = tid + min; i < max; i += stride) {
    auto val = gConstSwitchChan.reduce<mscclpp::f32x1>(i);
    gConstSwitchChan.broadcast(i, val);
  }
}

int worker(int myRank, int gpuId, const std::string &ipPort) {
  MSCCLPP_CUDATHROW(cudaSetDevice(gpuId));
  const int nRanks = 2;
  const int iter = 1000;
  const size_t bufferBytes = 128 * 1024 * 1024;

  log("Rank ", myRank, " (GPU ", gpuId, "): Preparing for tests ...");

  // Build a connection and a semaphore
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, nRanks);
  bootstrap->initialize(ipPort);
  std::shared_ptr<mscclpp::Communicator> comm = std::make_shared<mscclpp::Communicator>(bootstrap);

  std::vector<int> ranks;
  ranks.reserve(nRanks);
  for (int i = 0; i < nRanks; i++) ranks.push_back(i);

  auto buffer = mscclpp::GpuBuffer<float>(bufferBytes);

  auto nvlsConnection = mscclpp::connectNvlsCollective(comm, ranks, bufferBytes);

  auto switchChannel = nvlsConnection->bindAllocatedMemory(CUdeviceptr(buffer.data()), bufferBytes);

  auto deviceHandle = switchChannel.deviceHandle();

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gConstSwitchChan, &deviceHandle, sizeof(deviceHandle)));
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  // Call the kernel in a loop for perf evaluation

  for (size_t numElements : {1024, 1024 * 1024, 32 * 1024 * 1024}) {
    cudaEvent_t start, end;
    if (myRank == 0) {
      MSCCLPP_CUDATHROW(cudaEventCreate(&start));
      MSCCLPP_CUDATHROW(cudaEventCreate(&end));
    }
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    bootstrap->barrier();

    if (myRank == 0) {
      MSCCLPP_CUDATHROW(cudaEventRecord(start, 0));
    }

    for (int i = 0; i < iter; ++i) {
      kernelSwitchReduce<<<256, 1024>>>(myRank, numElements);
    }

    MSCCLPP_CUDATHROW(cudaGetLastError());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

    comm->bootstrap()->barrier();

    if (myRank == 0) {
      MSCCLPP_CUDATHROW(cudaEventRecord(end, 0));
      MSCCLPP_CUDATHROW(cudaEventSynchronize(end));
      float elapsedTime;
      float elapsedTimePerIter;
      float gbps;
      MSCCLPP_CUDATHROW(cudaEventElapsedTime(&elapsedTime, start, end));
      elapsedTimePerIter = elapsedTime / iter;
      float dataSize = numElements * 4;
      gbps = dataSize / elapsedTimePerIter * 1e-6f;
      log("Rank ", myRank, " (GPU ", gpuId, "): bytes ", dataSize, ", elapsed ", elapsedTimePerIter, " ms/iter, BW ",
          gbps, " GB/s");
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    int pid0 = spawn_process([]() {
      int rc = worker(0, 0, "lo:127.0.0.1:" PORT_NUMBER);
      exit(rc);
    });
    int pid1 = spawn_process([]() {
      int rc = worker(1, 1, "lo:127.0.0.1:" PORT_NUMBER);
      exit(rc);
    });
    if (pid0 < 0 || pid1 < 0) {
      log("Failed to spawn processes.");
      return -1;
    }
    int status0 = wait_process(pid0);
    int status1 = wait_process(pid1);
    if (status0 < 0 || status1 < 0) {
      log("Failed to wait for processes.");
      return -1;
    }
    if (status0 != 0 || status1 != 0) {
      log("One of the processes failed.");
      return -1;
    }
    log("Succeed!");
    return 0;
  } else if (argc == 4) {
    std::string ipPort = argv[1];
    int rank, gpuId;
    try {
      rank = std::stoi(argv[2]);
      gpuId = std::stoi(argv[3]);
    } catch (const std::exception &) {
      log("Error: rank and gpu_id must be valid integers.");
      return -1;
    }
    if (rank < 0 || rank > 2 || gpuId < 0) {
      log("Error: rank must be between 0 and 1 and gpu_id must be non-negative.");
      return -1;
    }
    if (worker(rank, gpuId, ipPort) == 0) {
      log("Rank ", rank, ": Succeed!");
      return 0;
    } else {
      return -1;
    }
  } else {
    std::cerr << "Usage:\n"
              << "  " << argv[0] << "                Run in intra-node mode\n"
              << "  " << argv[0] << " <ip_port> <rank> <gpu_id>   Run in inter-node mode\n";
    return -1;
  }
}
