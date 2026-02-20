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

__global__ void kernelSwitchReduce(int indx) {
  auto val = gConstSwitchChan.reduce<mscclpp::f32x1>(indx);
  gConstSwitchChan.broadcast(indx, val);
}

void worker(int myRank, int gpuId, const std::string &ipPort) {
  MSCCLPP_CUDATHROW(cudaSetDevice(gpuId));
  const int nRanks = 2;
  const int iter = 1000;
  const size_t bufferBytes = 256 * 1024 * 1024;

  log("Rank ", myRank, " (GPU ", gpuId, "): Preparing for tests ...");

  // Build a connection and a semaphore
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, nRanks);
  bootstrap->initialize(ipPort);
  std::shared_ptr<mscclpp::Communicator> comm = std::make_shared<mscclpp::Communicator>(bootstrap);

  std::vector<int> ranks;
  ranks.reserve(nRanks);
  for (int i = 0; i < nRanks; i++) ranks.push_back(i);

  float data[1024];
  for (int i = 0; i < 1024; ++i) {
    data[i] = static_cast<float>(myRank) + 1.0f;
  }

  auto buffer = mscclpp::GpuBuffer<float>(1024);
  cudaMemcpy(buffer.data(), data, sizeof(data), cudaMemcpyHostToDevice);

  auto nvlsConnection = mscclpp::connectNvlsCollective(comm, ranks, 1024);

  auto switchChannel = nvlsConnection->bindAllocatedMemory(CUdeviceptr(buffer.data()), 1024);

  auto deviceHandle = switchChannel.deviceHandle();

  cudaMemcpyToSymbol(gConstSwitchChan, &deviceHandle, sizeof(deviceHandle));
  cudaDeviceSynchronize();

  comm->bootstrap()->barrier();

  if (myRank == 0) {
    kernelSwitchReduce<<<1, 1>>>(myRank*512);
    cudaGetLastError();
    cudaDeviceSynchronize();
  }

  comm->bootstrap()->barrier();

  float dataout[1024];
  for (int i = 0; i < 1024; ++i) {
    dataout[i] = static_cast<float>(0.0f);
  }
  cudaMemcpy(dataout, buffer.data(), sizeof(dataout), cudaMemcpyDeviceToHost);

}

int main(int argc, char **argv) {
  if (argc == 1) {
    int pid0 = spawn_process([]() { worker(0, 0, "lo:127.0.0.1:" PORT_NUMBER); });
    int pid1 = spawn_process([]() { worker(1, 1, "lo:127.0.0.1:" PORT_NUMBER); });
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
    worker(rank, gpuId, ipPort);
    log("Rank ", rank, ": Succeed!");
    return 0;
  } else {
    std::cerr << "Usage:\n"
              << "  " << argv[0] << "                Run in intra-node mode\n"
              << "  " << argv[0] << " <ip_port> <rank> <gpu_id>   Run in inter-node mode\n";
    return -1;
  }
}
