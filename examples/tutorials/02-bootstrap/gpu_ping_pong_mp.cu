// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <iostream>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <sys/wait.h>
#include <sstream>

#define PORT_NUMER "50505"

__global__ void gpuKernel0(mscclpp::BaseMemoryChannelDeviceHandle *devHandle, int iter) {
  if (threadIdx.x + blockIdx.x * gridDim.x == 0) {
    for (int i = 0; i < iter; ++i) {
      devHandle->relaxedWait();
      // sleep (roughly) 1ms
      __nanosleep(1e6);
      devHandle->relaxedSignal();
    }
  }
}

__global__ void gpuKernel1(mscclpp::BaseMemoryChannelDeviceHandle *devHandle, int iter) {
  if (threadIdx.x + blockIdx.x * gridDim.x == 0) {
    for (int i = 0; i < iter; ++i) {
      devHandle->relaxedSignal();
      devHandle->relaxedWait();
    }
  }
}

void log(int gpuId, const std::string& msg) {
  std::stringstream ss;
  ss << "GPU " << gpuId << ": " << msg << std::endl;
  std::cout << ss.str();
}

void worker(int gpuId) {
  // Optional: check if we have at least two GPUs
  int deviceCount;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    std::cout << "Error: At least two GPUs are required." << std::endl;
    return;
  }

  // Optional: check if the two GPUs can peer-to-peer access each other
  int canAccessPeer;
  MSCCLPP_CUDATHROW(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));
  if (!canAccessPeer) {
    std::cout
        << "Error: GPU 0 cannot access GPU 1. Make sure that the GPUs are connected peer-to-peer. You can check this "
           "by running `nvidia-smi topo -m` (the connection between GPU 0 and 1 should be either NV# or PIX)."
        << std::endl;
    return;
  }

  MSCCLPP_CUDATHROW(cudaSetDevice(gpuId));
  const int myRank = gpuId;
  const int remoteRank = myRank == 0 ? 1 : 0;
  const int nRanks = 2;
  const int iter = 100;
  const mscclpp::Transport transport = mscclpp::Transport::CudaIpc;

  log(gpuId, "Initializing a bootstrap ...");

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, nRanks);
  bootstrap->initialize("lo:127.0.0.1:" PORT_NUMER);
  mscclpp::Communicator comm(bootstrap);

  log(gpuId, "Creating a connection ...");

  auto connFuture = comm.connect({transport, {mscclpp::DeviceType::GPU, gpuId}}, remoteRank);
  auto conn = connFuture.get();

  log(gpuId, "Creating a semaphore ...");

  auto semaFuture = comm.buildSemaphore(conn, remoteRank);
  auto sema = semaFuture.get();

  log(gpuId, "Creating a channel ...");

  mscclpp::BaseMemoryChannel memChan(sema);
  auto memChanHandle = memChan.deviceHandle();
  void *devHandle;
  MSCCLPP_CUDATHROW(cudaMalloc(&devHandle, sizeof(memChanHandle)));
  MSCCLPP_CUDATHROW(cudaMemcpy(devHandle, &memChanHandle, sizeof(memChanHandle), cudaMemcpyHostToDevice));

  log(gpuId, "Launching a GPU kernel ...");

  if (gpuId == 0) {
    gpuKernel0<<<1, 1>>>(reinterpret_cast<mscclpp::BaseMemoryChannelDeviceHandle *>(devHandle), iter);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  } else {
    cudaEvent_t start, end;
    MSCCLPP_CUDATHROW(cudaEventCreate(&start));
    MSCCLPP_CUDATHROW(cudaEventCreate(&end));
    MSCCLPP_CUDATHROW(cudaEventRecord(start));
    gpuKernel1<<<1, 1>>>(reinterpret_cast<mscclpp::BaseMemoryChannelDeviceHandle *>(devHandle), iter);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    MSCCLPP_CUDATHROW(cudaEventRecord(end));
    MSCCLPP_CUDATHROW(cudaEventSynchronize(end));

    float elapsedMs;
    MSCCLPP_CUDATHROW(cudaEventElapsedTime(&elapsedMs, start, end));

    float msPerIter = elapsedMs / iter;
    std::cout << "Elapsed " << msPerIter << " ms per iteration (" << iter << ")" << std::endl;
    if (msPerIter < 1.0f) {
      std::cout << "Failed: the elapsed time per iteration is less than 1 ms, which may indicate that the relaxedSignal "
                   "and relaxedWait are not working as expected."
                << std::endl;
    }
  }

  bootstrap->barrier();
}

int main() {
  pid_t pid = fork();
  if (pid < 0) {
    std::cout << "Error: fork() failed." << std::endl;
    return 1;
  } else if (pid == 0) {
    // Child process: use GPU 1
    worker(1);
    exit(0);
  } else {
    // Parent process: use GPU 0
    worker(0);
    // Wait for child to finish
    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
      std::cout << "Child process exited with error code: " << WEXITSTATUS(status) << std::endl;
      return 1;
    }
  }
  std::cout << "Succeed!" << std::endl;
  return 0;
}
