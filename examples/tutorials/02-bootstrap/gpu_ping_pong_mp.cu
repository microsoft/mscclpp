// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sys/wait.h>
#include <unistd.h>

#include <iostream>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <sstream>

#define PORT_NUMBER "50505"

template <typename... Args>
void log(Args&&... args) {
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

__device__ void spin_cycles(unsigned long long cycles) {
  unsigned long long start = clock64();
  while (clock64() - start < cycles) {
    // spin
  }
}

__global__ void gpuKernel0(mscclpp::BaseMemoryChannelDeviceHandle* devHandle, int iter) {
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    for (int i = 0; i < iter; ++i) {
      devHandle->relaxedWait();
      // spin for a few ms
      spin_cycles(1e7);
      devHandle->relaxedSignal();
    }
  }
}

__global__ void gpuKernel1(mscclpp::BaseMemoryChannelDeviceHandle* devHandle, int iter) {
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    for (int i = 0; i < iter; ++i) {
      devHandle->relaxedSignal();
      devHandle->relaxedWait();
    }
  }
}

void worker(int gpuId) {
  // Optional: check if we have at least two GPUs
  int deviceCount;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    log("Error: At least two GPUs are required.");
    std::exit(1);
  }

  // Optional: check if the two GPUs can peer-to-peer access each other
  int canAccessPeer;
  MSCCLPP_CUDATHROW(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));
  if (!canAccessPeer) {
    log("Error: GPU 0 cannot access GPU 1. Make sure that the GPUs are connected peer-to-peer. You can check this "
        "by running `nvidia-smi topo -m` (the connection between GPU 0 and 1 should be either NV# or PIX).");
    std::exit(1);
  }

  MSCCLPP_CUDATHROW(cudaSetDevice(gpuId));
  const int myRank = gpuId;
  const int remoteRank = myRank == 0 ? 1 : 0;
  const int nRanks = 2;
  const int iter = 100;
  const mscclpp::Transport transport = mscclpp::Transport::CudaIpc;

  log("GPU ", gpuId, ": Initializing a bootstrap ...");

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, nRanks);
  bootstrap->initialize("lo:127.0.0.1:" PORT_NUMBER);
  mscclpp::Communicator comm(bootstrap);

  log("GPU ", gpuId, ": Creating a connection ...");

  auto connFuture = comm.connect({transport, {mscclpp::DeviceType::GPU, gpuId}}, remoteRank);
  auto conn = connFuture.get();

  log("GPU ", gpuId, ": Creating a semaphore ...");

  auto semaFuture = comm.buildSemaphore(conn, remoteRank);
  auto sema = semaFuture.get();

  log("GPU ", gpuId, ": Creating a channel ...");

  mscclpp::BaseMemoryChannel memChan(sema);
  auto memChanHandle = memChan.deviceHandle();
  void* devHandle;
  MSCCLPP_CUDATHROW(cudaMalloc(&devHandle, sizeof(memChanHandle)));
  MSCCLPP_CUDATHROW(cudaMemcpy(devHandle, &memChanHandle, sizeof(memChanHandle), cudaMemcpyHostToDevice));

  log("GPU ", gpuId, ": Launching a GPU kernel ...");

  if (gpuId == 0) {
    gpuKernel0<<<1, 1>>>(reinterpret_cast<mscclpp::BaseMemoryChannelDeviceHandle*>(devHandle), iter);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  } else {
    cudaEvent_t start, end;
    MSCCLPP_CUDATHROW(cudaEventCreate(&start));
    MSCCLPP_CUDATHROW(cudaEventCreate(&end));
    MSCCLPP_CUDATHROW(cudaEventRecord(start));
    gpuKernel1<<<1, 1>>>(reinterpret_cast<mscclpp::BaseMemoryChannelDeviceHandle*>(devHandle), iter);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    MSCCLPP_CUDATHROW(cudaEventRecord(end));
    MSCCLPP_CUDATHROW(cudaEventSynchronize(end));

    float elapsedMs;
    MSCCLPP_CUDATHROW(cudaEventElapsedTime(&elapsedMs, start, end));

    float msPerIter = elapsedMs / iter;
    log("Elapsed ", msPerIter, " ms per iteration (", iter, ")");
    if (msPerIter < 1.0f) {
      log("Failed: the elapsed time per iteration is less than 1 ms, which may indicate that the relaxedSignal "
          "and relaxedWait are not working as expected.");
    }
  }

  bootstrap->barrier();
}

int main() {
  int pid0 = spawn_process([]() { worker(0); });
  int pid1 = spawn_process([]() { worker(1); });
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
}
