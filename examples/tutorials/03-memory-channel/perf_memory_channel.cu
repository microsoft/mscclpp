// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <iostream>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/concurrency_device.hpp>
#include <sys/wait.h>
#include <sstream>

#define PORT_NUMER "50505"

__device__ mscclpp::DeviceSyncer devSyncer;

__global__ void bidirCopyKernel(mscclpp::MemoryChannelDeviceHandle *devHandle, size_t copyBytes, int myRank) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    devHandle->relaxedSignal();
    devHandle->relaxedWait();
  }
  devSyncer.sync(gridDim.x);

  uint64_t offset = myRank * copyBytes;
  devHandle->put(offset, copyBytes, tid, blockDim.x * gridDim.x);
  devSyncer.sync(gridDim.x);
  if (tid == 0) {
    devHandle->signal();
    devHandle->wait();
  }
}

template <typename ...Args>
void log(int gpuId, Args&&... args) {
  std::stringstream ss;
  ss << "GPU " << gpuId << ": ";
  (ss << ... << args);
  ss << std::endl;
  std::cout << ss.str();
}

void worker(int gpuId) {
  MSCCLPP_CUDATHROW(cudaSetDevice(gpuId));
  const int myRank = gpuId;
  const int remoteRank = myRank == 0 ? 1 : 0;
  const int nRanks = 2;
  const int iter = 1000;
  const mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
  const size_t bufferBytes = 256 * 1024 * 1024;
  const size_t pktBufferBytes = 512 * 1024 * 1024;

  log(gpuId, "Preparing for bidirectional copy tests ...");

  // Build a connection and a semaphore
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, nRanks);
  bootstrap->initialize("lo:127.0.0.1:" PORT_NUMER);
  mscclpp::Communicator comm(bootstrap);
  auto conn = comm.connect({transport, {mscclpp::DeviceType::GPU, gpuId}}, remoteRank).get();
  auto sema = comm.buildSemaphore(conn, remoteRank).get();

  mscclpp::GpuBuffer buffer(bufferBytes);
  mscclpp::RegisteredMemory localRegMem = comm.registerMemory(buffer.data(), buffer.bytes(), transport);

  comm.sendMemory(localRegMem, remoteRank);
  auto remoteRegMemFuture = comm.recvMemory(remoteRank);
  mscclpp::RegisteredMemory remoteRegMem = remoteRegMemFuture.get();

  mscclpp::GpuBuffer pktBuffer(pktBufferBytes);
  mscclpp::MemoryChannel memChan(sema, remoteRegMem, localRegMem, pktBuffer.data());
  auto memChanHandle = memChan.deviceHandle();
  void *devHandle;
  MSCCLPP_CUDATHROW(cudaMalloc(&devHandle, sizeof(memChanHandle)));
  MSCCLPP_CUDATHROW(cudaMemcpy(devHandle, &memChanHandle, sizeof(memChanHandle), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaEvent_t start, end;
  if (gpuId == 0) {
    MSCCLPP_CUDATHROW(cudaEventCreate(&start));
    MSCCLPP_CUDATHROW(cudaEventCreate(&end));
  }
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  bootstrap->barrier();

  for (size_t copyBytes : {1024, 1024 * 1024, 128 * 1024 * 1024}) {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    MSCCLPP_CUDATHROW(cudaGraphCreate(&graph, 0));
    MSCCLPP_CUDATHROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    for (int i = 0; i < iter; ++i) {
      bidirCopyKernel<<<32, 1024, 0, stream>>>(reinterpret_cast<mscclpp::MemoryChannelDeviceHandle *>(devHandle), copyBytes, myRank);
    }

    MSCCLPP_CUDATHROW(cudaStreamEndCapture(stream, &graph));
    MSCCLPP_CUDATHROW(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Synchronize before timing
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    bootstrap->barrier();
    
    if (gpuId == 0) {
      MSCCLPP_CUDATHROW(cudaEventRecord(start, stream));
    }

    MSCCLPP_CUDATHROW(cudaGraphLaunch(graphExec, stream));

    if (gpuId == 0) {
      MSCCLPP_CUDATHROW(cudaEventRecord(end, stream));
      MSCCLPP_CUDATHROW(cudaEventSynchronize(end));
      float elapsedTime;
      float elapsedTimePerIter;
      float gbps;
      MSCCLPP_CUDATHROW(cudaEventElapsedTime(&elapsedTime, start, end));
      elapsedTimePerIter = elapsedTime / iter;
      gbps = float(copyBytes) / elapsedTimePerIter * 1e-6f;
      log(gpuId, "bytes ", copyBytes, ", elapsed ", elapsedTimePerIter, " ms/iter, BW ", gbps, " GB/s");
    }
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    MSCCLPP_CUDATHROW(cudaGraphExecDestroy(graphExec));
    MSCCLPP_CUDATHROW(cudaGraphDestroy(graph));
  }

  bootstrap->barrier();
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

int main() {
  int pid0 = spawn_process([]() { worker(0); });
  int pid1 = spawn_process([]() { worker(1); });
  if (pid0 < 0 || pid1 < 0) {
    std::cout << "Failed to spawn processes." << std::endl;
    return -1;
  }
  int status0 = wait_process(pid0);
  int status1 = wait_process(pid1);
  if (status0 < 0 || status1 < 0) {
    std::cout << "Failed to wait for processes." << std::endl;
    return -1;
  }
  if (status0 != 0 || status1 != 0) {
    std::cout << "One of the processes failed." << std::endl;
    return -1;
  }
  std::cout << "Succeed!" << std::endl;
  return 0;
}
