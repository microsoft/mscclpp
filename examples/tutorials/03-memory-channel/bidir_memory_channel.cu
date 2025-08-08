// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sys/wait.h>
#include <unistd.h>

#include <functional>
#include <iostream>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
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

__device__ mscclpp::DeviceSyncer devSyncer;

__global__ void bidirPutKernel(mscclpp::MemoryChannelDeviceHandle *devHandle, size_t copyBytes, int myRank) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    devHandle->relaxedSignal();
    devHandle->relaxedWait();
  }
  devSyncer.sync(gridDim.x);

  const uint64_t srcOffset = myRank * copyBytes;
  const uint64_t dstOffset = srcOffset;
  devHandle->put(dstOffset, srcOffset, copyBytes, /*threadId*/ tid, /*numThreads*/ blockDim.x * gridDim.x);
  devSyncer.sync(gridDim.x);
  if (tid == 0) {
    devHandle->signal();
    devHandle->wait();
  }
}

__global__ void bidirGetKernel(mscclpp::MemoryChannelDeviceHandle *devHandle, size_t copyBytes, int myRank) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    devHandle->relaxedSignal();
    devHandle->relaxedWait();
  }
  devSyncer.sync(gridDim.x);

  const int remoteRank = myRank ^ 1;
  const uint64_t srcOffset = remoteRank * copyBytes;
  const uint64_t dstOffset = srcOffset;
  devHandle->get(srcOffset, dstOffset, copyBytes, /*threadId*/ tid, /*numThreads*/ blockDim.x * gridDim.x);
}

__global__ void bidirPutPacketKernel(mscclpp::MemoryChannelDeviceHandle *devHandle, size_t copyBytes, int myRank,
                                     uint32_t flag) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    devHandle->relaxedSignal();
    devHandle->relaxedWait();
  }
  devSyncer.sync(gridDim.x);

  const uint64_t srcOffset = myRank * copyBytes;
  const uint64_t dstOffset = srcOffset;
  const uint64_t pktBufOffset = 0;
  devHandle->putPackets(pktBufOffset, srcOffset, copyBytes, tid, blockDim.x * gridDim.x, flag);
  devHandle->unpackPackets(pktBufOffset, dstOffset, copyBytes, tid, blockDim.x * gridDim.x, flag);
}

void worker(int gpuId) {
  MSCCLPP_CUDATHROW(cudaSetDevice(gpuId));
  const int myRank = gpuId;
  const int remoteRank = myRank == 0 ? 1 : 0;
  const int nRanks = 2;
  const int iter = 1000;
  const mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
  const size_t bufferBytes = 256 * 1024 * 1024;
  const size_t pktBufferBytes = 256 * 1024 * 1024;

  log("GPU ", gpuId, ": Preparing for tests ...");

  // Build a connection and a semaphore
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, nRanks);
  bootstrap->initialize("lo:127.0.0.1:" PORT_NUMBER);
  mscclpp::Communicator comm(bootstrap);
  auto conn = comm.connect({transport, {mscclpp::DeviceType::GPU, gpuId}}, remoteRank).get();
  auto sema = comm.buildSemaphore(conn, remoteRank).get();

  mscclpp::GpuBuffer buffer(bufferBytes);
  mscclpp::GpuBuffer pktBuffer(pktBufferBytes);
  mscclpp::RegisteredMemory localRegMem = comm.registerMemory(buffer.data(), buffer.bytes(), transport);
  mscclpp::RegisteredMemory localPktRegMem = comm.registerMemory(pktBuffer.data(), pktBuffer.bytes(), transport);

  comm.sendMemory(localRegMem, remoteRank);
  comm.sendMemory(localPktRegMem, remoteRank);
  auto remoteRegMemFuture = comm.recvMemory(remoteRank);
  auto remotePktRegMemFuture = comm.recvMemory(remoteRank);
  mscclpp::RegisteredMemory remoteRegMem = remoteRegMemFuture.get();
  mscclpp::RegisteredMemory remotePktRegMem = remotePktRegMemFuture.get();

  mscclpp::MemoryChannel memChan(sema, /*dst*/ remoteRegMem, /*src*/ localRegMem);
  mscclpp::MemoryChannel memPktChan(sema, /*dst*/ remotePktRegMem, /*src*/ localRegMem,
                                    /*packetBuffer*/ localPktRegMem.data());

  auto memChanHandle = memChan.deviceHandle();
  auto memPktChanHandle = memPktChan.deviceHandle();

  void *devHandle;
  void *devPktHandle;
  MSCCLPP_CUDATHROW(cudaMalloc(&devHandle, sizeof(memChanHandle)));
  MSCCLPP_CUDATHROW(cudaMalloc(&devPktHandle, sizeof(memPktChanHandle)));
  MSCCLPP_CUDATHROW(cudaMemcpy(devHandle, &memChanHandle, sizeof(memChanHandle), cudaMemcpyHostToDevice));
  MSCCLPP_CUDATHROW(cudaMemcpy(devPktHandle, &memPktChanHandle, sizeof(memPktChanHandle), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  std::function<void(size_t)> kernels[3];

  kernels[0] = [&](size_t copyBytes) {
    bidirPutKernel<<<32, 1024, 0, stream>>>(reinterpret_cast<mscclpp::MemoryChannelDeviceHandle *>(devHandle),
                                            copyBytes, myRank);
  };

  kernels[1] = [&](size_t copyBytes) {
    bidirGetKernel<<<32, 1024, 0, stream>>>(reinterpret_cast<mscclpp::MemoryChannelDeviceHandle *>(devHandle),
                                            copyBytes, myRank);
  };

  kernels[2] = [&](size_t copyBytes) {
    static uint32_t flag = 1;
    bidirPutPacketKernel<<<32, 1024, 0, stream>>>(reinterpret_cast<mscclpp::MemoryChannelDeviceHandle *>(devPktHandle),
                                                  copyBytes, myRank, flag++);
  };

  cudaEvent_t start, end;
  if (gpuId == 0) {
    MSCCLPP_CUDATHROW(cudaEventCreate(&start));
    MSCCLPP_CUDATHROW(cudaEventCreate(&end));
  }
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  bootstrap->barrier();

  for (int kernelId = 0; kernelId < 3; ++kernelId) {
    const std::string testName = (kernelId == 0) ? "Bidir Put" : (kernelId == 1) ? "Bidir Get" : "Bidir Put Packets";
    for (size_t copyBytes : {1024, 1024 * 1024, 128 * 1024 * 1024}) {
      cudaGraph_t graph;
      cudaGraphExec_t graphExec;

      MSCCLPP_CUDATHROW(cudaGraphCreate(&graph, 0));
      MSCCLPP_CUDATHROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

      for (int i = 0; i < iter; ++i) {
        kernels[kernelId](copyBytes);
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
        log("GPU ", gpuId, ": [", testName, "] bytes ", copyBytes, ", elapsed ", elapsedTimePerIter, " ms/iter, BW ",
            gbps, " GB/s");
      }
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      MSCCLPP_CUDATHROW(cudaGraphExecDestroy(graphExec));
      MSCCLPP_CUDATHROW(cudaGraphDestroy(graph));
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
