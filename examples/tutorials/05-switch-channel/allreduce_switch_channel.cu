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

void worker(int gpuId) {
  MSCCLPP_CUDATHROW(cudaSetDevice(gpuId));
  const int myRank = gpuId;
  const int remoteRank = myRank == 0 ? 1 : 0;
  const int nRanks = 2;
  const int iter = 1000;
  const mscclpp::Transport transport = mscclpp::Transport::CudaIpc;

  log("GPU ", gpuId, ": Preparing for tests ...");

  // Build a connection and a semaphore
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, nRanks);
  bootstrap->initialize("lo:127.0.0.1:" PORT_NUMBER);
  mscclpp::Communicator comm(bootstrap);

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
