// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_UNIX_SOCKET_HPP_
#define MSCCLPP_UNIX_SOCKET_HPP_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace mscclpp {

class UnixSocketServer {
 public:
  static UnixSocketServer& instance();
  static std::string generateSocketPath(int socketId);

  void start();
  void stop();
  uint32_t registerFd(int fd);
  void unregisterFd(uint32_t fdId);
  std::string getSocketPath() const;

 private:
  int listenUnixSockFd_ = -1;
  std::string listenUnixSockPath_;
  std::thread mainThread_;
  std::unique_ptr<uint32_t> abortFlagStorage_;
  volatile uint32_t* abortFlag_;
  std::mutex mutex_;
  std::unordered_map<uint32_t, int> fdMap_;

  UnixSocketServer();
  void mainLoop(int listenUnixSockFd);
};

class UnixSocketClient {
 public:
  static UnixSocketClient& instance();

  int requestFd(const std::string& socketPath, uint32_t fdId);
  ~UnixSocketClient();

 private:
  std::unordered_map<std::string, int> cachedFds_;
  std::mutex mutex_;

  UnixSocketClient() = default;
  int requestFdInternal(int connFd, uint32_t fdId);
};

}  // namespace mscclpp

#endif  // MSCCLPP_UNIX_SOCKET_HPP_
