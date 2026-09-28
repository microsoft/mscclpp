// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <memory>
#include <mscclpp/utils.hpp>
#include <thread>

#include "../framework.hpp"
#include "socket.h"
#include "utils_internal.hpp"

namespace {

struct ConnectedSockets {
  std::unique_ptr<mscclpp::Socket> sender;
  std::unique_ptr<mscclpp::Socket> receiver;
};

ConnectedSockets connectSockets() {
  mscclpp::SocketAddress listenAddr;
  mscclpp::SocketGetAddrFromString(&listenAddr, "127.0.0.1:0");

  mscclpp::Socket listenSocket(&listenAddr);
  listenSocket.bindAndListen();
  auto connectAddr = listenSocket.getAddr();

  ConnectedSockets sockets;
  sockets.sender = std::make_unique<mscclpp::Socket>(&connectAddr);
  sockets.receiver = std::make_unique<mscclpp::Socket>();

  std::thread connectThread([&]() { sockets.sender->connect(); });
  sockets.receiver->accept(&listenSocket);
  connectThread.join();
  return sockets;
}

bool waitUntilRecvSyscall(pid_t tid) {
  const std::string path = "/proc/self/task/" + std::to_string(tid) + "/syscall";
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream syscallFile(path);
    long syscallNumber = -1;
    if (syscallFile >> syscallNumber && (syscallNumber == SYS_recvfrom || syscallNumber == SYS_recvmsg)) return true;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

}  // namespace

TEST(Socket, ListenAndConnect) {
  auto sockets = connectSockets();
  ASSERT_EQ(sockets.sender->getState(), mscclpp::SocketStateReady);
  ASSERT_EQ(sockets.receiver->getState(), mscclpp::SocketStateReady);
}

TEST(Socket, RecvUntilEndRetriesUntilExactSize) {
  auto sockets = connectSockets();
  int flags = fcntl(sockets.receiver->getFd(), F_GETFL);
  ASSERT_NE(flags, -1);
  ASSERT_NE(fcntl(sockets.receiver->getFd(), F_SETFL, flags | O_NONBLOCK), -1);

  std::thread sender([&]() {
    char first = 'a';
    char second = 'b';
    sockets.sender->send(&first, sizeof(first));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    sockets.sender->send(&second, sizeof(second));
  });

  char bytes[2] = {};
  auto result = sockets.receiver->recvUntilEnd(bytes, sizeof(bytes));
  sender.join();
  ASSERT_TRUE(result == mscclpp::SocketRecvResult::Success);
  ASSERT_EQ(bytes[0], 'a');
  ASSERT_EQ(bytes[1], 'b');
}

TEST(Socket, RecvUntilEndDistinguishesBoundaryAndTruncatedEof) {
  {
    auto sockets = connectSockets();
    sockets.sender->close();
    char byte;
    ASSERT_TRUE(sockets.receiver->recvUntilEnd(&byte, sizeof(byte)) == mscclpp::SocketRecvResult::Closed);
  }
  {
    auto sockets = connectSockets();
    char byte = 'a';
    sockets.sender->send(&byte, sizeof(byte));
    sockets.sender->close();
    char bytes[2] = {};
    ASSERT_TRUE(sockets.receiver->recvUntilEnd(bytes, sizeof(bytes)) == mscclpp::SocketRecvResult::Truncated);
  }
}

TEST(Socket, BlockedReceiverShutdownJoinCloseIsBounded) {
  constexpr int Iterations = 50;
  constexpr auto MaxShutdownLatency = std::chrono::milliseconds(500);

  for (int i = 0; i < Iterations; ++i) {
    auto sockets = connectSockets();
    const int receiverFd = sockets.receiver->getFd();

    // This timeout bounds the test even if shutdown stops waking recv. Correct
    // shutdown should return far sooner than this fallback.
    struct timeval timeout = {1, 0};
    ASSERT_EQ(setsockopt(receiverFd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);

    std::atomic<bool> recvStarted{false};
    std::atomic<pid_t> receiverTid{0};
    int recvResult = -1;
    int recvError = 0;
    std::thread receiver([&]() {
      receiverTid.store(static_cast<pid_t>(syscall(SYS_gettid)), std::memory_order_release);
      recvStarted.store(true, std::memory_order_release);
      char byte;
      recvResult = ::recv(receiverFd, &byte, sizeof(byte), 0);
      if (recvResult < 0) recvError = errno;
    });

    while (!recvStarted.load(std::memory_order_acquire)) std::this_thread::yield();
    pid_t tid;
    while ((tid = receiverTid.load(std::memory_order_acquire)) == 0) std::this_thread::yield();
    const bool receiverBlocked = waitUntilRecvSyscall(tid);

    const auto shutdownStart = std::chrono::steady_clock::now();
    sockets.receiver->shutdown();
    sockets.sender->shutdown();
    sockets.receiver->shutdown();
    sockets.sender->shutdown();

    // shutdown must not release the descriptor before the receiver is joined.
    const int descriptorFlagsBeforeClose = ::fcntl(receiverFd, F_GETFD);

    receiver.join();
    const auto shutdownLatency =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - shutdownStart);
    sockets.receiver->close();
    errno = 0;
    const int descriptorFlagsAfterClose = ::fcntl(receiverFd, F_GETFD);
    const int descriptorErrorAfterClose = errno;
    sockets.sender->close();

    ASSERT_TRUE(receiverBlocked);
    const bool localShutdownResult =
        recvResult == 0 || (recvResult < 0 && (recvError == ENOTCONN || recvError == ESHUTDOWN));
    ASSERT_NE(descriptorFlagsBeforeClose, -1);
    ASSERT_EQ(descriptorFlagsAfterClose, -1);
    ASSERT_EQ(descriptorErrorAfterClose, EBADF);
    ASSERT_TRUE(localShutdownResult);
    ASSERT_LT(shutdownLatency.count(), MaxShutdownLatency.count());
  }
}
