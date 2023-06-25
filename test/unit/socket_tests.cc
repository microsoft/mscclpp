// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gtest/gtest.h>

#include <mscclpp/utils.hpp>
#include <thread>

#include "socket.h"

TEST(Socket, ListenAndConnect) {
  mscclpp::Timer timeout(3);

  std::string ipPortPair = "127.0.0.1:51512";
  mscclppSocketAddress listenAddr;

  ASSERT_NO_THROW(mscclppSocketGetAddrFromString(&listenAddr, ipPortPair.c_str()));

  mscclpp::Socket listenSock(&listenAddr);
  listenSock.listen();

  std::thread clientThread([&listenAddr]() {
    mscclpp::Socket sock(&listenAddr);
    sock.connect();
  });

  mscclpp::Socket sock;
  sock.accept(&listenSock);

  clientThread.join();
}
