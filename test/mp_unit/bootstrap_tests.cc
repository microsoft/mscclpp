// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <fcntl.h>
#include <mpi.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <future>
#include <mscclpp/errors.hpp>

#include "mp_unit_tests.hpp"
#include "socket.h"
#include "unix_socket.hpp"

void BootstrapTest::bootstrapTestAllGather(std::shared_ptr<mscclpp::Bootstrap> bootstrap) {
  std::vector<int> tmp(bootstrap->getNranks(), 0);
  tmp[bootstrap->getRank()] = bootstrap->getRank() + 1;
  bootstrap->allGather(tmp.data(), sizeof(int));
  for (int i = 0; i < bootstrap->getNranks(); ++i) {
    EXPECT_EQ(tmp[i], i + 1);
  }
}

void BootstrapTest::bootstrapTestBarrier(std::shared_ptr<mscclpp::Bootstrap> bootstrap) { bootstrap->barrier(); }

void BootstrapTest::bootstrapTestSendRecv(std::shared_ptr<mscclpp::Bootstrap> bootstrap) {
  for (int i = 0; i < bootstrap->getNranks(); i++) {
    if (bootstrap->getRank() == i) continue;
    int msg1 = (bootstrap->getRank() + 1) * 3;
    int msg2 = (bootstrap->getRank() + 1) * 3 + 1;
    int msg3 = (bootstrap->getRank() + 1) * 3 + 2;
    bootstrap->send(&msg1, sizeof(int), i, 0);
    bootstrap->send(&msg2, sizeof(int), i, 1);
    bootstrap->send(&msg3, sizeof(int), i, 2);
  }

  for (int i = 0; i < bootstrap->getNranks(); i++) {
    if (bootstrap->getRank() == i) continue;
    int msg1 = 0;
    int msg2 = 0;
    int msg3 = 0;
    // recv them in the opposite order to check correctness
    bootstrap->recv(&msg2, sizeof(int), i, 1);
    bootstrap->recv(&msg3, sizeof(int), i, 2);
    bootstrap->recv(&msg1, sizeof(int), i, 0);
    EXPECT_EQ(msg1, (i + 1) * 3);
    EXPECT_EQ(msg2, (i + 1) * 3 + 1);
    EXPECT_EQ(msg3, (i + 1) * 3 + 2);
  }
}

void BootstrapTest::bootstrapTestIpcDomain(std::shared_ptr<mscclpp::Bootstrap> bootstrap) {
  int nRanksPerIpcDomain = bootstrap->getNranksPerIpcDomain();
  EXPECT_GT(nRanksPerIpcDomain, 0);
  EXPECT_LE(nRanksPerIpcDomain, bootstrap->getNranks());
}

void BootstrapTest::bootstrapTestAll(std::shared_ptr<mscclpp::Bootstrap> bootstrap) {
  bootstrapTestAllGather(bootstrap);
  bootstrapTestBarrier(bootstrap);
  bootstrapTestSendRecv(bootstrap);
  bootstrapTestIpcDomain(bootstrap);
}

static void checkConcurrentFdRequests() {
  auto& server = mscclpp::UnixSocketServer::instance();
  auto& client = mscclpp::UnixSocketClient::instance();
  const auto path = server.getSocketPath();
  int ends[2];
  if (pipe(ends) != 0) throw mscclpp::SysError("pipe() failed", errno);
  for (int fd : ends) server.registerFd(fd);
  auto cleanup = [&] {
    for (int fd : ends) {
      server.unregisterFd(fd);
      close(fd);
    }
  };
  try {
    close(client.requestFd(path, ends[0]));
    std::vector<std::future<void>> requests;
    for (int i = 0; i < 4; ++i) {
      requests.emplace_back(std::async(std::launch::async, [&, end = i % 2] {
        for (int round = 0; round < 32; ++round) {
          int fd = client.requestFd(path, ends[end]);
          int flags = fcntl(fd, F_GETFL);
          close(fd);
          if (flags < 0 || (flags & O_ACCMODE) != (end == 0 ? O_RDONLY : O_WRONLY)) {
            throw std::runtime_error("received the wrong pipe end");
          }
        }
      }));
    }
    for (auto& request : requests) request.get();
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
}

TEST(BootstrapTest, WithId) {
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  bootstrapTestAll(bootstrap);
  checkConcurrentFdRequests();
}

TEST(BootstrapTest, WithIpPortPair) {
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  bootstrap->initialize(gEnv->args["ip_port"]);
  bootstrapTestAll(bootstrap);
}

TEST(BootstrapTest, ResumeWithId) {
  // This test may take a few minutes.
  bootstrapTestTimer.set(300);

  for (int i = 0; i < 10; ++i) {
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
    mscclpp::UniqueId id;
    if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    bootstrap->initialize(id, 300);
  }
}

TEST(BootstrapTest, ResumeWithIpPortPair) {
  for (int i = 0; i < 5; ++i) {
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
    bootstrap->initialize(gEnv->args["ip_port"]);
  }
}

TEST(BootstrapTest, ExitBeforeConnect) {
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  bootstrap->createUniqueId();
}

TEST(BootstrapTest, TimeoutWithId) {
  mscclpp::Timer timer;

  // All ranks initialize a bootstrap with their own id (will hang)
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  mscclpp::UniqueId id = bootstrap->createUniqueId();

  try {
    // Set bootstrap timeout to 1 second
    bootstrap->initialize(id, 1);
  } catch (const mscclpp::Error& e) {
    ASSERT_EQ(e.getErrorCode(), mscclpp::ErrorCode::Timeout);
  }

  // Timeout should be sligtly greater than 1 second
  ASSERT_GT(timer.elapsed(), 1000000);
  ASSERT_LT(timer.elapsed(), 1100000);
}

TEST(BootstrapTest, RejectInvalidRemoteAddresses) {
  struct UniqueIdInternal {
    uint64_t magic;
    mscclpp::SocketAddress addr;
  };
  struct ExtInfo {
    int rank;
    int nRanks;
    mscclpp::SocketAddress extAddressListenRoot;
    mscclpp::SocketAddress extAddressListen;
  };

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (gEnv->rank == 1) {
    UniqueIdInternal internalId{};
    std::memcpy(&internalId, id.data(), sizeof(internalId));
    volatile uint32_t abortFlag = 0;

    auto sendRawExtInfo = [&](int rank, int nRanks) {
      mscclpp::Socket sock(&internalId.addr, internalId.magic, mscclpp::SocketTypeBootstrap, &abortFlag);
      sock.connect();
      ExtInfo info{};
      info.rank = rank;
      info.nRanks = nRanks;
      std::memset(&info.extAddressListenRoot, 0x41, sizeof(info.extAddressListenRoot));
      std::memset(&info.extAddressListen, 0x41, sizeof(info.extAddressListen));
      int size = sizeof(info);
      sock.send(&size, sizeof(size));
      sock.send(&info, sizeof(info));
    };

    // Wrong socket type
    {
      mscclpp::Socket sock(&internalId.addr, internalId.magic, mscclpp::SocketTypeProxy, &abortFlag);
      sock.connect();
    }
    // Peer resets the connection before sending metadata
    {
      mscclpp::Socket sock(&internalId.addr, internalId.magic, mscclpp::SocketTypeBootstrap, &abortFlag);
      sock.connect();
      linger resetOnClose{1, 0};
      EXPECT_EQ(::setsockopt(sock.getFd(), SOL_SOCKET, SO_LINGER, &resetOnClose, sizeof(resetOnClose)), 0);
    }
    // Oversized, negative, and undersized messages
    {
      mscclpp::Socket sock(&internalId.addr, internalId.magic, mscclpp::SocketTypeBootstrap, &abortFlag);
      sock.connect();
      int badSize = sizeof(ExtInfo) + 64;
      sock.send(&badSize, sizeof(badSize));
    }
    {
      mscclpp::Socket sock(&internalId.addr, internalId.magic, mscclpp::SocketTypeBootstrap, &abortFlag);
      sock.connect();
      int badSize = -1;
      sock.send(&badSize, sizeof(badSize));
    }
    {
      mscclpp::Socket sock(&internalId.addr, internalId.magic, mscclpp::SocketTypeBootstrap, &abortFlag);
      sock.connect();
      struct {
        int size;
        int rank;
        int nRanks;
      } badFrame{sizeof(int) * 2, 1, gEnv->worldSize};
      sock.send(&badFrame, sizeof(badFrame));
    }
    // Mismatched nRanks
    sendRawExtInfo(1, gEnv->worldSize + 1);
    // Out-of-range ranks (negative, boundary, near-OOB, far-OOB)
    sendRawExtInfo(-1, gEnv->worldSize);
    sendRawExtInfo(gEnv->worldSize, gEnv->worldSize);
    sendRawExtInfo(gEnv->worldSize + 2, gEnv->worldSize);
    sendRawExtInfo(1000000, gEnv->worldSize);
  }

  bootstrap->initialize(id);
  bootstrapTestAll(bootstrap);
}

class MPIBootstrap : public mscclpp::Bootstrap {
 public:
  MPIBootstrap() : Bootstrap() {}
  int getRank() const override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }
  int getNranks() const override {
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    return worldSize;
  }
  int getNranksPerNode() const override {
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    int shmrank;
    MPI_Comm_size(shmcomm, &shmrank);
    return shmrank;
  }
  int getNranksPerIpcDomain() const override { return getNranksPerNode(); }
  void allGather(void* sendbuf, int size) override {
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, sendbuf, size, MPI_BYTE, MPI_COMM_WORLD);
  }
  void barrier() override { MPI_Barrier(MPI_COMM_WORLD); }
  void send(void* sendbuf, int size, int dest, int tag) override {
    MPI_Send(sendbuf, size, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
  }
  void recv(void* recvbuf, int size, int source, int tag) override {
    MPI_Recv(recvbuf, size, MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
};

TEST(BootstrapTest, MPIBootstrap) {
  auto bootstrap = std::make_shared<MPIBootstrap>();
  bootstrapTestAll(bootstrap);
}
