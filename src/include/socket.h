/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_SOCKET_H_
#define MSCCLPP_SOCKET_H_

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stddef.h>
#include <sys/socket.h>

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT 1000  // connection retry sleep interval in usec
#define SOCKET_NAME_MAXLEN (NI_MAXHOST + NI_MAXSERV)
#define MSCCLPP_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union mscclppSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum mscclppSocketState {
  mscclppSocketStateNone = 0,
  mscclppSocketStateInitialized = 1,
  mscclppSocketStateAccepting = 2,
  mscclppSocketStateAccepted = 3,
  mscclppSocketStateConnecting = 4,
  mscclppSocketStateConnectPolling = 5,
  mscclppSocketStateConnected = 6,
  mscclppSocketStateReady = 7,
  mscclppSocketStateClosed = 8,
  mscclppSocketStateError = 9,
  mscclppSocketStateNum = 10
};

enum mscclppSocketType {
  mscclppSocketTypeUnknown = 0,
  mscclppSocketTypeBootstrap = 1,
  mscclppSocketTypeProxy = 2,
  mscclppSocketTypeNetSocket = 3,
  mscclppSocketTypeNetIb = 4
};

const char* mscclppSocketToString(union mscclppSocketAddress* addr, char* buf, const int numericHostForm = 1);
void mscclppSocketGetAddrFromString(union mscclppSocketAddress* ua, const char* ip_port_pair);
int mscclppFindInterfaceMatchSubnet(char* ifNames, union mscclppSocketAddress* localAddrs,
                                    union mscclppSocketAddress* remoteAddr, int ifNameMaxSize, int maxIfs);
int mscclppFindInterfaces(char* ifNames, union mscclppSocketAddress* ifAddrs, int ifNameMaxSize, int maxIfs);

namespace mscclpp {

class Socket {
 public:
  Socket(const mscclppSocketAddress* addr = nullptr, uint64_t magic = MSCCLPP_SOCKET_MAGIC,
         enum mscclppSocketType type = mscclppSocketTypeUnknown, volatile uint32_t* abortFlag = nullptr,
         int asyncFlag = 0);
  ~Socket();

  void listen();
  void connect(int64_t timeout = -1);
  void accept(const Socket* listenSocket, int64_t timeout = -1);
  void send(void* ptr, int size);
  void recv(void* ptr, int size);
  void close();

  int getFd() const { return fd_; }
  int getAcceptFd() const { return acceptFd_; }
  int getConnectRetries() const { return connectRetries_; }
  int getAcceptRetries() const { return acceptRetries_; }
  volatile uint32_t* getAbortFlag() const { return abortFlag_; }
  int getAsyncFlag() const { return asyncFlag_; }
  enum mscclppSocketState getState() const { return state_; }
  uint64_t getMagic() const { return magic_; }
  enum mscclppSocketType getType() const { return type_; }
  mscclppSocketAddress getAddr() const { return addr_; }
  int getSalen() const { return salen_; }

 private:
  void tryAccept();
  void finalizeAccept();
  void startConnect();
  void pollConnect();
  void finalizeConnect();
  void progressState();

  void socketProgressOpt(int op, void* ptr, int size, int* offset, int block, int* closed);
  void socketProgress(int op, void* ptr, int size, int* offset);
  void socketWait(int op, void* ptr, int size, int* offset);

  int fd_;
  int acceptFd_;
  int connectRetries_;
  int acceptRetries_;
  volatile uint32_t* abortFlag_;
  int asyncFlag_;
  enum mscclppSocketState state_;
  uint64_t magic_;
  enum mscclppSocketType type_;

  union mscclppSocketAddress addr_;
  int salen_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SOCKET_H_
