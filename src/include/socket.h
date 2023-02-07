/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_SOCKET_H_
#define MSCCLPP_SOCKET_H_

#include "mscclpp.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT            1000 // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES   2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES    3 // connection timed out retry times (each one can take 20s)
#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)
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

struct mscclppSocket {
  int fd;
  int acceptFd;
  int timedOutRetries;
  int refusedRetries;
  union mscclppSocketAddress addr;
  volatile uint32_t* abortFlag;
  int asyncFlag;
  enum mscclppSocketState state;
  int salen;
  uint64_t magic;
  enum mscclppSocketType type;
};

const char *mscclppSocketToString(union mscclppSocketAddress *addr, char *buf, const int numericHostForm = 1);
mscclppResult_t mscclppSocketGetAddrFromString(union mscclppSocketAddress* ua, const char* ip_port_pair);
int mscclppFindInterfaceMatchSubnet(char* ifNames, union mscclppSocketAddress* localAddrs, union mscclppSocketAddress* remoteAddr, int ifNameMaxSize, int maxIfs);
int mscclppFindInterfaces(char* ifNames, union mscclppSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs);

// Initialize a socket
mscclppResult_t mscclppSocketInit(struct mscclppSocket* sock, union mscclppSocketAddress* addr = NULL, uint64_t magic = MSCCLPP_SOCKET_MAGIC, enum mscclppSocketType type = mscclppSocketTypeUnknown, volatile uint32_t* abortFlag = NULL, int asyncFlag = 0);
// Create a listening socket. sock->addr can be pre-filled with IP & port info. sock->fd is set after a successful call
mscclppResult_t mscclppSocketListen(struct mscclppSocket* sock);
mscclppResult_t mscclppSocketGetAddr(struct mscclppSocket* sock, union mscclppSocketAddress* addr);
// Connect to sock->addr. sock->fd is set after a successful call.
mscclppResult_t mscclppSocketConnect(struct mscclppSocket* sock);
// Return socket connection state.
// mscclppResult_t mscclppSocketReady(struct mscclppSocket* sock, int *running);
// Accept an incoming connection from listenSock->fd and keep the file descriptor in sock->fd, with the remote side IP/port in sock->addr.
mscclppResult_t mscclppSocketAccept(struct mscclppSocket* sock, struct mscclppSocket* ulistenSock);
// mscclppResult_t mscclppSocketGetFd(struct mscclppSocket* sock, int* fd);
// mscclppResult_t mscclppSocketSetFd(int fd, struct mscclppSocket* sock);

#define MSCCLPP_SOCKET_SEND 0
#define MSCCLPP_SOCKET_RECV 1

mscclppResult_t mscclppSocketProgress(int op, struct mscclppSocket* sock, void* ptr, int size, int* offset);
// mscclppResult_t mscclppSocketWait(int op, struct mscclppSocket* sock, void* ptr, int size, int* offset);
mscclppResult_t mscclppSocketSend(struct mscclppSocket* sock, void* ptr, int size);
mscclppResult_t mscclppSocketRecv(struct mscclppSocket* sock, void* ptr, int size);
// mscclppResult_t mscclppSocketTryRecv(struct mscclppSocket* sock, void* ptr, int size, int* closed);
mscclppResult_t mscclppSocketClose(struct mscclppSocket* sock);
#endif
