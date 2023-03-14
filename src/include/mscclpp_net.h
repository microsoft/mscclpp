/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
// TODO(saemal): this file is to be removed.
#ifndef MSCCLPP_NET_H_
#define MSCCLPP_NET_H_

#include "mscclpp.h"
#include <stdint.h>

#define MSCCLPP_NET_HANDLE_MAXSIZE 128

#define MSCCLPP_PTR_HOST 0x1
#define MSCCLPP_PTR_CUDA 0x2
#define MSCCLPP_PTR_DMABUF 0x4

// Maximum number of requests per comm object
#define MSCCLPP_NET_MAX_REQUESTS 8

typedef enum {MSCCLPP_LOG_NONE=0, MSCCLPP_LOG_VERSION=1, MSCCLPP_LOG_WARN=2, MSCCLPP_LOG_INFO=3, MSCCLPP_LOG_ABORT=4, MSCCLPP_LOG_TRACE=5} mscclppDebugLogLevel;
typedef enum {MSCCLPP_INIT=1, MSCCLPP_COLL=2, MSCCLPP_P2P=4, MSCCLPP_SHM=8, MSCCLPP_NET=16, MSCCLPP_GRAPH=32, MSCCLPP_TUNING=64, MSCCLPP_ENV=128, MSCCLPP_ALLOC=256, MSCCLPP_CALL=512, MSCCLPP_ALL=~0} mscclppDebugLogSubSys;

typedef void (*mscclppDebugLogger_t)(mscclppDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

typedef struct {
  char* name;     // Used mostly for logging.
  char* pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // [MSCCLPP_PTR_HOST|MSCCLPP_PTR_CUDA|MSCCLPP_PTR_DMABUF]
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  float latency;  // Network latency
  int maxComms;   // Maximum number of comms we can create
  int maxRecvs;   // Maximum number of grouped receives.
}mscclppNetProperties_v6_t;

typedef mscclppNetProperties_v6_t mscclppNetProperties_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  mscclppResult_t (*init)(mscclppDebugLogger_t logFunction);
  // Return the number of adapters.
  mscclppResult_t (*devices)(int* ndev);
  // Get various device properties.
  mscclppResult_t (*getProperties)(int dev, mscclppNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to MSCCLPP_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  mscclppResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  mscclppResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  mscclppResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either MSCCLPP_PTR_HOST or MSCCLPP_PTR_CUDA.
  mscclppResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  mscclppResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  mscclppResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  mscclppResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  mscclppResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with MSCCLPP_PTR_CUDA is
  // visible to the GPU
  mscclppResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  mscclppResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  mscclppResult_t (*closeSend)(void* sendComm);
  mscclppResult_t (*closeRecv)(void* recvComm);
  mscclppResult_t (*closeListen)(void* listenComm);
} mscclppNet_v6_t;

typedef mscclppNet_v6_t mscclppNet_t;

#define MSCCLPP_PLUGIN_SYMBOL mscclppNetPlugin_v6

typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  mscclppResult_t (*init)(mscclppDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  mscclppResult_t (*devices)(int* ndev);
  // Get various device properties.
  mscclppResult_t (*getProperties)(int dev, mscclppNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to MSCCLPP_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  mscclppResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  mscclppResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  mscclppResult_t (*reduceSupport)(mscclppDataType_t dataType, mscclppRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either MSCCLPP_PTR_HOST or MSCCLPP_PTR_CUDA.
  mscclppResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  mscclppResult_t (*regMrDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  mscclppResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  mscclppResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count,
      mscclppDataType_t dataType, mscclppRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with MSCCLPP_PTR_CUDA is
  // visible to the GPU
  mscclppResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  mscclppResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  mscclppResult_t (*closeColl)(void* collComm);
  mscclppResult_t (*closeListen)(void* listenComm);
} mscclppCollNet_v6_t;

typedef mscclppCollNet_v6_t mscclppCollNet_t;

#define MSCCLPP_COLLNET_PLUGIN_SYMBOL mscclppCollNetPlugin_v6

// v5 struct for backwards compatibility
typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  mscclppResult_t (*init)(mscclppDebugLogger_t logFunction);
  // Return the number of adapters.
  mscclppResult_t (*devices)(int* ndev);
  // Get various device properties.
  mscclppResult_t (*getProperties)(int dev, mscclppNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to MSCCLPP_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  mscclppResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  mscclppResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  mscclppResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either MSCCLPP_PTR_HOST or MSCCLPP_PTR_CUDA.
  mscclppResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  mscclppResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  mscclppResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  mscclppResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with MSCCLPP_PTR_CUDA is
  // visible to the GPU
  mscclppResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  mscclppResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  mscclppResult_t (*closeSend)(void* sendComm);
  mscclppResult_t (*closeRecv)(void* recvComm);
  mscclppResult_t (*closeListen)(void* listenComm);
} mscclppNet_v5_t;

// v5 struct for backwards compatibility
typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  mscclppResult_t (*init)(mscclppDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  mscclppResult_t (*devices)(int* ndev);
  // Get various device properties.
  mscclppResult_t (*getProperties)(int dev, mscclppNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to MSCCLPP_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  mscclppResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  mscclppResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  mscclppResult_t (*reduceSupport)(mscclppDataType_t dataType, mscclppRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either MSCCLPP_PTR_HOST or MSCCLPP_PTR_CUDA.
  mscclppResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  mscclppResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  mscclppResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count,
      mscclppDataType_t dataType, mscclppRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with MSCCLPP_PTR_CUDA is
  // visible to the GPU
  mscclppResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  mscclppResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  mscclppResult_t (*closeColl)(void* collComm);
  mscclppResult_t (*closeListen)(void* listenComm);
} mscclppCollNet_v5_t;

// v4 struct for backwards compatibility
typedef struct {
  char* name;     // Used mostly for logging.
  char* pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // MSCCLPP_PTR_HOST or MSCCLPP_PTR_HOST|MSCCLPP_PTR_CUDA
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  int maxComms;   // Maximum number of comms we can create
} mscclppNetProperties_v4_t;

// v4 struct for backwards compatibility
typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  mscclppResult_t (*init)(mscclppDebugLogger_t logFunction);
  // Return the number of adapters.
  mscclppResult_t (*devices)(int* ndev);
  // Get various device properties.
  mscclppResult_t (*getProperties)(int dev, mscclppNetProperties_v4_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to MSCCLPP_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  mscclppResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  mscclppResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connectHandle
  mscclppResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either MSCCLPP_PTR_HOST or MSCCLPP_PTR_CUDA.
  mscclppResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  mscclppResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  mscclppResult_t (*isend)(void* sendComm, void* data, int size, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  mscclppResult_t (*irecv)(void* recvComm, void* data, int size, void* mhandle, void** request);
  // Perform a flush/fence to make sure all data received with MSCCLPP_PTR_CUDA is
  // visible to the GPU
  mscclppResult_t (*iflush)(void* recvComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  mscclppResult_t (*test)(void* request, int* done, int* size);
  // Close and free send/recv comm objects
  mscclppResult_t (*closeSend)(void* sendComm);
  mscclppResult_t (*closeRecv)(void* recvComm);
  mscclppResult_t (*closeListen)(void* listenComm);
} mscclppNet_v4_t;

// v4 struct for backwards compatibility
typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  mscclppResult_t (*init)(mscclppDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  mscclppResult_t (*devices)(int* ndev);
  // Get various device properties.
  mscclppResult_t (*getProperties)(int dev, mscclppNetProperties_v4_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to MSCCLPP_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  mscclppResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  mscclppResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  mscclppResult_t (*reduceSupport)(mscclppDataType_t dataType, mscclppRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either MSCCLPP_PTR_HOST or MSCCLPP_PTR_CUDA.
  mscclppResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  mscclppResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  mscclppResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count,
      mscclppDataType_t dataType, mscclppRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with MSCCLPP_PTR_CUDA is
  // visible to the GPU
  mscclppResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  mscclppResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  mscclppResult_t (*closeColl)(void* collComm);
  mscclppResult_t (*closeListen)(void* listenComm);
} mscclppCollNet_v4_t;

#endif // end include guard
