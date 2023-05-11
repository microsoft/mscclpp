/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_UTILS_H_
#define MSCCLPP_UTILS_H_

#include <stdint.h>

#include <chrono>

#include "alloc.h"
#include "mscclpp.h"

// int mscclppCudaCompCap();

// PCI Bus ID <-> int64 conversion functions
mscclppResult_t int64ToBusId(int64_t id, char* busId);
mscclppResult_t busIdToInt64(const char* busId, int64_t* id);

mscclppResult_t getBusId(int cudaDev, int64_t* busId);
mscclppResult_t getDeviceNumaNode(int cudaDev, int* numaNode);

mscclppResult_t getHostName(char* hostname, int maxlen, const char delim);
uint64_t getHash(const char* string, int n);
uint64_t getHostHash();
uint64_t getPidHash();
mscclppResult_t getRandomData(void* buffer, size_t bytes);

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);

static long log2i(long n) {
  long l = 0;
  while (n >>= 1) l++;
  return l;
}

typedef std::chrono::steady_clock::time_point mscclppTime_t;
mscclppTime_t getClock();
int64_t elapsedClock(mscclppTime_t start, mscclppTime_t end);

/* get any bytes of random data from /dev/urandom, return 0 if it succeeds; else
 * return -1 */
inline mscclppResult_t getRandomData(void* buffer, size_t bytes) {
  mscclppResult_t ret = mscclppSuccess;
  if (bytes > 0) {
    const size_t one = 1UL;
    FILE* fp = fopen("/dev/urandom", "r");
    if (buffer == NULL || fp == NULL || fread(buffer, bytes, one, fp) != one) ret = mscclppSystemError;
    if (fp) fclose(fp);
  }
  return ret;
}

mscclppResult_t numaBind(int node);

typedef struct bitmask* mscclppNumaState;

mscclppResult_t getNumaState(mscclppNumaState* state);

mscclppResult_t setNumaState(mscclppNumaState state);

#endif
