/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_UTILS_INTERNAL_HPP_
#define MSCCLPP_UTILS_INTERNAL_HPP_

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mscclpp/utils.hpp>

namespace mscclpp {

// PCI Bus ID <-> int64 conversion functions
std::string int64ToBusId(int64_t id);
int64_t busIdToInt64(const std::string busId);

uint64_t getHash(const char* string, int n);
uint64_t getHostHash();
uint64_t getPidHash();
void getRandomData(void* buffer, size_t bytes);

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);

using TimePoint = std::chrono::steady_clock::time_point;
TimePoint getClock();
int64_t elapsedClock(TimePoint start, TimePoint end);

/* get any bytes of random data from /dev/urandom */
inline void getRandomData(void* buffer, size_t bytes) {
  if (bytes > 0) {
    const size_t one = 1UL;
    FILE* fp = fopen("/dev/urandom", "r");
    if (buffer == NULL || fp == NULL || fread(buffer, bytes, one, fp) != one) {
      throw Error("Failed to read random data", ErrorCode::SystemError);
    }
    if (fp) fclose(fp);
  }
}

}  // namespace mscclpp

#endif
