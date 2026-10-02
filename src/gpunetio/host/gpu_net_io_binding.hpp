// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_BINDING_HPP_
#define MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_BINDING_HPP_

#include <cstdio>
#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>
#include <stdexcept>
#include <vector>

namespace mscclpp::detail {

struct GpuNetIoSetupStatus {
  int code;
  char message[256];
};

template <class Operation>
void runGpuNetIoSetupPhase(Bootstrap& bootstrap, std::vector<GpuNetIoSetupStatus>& statuses, const char* phase,
                           Operation operation) {
  auto& local = statuses[bootstrap.getRank()];
  local = {};
  try {
    operation();
  } catch (const Error& error) {
    local.code = error.getErrorCode() == ErrorCode::InvalidUsage ? 1 : 2;
    std::snprintf(local.message, sizeof(local.message), "%s", error.what());
  } catch (const std::invalid_argument& error) {
    local.code = 1;
    std::snprintf(local.message, sizeof(local.message), "%s", error.what());
  } catch (const std::exception& error) {
    local.code = 2;
    std::snprintf(local.message, sizeof(local.message), "%s", error.what());
  } catch (...) {
    local.code = 2;
    std::snprintf(local.message, sizeof(local.message), "%s", "Unknown setup exception");
  }
  bootstrap.allGather(statuses.data(), sizeof(GpuNetIoSetupStatus));
  for (size_t peer = 0; peer < statuses.size(); ++peer) {
    const auto& status = statuses[peer];
    if (status.code != 0) {
      throw Error(std::string("GPUNetIO setup phase '") + phase + "' failed on rank " + std::to_string(peer) + ": " +
                      status.message,
                  status.code == 1 ? ErrorCode::InvalidUsage : ErrorCode::SystemError);
    }
  }
}

struct GpuNetIoMemoryExchange {
  uint64_t base;
  uint64_t bytes;
  uint32_t rkey;
  uint32_t kind;
  int32_t rank;
  int32_t peer;
  int32_t qpIndex;
  uint32_t version;
};

inline GpuNetIoMemoryExchange exchangeGpuNetIoMemory(Bootstrap& bootstrap, GpuNetIoMemoryExchange outgoing, int tag) {
  GpuNetIoMemoryExchange incoming{};
  if (outgoing.rank < outgoing.peer) {
    bootstrap.send(&outgoing, sizeof(outgoing), outgoing.peer, tag);
    bootstrap.recv(&incoming, sizeof(incoming), outgoing.peer, tag);
  } else {
    bootstrap.recv(&incoming, sizeof(incoming), outgoing.peer, tag);
    bootstrap.send(&outgoing, sizeof(outgoing), outgoing.peer, tag);
  }
  if (incoming.version != 1 || incoming.kind != outgoing.kind || incoming.rank != outgoing.peer ||
      incoming.peer != outgoing.rank || incoming.qpIndex != outgoing.qpIndex || incoming.base == 0 ||
      incoming.bytes == 0 || incoming.bytes > UINTPTR_MAX - incoming.base) {
    throw Error("Mismatched GPUNetIO connection or memory exchange", ErrorCode::InvalidUsage);
  }
  return incoming;
}

}  // namespace mscclpp::detail

#endif