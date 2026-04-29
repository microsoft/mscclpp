// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "NcclFallback.hpp"

#include <dlfcn.h>
#include <nccl.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mscclpp/gpu_utils.hpp>
#include <stdexcept>
#include <vector>

namespace torch::comms {

// --- NCCL C-API ABI mirror ---
//
// We use dlsym for runtime binding (no link-time NCCL dependency), but include
// nccl.h so enum/type names stay source-of-truth (no hardcoded numeric values).
namespace {
bool torchcommTraceEnabled() {
  const char* value = std::getenv("MSCCLPP_TORCHCOMMS_TRACE");
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

// Function pointer types (signatures from nccl.h).
using GetUniqueIdFn = ncclResult_t (*)(ncclUniqueId*);
using CommInitRankFn = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int);
using CommDestroyFn = ncclResult_t (*)(ncclComm_t);
using ReduceScatterFn = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
                                         cudaStream_t);
using BroadcastFn = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
using AllReduceFn =
    ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);

ncclDataType_t torchDtypeToNccl(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return ncclFloat32;
    case at::kHalf:
      return ncclFloat16;
    case at::kBFloat16:
      return ncclBfloat16;
    case at::kInt:
      return ncclInt32;
    case at::kUInt32:
      return ncclUint32;
    default:
      throw std::runtime_error("[NcclFallback] unsupported dtype " + std::string(at::toString(dtype)));
  }
}

ncclRedOp_t torchReduceOpToNccl(const ReduceOp& op) {
  using RedOpType = ReduceOp::RedOpType;
  switch (op.type()) {
    case RedOpType::SUM:
    case RedOpType::PREMUL_SUM:  // caller has already applied the scaling
      return ncclSum;
    case RedOpType::AVG:
      return ncclAvg;
    case RedOpType::MIN:
      return ncclMin;
    default:
      throw std::runtime_error("[NcclFallback] unsupported reduce op type " +
                               std::to_string(static_cast<int>(op.type())));
  }
}
}  // namespace

// --- Lifecycle ---

std::unique_ptr<NcclFallback> NcclFallback::tryCreate(const std::shared_ptr<mscclpp::Communicator>& comm, int rank,
                                                     int worldSize) {
  std::unique_ptr<NcclFallback> fb(new NcclFallback());

  // Search candidates for libnccl. MSCCLPP_NCCL_LIB_PATH matches the
  // existing src/ext/nccl behavior; bare "libnccl.so.2" picks up PyTorch's
  // bundled copy via the Python rpath.
  std::vector<std::string> candidates;
  if (const char* envPath = std::getenv("MSCCLPP_NCCL_LIB_PATH"); envPath && envPath[0]) {
    candidates.emplace_back(envPath);
  }
  candidates.emplace_back("libnccl.so.2");
  candidates.emplace_back("libnccl.so");

  for (const auto& path : candidates) {
    fb->dlHandle_ = dlopen(path.c_str(), RTLD_LAZY | RTLD_NODELETE);
    if (fb->dlHandle_) break;
  }
  if (!fb->dlHandle_) {
    if (rank == 0) {
      const char* err = dlerror();
      std::cerr << "[NcclFallback] could not dlopen libnccl.so.2; fallback disabled. dlerror="
                << (err ? err : "(null)") << std::endl;
    }
    return nullptr;
  }

  auto sym = [&](const char* name) -> void* {
    void* p = dlsym(fb->dlHandle_, name);
    if (!p && rank == 0) {
      std::cerr << "[NcclFallback] dlsym(" << name << ") failed: " << dlerror() << std::endl;
    }
    return p;
  };

  fb->getUniqueIdFn_ = sym("ncclGetUniqueId");
  fb->commInitRankFn_ = sym("ncclCommInitRank");
  fb->commDestroyFn_ = sym("ncclCommDestroy");
  fb->reduceScatterFn_ = sym("ncclReduceScatter");
  fb->broadcastFn_ = sym("ncclBroadcast");
  fb->allReduceFn_ = sym("ncclAllReduce");
  if (!fb->getUniqueIdFn_ || !fb->commInitRankFn_ || !fb->commDestroyFn_ || !fb->reduceScatterFn_ ||
      !fb->broadcastFn_ || !fb->allReduceFn_) {
    return nullptr;  // dtor cleans up dlHandle_
  }

  // Distribute the ncclUniqueId via the MSCCL++ bootstrap. The base Bootstrap
  // interface exposes allGather() but not broadcast(), so we use allGather:
  // rank 0 fills its own slot, others zero theirs, then everyone reads slot 0.
  std::vector<ncclUniqueId> allIds(worldSize);
  if (rank == 0) {
    int rc = reinterpret_cast<GetUniqueIdFn>(fb->getUniqueIdFn_)(&allIds[0]);
    if (rc != 0) {
      std::cerr << "[NcclFallback] ncclGetUniqueId failed: rc=" << rc << std::endl;
      return nullptr;
    }
  } else {
    std::memset(&allIds[rank], 0, sizeof(ncclUniqueId));
  }
  comm->bootstrap()->allGather(allIds.data(), sizeof(ncclUniqueId));

  int rc = reinterpret_cast<CommInitRankFn>(fb->commInitRankFn_)(reinterpret_cast<ncclComm_t*>(&fb->ncclComm_),
                                                                 worldSize, allIds[0], rank);
  if (rc != 0) {
    std::cerr << "[NcclFallback] ncclCommInitRank failed: rc=" << rc << std::endl;
    return nullptr;
  }

  // Persistent 4-byte device buffer for barrier-as-allreduce. This is local,
  // never shared cross-rank, so plain cudaMalloc/cudaFree is appropriate.
  MSCCLPP_CUDATHROW(cudaMalloc(&fb->barrierBuf_, sizeof(int)));
  MSCCLPP_CUDATHROW(cudaMemset(fb->barrierBuf_, 0, sizeof(int)));

  if (rank == 0) {
    std::cerr << "[NcclFallback] enabled (libnccl.so.2 dlopened)." << std::endl;
  }
  return fb;
}

NcclFallback::~NcclFallback() {
  if (barrierBuf_) cudaFree(barrierBuf_);
  if (ncclComm_ && commDestroyFn_) {
    reinterpret_cast<CommDestroyFn>(commDestroyFn_)(reinterpret_cast<ncclComm_t>(ncclComm_));
  }
  if (dlHandle_) dlclose(dlHandle_);
}

// --- Dispatchers ---
// Keep fallback narrowly scoped to collectives that currently need it.
// Unsupported collectives remain explicit in TorchCommMSCCLPP to preserve
// TorchComm API semantics and clear error messaging.

void NcclFallback::reduceScatter(const void* sendbuf, void* recvbuf, size_t recvCount, at::ScalarType dtype,
                                 const ReduceOp& op, cudaStream_t stream) {
  if (torchcommTraceEnabled()) {
    std::cerr << "[NcclFallback] reduce_scatter -> NCCL recvCount=" << recvCount
              << " dtype=" << static_cast<int>(torchDtypeToNccl(dtype))
              << " op=" << static_cast<int>(torchReduceOpToNccl(op)) << std::endl;
  }
  int rc = reinterpret_cast<ReduceScatterFn>(reduceScatterFn_)(sendbuf, recvbuf, recvCount, torchDtypeToNccl(dtype),
                                                                torchReduceOpToNccl(op),
                                                                reinterpret_cast<ncclComm_t>(ncclComm_), stream);
  if (rc != 0) throw std::runtime_error("[NcclFallback] ncclReduceScatter rc=" + std::to_string(rc));
}

void NcclFallback::broadcast(const void* sendbuf, void* recvbuf, size_t count, at::ScalarType dtype, int root,
                             cudaStream_t stream) {
  if (torchcommTraceEnabled()) {
    std::cerr << "[NcclFallback] broadcast -> NCCL count=" << count
              << " dtype=" << static_cast<int>(torchDtypeToNccl(dtype)) << " root=" << root << std::endl;
  }
  int rc = reinterpret_cast<BroadcastFn>(broadcastFn_)(sendbuf, recvbuf, count, torchDtypeToNccl(dtype), root,
                                                       reinterpret_cast<ncclComm_t>(ncclComm_), stream);
  if (rc != 0) throw std::runtime_error("[NcclFallback] ncclBroadcast rc=" + std::to_string(rc));
}

void NcclFallback::barrier(cudaStream_t stream) {
  if (torchcommTraceEnabled()) {
    std::cerr << "[NcclFallback] barrier -> NCCL allreduce" << std::endl;
  }
  int rc = reinterpret_cast<AllReduceFn>(allReduceFn_)(barrierBuf_, barrierBuf_, 1, ncclInt32, ncclSum,
                                                       reinterpret_cast<ncclComm_t>(ncclComm_), stream);
  if (rc != 0) throw std::runtime_error("[NcclFallback] barrier (allreduce) rc=" + std::to_string(rc));
}

}  // namespace torch::comms
