// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ibgda.hpp"

#if defined(USE_IBVERBS) && defined(MSCCLPP_USE_MLX5DV) && !defined(MSCCLPP_USE_ROCM)

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "logger.hpp"
#include "mlx5dv_wrapper.hpp"

namespace mscclpp {

struct IbgdaResources::Impl {
  void* sq_buf_host = nullptr;
  size_t sq_bytes = 0;
  void* dbrec_host = nullptr;        // not necessarily aligned to anything > 8B
  void* dbrec_register_addr = nullptr;
  size_t dbrec_register_bytes = 0;
  void* uar_page_host = nullptr;     // page-aligned base of the UAR
  void* state_dev = nullptr;         // device-resident state
  bool sq_registered = false;
  bool dbrec_registered = false;
  bool uar_registered = false;
};

namespace {

inline uintptr_t pageMask() {
  static const uintptr_t mask = static_cast<uintptr_t>(::sysconf(_SC_PAGESIZE)) - 1;
  return mask;
}

}  // namespace

IbgdaResources::IbgdaResources(ibv_qp* qp) : pimpl_(std::make_unique<Impl>()) {
  if (qp == nullptr) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "IbgdaResources: qp is null");
  }
  if (!MLX5DV::isAvailable()) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "IbgdaResources: libmlx5 not available");
  }

  // 1. Extract sq.buf / dbrec / bf via mlx5dv_init_obj.
  struct mlx5dv_qp dvQp{};
  if (MLX5DV::mlx5dv_init_obj_qp(qp, &dvQp) != 0) {
    THROW(NET, IbError, errno, "mlx5dv_init_obj(QP) failed (errno ", errno, ")");
  }

  pimpl_->sq_buf_host = dvQp.sq.buf;
  pimpl_->sq_bytes = static_cast<size_t>(dvQp.sq.wqe_cnt) * dvQp.sq.stride;
  pimpl_->dbrec_host = dvQp.dbrec;
  if (pimpl_->sq_buf_host == nullptr || pimpl_->dbrec_host == nullptr || dvQp.bf.reg == nullptr) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "mlx5dv_qp returned null pointers");
  }
  if (dvQp.sq.wqe_cnt == 0 || (dvQp.sq.wqe_cnt & (dvQp.sq.wqe_cnt - 1)) != 0) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "sq.wqe_cnt must be a power of 2, got ", dvQp.sq.wqe_cnt);
  }

  // 2. Register the SQ buffer with CUDA (host-mapped). We register the
  //    enclosing whole pages because cudaHostRegister requires that.
  {
    uintptr_t base = reinterpret_cast<uintptr_t>(pimpl_->sq_buf_host);
    uintptr_t pageBase = base & ~pageMask();
    size_t pad = static_cast<size_t>(base - pageBase);
    size_t regBytes = (pad + pimpl_->sq_bytes + pageMask()) & ~pageMask();
    void* regAddr = reinterpret_cast<void*>(pageBase);
    cudaError_t e = cudaHostRegister(regAddr, regBytes, cudaHostRegisterDefault);
    if (e != cudaSuccess) {
      THROW(NET, SysError, static_cast<int>(e), "cudaHostRegister(sq.buf) failed: ",
            cudaGetErrorString(e));
    }
    pimpl_->sq_registered = true;
    void* dev = nullptr;
    MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dev, pimpl_->sq_buf_host, 0));
    handle_.sq_buf = dev;
  }

  // 3. Register the DBR record. mlx5 hands us a pointer into a small block
  //    of DBR records (8 B each); register a whole page starting from its
  //    page base.
  {
    uintptr_t base = reinterpret_cast<uintptr_t>(pimpl_->dbrec_host);
    uintptr_t pageBase = base & ~pageMask();
    size_t regBytes = pageMask() + 1;  // 1 page
    pimpl_->dbrec_register_addr = reinterpret_cast<void*>(pageBase);
    pimpl_->dbrec_register_bytes = regBytes;
    cudaError_t e = cudaHostRegister(pimpl_->dbrec_register_addr, regBytes, cudaHostRegisterDefault);
    if (e == cudaErrorHostMemoryAlreadyRegistered) {
      // The DBR page may overlap with another QP we already registered.
      // Treat as success but skip unregister on shutdown. Clear sticky error.
      pimpl_->dbrec_registered = false;
      (void)cudaGetLastError();
    } else if (e != cudaSuccess) {
      THROW(NET, SysError, static_cast<int>(e), "cudaHostRegister(dbrec) failed: ",
            cudaGetErrorString(e));
    } else {
      pimpl_->dbrec_registered = true;
    }
    void* dev = nullptr;
    MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&dev, pimpl_->dbrec_host, 0));
    handle_.dbrec = static_cast<uint32_t*>(dev);
  }

  // 4. Map the UAR page (NIC MMIO) into GPU VA via cuMemHostRegister IOMEMORY.
  {
    uintptr_t bfAddr = reinterpret_cast<uintptr_t>(dvQp.bf.reg);
    uintptr_t pageAddr = bfAddr & ~uintptr_t(4095);  // UAR pages are 4 KB
    uintptr_t bfOffset = bfAddr - pageAddr;
    pimpl_->uar_page_host = reinterpret_cast<void*>(pageAddr);
    CUresult cuRes =
        cuMemHostRegister(pimpl_->uar_page_host, 4096, CU_MEMHOSTREGISTER_IOMEMORY);
    if (cuRes == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
      // The UAR page is shared across QPs on the same context; if a sibling
      // QP already registered it, treat as success and skip unregister.
      pimpl_->uar_registered = false;
      (void)cudaGetLastError();
    } else if (cuRes != CUDA_SUCCESS) {
      const char* s = nullptr;
      cuGetErrorString(cuRes, &s);
      THROW(NET, SysError, static_cast<int>(cuRes),
            "cuMemHostRegister(UAR, IOMEMORY) failed: ", s ? s : "?",
            ". Ensure 'options nvidia NVreg_RegistryDwords=\"PeerMappingOverride=1;\"' "
            "is set and nvidia_peermem is loaded.");
    } else {
      pimpl_->uar_registered = true;
    }
    CUdeviceptr dPage = 0;
    MSCCLPP_CUTHROW(cuMemHostGetDevicePointer(&dPage, pimpl_->uar_page_host, 0));
    handle_.bf_reg = reinterpret_cast<uint64_t*>(dPage + bfOffset);
    handle_.bf_offset = static_cast<uint32_t>(bfOffset);
  }

  // 5. Allocate GPU-resident state (zero-initialized).
  {
    constexpr size_t kStateBytes = 4 * sizeof(uint64_t);  // resv/ready/prod/lock
    MSCCLPP_CUDATHROW(cudaMalloc(&pimpl_->state_dev, kStateBytes));
    MSCCLPP_CUDATHROW(cudaMemset(pimpl_->state_dev, 0, kStateBytes));
    handle_.state = static_cast<uint64_t*>(pimpl_->state_dev);
  }

  // 6. Fill the rest of the handle.
  handle_.qpn = qp->qp_num;
  handle_.wqe_cnt = dvQp.sq.wqe_cnt;
  handle_.stride = dvQp.sq.stride;
}

IbgdaResources::~IbgdaResources() {
  if (!pimpl_) return;
  if (pimpl_->state_dev) {
    cudaFree(pimpl_->state_dev);
  }
  if (pimpl_->uar_registered && pimpl_->uar_page_host) {
    cuMemHostUnregister(pimpl_->uar_page_host);
  }
  if (pimpl_->dbrec_registered && pimpl_->dbrec_register_addr) {
    cudaHostUnregister(pimpl_->dbrec_register_addr);
  }
  if (pimpl_->sq_registered && pimpl_->sq_buf_host) {
    uintptr_t base = reinterpret_cast<uintptr_t>(pimpl_->sq_buf_host);
    void* regAddr = reinterpret_cast<void*>(base & ~pageMask());
    cudaHostUnregister(regAddr);
  }
}

}  // namespace mscclpp

#endif  // USE_IBVERBS && MSCCLPP_USE_MLX5DV && !MSCCLPP_USE_ROCM
