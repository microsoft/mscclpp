// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// IBGDA-style GPU-direct work-request submission for an mlx5 QP.
//
// Given an existing ibv_qp (RC, plain ibv_create_qp), IbgdaResources extracts
// the SQ ring buffer, doorbell record, and BlueFlame UAR via mlx5dv_init_obj
// and maps them into GPU virtual address space:
//   - SQ buf and DBR are host RAM, mapped via cudaHostRegister.
//   - The BF UAR is device MMIO, mapped via cuMemHostRegister(IOMEMORY).
// It also allocates a small GPU-resident state struct (resv_head / ready_head
// / prod_idx / lock) used by the device-side WQE writer.
//
// The host retains ownership of the QP and never touches sq.buf/dbrec/bf
// itself once GPU posting starts — completion polling on the send CQ remains
// a host-side ibv_poll_cq() call.

#ifndef MSCCLPP_IBGDA_HPP_
#define MSCCLPP_IBGDA_HPP_

#if defined(USE_IBVERBS) && defined(MSCCLPP_USE_MLX5DV) && !defined(MSCCLPP_USE_ROCM)

#include <cstdint>
#include <memory>

struct ibv_qp;

namespace mscclpp {

// POD copied to the GPU; consumed by the device-side WQE writer (added in
// Stage 2). Field order/sizes must match include/mscclpp/ibgda_device.hpp.
struct IbgdaQpHandle {
  // Device-mapped pointers (MUST be dereferenceable from the GPU only).
  void* sq_buf;          // SQ WQE ring (host RAM, cudaHostRegister)
  uint32_t* dbrec;       // 32-bit doorbell record (host RAM, cudaHostRegister)
  uint64_t* bf_reg;      // BlueFlame doorbell (NIC MMIO, cuMemHostRegister IOMEMORY)
  // GPU-resident bookkeeping state (allocated by IbgdaResources).
  // Layout: { uint64_t resv_head; uint64_t ready_head; uint64_t prod_idx; int post_send_lock; }.
  uint64_t* state;
  // Constants (host-order).
  uint32_t qpn;
  uint32_t wqe_cnt;      // SQ length in WQEBBs (power of 2)
  uint32_t stride;       // bytes per WQEBB (typically 64)
  uint32_t bf_offset;    // byte offset of BF doorbell within its UAR page
};

// Wraps an existing ibv_qp and prepares the GPU mappings + GPU state. Owns
// the cudaHostRegister/cuMemHostRegister registrations + GPU state buffer.
// Lifetime must enclose any kernel launch that uses getHandle().
class IbgdaResources {
 public:
  // The qp must already be modified to RTS before kernel posting; that is
  // the caller's responsibility.
  explicit IbgdaResources(ibv_qp* qp);
  ~IbgdaResources();

  IbgdaResources(const IbgdaResources&) = delete;
  IbgdaResources& operator=(const IbgdaResources&) = delete;

  const IbgdaQpHandle& getHandle() const { return handle_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
  IbgdaQpHandle handle_{};
};

}  // namespace mscclpp

#endif  // USE_IBVERBS && MSCCLPP_USE_MLX5DV && !MSCCLPP_USE_ROCM
#endif  // MSCCLPP_IBGDA_HPP_
