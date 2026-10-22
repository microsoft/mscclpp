/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file doca_gpunetio_dev_verbs_cq.cuh
 * @brief GDAKI CUDA device functions for CQ management
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEV_VERBS_CQ_H
#define DOCA_GPUNETIO_DEV_VERBS_CQ_H

#include <errno.h>
#include <stddef.h>

#include "doca_gpunetio_dev_verbs_common.cuh"

/**
 * @brief Return device CQ SQ pointer from a device QP
 *
 * @param[in] qp - Dev QP pointer
 *
 * @return Dev CQ pointer
 */
__device__ static __forceinline__ struct doca_gpu_dev_verbs_cq *doca_gpu_dev_verbs_qp_get_cq_sq(
    struct doca_gpu_dev_verbs_qp *qp) {
  return &(qp->cq_sq);
}

/**
 * @brief Increament and round up CQE id
 *
 * @param[in] cqe_idx - cqe idx
 * @param[in] increment - cqe idx increment
 *
 * @return cqe incremented idx
 */
__device__ static __forceinline__ uint32_t doca_gpu_dev_verbs_cqe_idx_inc_mask(uint32_t cqe_idx, uint32_t increment) {
  return (cqe_idx + increment) & DOCA_GPUNETIO_VERBS_CQE_CI_MASK;
}

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
/**
 * @brief Print error CQE values
 *
 * @param[in] cqe64 - erroneous cqe
 *
 * @return
 */
__device__ static __forceinline__ void doca_gpu_dev_verbs_cq_print_cqe_err(struct doca_gpunetio_ib_mlx5_cqe64 *cqe64) {
  struct doca_gpunetio_ib_mlx5_err_cqe_ex *err_cqe = (struct doca_gpunetio_ib_mlx5_err_cqe_ex *)cqe64;

  printf(
      "got completion with err: "
      "syndrome=%#x, vendor_err_synd=%#x, "
      "hw_err_synd=%#x, hw_synd_type=%#x, wqe_counter=%u\n",
      err_cqe->syndrome, err_cqe->vendor_err_synd, err_cqe->hw_err_synd, err_cqe->hw_synd_type,
      doca_gpu_dev_verbs_bswap16(err_cqe->wqe_counter));
}
#endif

/**
 * @brief [Internal] Poll the Completion Queue (CQ) at a specific index.
 * This function does not update the SW consumer index nor guarantees the ordering.
 * It also does not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 */
template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_one_cq_at_legacy(struct doca_gpu_dev_verbs_cq *cq,
                                                                             uint64_t cons_index) {
  uint8_t *cqe = (uint8_t *)__ldg((uintptr_t *)&cq->cqe_daddr);
  const uint32_t cqe_num = __ldg(&cq->cqe_num);
  const uint64_t cqe_rsvd = __ldg(&cq->cqe_rsvd);
  uint64_t cons_index_in_cq = cons_index + cqe_rsvd;
  uint32_t idx = cons_index_in_cq & (cqe_num - 1);
  struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 =
      (struct doca_gpunetio_ib_mlx5_cqe64 *)(cqe + (idx * DOCA_GPUNETIO_VERBS_CQE_SIZE));

  uint64_t cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);

  if (cons_index < cqe_ci) return 0;
  if (cons_index >= cqe_ci + cqe_num) return EBUSY;

  uint8_t opown;
  uint8_t opcode;
  bool observed_completion;

#if __CUDA_ARCH__ >= 900
  opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);

  observed_completion = !((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num));
#else
  uint32_t cqe_chunk;
  uint16_t wqe_counter;

  cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&cqe64->wqe_counter);
  cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
  wqe_counter = cqe_chunk >> 16;
  opown = cqe_chunk & 0xff;

  observed_completion = !((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num)) &&
                        (wqe_counter == ((uint32_t)cons_index & 0xffff));
#endif

  if (!observed_completion) return EBUSY;

  opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
  if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif
  return (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) * -EIO;
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index. This function does
 * not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_one_cq_at_legacy() returns 0. If the completion is
 * not available, returns EBUSY. If it is a completion with error, returns a
 * negative value.
 */
template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_one_cq_at_legacy(struct doca_gpu_dev_verbs_cq *cq,
                                                                        uint64_t cons_index) {
  int status = doca_priv_gpu_dev_verbs_poll_one_cq_at_legacy<resource_sharing_mode, qp_type>(cq, cons_index);
  if (status == 0) {
    doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
  }
  return status;
}

/**
 * @brief [Internal] Poll the Completion Queue (CQ) at a specific index.
 * This function does not update the SW consumer index nor guarantees the ordering.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 */
template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_cq_at_legacy(struct doca_gpu_dev_verbs_cq *cq,
                                                                         uint64_t cons_index) {
  struct doca_gpunetio_ib_mlx5_cqe64 *cqe = (struct doca_gpunetio_ib_mlx5_cqe64 *)__ldg((uintptr_t *)&cq->cqe_daddr);
  const uint32_t cqe_num = __ldg(&cq->cqe_num);
  const uint64_t cqe_rsvd = __ldg(&cq->cqe_rsvd);
  uint64_t cons_index_in_cq = cons_index + cqe_rsvd;
  uint32_t idx = cons_index_in_cq & (cqe_num - 1);
  struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 = &cqe[idx];
  uint8_t opown;
  uint8_t opcode;
  uint64_t cqe_ci;
#if __CUDA_ARCH__ >= 900
  do {
    cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
    [[unlikely]] if (cons_index < cqe_ci)
      return 0;
    opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
  } while (
      (cons_index >= cqe_ci + cqe_num) ||
      ((cqe_ci <= cons_index) && ((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num))));
#else
  uint32_t cqe_chunk;
  uint16_t wqe_counter;

  do {
    cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
    [[unlikely]] if (cons_index < cqe_ci)
      return 0;
    cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&cqe64->wqe_counter);
    cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
    wqe_counter = cqe_chunk >> 16;
    opown = cqe_chunk & 0xff;
  } while (
      (cons_index >= cqe_ci + cqe_num) ||
      ((cqe_ci <= cons_index) && (((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num)) ||
                                  (wqe_counter != ((uint32_t)cons_index & 0xffff)))));
#endif

  opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
  if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif
  return (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) * -EIO;
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index. This function waits for the completion
 * to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_cq_at_legacy() returns 0. If it is a completion with
 * error, returns a negative value.
 */
// WQEBB-frontier completion tracking. cons_index is a WQEBB index (the ticket returned by reservation).
// CQEs are consumed in order; each CQE's wqe_counter gives the start WQEBB of the completed WQE and the WQE's
// DS count (read from its control segment in the SQ ring) gives its size, so multi-WQEBB WQEs are supported.
// cq->cqe_ci holds the frontier (all WQEBBs below it completed); cq->cqe_slot counts consumed CQEs.
struct doca_priv_gpu_dev_verbs_cq_walk {
  struct doca_gpu_dev_verbs_cq *cq;
  uint8_t *cqe_base;
  uint32_t cqe_num;
  uint64_t cqe_rsvd;
  uint16_t wqe_mask;
  uintptr_t wqe_base;
  uint64_t frontier;  // every WQEBB below it is complete
  uint64_t slot;      // CQEs consumed (relative to cqe_rsvd); the CQE at slot belongs to the WQE starting at frontier
  uint64_t published_slot;
};

__device__ static __forceinline__ void doca_priv_gpu_dev_verbs_cq_walk_init(struct doca_priv_gpu_dev_verbs_cq_walk &w,
                                                                           struct doca_gpu_dev_verbs_cq *cq) {
  w.cq = cq;
  w.cqe_base = (uint8_t *)__ldg((uintptr_t *)&cq->cqe_daddr);
  w.cqe_num = __ldg(&cq->cqe_num);
  w.cqe_rsvd = __ldg(&cq->cqe_rsvd);
  struct doca_gpu_dev_verbs_qp *qp =
      (struct doca_gpu_dev_verbs_qp *)((uint8_t *)cq - offsetof(struct doca_gpu_dev_verbs_qp, cq_sq));
  w.wqe_mask = __ldg(&qp->sq_wqe_mask);
  w.wqe_base = __ldg((uintptr_t *)&qp->sq_wqe_daddr);
}

// Byte-swapped dword holding wqe_counter (bits 31..16) and op_own (bits 7..0) of CQE slot (relative to cqe_rsvd).
__device__ static __forceinline__ uint32_t doca_priv_gpu_dev_verbs_cq_word(const struct doca_priv_gpu_dev_verbs_cq_walk &w,
                                                                         uint64_t slot) {
  const uint64_t slot_in_cq = slot + w.cqe_rsvd;
  return doca_gpu_dev_verbs_bswap32(doca_gpu_dev_verbs_load_relaxed_sys_global(
      (uint32_t *)&((struct doca_gpunetio_ib_mlx5_cqe64 *)(w.cqe_base + (slot_in_cq & (w.cqe_num - 1)) *
                                                                            DOCA_GPUNETIO_VERBS_CQE_SIZE))
          ->wqe_counter));
}

__device__ static __forceinline__ bool doca_priv_gpu_dev_verbs_cq_word_arrived(const struct doca_priv_gpu_dev_verbs_cq_walk &w,
                                                                              uint64_t slot, uint32_t word) {
  return !((word & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!((slot + w.cqe_rsvd) & w.cqe_num));
}

__device__ static __forceinline__ bool doca_priv_gpu_dev_verbs_cq_word_error(uint32_t word) {
  return ((word & 0xff) >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR;
}

__device__ static __forceinline__ uint64_t doca_priv_gpu_dev_verbs_cq_word_start(const struct doca_priv_gpu_dev_verbs_cq_walk &w,
                                                                                uint32_t word) {
  return w.frontier + (int64_t)(int16_t)(uint16_t)((word >> 16) - (uint16_t)w.frontier);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ void doca_priv_gpu_dev_verbs_cq_walk_publish(
    struct doca_priv_gpu_dev_verbs_cq_walk &w) {
  if (w.slot == w.published_slot) return;
  doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&w.cq->cqe_slot, w.slot);
  doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&w.cq->cqe_ci, w.frontier);
  w.published_slot = w.slot;
}

// Speculative O(1) check: every WQE has >= 1 WQEBB, so the CQE at slot + (cons_index - frontier) belongs to a WQE
// starting at or after cons_index; if it has arrived, cons_index is complete (CQEs arrive in order). Without
// multi-WQEBB WQEs this is exactly the legacy one-CQE-per-WQEBB slot.
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_cq_try_spec(struct doca_priv_gpu_dev_verbs_cq_walk &w,
                                                                         uint64_t cons_index) {
  if (cons_index - w.frontier >= w.cqe_num) return EBUSY;
  const uint64_t spec = w.slot + (cons_index - w.frontier);
  const uint32_t word = doca_priv_gpu_dev_verbs_cq_word(w, spec);
  if (!doca_priv_gpu_dev_verbs_cq_word_arrived(w, spec, word)) return EBUSY;
  if (doca_priv_gpu_dev_verbs_cq_word_error(word)) return -EIO;
  const uint64_t start = doca_priv_gpu_dev_verbs_cq_word_start(w, word);
  if (start < cons_index) return EBUSY;
  w.frontier = start;
  w.slot = spec;
  return 0;
}

// In-order walk from w.slot. Returns 0 once cons_index < frontier, EBUSY when the head CQE has not arrived.
// Only local state is updated; callers publish.
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_cq_walk_step(struct doca_priv_gpu_dev_verbs_cq_walk &w,
                                                                          uint64_t cons_index) {
  constexpr int kBatch = 8;
  for (;;) {
    if (!doca_priv_gpu_dev_verbs_cq_word_arrived(w, w.slot, doca_priv_gpu_dev_verbs_cq_word(w, w.slot))) return EBUSY;
    uint32_t words[kBatch];
#pragma unroll
    for (int i = 0; i < kBatch; ++i) words[i] = doca_priv_gpu_dev_verbs_cq_word(w, w.slot + i);
    int arrived = 0;
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      if (arrived != i || !doca_priv_gpu_dev_verbs_cq_word_arrived(w, w.slot + i, words[i])) continue;
      if (doca_priv_gpu_dev_verbs_cq_word_error(words[i])) {
#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
        doca_gpu_dev_verbs_cq_print_cqe_err((struct doca_gpunetio_ib_mlx5_cqe64 *)(w.cqe_base +
                                                                                   ((w.slot + i + w.cqe_rsvd) &
                                                                                    (w.cqe_num - 1)) *
                                                                                       DOCA_GPUNETIO_VERBS_CQE_SIZE));
#endif
        return -EIO;
      }
      arrived = i + 1;
    }
    if (arrived == 0) return EBUSY;  // head re-read raced; retry later
    uint32_t last = words[0];
#pragma unroll
    for (int i = 1; i < kBatch; ++i)
      if (i < arrived) last = words[i];
    const uint64_t start = doca_priv_gpu_dev_verbs_cq_word_start(w, last);
    // Every WQE is signaled, so all WQEBBs before the last completed WQE are complete too.
    if (cons_index < start) {
      w.slot += arrived - 1;
      w.frontier = start;
      return 0;
    }
    const uint64_t ctrl =
        *(volatile uint64_t *)(w.wqe_base + ((uintptr_t)(start & w.wqe_mask) << DOCA_GPUNETIO_IB_MLX5_WQE_SQ_SHIFT));
    const uint32_t opmod_idx = doca_gpu_dev_verbs_bswap32((uint32_t)ctrl);
    const uint32_t qpn_ds = doca_gpu_dev_verbs_bswap32((uint32_t)(ctrl >> 32));
    // A recycled ring slot means this completion was already published (reuse waits on it).
    const uint64_t size = (((opmod_idx >> DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) & 0xffff) == (start & 0xffff))
                              ? (((qpn_ds & 0x3f) + 3) >> 2)
                              : 1;
    w.slot += arrived;
    w.frontier = start + size;
    if (cons_index < w.frontier) return 0;
  }
}

// WQEBB-frontier completion tracking. cons_index is a WQEBB index (the ticket returned by reservation).
// CQEs are consumed in order; each CQE's wqe_counter gives the start WQEBB of the completed WQE and the WQE's
// DS count (read from its control segment in the SQ ring) gives its size, so multi-WQEBB WQEs are supported.
// cq->cqe_ci holds the frontier (all WQEBBs below it completed); cq->cqe_slot counts consumed CQEs.
// Shared fields live in the QP's cache lines, so they are written only when a wait succeeds.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          bool blocking = false>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_cq_advance(struct doca_gpu_dev_verbs_cq *cq,
                                                                         uint64_t cons_index) {
  struct doca_priv_gpu_dev_verbs_cq_walk w;
  w.frontier = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
  if (cons_index < w.frontier) return 0;
  doca_priv_gpu_dev_verbs_cq_walk_init(w, cq);
  w.slot = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_slot);
  w.published_slot = w.slot;
  uint32_t multi = 0;
  for (uint32_t spin = 0;; ++spin) {
    const int spec = doca_priv_gpu_dev_verbs_cq_try_spec(w, cons_index);
    if (spec == 0) break;
    if (spec < 0) return spec;
    // Without multi-WQEBB WQEs the speculative slot is exact, so a miss means not complete (legacy cost).
    if ((spin & 255) == 0) multi = *(volatile uint32_t *)&cq->multi_bb;
    if (multi) {
      const int status = doca_priv_gpu_dev_verbs_cq_walk_step(w, cons_index);
      if (status == 0) break;
      if (status < 0) return status;
    }
    if (!blocking) return EBUSY;
  }
  doca_priv_gpu_dev_verbs_cq_walk_publish<resource_sharing_mode>(w);
  return 0;
}

template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_one_cq_at(struct doca_gpu_dev_verbs_cq *cq,
                                                                             uint64_t cons_index) {
  return doca_priv_gpu_dev_verbs_cq_advance<resource_sharing_mode>(cq, cons_index);
}

template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_one_cq_at(struct doca_gpu_dev_verbs_cq *cq,
                                                                        uint64_t cons_index) {
  int status = doca_priv_gpu_dev_verbs_cq_advance<resource_sharing_mode>(cq, cons_index);
  if (status == 0) doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
  return status;
}

template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_cq_at(struct doca_gpu_dev_verbs_cq *cq,
                                                                         uint64_t cons_index) {
  int status;
  status = doca_priv_gpu_dev_verbs_cq_advance<resource_sharing_mode, true>(cq, cons_index);
  return status;
}

template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_at(struct doca_gpu_dev_verbs_cq *cq,
                                                                    uint64_t cons_index) {
  int status = doca_priv_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
  if (status == 0) doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
  return status;
}

// Wait until every WQE reserved below end_index (the current sq_rsvd_index) completed. Requires that all those
// WQEs were posted and that no other thread posts concurrently, so the WQE count is end_index - multi_bb and
// the last CQE slot is known: spin on that single CQE (legacy cost) instead of walking the CQ.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_drain(struct doca_gpu_dev_verbs_cq *cq,
                                                                       uint64_t end_index) {
  struct doca_priv_gpu_dev_verbs_cq_walk w;
  w.frontier = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
  if (end_index <= w.frontier) return 0;
  doca_priv_gpu_dev_verbs_cq_walk_init(w, cq);
  const uint64_t last = end_index - *(volatile uint32_t *)&cq->multi_bb - 1;
  for (uint32_t spin = 1;; ++spin) {
    const uint32_t word = doca_priv_gpu_dev_verbs_cq_word(w, last);
    if (doca_priv_gpu_dev_verbs_cq_word_arrived(w, last, word)) {
      if (doca_priv_gpu_dev_verbs_cq_word_error(word)) return -EIO;
      const uint64_t start = doca_priv_gpu_dev_verbs_cq_word_start(w, word);
      // Valid only if this CQE completes the WQE ending at end_index (in-order completion covers the rest). No WQE
      // extends past end_index, so a WQE starting at end_index - 1 qualifies without reading its size.
      bool exact = start + 1 == end_index;
      if (!exact && start < end_index) {
        const uint64_t ctrl =
            *(volatile uint64_t *)(w.wqe_base + ((uintptr_t)(start & w.wqe_mask) << DOCA_GPUNETIO_IB_MLX5_WQE_SQ_SHIFT));
        const uint32_t opmod_idx = doca_gpu_dev_verbs_bswap32((uint32_t)ctrl);
        const uint32_t qpn_ds = doca_gpu_dev_verbs_bswap32((uint32_t)(ctrl >> 32));
        exact = ((opmod_idx >> DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) & 0xffff) == (start & 0xffff) &&
                start + (((qpn_ds & 0x3f) + 3) >> 2) == end_index;
      }
      if (exact) {
        w.published_slot = w.slot = last + 1;
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_slot, last + 1);
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, end_index);
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        return 0;
      }
      return doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode>(cq, end_index - 1);
    }
    // Concurrent posting can make the count stale; the general path guarantees progress.
    if ((spin & 1023) == 0) {
      const int status = doca_priv_gpu_dev_verbs_cq_advance<resource_sharing_mode>(cq, end_index - 1);
      if (status != EBUSY) {
        if (status == 0) doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        return status;
      }
    }
  }
  return 0;
}

template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_at_legacy(struct doca_gpu_dev_verbs_cq *cq,
                                                                    uint64_t cons_index) {
  int status = doca_priv_gpu_dev_verbs_poll_cq_at_legacy<resource_sharing_mode, qp_type>(cq, cons_index);
  if (status == 0) {
    doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
  }
  return status;
}

/**
 * @brief [Internal] Poll the Collapsed Completion Queue (CQ) at a specific index. This function
 * waits for the completion to arrive.
 *
 * @param cq - Collapsed Completion Queue (CQ)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @param new_cqe_ci - New CQ Consumer Index
 * @return On success, doca_priv_gpu_dev_verbs_poll_cq_collapsed_at() returns 0. If it is a
 * completion with error, returns a negative value.
 */
template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_cq_collapsed_at(struct doca_gpu_dev_verbs_qp *qp,
                                                                                   struct doca_gpu_dev_verbs_cq *cq,
                                                                                   uint64_t cons_index,
                                                                                   uint64_t *new_cqe_ci) {
  struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 = (struct doca_gpunetio_ib_mlx5_cqe64 *)__ldg((uintptr_t *)&cq->cqe_daddr);
  const uint32_t cqe_num = __ldg(&cq->cqe_num);
  uint8_t opown;
  uint8_t opcode;
  uint64_t cqe_ci;
  uint32_t cqe_chunk;
  uint16_t wqe_counter;

  // If idx is a lot greater than cons_idx, we might get incorrect result due
  // to wqe_counter wraparound. We need to check prod_idx to be sure that idx
  // has already been submitted.
  while (doca_gpu_dev_verbs_atomic_read<uint64_t, resource_sharing_mode>(&qp->sq_wqe_pi) < cons_index);
  doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();

  do {
    cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
    [[unlikely]] if (cons_index < cqe_ci)
      return 0;
    cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&cqe64->wqe_counter);
    cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
    wqe_counter = cqe_chunk >> 16;
    opown = cqe_chunk & 0xff;
    opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
  }
  // NOTE: This while loop is part of do while above.
  // wqe_counter is the HW consumer index. However, we always maintain index
  // + 1 in SW. To be able to compare with idx, we need to use wqe_counter +
  // 1. Because wqe_counter is uint16_t, it may wraparound. Still we know for
  // sure that if idx - wqe_counter - 1 < ncqes, wqe_counter + 1 is less than
  // idx, and thus we need to wait. We don't need to wait when idx ==
  // wqe_counter + 1. That's why we use - (uint16_t)2 here to make this case
  // wraparound.
  while ((opcode == DOCA_GPUNETIO_IB_MLX5_CQE_INVALID) ||
         ((cqe_ci <= cons_index) && ((uint16_t)((uint16_t)cons_index - wqe_counter - (uint16_t)2) < cqe_num)));

  ++wqe_counter;
  *new_cqe_ci = (cons_index & ~(0xFFFFULL) | wqe_counter) + (((uint16_t)cons_index > wqe_counter) ? 0x10000ULL : 0x0);

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
  if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif

  return ((opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) * -EIO);
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index. This function waits for the completion
 * to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_cq_collapsed_at() returns 0. If it is a completion
 * with error, returns a negative value.
 */
template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_collapsed_at(struct doca_gpu_dev_verbs_qp *qp,
                                                                              uint64_t cons_index) {
  uint64_t new_cqe_ci = 0;
  int status = doca_priv_gpu_dev_verbs_poll_cq_collapsed_at<resource_sharing_mode, qp_type>(
      qp, doca_gpu_dev_verbs_qp_get_cq_sq(qp), cons_index, &new_cqe_ci);
  if (status == 0) {
    doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&(doca_gpu_dev_verbs_qp_get_cq_sq(qp)->cqe_ci),
                                                                   new_cqe_ci);
  }
  return status;
}

/**
 * @brief Poll the Completion Queue (CQ). This function waits for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param count - Number of completions to poll
 * @return On success, doca_gpu_dev_verbs_poll_cq() returns 0. If it is a completion with
 * error, returns a negative value.
 */
template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq(struct doca_gpu_dev_verbs_cq *cq, uint32_t count) {
  [[unlikely]] if (count == 0)
    return 0;
  uint64_t cons_index = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci) + count - 1;
  return doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
}

/**
 * @brief Increment CQ DBREC
 *
 * @param[in] cq - GPU Completion Queue
 * @param[in] cqe_num - CQE num to increment
 *
 * @return new CQE consumer index
 */
template <bool is_overrun>
__device__ static __forceinline__ uint32_t doca_gpu_dev_verbs_cq_update_dbrec(struct doca_gpu_dev_verbs_cq *cq,
                                                                              uint32_t cqe_num) {
  uint32_t cqe_ci = DOCA_GPUNETIO_VOLATILE(cq->cqe_ci);

  cqe_ci = (cqe_ci + cqe_num) & DOCA_GPUNETIO_VERBS_CQE_CI_MASK;
  if (is_overrun == false) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;"
                 :
                 : "l"(cq->dbrec), "r"(doca_gpu_dev_verbs_bswap32(cqe_ci)));
  }

  DOCA_GPUNETIO_VOLATILE(cq->cqe_ci) = cqe_ci;

  return cqe_ci;
}

#endif /* DOCA_GPUNETIO_DEV_VERBS_CQ_H */

/** @} */
