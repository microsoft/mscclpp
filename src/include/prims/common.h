/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

// #include "collectives.h"
#include "devcomm.h"
#include "op128.h"

#if __CUDA_ARCH__ >= 800
#define COLL_UNROLL 8
#else
#define COLL_UNROLL 4
#endif
#define NCCL_MAX_SLICE_PER_CHUNK 2

#define NCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree

typedef void(*ncclKern_t)();

enum ncclDevRedOp_t {
  ncclDevSum, ncclDevProd, ncclDevMax, ncclDevMin,
  ncclDevPreMulSum, ncclDevSumPostDiv,
  ncclNumDevRedOps
};
struct ncclDevRedOpFull {
  ncclDevRedOp_t op;
  bool scalarArgIsPtr;
  uint64_t scalarArg;
};

struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_DIRECT_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_DIRECT_ARITY];
  void* srcs[NCCL_MAX_DIRECT_ARITY+1];
  void* dsts[NCCL_MAX_DIRECT_ARITY+1];
  int totalSendSize[NCCL_MAX_SLICE_PER_CHUNK];
};

struct ncclShmemData {
  union {
    uint64_t ll128warp[NCCL_LL128_MAX_NTHREADS/WARP_SIZE][NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE];
    struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  };
  uint64_t redOpArgs[NCCL_MAX_DIRECT_ARITY+1];
  int channelId;
  int aborted;
  alignas(16) struct ncclDevComm comm;
  alignas(16) struct ncclDevChannel channel;
  alignas(16) struct ncclWork work;
};
static_assert(offsetof(struct ncclShmemData, work)%16 == 0, "shmem.work needs to be 16B aligned");

extern __shared__ ncclShmemData ncclShmem;

__device__ inline bool barrierReduceAny(int bit) {
  uint32_t popc;
  asm ("{"
    ".reg .pred barr_pred;"
    "setp.eq.u32 barr_pred, %1, 1;"
    "bar.red.popc.u32 %0, 2, barr_pred;"
  "}" : "=r"(popc) : "r"(bit));
  return popc != 0;
}

// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    uint64_t a=0, b=0;
    asm("ld.v2.u64 {%0,%1},[%2];" : "=l"(a),"=l"(b) : "l"((char const*)src + offset));
    asm volatile("st.v2.u64 [%0],{%1,%2};" :: "l"((char*)dst + offset), "l"(a), "l"(b));
  }
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkElement {
  __device__ void run(ncclWorkElem*) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWork {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run(ncclWork *w) {
    int wid = threadIdx.x / WARP_SIZE;
    ncclWorkElem* we = w->header.type == ncclWorkTypeRegColl ? &w->regElems[0].elem : &w->elems[0];
    int stride = w->header.type == ncclWorkTypeRegColl ? sizeof(ncclWorkElemReg) : sizeof(ncclWorkElem);
    #pragma unroll 1
    while ((char*)we + stride <= (char*)(w+1) && we->isUsed) {
      if (wid < we->nWarps) {
        RunWorkElement<Fn, T, RedOp, Algo, Proto>().run(we);
      }
      we = (ncclWorkElem*)((char*)we + stride);
    }
  }
};

static __device__ void ncclRedopPtrDeref(struct ncclWorkElem* we) {
  if (we->isUsed && we->redOpArgIsPtr) {
    /* redOpArg is a pointer to the scalar value, so we'll dereference it
     * here so that redOpArg holds the bits of the scalar going forward.
     * The tricky thing is we don't know its type T since that's encoded in
     * the funcIndex. Because it would be difficult to get sizeof(T) from
     * funcIndex, we'll cheat and just dereference the largest possible size
     * given the alignment of the pointer. We might be reading in more bytes
     * than we need but that's harmless.
     */
    if (we->redOpArg%2 != 0)
      we->redOpArg = *reinterpret_cast<uint8_t*>(we->redOpArg);
    else if (we->redOpArg%4 != 0)
      we->redOpArg = *reinterpret_cast<uint16_t*>(we->redOpArg);
    else if (we->redOpArg%8 != 0)
      we->redOpArg = *reinterpret_cast<uint32_t*>(we->redOpArg);
    else
      we->redOpArg = *reinterpret_cast<uint64_t*>(we->redOpArg);
  }
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int FnIndex>
__device__ void ncclKernel(
    struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead
  )  {
  int tid = threadIdx.x;

  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  if (tid < WARP_SIZE) {
    int x = tid;
    if (channelMask & (1ull<<x)) {
      int y = __popcll(channelMask & ((1ull<<x)-1));
      if (blockIdx.x == y) ncclShmem.channelId = x;
    }
    if (32 < MAXCHANNELS) {
      x = 32 + tid;
      if (channelMask & (1ull<<x)) {
        int y = __popcll(channelMask & ((1ull<<x)-1));
        if (blockIdx.x == y) ncclShmem.channelId = x;
      }
    }
  }
  __syncthreads(); // publish ncclShmem.channelId
  int channelId = ncclShmem.channelId;
  /* set abort flag to 0 */
  if (tid == 0) ncclShmem.aborted = 0;

  if (true) {
    void *dst, *src;
    int bytes;
    // Use first 3 warps to load comm, channel, and work into ncclShmem
    switch (tid/WARP_SIZE) {
    case 0:
      dst = &ncclShmem.comm;
      src = comm;
      bytes = sizeof(ncclDevComm);
      static_assert(sizeof(ncclDevComm) <= 16*WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
      break;
    case 1:
      // Get address of channel without incurring indirect load from ncclDevComm::channels
      dst = &ncclShmem.channel;
      src = &((ncclDevCommAndChannels*)comm)->channels[channelId];
      bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      break;
    case 2:
      dst = &ncclShmem.work;
      src = workHead + blockIdx.x;
      bytes = sizeof(ncclWork);
      static_assert(sizeof(ncclWork) <= 16*WARP_SIZE, "ncclWork cannot be loaded by a single warp in one insn.");
      break;
    default:
      bytes = 0;
      break;
    }
    copyToShmem16(tid%WARP_SIZE, dst, src, bytes);
  }
  __syncthreads(); // publish ncclShmem

  while (true) {
    // Notify host that all fifo reads are complete.
    if (tid == 0 && ncclShmem.work.header.isLast && ncclShmem.work.header.inFifo) {
      *ncclShmem.channel.workFifoDone = ncclShmem.work.header.doneAcks;
    }

    __syncwarp();
    if (ncclShmem.work.header.type == ncclWorkTypeColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS) ncclRedopPtrDeref(&ncclShmem.work.elems[tid]);
    } else if (ncclShmem.work.header.type == ncclWorkTypeRegColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS_REG) ncclRedopPtrDeref(&ncclShmem.work.regElems[tid].elem);
    }
    __syncthreads();

    if (ncclShmem.work.header.funcIndex == FnIndex) {
      RunWork<Fn, T, RedOp, Algo, Proto>().run(&ncclShmem.work);
    } else {
    }

    int workIxNext = ncclShmem.work.header.workNext;
    __syncthreads();
    if (ncclShmem.work.header.isLast) break;

    copyToShmem16(tid, &ncclShmem.work, workHead + workIxNext, sizeof(ncclWork));

    { // Check whether the last operation was aborted and make sure all threads exit
      int aborted = tid == 0 ? *comm->abortFlag : 0;
      if (barrierReduceAny(aborted)) // publish ncclShmem.work
        break;
    }
  }
}

// Only generate kernels for SUM
#if NCCL_OP == 0
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
__global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)( \
    struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead \
  ) { \
  ncclKernel<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto, fIndex> \
    (comm, channelMask, workHead); \
}
#else
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fInded)
#endif

// Examples :     AllReduce, RING, LL,    Sum,   uint8
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__ void NCCL_FUNC_NAME(func, algo, proto, devredop, type)() { \
  RunWork<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto>().run(&ncclShmem.work); \
}

// Only generate inline kernels for LL
#define IMPL_COLL4(func, algo, devredop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     devredop, type) \
  IMPL_COLL_FUNC(func, algo, LL128,  devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type) \
  IMPL_COLL_KERN(func, algo, LL,     devredop, type, FUNC_INDEX(ncclFunc##func, ncclDev##devredop, ncclType, NCCL_ALGO_##algo, NCCL_PROTO_LL)) \

#define IMPL_COLL3(func, devredop, type, ncclType) \
  IMPL_COLL4(func, TREE,    devredop, type, ncclType) \
  IMPL_COLL4(func, RING,    devredop, type, ncclType) \
  IMPL_COLL4(func, COLLNET_DIRECT, devredop, type, ncclType) \
  IMPL_COLL4(func, COLLNET_CHAIN, devredop, type, ncclType)

#if NCCL_TYPE == 0
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int8_t,   ncclInt8)
#elif NCCL_TYPE == 1
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint8_t,  ncclUint8)
#elif NCCL_TYPE == 2
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int32_t,  ncclInt32)
#elif NCCL_TYPE == 3
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint32_t, ncclUint32)
#elif NCCL_TYPE == 4
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int64_t,  ncclInt64)
#elif NCCL_TYPE == 5
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint64_t, ncclUint64)
#elif NCCL_TYPE == 6
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, half,     ncclFloat16)
#elif NCCL_TYPE == 7
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, float,    ncclFloat32)
#elif NCCL_TYPE == 8
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, double,   ncclFloat64)
#elif NCCL_TYPE == 9 && defined(__CUDA_BF16_TYPES_EXIST__)
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, __nv_bfloat16, ncclBfloat16)
#endif

// Reduction define all functions
#if NCCL_OP == 0
#define IMPL_COLL_R(func) IMPL_COLL2(func, Sum);
#elif NCCL_OP == 1
#define IMPL_COLL_R(func) IMPL_COLL2(func, Prod);
#elif NCCL_OP == 2
#define IMPL_COLL_R(func) IMPL_COLL2(func, Min);
#elif NCCL_OP == 3
#define IMPL_COLL_R(func) IMPL_COLL2(func, Max);
#elif NCCL_OP == 4
#define IMPL_COLL_R(func) IMPL_COLL2(func, PreMulSum);
#elif NCCL_OP == 5
  #if NCCL_TYPE < 6
    #define IMPL_COLL_R(func) IMPL_COLL2(func, SumPostDiv);
  #else
    #define IMPL_COLL_R(func) // skip SumPostDiv for floating point
  #endif
#endif

#if NCCL_OP == 0 && NCCL_TYPE == 0
// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE, Sum, int8_t, FUNC_INDEX_P2P);
#else
#define IMPL_COLL_C(func)
#define IMPL_COLL_P(func)
#endif

#endif
