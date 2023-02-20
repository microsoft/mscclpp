// #include "reduce_kernel.h" // for reduction funcs
#ifndef PRIMS_LL_H_
#define PRIMS_LL_H_
union ncclLLFifoLine {
    /* Flags have to be *after* data, because otherwise, an incomplete receive
       from the network may receive the flag but not the data.
       Note this is assuming that either we receive contiguous chunks of data
       (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
    struct {
        uint32_t data1;
        uint32_t flag1;
        uint32_t data2;
        uint32_t flag2;
    };
    uint64_t v[2];
    int4 i4;
};
#define NCCL_STEPS 8
template <typename T> class Primitives_LL
{
public:
    // In the case of Fan::MaxRecv == 0, we need to force MaxRecv to 1 for this
    // to compile This is because of a recv buffer which is allocated to MaxRecv
    // length in send-only cases
    // static constexpr int MaxRecv = 1;
    // static constexpr int MaxSend = 1;
    static constexpr int Input = 0, Output = 1;
    uint64_t redOp;
    const int tid;
    const int nthreads;
    // const int wid;
    const int group;
    const int stepLines;
    // Fan fan;
    T *data_src;
    T *data_dst;
    T *userBufs[2];
    volatile uint64_t *recvConnHeadPtr = NULL;
    uint64_t recvConnHead;

    volatile uint64_t *sendConnHeadPtr = NULL;
    uint64_t sendConnHead;

    uint64_t recvStep;
    uint64_t sendStep;
    union ncclLLFifoLine *recvBuff;
    union ncclLLFifoLine *sendBuff;

    inline __device__ int recvOffset(int i)
    {
        return (recvStep % NCCL_STEPS) * stepLines;
    }
    inline __device__ int sendOffset(int i)
    {
        return (sendStep % NCCL_STEPS) * stepLines;
    }
    inline __device__ union ncclLLFifoLine *recvPtr(int i)
    {
        return recvBuff + recvOffset(i);
    }
    inline __device__ union ncclLLFifoLine *sendPtr(int i)
    {
        return sendBuff + sendOffset(i);
    }
    inline __device__ uint32_t recvFlag(int i)
    {
        return (uint32_t)(recvStep + 1);
    }
    inline __device__ uint32_t sendFlag(int i)
    {
        return (uint32_t)(sendStep + 1);
    }

    inline __device__ void barrier()
    {
        constexpr int WARP_SIZE = 32;
        if (nthreads == WARP_SIZE)
            __syncwarp();
        else
            asm volatile("bar.sync %1, %0;" ::"r"(nthreads), "r"(15 - group));
    }

    inline __device__ void waitSend()
    {
        uint64_t sendConnHeadCache = *sendConnHeadPtr; // Cache last seen value
        while (sendConnHeadCache < sendConnHead) {
            sendConnHeadCache = *sendConnHeadPtr;
        }
        printf("sendConnHeadCache: %d", sendConnHeadCache);
        sendConnHead += 1;
        barrier();
    }

    inline __device__ void postRecv()
    {
        barrier();
        recvConnHead += 1;
        *recvConnHeadPtr = recvConnHead;
    }

    __device__ uint64_t readLL(union ncclLLFifoLine *src_, int offset,
                               uint32_t flag)
    {
        union ncclLLFifoLine *src = src_ + offset;
        uint32_t data1, flag1, data2, flag2;
        int spins = 0;
        do {
            asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                         : "l"(&src->i4));
            // if (checkAbort(spins, 0)) break;
        } while ((flag1 != flag) || (flag2 != flag));
        uint64_t val64 = data1 + (((uint64_t)data2) << 32);
        src->i4 = make_int4(0, 0, 0, 0);
        return val64;
    }

    __device__ void storeLL(union ncclLLFifoLine *dst, uint64_t val,
                            uint32_t flag)
    {
        asm volatile(
            "st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(&dst->i4),
            "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)),
            "r"(flag));
    }

    static constexpr int EltPerLine = sizeof(uint64_t) / sizeof(T);

    template <typename U> __device__ static U load(U *src)
    {
        union {
            U elt;
            uint16_t u2;
            uint32_t u4;
            uint64_t u8;
        };
        if (sizeof(U) == 1)
            asm("ld.volatile.global.b8 %0,[%1];" : "=r"(u4) : "l"(src));
        else if (sizeof(U) == 2)
            asm("ld.volatile.global.b16 %0,[%1];" : "=h"(u2) : "l"(src));
        else if (sizeof(U) == 4)
            asm("ld.volatile.global.b32 %0,[%1];" : "=r"(u4) : "l"(src));
        else
            asm("ld.volatile.global.b64 %0,[%1];" : "=l"(u8) : "l"(src));
        return elt;
    }

    template <typename U> __device__ static void store(U *dst, U val)
    {
        union {
            U elt;
            uint16_t u2;
            uint32_t u4;
            uint64_t u8;
        };
        elt = val;
        if (sizeof(U) == 1)
            asm("st.volatile.global.b8 [%0],%1;" ::"l"(dst), "r"(u4));
        else if (sizeof(U) == 2)
            asm("st.volatile.global.b16 [%0],%1;" ::"l"(dst), "h"(u2));
        else if (sizeof(U) == 4)
            asm("st.volatile.global.b32 [%0],%1;" ::"l"(dst), "r"(u4));
        else
            asm("st.volatile.global.b64 [%0],%1;" ::"l"(dst), "l"(u8));
    }

    struct DataLoader {
        int misalign;
        union {
            uint32_t u4[sizeof(T) <= 2 ? 3 : 2];
            uint64_t u8;
            T elt[EltPerLine];
        };

        __device__ void loadBegin(T *src, int eltN)
        {
            if (sizeof(T) <= 2) {
                misalign = reinterpret_cast<uintptr_t>(src) % 4;
                uint32_t *p = reinterpret_cast<uint32_t *>(
                    reinterpret_cast<uintptr_t>(src) & -uintptr_t(4));
                u4[0] = load(p + 0);
                u4[1] = misalign + eltN * sizeof(T) > 4 ? load(p + 1) : 0;
                // u4[2] would be simpler, but that throws warnings on some
                // compilers
                u4[sizeof(T) <= 2 ? 2 : 0] =
                    misalign + eltN * sizeof(T) > 8 ? load(p + 2) : 0;
            } else {
#pragma unroll
                for (int i = 0; i < EltPerLine; i++) {
                    if (i == 0 || i < eltN)
                        elt[i] = load(src + i);
                }
            }
        }

        __device__ uint64_t loadFinish()
        {
            if (sizeof(T) <= 2) {
                u4[0] = __funnelshift_r(u4[0], u4[1], 8 * misalign);
                // u4[2] would be simpler, but that throws warnings on some
                // compilers
                u4[1] = __funnelshift_r(u4[1], u4[sizeof(T) <= 2 ? 2 : 0],
                                        8 * misalign);
            }
            return u8;
        }
    };

    __device__ void storeData(T *dst, uint64_t val, int eltN)
    {
        union {
            uint64_t u8;
            T elt[EltPerLine];
        };
        u8 = val;
#pragma unroll
        for (int i = 0; i < EltPerLine; i++) {
            if (i == 0 || i < eltN)
                // store(dst+i, elt[i]);
                dst[i] = elt[i];
        }
    }

    union converter {
        uint64_t storage;
        struct {
            float a, b;
        };
    };

    __device__ uint64_t floatsum(const uint64_t x, const uint64_t y)
    {
        converter cx, cy, cr;
        cx.storage = x;
        cy.storage = y;

        cr.a = cx.a + cy.a;
        cr.b = cx.b + cy.b;

        return cr.storage;
    }

    template <int RECV, int SEND, int SrcBuf, int DstBuf>
    __device__ __forceinline__ void LLGenericOp(intptr_t srcIx, intptr_t dstIx,
                                                int nelem, bool postOp)
    {
        constexpr int SRC = SrcBuf != -1 ? 1 : 0;
        constexpr int DST = DstBuf != -1 ? 1 : 0;
        T *srcElts = SrcBuf == -1 ? nullptr : data_src + srcIx;
        T *dstElts = DstBuf == -1 ? nullptr : data_dst + dstIx;
        int offset = tid;

        // Always waitSend in case of cleanup
        // nelem = nelem < 0 ? 0 : nelem;
        if (SEND)
            waitSend();
        nelem -= tid * EltPerLine;
        srcElts += tid * EltPerLine;
        dstElts += tid * EltPerLine;
        int eltPerTrip = nthreads * EltPerLine;
        while (nelem > 0) {
            int eltInLine = EltPerLine < nelem ? EltPerLine : nelem;

            DataLoader dl;
            // ncclLLFifoLine line[MaxRecv];
            uint64_t data, peerData;
            if (SRC) {
                dl.loadBegin(srcElts, eltInLine);
                srcElts += eltPerTrip;
            }
            if (RECV) {
                printf("readLLBeginAll");
                // readLLBeginAll<1>(offset, line);
                peerData = readLL(recvBuff, offset, 1);
                printf("readLLBegindone");
            }
            if (SRC) {
                data = dl.loadFinish();
                // if (SrcBuf == Input)
                //     data = MULTI<RedOp, T>().preOp(redOp, data);
            }
            if (RECV) {
                data = !SRC ? peerData : floatsum(peerData, data);
            }

            // if (postOp)
            //     data = MULTI<RedOp, T>().postOp(redOp, data);

            if (SEND) {
                printf("sendBuff = %p\n", sendBuff);
                storeLL(sendBuff + offset, data, 1);
            }
            if (DST) {
                storeData(dstElts, data, eltInLine);
                dstElts += eltPerTrip;
            }
            nelem -= eltPerTrip;
            offset += nthreads;
        }

        if (RECV) {
            recvStep++;
            postRecv();
        }
        if (SEND) {
            sendStep++;
        }
    }

    __device__ Primitives_LL(const int tid, const int nthreads,
                             uint64_t redOpArg, int group, const int stepLines)
        : redOp(redOpArg), tid(tid), nthreads(nthreads),
          group(group & (uint16_t)0xFFFF), stepLines(stepLines)
    {
        sendConnHead = 0;
        recvConnHead = 0;
        sendStep = 0;
        recvStep = 0;
    }

    __device__ void send(intptr_t inpIx, int eltN)
    {
        return LLGenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
    }
    __device__ void sendFromOutput(intptr_t outIx, int eltN)
    {
        return LLGenericOp<0, 1, Output, -1>(outIx, -1, eltN, false);
    }
    __device__ void recv(intptr_t outIx, int eltN, bool postOp = false)
    {
        return LLGenericOp<1, 0, -1, Output>(-1, outIx, eltN, postOp);
    }
    __device__ void recvReduceSend(intptr_t inpIx, int eltN)
    {
        return LLGenericOp<1, 1, Input, -1>(inpIx, -1, eltN, false);
    }
    __device__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN,
                                   bool postOp = false)
    {
        return LLGenericOp<1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
    }
    __device__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN,
                             bool postOp = false)
    {
        return LLGenericOp<0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
    }
    __device__ void recvCopySend(intptr_t outIx, int eltN, bool postOp = false)
    {
        return LLGenericOp<1, 1, -1, Output>(-1, outIx, eltN, postOp);
    }
    __device__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN,
                                       bool postOp = false)
    {
        return LLGenericOp<1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
    }
};

#endif