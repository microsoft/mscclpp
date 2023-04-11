#ifndef MSCCLPP_IB_H_
#define MSCCLPP_IB_H_

#include "mscclpp.h"
#include <infiniband/verbs.h>
#include <list>
#include <memory>
#include <string>

#define MSCCLPP_IB_CQ_SIZE 1024
#define MSCCLPP_IB_CQ_POLL_NUM 4
#define MSCCLPP_IB_MAX_SENDS 64
#define MSCCLPP_IB_MAX_DEVS 8

// QP info to be shared with the remote peer
struct mscclppIbQpInfo
{
  uint16_t lid;
  uint8_t port;
  uint8_t linkLayer;
  uint32_t qpn;
  uint64_t spn;
  ibv_mtu mtu;
};

// IB queue pair
struct mscclppIbQp
{
  struct ibv_qp* qp;
  struct mscclppIbQpInfo info;
  struct ibv_send_wr* wrs;
  struct ibv_sge* sges;
  struct ibv_cq* cq;
  struct ibv_wc* wcs;
  int wrn;

  int rtr(const mscclppIbQpInfo* info);
  int rts();
  int stageSend(struct mscclppIbMr* ibMr, const mscclppIbMrInfo* info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                uint64_t dstOffset, bool signaled);
  int stageSendWithImm(struct mscclppIbMr* ibMr, const mscclppIbMrInfo* info, uint32_t size, uint64_t wrId,
                       uint64_t srcOffset, uint64_t dstOffset, bool signaled, unsigned int immData);
  int postSend();
  int postRecv(uint64_t wrId);
  int pollCq();
};

// Holds resources of a single IB device.
struct mscclppIbContext
{
  struct ibv_context* ctx;
  struct ibv_pd* pd;
  int* ports;
  int nPorts;
  struct mscclppIbQp* qps;
  int nQps;
  int maxQps;
  struct mscclppIbMr* mrs;
  int nMrs;
  int maxMrs;
};

mscclppResult_t mscclppIbContextCreate(struct mscclppIbContext** ctx, const char* ibDevName);
mscclppResult_t mscclppIbContextDestroy(struct mscclppIbContext* ctx);
mscclppResult_t mscclppIbContextCreateQp(struct mscclppIbContext* ctx, struct mscclppIbQp** ibQp, int port = -1);
mscclppResult_t mscclppIbContextRegisterMr(struct mscclppIbContext* ctx, void* buff, size_t size,
                                           struct mscclppIbMr** ibMr);
mscclppResult_t mscclppIbContextRegisterMr2(struct mscclppIbContext* ctx, void* buff, size_t size,
                                           struct mscclppIbMr* ibMr);

#endif
