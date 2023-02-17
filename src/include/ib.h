#ifndef MSCCLPP_IB_H_
#define MSCCLPP_IB_H_

#include "mscclpp.h"
#include <list>
#include <memory>
#include <string>
#include <infiniband/verbs.h>

#define MSCCLPP_IB_CQ_SIZE 1024
#define MSCCLPP_IB_CQ_POLL_NUM 4
#define MSCCLPP_IB_MAX_SENDS 64
#define MSCCLPP_IB_MAX_DEVS 8

// MR info to be shared with the remote peer
struct mscclppIbMrInfo {
  uint64_t addr;
  uint32_t rkey;
};

// IB memory region
struct mscclppIbMr {
  struct ibv_mr *mr;
  void *buff;
  struct mscclppIbMrInfo info;
};

// QP info to be shared with the remote peer
struct mscclppIbQpInfo {
  uint16_t lid;
  uint8_t port;
  uint8_t linkLayer;
  uint32_t qpn;
  uint64_t spn;
  int mtu;
};

// IB queue pair
struct mscclppIbQp {
  struct ibv_qp *qp;
  struct mscclppIbQpInfo info;
  struct ibv_send_wr *wrs;
  struct ibv_sge *sges;
  int wrn;

  int rtr(const mscclppIbQpInfo *info);
  int rts();
  int stageSend(struct mscclppIbMr *ibMr, const mscclppIbMrInfo *info, int size,
                uint64_t wrId, unsigned int immData, int offset = 0);
  int postSend();
  int postRecv(uint64_t wrId);
};

// Holds resources of a single IB device.
struct mscclppIbContext {
  int numa_node;
  struct ibv_context *ctx;
  struct ibv_cq *cq;
  struct ibv_pd *pd;
  struct ibv_wc *wcs;
  int wcn;
  int *ports;
  int nPorts;
  struct mscclppIbQp *qps;
  int nQps;
  int maxQps;
  struct mscclppIbMr *mrs;
  int nMrs;
  int maxMrs;
};

mscclppResult_t mscclppIbContextCreate(struct mscclppIbContext **ctx, const char *ibDevName);
mscclppResult_t mscclppIbContextDestroy(struct mscclppIbContext *ctx);
mscclppResult_t mscclppIbContextCreateQp(struct mscclppIbContext *ctx, struct mscclppIbQp **ibQp, int port = -1);
mscclppResult_t mscclppIbContextRegisterMr(struct mscclppIbContext *ctx, void *buff, size_t size, struct mscclppIbMr **ibMr);
mscclppResult_t mscclppIbContextPollCq(struct mscclppIbContext *ctx, int *wcNum);

#endif
