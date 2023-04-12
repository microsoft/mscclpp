#include <cassert>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <unistd.h>
#include <vector>

#include "alloc.h"
#include "comm.h"
#include "debug.h"
#include "ib.h"

static int getIbDevNumaNode(const char* ibDevPath)
{
  if (ibDevPath == NULL) {
    WARN("ibDevPath is NULL");
    return -1;
  }
  const char* postfix = "/device/numa_node";
  FILE* fp = NULL;
  char* filePath = NULL;
  int node = -1;
  int res;
  if (mscclppCalloc(&filePath, strlen(ibDevPath) + strlen(postfix) + 1) != mscclppSuccess) {
    WARN("mscclppCalloc failed");
    goto exit;
  }
  memcpy(filePath, ibDevPath, strlen(ibDevPath) * sizeof(char));
  filePath[strlen(ibDevPath)] = '\0';
  if (strncat(filePath, postfix, strlen(postfix)) == NULL) {
    WARN("strncat failed");
    goto exit;
  }
  fp = fopen(filePath, "r");
  if (fp == NULL) {
    WARN("fopen failed (errno %d, path %s)", errno, filePath);
    goto exit;
  }
  res = fscanf(fp, "%d", &node);
  if (res != 1) {
    WARN("fscanf failed (errno %d, path %s)", errno, filePath);
    node = -1;
    goto exit;
  }
exit:
  if (filePath != NULL) {
    free(filePath);
  }
  if (fp != NULL) {
    fclose(fp);
  }
  return node;
}

mscclppResult_t mscclppIbContextCreate(struct mscclppIbContext** ctx, const char* ibDevName)
{
  struct mscclppIbContext* _ctx;
  MSCCLPPCHECK(mscclppCalloc(&_ctx, 1));

  std::vector<int> ports;

  int num;
  struct ibv_device** devices = ibv_get_device_list(&num);
  for (int i = 0; i < num; ++i) {
    if (strncmp(devices[i]->name, ibDevName, IBV_SYSFS_NAME_MAX) == 0) {
      _ctx->ctx = ibv_open_device(devices[i]);
      break;
    }
  }
  ibv_free_device_list(devices);
  if (_ctx->ctx == nullptr) {
    WARN("ibv_open_device failed (errno %d, device name %s)", errno, ibDevName);
    goto fail;
  }

  // Check available ports
  struct ibv_device_attr devAttr;
  if (ibv_query_device(_ctx->ctx, &devAttr) != 0) {
    WARN("ibv_query_device failed (errno %d, device name %s)", errno, ibDevName);
    goto fail;
  }

  for (uint8_t i = 1; i <= devAttr.phys_port_cnt; ++i) {
    struct ibv_port_attr portAttr;
    if (ibv_query_port(_ctx->ctx, i, &portAttr) != 0) {
      WARN("ibv_query_port failed (errno %d, port %d)", errno, i);
      goto fail;
    }
    if (portAttr.state != IBV_PORT_ACTIVE) {
      continue;
    }
    if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) {
      continue;
    }
    ports.push_back((int)i);
  }
  if (ports.size() == 0) {
    WARN("no active IB port found");
    goto fail;
  }
  MSCCLPPCHECK(mscclppCalloc(&_ctx->ports, ports.size()));
  _ctx->nPorts = (int)ports.size();
  for (int i = 0; i < _ctx->nPorts; ++i) {
    _ctx->ports[i] = ports[i];
  }

  _ctx->pd = ibv_alloc_pd(_ctx->ctx);
  if (_ctx->pd == NULL) {
    WARN("ibv_alloc_pd failed (errno %d)", errno);
    goto fail;
  }

  *ctx = _ctx;
  return mscclppSuccess;
fail:
  *ctx = NULL;
  if (_ctx->ports != NULL) {
    free(_ctx->ports);
  }
  free(_ctx);
  return mscclppInternalError;
}

mscclppResult_t mscclppIbContextDestroy(struct mscclppIbContext* ctx)
{
  for (int i = 0; i < ctx->nMrs; ++i) {
    if (ctx->mrs[i].mr) {
      ibv_dereg_mr(ctx->mrs[i].mr);
    }
  }
  for (int i = 0; i < ctx->nQps; ++i) {
    if (ctx->qps[i].qp) {
      ibv_destroy_qp(ctx->qps[i].qp);
    }
    ibv_destroy_cq(ctx->qps[i].cq);
    free(ctx->qps[i].wcs);
    free(ctx->qps[i].sges);
    free(ctx->qps[i].wrs);
  }
  if (ctx->pd != NULL) {
    ibv_dealloc_pd(ctx->pd);
  }
  if (ctx->ctx != NULL) {
    ibv_close_device(ctx->ctx);
  }
  free(ctx->mrs);
  free(ctx->qps);
  free(ctx->ports);
  free(ctx);
  return mscclppSuccess;
}

mscclppResult_t mscclppIbContextCreateQp(struct mscclppIbContext* ctx, struct mscclppIbQp** ibQp, int port /*=-1*/)
{
  if (port < 0) {
    port = ctx->ports[0];
  } else {
    bool found = false;
    for (int i = 0; i < ctx->nPorts; ++i) {
      if (ctx->ports[i] == port) {
        found = true;
        break;
      }
    }
    if (!found) {
      WARN("invalid IB port: %d", port);
      return mscclppInternalError;
    }
  }

  struct ibv_cq* cq = ibv_create_cq(ctx->ctx, MSCCLPP_IB_CQ_SIZE, NULL, NULL, 0);
  if (cq == NULL) {
    WARN("ibv_create_cq failed (errno %d)", errno);
    return mscclppInternalError;
  }

  struct ibv_qp_init_attr qp_init_attr;
  std::memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
  qp_init_attr.sq_sig_all = 0;
  qp_init_attr.send_cq = cq;
  qp_init_attr.recv_cq = cq;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.cap.max_send_wr = MAXCONNECTIONS * MSCCLPP_PROXY_FIFO_SIZE;
  qp_init_attr.cap.max_recv_wr = MAXCONNECTIONS * MSCCLPP_PROXY_FIFO_SIZE;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_init_attr.cap.max_inline_data = 0;
  struct ibv_qp* qp = ibv_create_qp(ctx->pd, &qp_init_attr);
  if (qp == nullptr) {
    WARN("ibv_create_qp failed (errno %d)", errno);
    return mscclppInternalError;
  }
  struct ibv_port_attr port_attr;
  if (ibv_query_port(ctx->ctx, port, &port_attr) != 0) {
    WARN("ibv_query_port failed (errno %d, port %d)", errno, port);
    return mscclppInternalError;
  }

  // Register QP to this ctx
  qp->context = ctx->ctx;
  if (qp->context == NULL) {
    WARN("IB context is NULL");
    return mscclppInternalError;
  }
  ctx->nQps++;
  if (ctx->qps == NULL) {
    MSCCLPPCHECK(mscclppCalloc(&ctx->qps, MAXCONNECTIONS));
    ctx->maxQps = MAXCONNECTIONS;
  }
  if (ctx->maxQps < ctx->nQps) {
    WARN("too many QPs");
    return mscclppInternalError;
  }
  struct mscclppIbQp* _ibQp = &ctx->qps[ctx->nQps - 1];
  _ibQp->qp = qp;
  _ibQp->info.lid = port_attr.lid;
  _ibQp->info.port = port;
  _ibQp->info.linkLayer = port_attr.link_layer;
  _ibQp->info.qpn = qp->qp_num;
  _ibQp->info.mtu = port_attr.active_mtu;
  _ibQp->info.is_global = (port_attr.flags & IBV_QPF_GRH_REQUIRED);
  if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND || _ibQp->info.is_global) {
    union ibv_gid gid;
    if (ibv_query_gid(ctx->ctx, port, 0, &gid) != 0) {
      WARN("ibv_query_gid failed (errno %d)", errno);
      return mscclppInternalError;
    }
    _ibQp->info.spn = gid.global.subnet_prefix;
    _ibQp->info.iid = gid.global.interface_id;
  }

  struct ibv_qp_attr qp_attr;
  std::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = port;
  qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
    WARN("ibv_modify_qp failed (errno %d)", errno);
    return mscclppInternalError;
  }

  MSCCLPPCHECK(mscclppCalloc(&_ibQp->wrs, MSCCLPP_IB_MAX_SENDS));
  MSCCLPPCHECK(mscclppCalloc(&_ibQp->sges, MSCCLPP_IB_MAX_SENDS));
  MSCCLPPCHECK(mscclppCalloc(&_ibQp->wcs, MSCCLPP_IB_CQ_POLL_NUM));
  _ibQp->cq = cq;

  *ibQp = _ibQp;

  return mscclppSuccess;
}

mscclppResult_t mscclppIbContextRegisterMr(struct mscclppIbContext* ctx, void* buff, size_t size,
                                           struct mscclppIbMr** ibMr)
{
  if (ctx->mrs == NULL) {
    MSCCLPPCHECK(mscclppCalloc(&ctx->mrs, MAXCONNECTIONS));
    ctx->maxMrs = MAXCONNECTIONS;
  }
  ctx->nMrs++;
  if (ctx->maxMrs < ctx->nMrs) {
    WARN("too many MRs");
    return mscclppInternalError;
  }
  struct mscclppIbMr* _ibMr = &ctx->mrs[ctx->nMrs - 1];
  *ibMr = _ibMr;
  return mscclppIbContextRegisterMr2(ctx, buff, size, _ibMr);
}


mscclppResult_t mscclppIbContextRegisterMr2(struct mscclppIbContext* ctx, void* buff, size_t size,
                                           struct mscclppIbMr* ibMr)
{
  if (size == 0) {
    WARN("invalid size: %zu", size);
    return mscclppInvalidArgument;
  }
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) {
    pageSize = sysconf(_SC_PAGESIZE);
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(buff) & -pageSize;
  size_t pages = (size + (reinterpret_cast<uintptr_t>(buff) - addr) + pageSize - 1) / pageSize;
  struct ibv_mr* mr =
    ibv_reg_mr(ctx->pd, reinterpret_cast<void*>(addr), pages * pageSize,
               IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
  if (mr == nullptr) {
    WARN("ibv_reg_mr failed (errno %d)", errno);
    return mscclppInternalError;
  }
  ibMr->mr = mr;
  ibMr->buff = buff;
  ibMr->info.addr = (uint64_t)buff;
  ibMr->info.rkey = mr->rkey;
  return mscclppSuccess;
}

//////////////////////////////////////////////////////////////////////////////

int mscclppIbQp::rtr(const mscclppIbQpInfo* info)
{
  struct ibv_qp_attr qp_attr;
  std::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.path_mtu = info->mtu;
  qp_attr.dest_qp_num = info->qpn;
  qp_attr.rq_psn = 0;
  qp_attr.max_dest_rd_atomic = 1;
  qp_attr.min_rnr_timer = 0x12;
  if (info->linkLayer == IBV_LINK_LAYER_ETHERNET || info->is_global) {
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
    qp_attr.ah_attr.grh.dgid.global.interface_id = info->iid;
    qp_attr.ah_attr.grh.flow_label = 0;
    qp_attr.ah_attr.grh.sgid_index = 0;
    qp_attr.ah_attr.grh.hop_limit = 255;
    qp_attr.ah_attr.grh.traffic_class = 0;
    qp_attr.ah_attr.dlid = info->lid;
  } else {
    qp_attr.ah_attr.is_global = 0;
    qp_attr.ah_attr.dlid = info->lid;
  }
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = info->port;
  return ibv_modify_qp(this->qp, &qp_attr,
                       IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                         IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

int mscclppIbQp::rts()
{
  struct ibv_qp_attr qp_attr;
  std::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.timeout = 18;
  qp_attr.retry_cnt = 7;
  qp_attr.rnr_retry = 7;
  qp_attr.sq_psn = 0;
  qp_attr.max_rd_atomic = 1;
  return ibv_modify_qp(this->qp, &qp_attr,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                         IBV_QP_MAX_QP_RD_ATOMIC);
}

int mscclppIbQp::stageSend(struct mscclppIbMr* ibMr, const mscclppIbMrInfo* info, uint32_t size, uint64_t wrId,
                           uint64_t srcOffset, uint64_t dstOffset, bool signaled)
{
  if (this->wrn >= MSCCLPP_IB_MAX_SENDS) {
    return -1;
  }
  int wrn = this->wrn;
  struct ibv_send_wr* wr_ = &this->wrs[wrn];
  struct ibv_sge* sge_ = &this->sges[wrn];
  // std::memset(wr_, 0, sizeof(struct ibv_send_wr));
  // std::memset(sge_, 0, sizeof(struct ibv_sge));
  wr_->wr_id = wrId;
  wr_->sg_list = sge_;
  wr_->num_sge = 1;
  wr_->opcode = IBV_WR_RDMA_WRITE;
  wr_->send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr_->wr.rdma.remote_addr = (uint64_t)(info->addr) + dstOffset;
  wr_->wr.rdma.rkey = info->rkey;
  wr_->next = nullptr;
  sge_->addr = (uint64_t)(ibMr->buff) + srcOffset;
  sge_->length = size;
  sge_->lkey = ibMr->mr->lkey;
  if (wrn > 0) {
    this->wrs[wrn - 1].next = wr_;
  }
  this->wrn++;
  return this->wrn;
}

int mscclppIbQp::stageSendWithImm(struct mscclppIbMr* ibMr, const mscclppIbMrInfo* info, uint32_t size, uint64_t wrId,
                                  uint64_t srcOffset, uint64_t dstOffset, bool signaled, unsigned int immData)
{
  int wrn = this->stageSend(ibMr, info, size, wrId, srcOffset, dstOffset, signaled);
  this->wrs[wrn - 1].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  this->wrs[wrn - 1].imm_data = immData;
  return wrn;
}

int mscclppIbQp::postSend()
{
  if (this->wrn == 0) {
    return 0;
  }

  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(this->qp, this->wrs, &bad_wr);
  if (ret != 0) {
    return ret;
  }
  this->wrn = 0;
  return 0;
}

int mscclppIbQp::postRecv(uint64_t wrId)
{
  struct ibv_recv_wr wr, *bad_wr;
  wr.wr_id = wrId;
  wr.sg_list = nullptr;
  wr.num_sge = 0;
  wr.next = nullptr;
  return ibv_post_recv(this->qp, &wr, &bad_wr);
}

int mscclppIbQp::pollCq()
{
  return ibv_poll_cq(this->cq, MSCCLPP_IB_CQ_POLL_NUM, this->wcs);
}
