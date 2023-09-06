// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_IB_HPP_
#define MSCCLPP_IB_HPP_

#include <list>
#include <memory>
#include <string>
#include <vector>

// Forward declarations of IB structures
struct ibv_context;
struct ibv_device_attr;
struct ibv_pd;
struct ibv_mr;
struct ibv_qp;
struct ibv_cq;
struct ibv_wc;
struct ibv_send_wr;
struct ibv_sge;

namespace mscclpp {

struct IbMrInfo {
  uint64_t addr;
  uint32_t rkey;
};

class IbMr {
 public:
  ~IbMr();

  IbMrInfo getInfo() const;
  const void* getBuff() const;
  uint32_t getLkey() const;

 private:
  IbMr(ibv_pd* pd, void* buff, size_t alignedSize);

  ibv_mr* mr;
  void* buff;
  size_t size;

  friend class IbCtx;
};

// QP info to be shared with the remote peer
struct IbQpInfo {
  uint16_t lid;
  uint8_t port;
  uint8_t linkLayer;
  uint32_t qpn;
  uint64_t spn;
  int mtu;
  uint64_t iid;
  bool is_grh;
};

class IbQp {
 public:
  ~IbQp();

  void rtr(const IbQpInfo& info);
  void rts();
  void stageSend(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint32_t srcOffset,
                 uint32_t dstOffset, bool signaled);
  void stageAtomicAdd(const IbMr* mr, const IbMrInfo& info, uint64_t wrId, uint32_t dstOffset, uint64_t addVal);
  void stageSendWithImm(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint32_t srcOffset,
                        uint32_t dstOffset, bool signaled, unsigned int immData);
  void stageSendGather(const std::vector<const IbMr*>& srcMrList, const IbMrInfo& dstMrInfo,
                       const std::vector<uint32_t>& srcSizeList, uint64_t wrId,
                       const std::vector<uint32_t>& srcOffsetList, uint32_t dstOffset, bool signaled);
  void postSend();
  void postRecv(uint64_t wrId);
  int pollCq();

  IbQpInfo& getInfo();
  const ibv_wc* getWc(int idx) const;

 private:
  struct WrInfo {
    ibv_send_wr* wr;
    ibv_sge* sge;
  };

  IbQp(ibv_context* ctx, ibv_pd* pd, int port, int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr,
       int maxWrPerSend, int maxNumSgesPerWr);
  WrInfo getNewWrInfo(int numSges);

  IbQpInfo info;

  ibv_qp* qp;
  ibv_cq* cq;
  std::unique_ptr<ibv_wc[]> wcs;
  std::unique_ptr<ibv_send_wr[]> wrs;
  std::unique_ptr<ibv_sge[]> sges;
  int numStagedWrs_;
  int numStagedSges_;

  const int maxCqPollNum_;
  const int maxWrPerSend_;
  const int maxNumSgesPerWr_;

  friend class IbCtx;
};

class IbCtx {
 public:
  IbCtx(const std::string& devName);
  ~IbCtx();

  IbQp* createQp(int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr, int maxWrPerSend, int maxNumSgesPerWr,
                 int port = -1);
  const IbMr* registerMr(void* buff, uint32_t size);

  const std::string& getDevName() const;

 private:
  bool isPortUsable(int port) const;
  int getAnyActivePort() const;
  void validateQpConfig(int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr, int maxWrPerSend,
                        int maxNumSgesPerWr, int port) const;

  const std::string devName;
  ibv_context* ctx;
  ibv_pd* pd;
  ibv_device_attr* devAttr;
  std::list<std::unique_ptr<IbQp>> qps;
  std::list<std::unique_ptr<IbMr>> mrs;
};

}  // namespace mscclpp

#endif  // MSCCLPP_IB_HPP_
