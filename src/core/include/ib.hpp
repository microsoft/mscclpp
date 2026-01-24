// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_IB_HPP_
#define MSCCLPP_IB_HPP_

#include <list>
#include <memory>
#include <mscclpp/core.hpp>
#include <string>

// Forward declarations of IB structures
struct ibv_context;
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
  virtual ~IbMr();

  virtual IbMrInfo getInfo() const;
  virtual const void* getBuff() const;
  virtual uint32_t getLkey() const;

 private:
  IbMr(ibv_pd* pd, void* buff, std::size_t size);

  ibv_mr* mr_;
  void* buff_;
  std::size_t size_;

  friend class IbCtx;
};

// QP info to be shared with the remote peer
struct IbQpInfo {
  uint16_t lid;
  uint8_t linkLayer;
  uint32_t qpn;
  uint64_t spn;
  int mtu;
  uint64_t iid;
  bool is_grh;
};

enum class WsStatus {
  Success,
};

class IbQp {
 public:
  virtual ~IbQp();

  virtual void rtr([[maybe_unused]] const IbQpInfo& info);
  virtual void rts();
  virtual void stageSend([[maybe_unused]] const IbMr* mr, [[maybe_unused]] const IbMrInfo& info,
                         [[maybe_unused]] uint32_t size, [[maybe_unused]] uint64_t wrId,
                         [[maybe_unused]] uint64_t srcOffset, [[maybe_unused]] uint64_t dstOffset,
                         [[maybe_unused]] bool signaled);
  virtual void stageAtomicAdd([[maybe_unused]] const IbMr* mr, [[maybe_unused]] const IbMrInfo& info,
                              [[maybe_unused]] uint64_t wrId, [[maybe_unused]] uint64_t dstOffset,
                              [[maybe_unused]] uint64_t addVal, [[maybe_unused]] bool signaled);
  virtual void stageSendWithImm([[maybe_unused]] const IbMr* mr, [[maybe_unused]] const IbMrInfo& info,
                                [[maybe_unused]] uint32_t size, [[maybe_unused]] uint64_t wrId,
                                [[maybe_unused]] uint64_t srcOffset, [[maybe_unused]] uint64_t dstOffset,
                                [[maybe_unused]] bool signaled, [[maybe_unused]] unsigned int immData);
  virtual void postSend();
  virtual int pollCq();

  IbQpInfo& getInfo() { return info_; }
  virtual int getWcStatus([[maybe_unused]] int idx) const;
  virtual std::string getWcStatusString([[maybe_unused]] int idx) const;
  virtual int getNumCqItems() const;

 private:
  struct WrInfo {
    ibv_send_wr* wr;
    ibv_sge* sge;
  };

  IbQp(ibv_context* ctx, ibv_pd* pd, int portNum, int gidIndex, int maxCqSize, int maxCqPollNum, int maxSendWr,
       int maxRecvWr, int maxWrPerSend);
  WrInfo getNewWrInfo();

  int portNum_;
  int gidIndex_;

  IbQpInfo info_;

  ibv_qp* qp_;
  ibv_cq* cq_;
  std::shared_ptr<std::vector<ibv_wc>> wcs_;
  std::shared_ptr<std::vector<ibv_send_wr>> wrs_;
  std::shared_ptr<std::vector<ibv_sge>> sges_;
  int wrn_;
  int numSignaledPostedItems_;
  int numSignaledStagedItems_;

  const int maxCqPollNum_;
  const int maxWrPerSend_;

  friend class IbCtx;
};

class IbCtx {
 public:
#if defined(USE_IBVERBS)
  IbCtx(const std::string& devName);
  ~IbCtx();

  std::shared_ptr<IbQp> createQp(int port, int gidIndex, int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr,
                                 int maxWrPerSend);
  std::unique_ptr<const IbMr> registerMr(void* buff, std::size_t size);
#else
  IbCtx([[maybe_unused]] const std::string& devName) {}
  ~IbCtx() {}

  std::shared_ptr<IbQp> createQp(int, int, int, int, int, int, int) { return nullptr; }
  std::unique_ptr<const IbMr> registerMr([[maybe_unused]] void* buff, [[maybe_unused]] std::size_t size) {
    return nullptr;
  }
#endif

  const std::string& getDevName() const { return devName_; };

 private:
  bool isPortUsable(int port, int gidIndex) const;
  int getAnyUsablePort(int gidIndex) const;

  const std::string devName_;
  ibv_context* ctx_;
  ibv_pd* pd_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_IB_HPP_
