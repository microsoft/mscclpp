#ifndef MSCCLPP_IB_HPP_
#define MSCCLPP_IB_HPP_

#include <list>
#include <memory>
#include <string>

#define MSCCLPP_IB_CQ_SIZE 1024
#define MSCCLPP_IB_CQ_POLL_NUM 1
#define MSCCLPP_IB_MAX_SENDS 64
#define MSCCLPP_IB_MAX_DEVS 8

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
  ~IbMr();

  IbMrInfo getInfo() const;
  const void* getBuff() const;
  uint32_t getLkey() const;

 private:
  IbMr(ibv_pd* pd, void* buff, std::size_t size);

  ibv_mr* mr;
  void* buff;
  std::size_t size;

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
  int stageSend(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                uint64_t dstOffset, bool signaled);
  int stageSendAtomic(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                uint64_t dstOffset, bool signaled);
  int stageSendWithImm(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                       uint64_t dstOffset, bool signaled, unsigned int immData);
  void postSend();
  void postRecv(uint64_t wrId);
  int pollCq();

  IbQpInfo& getInfo();
  const ibv_wc* getWc(int idx) const;

 private:
  IbQp(ibv_context* ctx, ibv_pd* pd, int port);

  IbQpInfo info;

  ibv_qp* qp;
  ibv_cq* cq;
  std::unique_ptr<ibv_wc[]> wcs;
  std::unique_ptr<ibv_send_wr[]> wrs;
  std::unique_ptr<ibv_sge[]> sges;
  int wrn;

  friend class IbCtx;
};

class IbCtx {
 public:
  IbCtx(const std::string& devName);
  ~IbCtx();

  IbQp* createQp(int port = -1);
  const IbMr* registerMr(void* buff, std::size_t size);

  const std::string& getDevName() const;

 private:
  bool isPortUsable(int port) const;
  int getAnyActivePort() const;

  const std::string devName;
  ibv_context* ctx;
  ibv_pd* pd;
  std::list<std::unique_ptr<IbQp>> qps;
  std::list<std::unique_ptr<IbMr>> mrs;
};

}  // namespace mscclpp

#endif  // MSCCLPP_IB_HPP_
