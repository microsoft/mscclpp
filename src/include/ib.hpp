#ifndef MSCCLPP_IB_HPP_
#define MSCCLPP_IB_HPP_

#include <memory>
#include <string>
#include <list>

#define MSCCLPP_IB_CQ_SIZE 1024
#define MSCCLPP_IB_CQ_POLL_NUM 1
#define MSCCLPP_IB_MAX_SENDS 64
#define MSCCLPP_IB_MAX_DEVS 8

namespace mscclpp {

struct IbMrInfo
{
  uint64_t addr;
  uint32_t rkey;
};

class IbMr
{
public:
  ~IbMr();

  IbMrInfo getInfo() const;
  const void* getBuff() const;
  uint32_t getLkey() const;

private:
  IbMr(void* pd, void* buff, std::size_t size);

  void* mr;
  void* buff;
  std::size_t size;

  friend class IbCtx;
};

// QP info to be shared with the remote peer
struct IbQpInfo
{
  uint16_t lid;
  uint8_t port;
  uint8_t linkLayer;
  uint32_t qpn;
  uint64_t spn;
  int mtu;
};

class IbQp
{
public:
  ~IbQp();

  void rtr(const IbQpInfo& info);
  void rts();
  int stageSend(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled);
  int stageSendWithImm(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled, unsigned int immData);
  void postSend();
  void postRecv(uint64_t wrId);
  int pollCq();

  IbQpInfo& getInfo();
  const void* getWc(int idx) const;

private:
  IbQp(void* ctx, void* pd, int port);

  IbQpInfo info;

  void* qp;
  void* cq;
  void* wcs;
  void* wrs;
  void* sges;
  int wrn;

  friend class IbCtx;
};

class IbCtx
{
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
  void* ctx;
  void* pd;
  std::list<std::unique_ptr<IbQp>> qps;
  std::list<std::unique_ptr<IbMr>> mrs;
};

} // namespace mscclpp

#endif // MSCCLPP_IB_HPP_
