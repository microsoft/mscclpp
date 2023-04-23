#ifndef MSCCLPP_IB_HPP_
#define MSCCLPP_IB_HPP_

#include <memory>
#include <string>
#include <list>

namespace mscclpp {

// QP info to be shared with the remote peer
struct IbQpInfo
{
  uint16_t lid;
  uint8_t port;
  uint8_t linkLayer;
  uint32_t qpn;
  uint64_t spn;
  uint32_t mtu;
};

class IbQp
{
public:
  ~IbQp();

  IbQpInfo info;

private:
  IbQp(void* ctx, void* pd, int port);

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
  IbCtx(const std::string& ibDevName);
  ~IbCtx();

  IbQp* createQp(int port = -1);

private:
  bool IbCtx::isPortUsable(int port) const;
  int IbCtx::getAnyActivePort() const;

  void* ctx;
  void* pd;
  std::list<std::unique_ptr<IbQp>> qps;
};

} // namespace mscclpp

#endif // MSCCLPP_IB_HPP_
