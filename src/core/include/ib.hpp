// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_IB_HPP_
#define MSCCLPP_IB_HPP_

#include <list>
#include <memory>
#include <mscclpp/core.hpp>
#include <string>

#include "rdma_provider.hpp"

// Forward declarations of IB structures
struct ibv_context;
struct ibv_pd;
struct ibv_mr;
struct ibv_qp;
struct ibv_cq;
struct ibv_wc;
struct ibv_send_wr;
struct ibv_recv_wr;
struct ibv_sge;

namespace mscclpp {

struct TransportInfo;
struct GpuNetIoDevice;
class GpuNetIoQp;

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
  IbMr(ibv_pd* pd, void* buff, std::size_t size, bool isDataDirect);

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
  bool isGrh;
};

enum class WsStatus {
  Success,
};

class IbQp {
 public:
  ~IbQp();

  void rtr(const IbQpInfo& info);
  void rts();
  void stageSendWrite(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                      uint64_t dstOffset, bool signaled);
  void stageSendAtomicAdd(const IbMr* mr, const IbMrInfo& info, uint64_t wrId, uint64_t dstOffset, uint64_t addVal,
                          bool signaled);
  void stageSendWriteWithImm(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                             uint64_t dstOffset, bool signaled, unsigned int immData);
  void postSend();

  void stageRecv(uint64_t wrId);
  void stageRecv(const IbMr* mr, uint64_t wrId, uint32_t size, uint64_t offset = 0);
  void postRecv();

  int pollSendCq();
  int pollRecvCq();

  IbQpInfo& getInfo() { return info_; }
  int getSendWcStatus(int idx) const;
  std::string getSendWcStatusString(int idx) const;
  int getNumSendCqItems() const;
  int getRecvWcStatus(int idx) const;
  std::string getRecvWcStatusString(int idx) const;
  unsigned int getRecvWcImmData(int idx) const;

 private:
  struct SendWrInfo {
    ibv_send_wr* wr;
    ibv_sge* sge;
  };

  struct RecvWrInfo {
    ibv_recv_wr* wr;
    ibv_sge* sge;
  };

  IbQp(ibv_context* ctx, ibv_pd* pd, int portNum, int gidIndex, int maxSendCqSize, int maxSendCqPollNum, int maxSendWr,
       int maxRecvWr, int maxWrPerSend, bool noAtomic);
  SendWrInfo getNewSendWrInfo();
  RecvWrInfo getNewRecvWrInfo();

  int portNum_;
  int gidIndex_;

  IbQpInfo info_;

  ibv_qp* qp_;
  ibv_cq* sendCq_;
  ibv_cq* recvCq_;
  std::shared_ptr<std::vector<ibv_wc>> sendWcs_;
  std::shared_ptr<std::vector<ibv_wc>> recvWcs_;
  std::shared_ptr<std::vector<ibv_send_wr>> sendWrs_;
  std::shared_ptr<std::vector<ibv_sge>> sendSges_;
  std::shared_ptr<std::vector<ibv_recv_wr>> recvWrs_;
  std::shared_ptr<std::vector<ibv_sge>> recvSges_;
  int numStagedSend_;
  int numStagedRecv_;
  int numPostedSignaledSend_;
  int numStagedSignaledSend_;

  const int maxSendCqSize_;
  const int maxSendCqPollNum_;
  const int maxSendWr_;
  const int maxWrPerSend_;
  const int maxRecvWr_;
  const bool noAtomic_;

  friend class IbCtx;
};

class IbCtx {
 public:
#if defined(USE_IBVERBS)
  IbCtx(const std::string& devName);
  ~IbCtx();

  std::shared_ptr<IbQp> createQp(int port, int gidIndex, int maxSendCqSize, int maxSendCqPollNum, int maxSendWr,
                                 int maxRecvWr, int maxWrPerSend, bool noAtomic);
  std::unique_ptr<const IbMr> registerMr(void* buff, std::size_t size);
  bool supportsRdmaAtomics() const;
  bool isMlx5() const;
  bool isDataDirect() const;
  bool isVirtualFunction() const;

  /// Resolve `port` (or pick any usable port if it is -1) and return the local port/GID addressing info that a
  /// remote QP needs to connect to a QP on this port. `qpn` is left as 0.
  IbQpInfo getPortInfo(int& port, int gidIndex) const;
  ibv_pd* getPd() const { return pd_; }

  /// DOCA GPUNetIO handles opened on this device's protection domain, created on first use. Requires GPUNetIO.
  GpuNetIoDevice& gpuNetIoDevice();
#else
  IbCtx([[maybe_unused]] const std::string& devName) {}
  ~IbCtx() {}

  std::shared_ptr<IbQp> createQp(int, int, int, int, int, int, int, bool) { return nullptr; }
  std::unique_ptr<const IbMr> registerMr([[maybe_unused]] void* buff, [[maybe_unused]] std::size_t size) {
    return nullptr;
  }
  bool supportsRdmaAtomics() const { return false; }
  bool isMlx5() const { return false; }
  bool isDataDirect() const { return false; }
  bool isVirtualFunction() const { return false; }
  IbQpInfo getPortInfo(int&, int) const { return {}; }
  ibv_pd* getPd() const { return nullptr; }
#endif

  const std::string& getDevName() const { return devName_; };

 private:
  bool isPortUsable(int port, int gidIndex) const;
  int getAnyUsablePort(int gidIndex) const;

  const std::string devName_;
  ibv_context* ctx_;
  ibv_pd* pd_;
  bool supportsRdmaAtomics_;
  bool isMlx5_;
  bool isDataDirect_;
  bool isVF_;
#if defined(USE_IBVERBS)
  std::unique_ptr<GpuNetIoDevice> gpuNetIoDevice_;
#endif
};

class IbMemory final : public RdmaMemory {
 public:
  IbMemory(std::shared_ptr<RdmaProvider> provider, std::unique_ptr<const IbMr> mr);
  explicit IbMemory(const IbMrInfo& info);
  bool isLocal() const override { return mr_ != nullptr; }
  void serialize(std::vector<char>& data) const override;
  const IbMr* mr() const { return mr_.get(); }
  const IbMrInfo& info() const { return info_; }

 private:
  std::unique_ptr<const IbMr> mr_;
  IbMrInfo info_;
};

class IbEndpoint final : public RdmaEndpoint {
 public:
  IbEndpoint(std::shared_ptr<RdmaProvider> provider, IbCtx& context, EndpointConfig& config);
  IbEndpoint(const IbQpInfo& info, bool noAtomic);
  bool isLocal() const override { return qp_ != nullptr || gpuNetIoQp_ != nullptr; }
  void serialize(std::vector<char>& data) const override;
  std::shared_ptr<BaseConnection> createConnection(std::shared_ptr<Context> context, const Endpoint& local,
                                                   const Endpoint& remote) const override;
  const std::shared_ptr<IbQp>& qp() const { return qp_; }
  const std::shared_ptr<GpuNetIoQp>& gpuNetIoQp() const { return gpuNetIoQp_; }
  const IbQpInfo& info() const { return info_; }
  bool noAtomic() const { return noAtomic_; }
  const IbCtx& context() const { return *context_; }

 private:
  IbCtx* context_ = nullptr;
  std::shared_ptr<IbQp> qp_;
  std::shared_ptr<GpuNetIoQp> gpuNetIoQp_;
  IbQpInfo info_;
  bool noAtomic_;
};

class IbProvider final : public RdmaProvider {
 public:
  explicit IbProvider(Transport transport);
  std::shared_ptr<RdmaEndpoint> createEndpoint(EndpointConfig& config) override;
  std::shared_ptr<RdmaMemory> registerMemory(void* data, size_t size) override;

 private:
  IbCtx context_;
};

const IbEndpoint& getIbEndpoint(const RdmaEndpoint& endpoint);
const IbMemory& getIbMemory(const TransportInfo& info);
std::shared_ptr<RdmaEndpoint> deserializeIbEndpoint(std::vector<char>::const_iterator& it);
std::shared_ptr<RdmaMemory> deserializeIbMemory(std::vector<char>::const_iterator& it);

}  // namespace mscclpp

#endif  // MSCCLPP_IB_HPP_
