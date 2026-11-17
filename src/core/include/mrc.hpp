// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_MRC_HPP_
#define MSCCLPP_MRC_HPP_

#if defined(MSCCLPP_USE_MRC)

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "rdma_provider.hpp"

struct ibv_context;
struct ibv_pd;
struct ibv_mr;
struct ibv_wc;
struct ibv_send_wr;
struct ibv_recv_wr;
struct ibv_sge;
struct mrc_context;
struct mrc_cq;
struct mrc_qp;

namespace mscclpp {

enum class Transport;

std::string getMrcDeviceName(Transport transport);

struct MrcMrInfo {
  uint64_t addr;
  uint32_t rkey;
};

class MrcMr {
 public:
  ~MrcMr();
  MrcMr(const MrcMr&) = delete;
  MrcMr& operator=(const MrcMr&) = delete;

  MrcMrInfo getInfo() const;
  const void* getBuff() const;
  uint32_t getLkey() const;

 private:
  MrcMr(ibv_pd* pd, void* buff, std::size_t size, const std::string& dataDirectPath);

  ibv_mr* mr_;
  void* buff_;

  friend class MrcCtx;
};

// QP info to be shared with the remote peer.
struct MrcQpInfo {
  uint16_t lid;
  uint8_t linkLayer;
  uint32_t qpn;
  uint64_t spn;
  int mtu;
  uint64_t iid;
  bool isGrh;
};

class MrcQp {
 public:
  ~MrcQp();
  MrcQp(const MrcQp&) = delete;
  MrcQp& operator=(const MrcQp&) = delete;

  void rtr(const MrcQpInfo& info);
  void rts();
  void stageSendWrite(const MrcMr* mr, const MrcMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                      uint64_t dstOffset, bool signaled);
  void stageSendWriteWithImm(const MrcMr* mr, const MrcMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                             uint64_t dstOffset, bool signaled, unsigned int immData);
  void postSend();
  void stageRecv(uint64_t wrId);
  void postRecv();
  int pollSendCq();
  int pollRecvCq();

  MrcQpInfo& getInfo() { return info_; }
  int getSendWcStatus(int idx) const;
  std::string getSendWcStatusString(int idx) const;
  int getNumSendCqItems() const;
  int getRecvWcStatus(int idx) const;
  std::string getRecvWcStatusString(int idx) const;
  unsigned int getRecvWcImmData(int idx) const;

 private:
  MrcQp(ibv_context* context, mrc_context* mrcContext, ibv_pd* pd, int port, int gidIndex, uint64_t evProfileId,
        int maxSendCqSize, int maxSendWr, int maxRecvWr);
  ibv_send_wr& stageSend(const MrcMr* mr, const MrcMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                         uint64_t dstOffset, bool signaled);

  int portNum_;
  int gidIndex_;
  MrcQpInfo info_;

  mrc_cq* sendCq_;
  mrc_cq* recvCq_;
  mrc_qp* qp_;
  std::unique_ptr<ibv_wc[]> sendWcs_;
  std::unique_ptr<ibv_send_wr[]> sendWrs_;
  std::unique_ptr<ibv_sge[]> sendSges_;
  std::unique_ptr<ibv_wc[]> recvWcs_;
  std::unique_ptr<ibv_recv_wr[]> recvWrs_;
  int numStagedSend_;
  int numStagedSignaledSend_;
  int numPostedSignaledSend_;
  int numSendWcs_;
  bool sendFailed_;
  int numStagedRecv_;
  int numPostedRecv_;
  int numRecvWcs_;
  bool recvFailed_;
  const int maxSendCqSize_;
  const int maxSendWr_;
  const int maxRecvWr_;

  friend class MrcCtx;
};

class MrcCtx {
 public:
  explicit MrcCtx(const std::string& devName);
  ~MrcCtx();
  MrcCtx(const MrcCtx&) = delete;
  MrcCtx& operator=(const MrcCtx&) = delete;

  // The installed provider requires an explicit port for INIT, although mrc.h lists IBV_QP_PORT as unsupported.
  // The EV profile must already be configured by the controller.
  std::shared_ptr<MrcQp> createQp(int port, int gidIndex, uint64_t evProfileId, int maxSendCqSize, int maxSendWr,
                                  int maxRecvWr);
  std::unique_ptr<const MrcMr> registerMr(void* buff, std::size_t size);

  const std::string& getDevName() const { return devName_; }

 private:
  const std::string devName_;
  ibv_context* context_;
  ibv_pd* pd_;
  mrc_context* mrcContext_;
  std::string dataDirectPath_;
};

class MrcMemory final : public RdmaMemory {
 public:
  MrcMemory(std::shared_ptr<RdmaProvider> provider, std::unique_ptr<const MrcMr> mr);
  explicit MrcMemory(const MrcMrInfo& info);
  bool isLocal() const override { return mr_ != nullptr; }
  uint64_t address() const { return info_.addr; }
  void serialize(std::vector<char>& data) const override;
  const MrcMr* mr() const { return mr_.get(); }
  const MrcMrInfo& info() const { return info_; }

 private:
  std::unique_ptr<const MrcMr> mr_;
  MrcMrInfo info_;
};

class MrcEndpoint final : public RdmaEndpoint {
 public:
  MrcEndpoint(std::shared_ptr<RdmaProvider> provider, MrcCtx& context, EndpointConfig& config);
  explicit MrcEndpoint(const MrcQpInfo& info);
  bool isLocal() const override { return qp_ != nullptr; }
  void serialize(std::vector<char>& data) const override;
  void connect(const RdmaEndpoint& remote);
  std::shared_ptr<BaseConnection> createConnection(std::shared_ptr<Context> context, const Endpoint& local,
                                                   const Endpoint& remote) const override;
  const std::shared_ptr<MrcQp>& qp() const { return qp_; }

 private:
  std::shared_ptr<MrcQp> qp_;
  MrcQpInfo info_;
};

class MrcProvider final : public RdmaProvider {
 public:
  explicit MrcProvider(Transport transport);
  std::shared_ptr<RdmaEndpoint> createEndpoint(EndpointConfig& config) override;
  std::shared_ptr<RdmaMemory> registerMemory(void* data, size_t size) override;

 private:
  MrcCtx context_;
};

std::shared_ptr<RdmaEndpoint> deserializeMrcEndpoint(const EndpointConfig& config,
                                                     std::vector<char>::const_iterator& it,
                                                     std::vector<char>::const_iterator end);
void prepareMrcEndpointConfig(EndpointConfig& config);
std::shared_ptr<RdmaMemory> deserializeMrcMemory(Transport transport, std::vector<char>::const_iterator& it,
                                                 std::vector<char>::const_iterator end, void* originalData, size_t size,
                                                 TransportFlags advertised, TransportFlags& seen);

}  // namespace mscclpp

#endif  // defined(MSCCLPP_USE_MRC)
#endif  // MSCCLPP_MRC_HPP_
