// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_MRC_HPP_
#define MSCCLPP_MRC_HPP_

#if defined(MSCCLPP_USE_MRC)

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "connection.hpp"
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

// The logical extent comes from RegisteredMemory, not from the native MR wire record.
struct MrcBuffer {
  MrcMrInfo info;
  uint64_t size;
};

class MrcMr {
 public:
  ~MrcMr();
  MrcMr(const MrcMr&) = delete;
  MrcMr& operator=(const MrcMr&) = delete;

  MrcBuffer getBuffer() const;

 private:
  MrcMr(ibv_pd* pd, void* buff, std::size_t size, const std::string& dataDirectPath);

  ibv_mr* mr_;
  void* buff_;
  std::size_t size_;

  friend class MrcCtx;
  friend class MrcQp;
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

class MrcSendProgress {
 public:
  void check() const;
  int pendingSignaledSends() const;
  bool drained() const;
  void posted(int result, int staged, int signaled);
  void polled(int count, const ibv_wc* wcs);

 private:
  std::exception_ptr failure_;
  int pending_ = 0;
  bool hasUnsignaled_ = false;
};

class MrcRecvProgress {
 public:
  void check() const;
  int pendingRecvs() const;
  void posted(int result, int staged);
  void polled(int count, const ibv_wc* wcs);
  const ibv_wc& completion(int idx, const ibv_wc* wcs) const;

 private:
  std::exception_ptr failure_;
  int pending_ = 0;
  int numWcs_ = 0;
};

class MrcQp {
 public:
  ~MrcQp();
  MrcQp(const MrcQp&) = delete;
  MrcQp& operator=(const MrcQp&) = delete;

  void rtr(const MrcQpInfo& info);
  void rts();
  void stageSendWrite(const MrcMr* mr, const MrcBuffer& dst, uint64_t size, uint64_t wrId, uint64_t srcOffset,
                      uint64_t dstOffset, bool signaled);
  void stageSendWriteWithImm(const MrcMr* mr, const MrcBuffer& dst, uint64_t size, uint64_t wrId, uint64_t srcOffset,
                             uint64_t dstOffset, bool signaled, unsigned int immData);
  void postSend();
  void stageRecv(uint64_t wrId);
  void postRecv();
  int pollSendCq();
  // Poll at most once. A checked drain requires every posted send on this QP to be signaled.
  bool progressSend();
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
  std::pair<uint64_t, uint64_t> checkedWriteAddresses(const MrcMr* mr, const MrcBuffer& dst, uint64_t size,
                                                      uint64_t srcOffset, uint64_t dstOffset) const;
  ibv_send_wr& stageSend(const MrcMr* mr, const MrcBuffer& dst, uint64_t size, uint64_t wrId, uint64_t srcOffset,
                         uint64_t dstOffset, bool signaled);

  ibv_pd* pd_;
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
  MrcSendProgress sendProgress_;
  int numSendWcs_;
  int numStagedRecv_;
  MrcRecvProgress recvProgress_;
  const int maxSendCqSize_;
  const int maxSendWr_;
  const int maxRecvWr_;

  // Keep the source registrations through a checked drain or QP destruction, even if the Connection goes away.
  std::vector<std::shared_ptr<RdmaMemory>> writeSources_;

  friend class MrcCtx;
  friend class MrcConnection;
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
  MrcMemory(const MrcMrInfo& info, size_t size);
  bool isLocal() const override { return mr_ != nullptr; }
  uint64_t address() const { return buffer_.info.addr; }
  void serialize(std::vector<char>& data) const override;
  const MrcMr* mr() const { return mr_.get(); }
  const MrcBuffer& buffer() const { return buffer_; }

 private:
  std::unique_ptr<const MrcMr> mr_;
  MrcBuffer buffer_;
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

// Internal data path only; MrcEndpoint::createConnection stays disabled until signaling and teardown are ready.
class MrcConnection final : public BaseConnection {
 public:
  MrcConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint, const Endpoint& remoteEndpoint);
  Transport transport() const override { return localEndpoint_.transport(); }
  Transport remoteTransport() const override { return remoteEndpoint_.transport(); }
  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;
  void accumulate(RegisteredMemory dst, uint64_t dstOffset, int64_t value) override;
  void flush(int64_t timeoutUsec = -1) override;
  void requestFlush() override;
  bool progressFlush() override;

 private:
  Endpoint remoteEndpoint_;
  std::shared_ptr<MrcQp> qp_;
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
