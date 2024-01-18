#include "endpoint.hpp"

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>

#include "api.h"
#include "context.hpp"
#include "utils_internal.hpp"

namespace mscclpp {

Endpoint::Impl::Impl(EndpointConfig config, Context::Impl& contextImpl)
    : transport_(config.transport), hostHash_(getHostHash()) {
  if (AllIBTransports.has(transport_)) {
    ibLocal_ = true;
    ibQp_ = contextImpl.getIbContext(transport_)
                ->createQp(config.ibMaxCqSize, config.ibMaxCqPollNum, config.ibMaxSendWr, 0, config.ibMaxWrPerSend);
    ibQpInfo_ = ibQp_->getInfo();
  }

  if (AllNvlsTransports.has(transport_)) {
    minMcGran_ = 0;
    mcGran_ = 0;
    mcProp_ = {};
    mcProp_.size = config.nvlsBufferSize;
    mcProp_.numDevices = config.nvlsNumDevices;
    mcProp_.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    MSCCLPP_CUTHROW(cuMulticastGetGranularity(&minMcGran_, &mcProp_, CU_MULTICAST_GRANULARITY_MINIMUM));
    MSCCLPP_CUTHROW(cuMulticastGetGranularity(&mcGran_, &mcProp_, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    mcProp_.size = ((mcProp_.size + mcGran_ - 1) / mcGran_) * mcGran_;
    printf("---> %ld %ld | %lld %lld\n", mcProp_.size, mcProp_.numDevices, mcGran_, minMcGran_);
    // create the mc handle now only on the root
    if (transport_ == Transport::NvlsRoot) {
      MSCCLPP_CUTHROW(cuMulticastCreate(&mcHandle_, &mcProp_));

      mcFileDesc_ = 0;
      MSCCLPP_CUTHROW(
          cuMemExportToShareableHandle(&mcFileDesc_, mcHandle_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));
      rootPid_ = getpid();
      printf("LLLLLLL %lld %lld\n", mcFileDesc_, rootPid_);
    }
  }
}

MSCCLPP_API_CPP Transport Endpoint::transport() { return pimpl_->transport_; }

MSCCLPP_API_CPP std::vector<char> Endpoint::serialize() {
  std::vector<char> data;
  std::copy_n(reinterpret_cast<char*>(&pimpl_->transport_), sizeof(pimpl_->transport_), std::back_inserter(data));
  std::copy_n(reinterpret_cast<char*>(&pimpl_->hostHash_), sizeof(pimpl_->hostHash_), std::back_inserter(data));
  if (AllIBTransports.has(pimpl_->transport_)) {
    std::copy_n(reinterpret_cast<char*>(&pimpl_->ibQpInfo_), sizeof(pimpl_->ibQpInfo_), std::back_inserter(data));
  }

  if (pimpl_->transport_ == Transport::NvlsRoot) {
    std::copy_n(reinterpret_cast<char*>(&pimpl_->mcFileDesc_), sizeof(pimpl_->mcFileDesc_), std::back_inserter(data));
    std::copy_n(reinterpret_cast<char*>(&pimpl_->rootPid_), sizeof(pimpl_->rootPid_), std::back_inserter(data));
  }
  return data;
}

MSCCLPP_API_CPP Endpoint Endpoint::deserialize(const std::vector<char>& data) {
  return Endpoint(std::make_shared<Impl>(data));
}

Endpoint::Impl::Impl(const std::vector<char>& serialization) {
  auto it = serialization.begin();
  std::copy_n(it, sizeof(transport_), reinterpret_cast<char*>(&transport_));
  it += sizeof(transport_);
  std::copy_n(it, sizeof(hostHash_), reinterpret_cast<char*>(&hostHash_));
  it += sizeof(hostHash_);
  if (AllIBTransports.has(transport_)) {
    ibLocal_ = false;
    std::copy_n(it, sizeof(ibQpInfo_), reinterpret_cast<char*>(&ibQpInfo_));
    it += sizeof(ibQpInfo_);
  }
  if (transport_ == Transport::NvlsRoot) {
    mcFileDesc_ = 0;
    std::copy_n(it, sizeof(mcFileDesc_), reinterpret_cast<char*>(&mcFileDesc_));
    it += sizeof(mcFileDesc_);
    std::copy_n(it, sizeof(rootPid_), reinterpret_cast<char*>(&rootPid_));
    it += sizeof(rootPid_);
    int rootPidFd = syscall(SYS_pidfd_open, rootPid_, 0);
    int mcRootFileDescFd = syscall(SYS_pidfd_getfd, rootPidFd, mcFileDesc_, 0);
    printf("==========> %lld %lld %lld\n", rootPidFd, mcRootFileDescFd, mcFileDesc_);
    MSCCLPP_CUTHROW(
        cuMemImportFromShareableHandle(&mcHandle_, (void*)mcRootFileDescFd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    // close(rootPidFd);
  }
}

MSCCLPP_API_CPP Endpoint::Endpoint(std::shared_ptr<mscclpp::Endpoint::Impl> pimpl) : pimpl_(pimpl) {}

}  // namespace mscclpp