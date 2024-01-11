#include "endpoint.hpp"

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
    mcProp_.size = config.nvlsBufferSize;
    mcProp_.handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    MSCCLPP_CUTHROW(cuMulticastGetGranularity(&minMcGran_, &config.mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));
    MSCCLPP_CUTHROW(cuMulticastGetGranularity(&mcGran_, &config.mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    mcProp_.size = ((mcProp_.size + mcGran_ - 1) / mcGran_) * mcGran_;
    // create the mc handle now only on the root
    if (transport_ == Transport::NvlsRoot){
      MSCCLPP_CUTHROW(cuMulticastCreate(&mcHandle_, &mcProp_));

      fileDesc_ = 0;
      MSCCLPP_CUTHROW(cuMemExportToShareableHandle(&fileDesc_, handle, handleType, 0 /*flags*/));
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

  if (transport_ == Transport::NvlsRoot) {
    std::copy_n(reinterpret_cast<char*>(&pimpl_->fileDesc_), sizeof(pimpl_->fileDesc_), std::back_inserter(data));
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
  if (transport_ == Transport::NvlsNonRoot) {
    fileDesc_ = 0;
    std::copy_n(it, sizeof(fileDesc_), reinterpret_cast<char*>(&fileDesc_));
    it += sizeof(fileDesc_);
    MSCCLPP_CUTHROW(cuMemImportFromShareableHandle(&mcHandle_, (void*)fileDesc_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  }
}

MSCCLPP_API_CPP Endpoint::Endpoint(std::shared_ptr<mscclpp::Endpoint::Impl> pimpl) : pimpl_(pimpl) {}

}  // namespace mscclpp