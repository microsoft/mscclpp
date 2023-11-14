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
}

MSCCLPP_API_CPP Transport Endpoint::transport() const { return pimpl_->transport_; }

MSCCLPP_API_CPP std::vector<char> Endpoint::serialize() const {
  std::vector<char> data;
  std::copy_n(reinterpret_cast<char*>(&pimpl_->transport_), sizeof(pimpl_->transport_), std::back_inserter(data));
  std::copy_n(reinterpret_cast<char*>(&pimpl_->hostHash_), sizeof(pimpl_->hostHash_), std::back_inserter(data));
  if (AllIBTransports.has(pimpl_->transport_)) {
    std::copy_n(reinterpret_cast<char*>(&pimpl_->ibQpInfo_), sizeof(pimpl_->ibQpInfo_), std::back_inserter(data));
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
}

MSCCLPP_API_CPP Endpoint::Endpoint(std::shared_ptr<mscclpp::Endpoint::Impl> pimpl) : pimpl_(pimpl) {}

}  // namespace mscclpp