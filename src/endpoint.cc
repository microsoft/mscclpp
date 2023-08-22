#include "endpoint.hpp"

#include <algorithm>

#include "api.h"
#include "context.hpp"

namespace mscclpp {

Endpoint::Impl::Impl(Transport transport, int ibMaxCqSize, int ibMaxCqPollNum, int ibMaxSendWr, int ibMaxWrPerSend,
                     Context::Impl& contextImpl)
    : transport_(transport), hostHash_(contextImpl.hostHash_) {
  if (AllIBTransports.has(transport)) {
    ibLocal_ = true;
    ibQp_ = contextImpl.getIbContext(transport)->createQp(ibMaxCqSize, ibMaxCqPollNum, ibMaxSendWr, 0, ibMaxWrPerSend);
    ibQpInfo_ = ibQp_->getInfo();
  }
}

MSCCLPP_API_CPP Transport Endpoint::transport() { return pimpl->transport_; }

MSCCLPP_API_CPP std::vector<char> Endpoint::serialize() {
  std::vector<char> data;
  std::copy_n(reinterpret_cast<char*>(&pimpl->transport_), sizeof(pimpl->transport_), std::back_inserter(data));
  if (AllIBTransports.has(pimpl->transport_)) {
    std::copy_n(reinterpret_cast<char*>(&pimpl->ibQpInfo_), sizeof(pimpl->ibQpInfo_), std::back_inserter(data));
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
  if (AllIBTransports.has(transport_)) {
    ibLocal_ = false;
    std::copy_n(it, sizeof(ibQpInfo_), reinterpret_cast<char*>(&ibQpInfo_));
    it += sizeof(ibQpInfo_);
  }
}

MSCCLPP_API_CPP Endpoint::Endpoint(std::shared_ptr<mscclpp::Endpoint::Impl> pimpl) : pimpl(pimpl) {}

}  // namespace mscclpp