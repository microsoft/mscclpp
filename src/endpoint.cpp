#include "endpoint.hpp"

#include "context.hpp"

namespace mscclpp {

Endpoint::Impl::Impl(Transport transport, int ibMaxCqSize, int ibMaxCqPollNum, int ibMaxSendWr, int ibMaxWrPerSend, Context::Impl& contextImpl) : transport_(transport), rank_(contextImpl.rank_), hostHash_(contextImpl.hostHash_) {
  if (AllIBTransports.has(transport)) {
    ibLocal_ = true;
    ibQp_ = contextImpl.getIbContext(transport)->createQp(maxCqSize, maxCqPollNum, maxSendWr, 0, maxWrPerSend);
    ibQpInfo_ = ibQp_->getInfo();
  }
}

Transport Endpoint::transport() {
  return impl_->transport_;
}

std::vector<char> Endpoint::serialize() {
  std::vector<char> data;
  std::copy_n(reinterpret_cast<char*>(&pimpl->transport_), sizeof(pimpl->transport_), std::back_inserter(data));
  if (AllIBTransports.has(pimpl->transport_)) {
    std::copy_n(reinterpret_cast<char*>(&pimpl->ibQpInfo_), sizeof(pimpl->ibQpInfo_), std::back_inserter(data));
  }
}

static Endpoint Endpoint::deserialize(const std::vector<char>& data) {
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

} // namespace mscclpp