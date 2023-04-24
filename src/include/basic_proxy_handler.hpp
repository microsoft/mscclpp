#ifndef MSCCLPP_BASIC_PROXY_SERVICE_HPP_
#define MSCCLPP_BASIC_PROXY_SERVICE_HPP_

#include "mscclpp.hpp"
#include "communicator.hpp"

namespace mscclpp {

ProxyHandler makeBasicProxyHandler(Communicator::Impl &comm);

}

#endif