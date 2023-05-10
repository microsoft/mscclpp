#ifndef MSCCLPP_BASIC_PROXY_SERVICE_HPP_
#define MSCCLPP_BASIC_PROXY_SERVICE_HPP_

#include "communicator.hpp"
#include <mscclpp/core.hpp>

namespace mscclpp {

ProxyHandler makeBasicProxyHandler(Communicator::Impl& comm);

}

#endif