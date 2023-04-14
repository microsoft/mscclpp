#ifndef MSCCLPP_HOST_CONNECTION_HPP_
#define MSCCLPP_HOST_CONNECTION_HPP_

#include "mscclpp.hpp"
#include "mscclpp.h"

namespace mscclpp {

struct HostConnection::Impl {
  mscclppHostConn_t* hostConn;

  Impl();

  ~Impl();

  void setup(mscclppHostConn_t *hostConn);
};

} // namespace mscclpp

#endif // MSCCLPP_HOST_CONNECTION_HPP_