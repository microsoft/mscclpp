#ifndef MSCCLPP_HOST_CONNECTION_HPP_
#define MSCCLPP_HOST_CONNECTION_HPP_

#include "comm.h"
#include "mscclpp.h"
#include "mscclpp.hpp"

namespace mscclpp {

struct HostConnection::Impl
{
  Communicator* comm;
  mscclppConn* conn;
  mscclppHostConn_t* hostConn;

  Impl(Communicator* comm, mscclppConn* conn);

  ~Impl();
};

} // namespace mscclpp

#endif // MSCCLPP_HOST_CONNECTION_HPP_