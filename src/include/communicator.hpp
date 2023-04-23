#ifndef MSCCL_COMMUNICATOR_HPP_
#define MSCCL_COMMUNICATOR_HPP_

#include "mscclpp.hpp"
#include "mscclpp.h"
#include "channel.hpp"
#include "proxy.hpp"

namespace mscclpp {

struct Communicator::Impl {
    mscclppComm_t comm;
    std::vector<std::shared_ptr<HostConnection>> connections;
    Proxy proxy;

    Impl();

    ~Impl();

    friend class HostConnection;
};

} // namespace mscclpp

#endif // MSCCL_COMMUNICATOR_HPP_
