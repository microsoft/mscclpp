// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <mscclpp/core.hpp>

namespace nb = nanobind;
using namespace mscclpp;

template<typename T>
void def_nonblocking_future(nb::handle &m, const std::string &typestr) {
    std::string pyclass_name = std::string("NonblockingFuture") + typestr;
    nb::class_<NonblockingFuture<T>>(m, pyclass_name.c_str())
      .def("ready", &NonblockingFuture<T>::ready)
      .def("get", &NonblockingFuture<T>::get);
}

NB_MODULE(mscclpp, m) {
  nb::class_<Bootstrap>(m, "Bootstrap")
    .def("getRank", &Bootstrap::getRank)
    .def("getNranks", &Bootstrap::getNranks)
    .def("send", (void (Bootstrap::*)(void*, int, int, int)) &Bootstrap::send, nb::arg("data"), nb::arg("size"), nb::arg("peer"), nb::arg("tag"))
    .def("recv", (void (Bootstrap::*)(void*, int, int, int)) &Bootstrap::recv, nb::arg("data"), nb::arg("size"), nb::arg("peer"), nb::arg("tag"))
    .def("allGather", &Bootstrap::allGather, nb::arg("allData"), nb::arg("size"))
    .def("barrier", &Bootstrap::barrier)
    .def("send", (void (Bootstrap::*)(const std::vector<char>&, int, int)) &Bootstrap::send, nb::arg("data"), nb::arg("peer"), nb::arg("tag"))
    .def("recv", (void (Bootstrap::*)(std::vector<char>&, int, int)) &Bootstrap::recv, nb::arg("data"), nb::arg("peer"), nb::arg("tag"));

  nb::class_<UniqueId>(m, "UniqueId");

  nb::class_<TcpBootstrap, Bootstrap>(m, "TcpBootstrap")
    .def(nb::init<int, int>(), nb::arg("rank"), nb::arg("nRanks"))
    .def("createUniqueId", &TcpBootstrap::createUniqueId)
    .def("getUniqueId", &TcpBootstrap::getUniqueId)
    .def("initialize", (void (TcpBootstrap::*)(UniqueId)) &TcpBootstrap::initialize, nb::arg("uniqueId"))
    .def("initialize", (void (TcpBootstrap::*)(std::string)) &TcpBootstrap::initialize, nb::arg("ifIpPortTrio"));

  nb::enum_<Transport>(m, "Transport")
    .value("Unknown", Transport::Unknown)
    .value("CudaIpc", Transport::CudaIpc)
    .value("IB0", Transport::IB0)
    .value("IB1", Transport::IB1)
    .value("IB2", Transport::IB2)
    .value("IB3", Transport::IB3)
    .value("IB4", Transport::IB4)
    .value("IB5", Transport::IB5)
    .value("IB6", Transport::IB6)
    .value("IB7", Transport::IB7)
    .value("NumTransports", Transport::NumTransports);

  nb::class_<TransportFlags>(m, "TransportFlags")
    .def(nb::init<>())
    .def(nb::init<Transport>(), nb::arg("transport"))
    .def("has", &TransportFlags::has, nb::arg("transport"))
    .def("none", &TransportFlags::none)
    .def("any", &TransportFlags::any)
    .def("all", &TransportFlags::all)
    .def("count", &TransportFlags::count)
    .def(nb::self |= nb::self)
    .def(nb::self | nb::self)
    .def(nb::self | Transport())
    .def(nb::self &= nb::self)
    .def(nb::self & nb::self)
    .def(nb::self & Transport())
    .def(nb::self ^= nb::self)
    .def(nb::self ^ nb::self)
    .def(nb::self ^ Transport())
    .def(~nb::self)
    .def(nb::self == nb::self)
    .def(nb::self != nb::self);

  nb::class_<RegisteredMemory>(m, "RegisteredMemory")
    .def(nb::init<>())
    .def("data", &RegisteredMemory::data)
    .def("size", &RegisteredMemory::size)
    .def("rank", &RegisteredMemory::rank)
    .def("transports", &RegisteredMemory::transports)
    .def("serialize", &RegisteredMemory::serialize)
    .def_static("deserialize", &RegisteredMemory::deserialize, nb::arg("data"));

  nb::class_<Connection>(m, "Connection")
    .def("write", &Connection::write, nb::arg("dst"), nb::arg("dstOffset"), nb::arg("src"), nb::arg("srcOffset"), nb::arg("size"))
    .def("updateAndSync", &Connection::updateAndSync, nb::arg("dst"), nb::arg("dstOffset"), nb::arg("src"), nb::arg("newValue"))
    .def("flush", &Connection::flush)
    .def("remoteRank", &Connection::remoteRank)
    .def("tag", &Connection::tag)
    .def("transport", &Connection::transport)
    .def("remoteTransport", &Connection::remoteTransport);

  def_nonblocking_future<RegisteredMemory>(m, "RegisteredMemory");

  nb::class_<Communicator>(m, "Communicator")
    .def(nb::init<std::shared_ptr<Bootstrap>>(), nb::arg("bootstrap"))
    .def("bootstrap", &Communicator::bootstrap)
    .def("registerMemory", &Communicator::registerMemory, nb::arg("ptr"), nb::arg("size"), nb::arg("transports"))
    .def("sendMemoryOnSetup", &Communicator::sendMemoryOnSetup, nb::arg("memory"), nb::arg("remoteRank"), nb::arg("tag"))
    .def("recvMemoryOnSetup", &Communicator::recvMemoryOnSetup, nb::arg("remoteRank"), nb::arg("tag"))
    .def("connectOnSetup", &Communicator::connectOnSetup, nb::arg("remoteRank"), nb::arg("tag"), nb::arg("transport"))
    .def("setup", &Communicator::setup);
}
