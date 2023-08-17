// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/core.hpp>

namespace nb = nanobind;
using namespace mscclpp;

extern void register_error(nb::module_& m);
extern void register_proxy_channel(nb::module_& m);
extern void register_sm_channel(nb::module_& m);
extern void register_fifo(nb::module_& m);
extern void register_semaphore(nb::module_& m);
extern void register_utils(nb::module_& m);
extern void register_numa(nb::module_& m);

template <typename T>
void def_nonblocking_future(nb::handle& m, const std::string& typestr) {
  std::string pyclass_name = std::string("NonblockingFuture") + typestr;
  nb::class_<NonblockingFuture<T>>(m, pyclass_name.c_str())
      .def("ready", &NonblockingFuture<T>::ready)
      .def("get", &NonblockingFuture<T>::get);
}

void register_core(nb::module_& m) {
  nb::class_<Bootstrap>(m, "Bootstrap")
      .def("get_rank", &Bootstrap::getRank)
      .def("get_n_ranks", &Bootstrap::getNranks)
      .def(
          "send",
          [](Bootstrap* self, uintptr_t ptr, size_t size, int peer, int tag) {
            void* data = reinterpret_cast<void*>(ptr);
            self->send(data, size, peer, tag);
          },
          nb::arg("data"), nb::arg("size"), nb::arg("peer"), nb::arg("tag"))
      .def(
          "recv",
          [](Bootstrap* self, uintptr_t ptr, size_t size, int peer, int tag) {
            void* data = reinterpret_cast<void*>(ptr);
            self->recv(data, size, peer, tag);
          },
          nb::arg("data"), nb::arg("size"), nb::arg("peer"), nb::arg("tag"))
      .def("all_gather", &Bootstrap::allGather, nb::arg("allData"), nb::arg("size"))
      .def("barrier", &Bootstrap::barrier)
      .def("send", (void (Bootstrap::*)(const std::vector<char>&, int, int)) & Bootstrap::send, nb::arg("data"),
           nb::arg("peer"), nb::arg("tag"))
      .def("recv", (void (Bootstrap::*)(std::vector<char>&, int, int)) & Bootstrap::recv, nb::arg("data"),
           nb::arg("peer"), nb::arg("tag"));

  nb::class_<UniqueId>(m, "UniqueId");

  nb::class_<TcpBootstrap, Bootstrap>(m, "TcpBootstrap")
      .def(nb::init<int, int>(), "Do not use this constructor. Use create instead.")
      .def_static(
          "create", [](int rank, int nRanks) { return std::make_shared<TcpBootstrap>(rank, nRanks); }, nb::arg("rank"),
          nb::arg("nRanks"))
      .def("create_unique_id", &TcpBootstrap::createUniqueId)
      .def("get_unique_id", &TcpBootstrap::getUniqueId)
      .def("initialize", (void (TcpBootstrap::*)(UniqueId, int64_t)) & TcpBootstrap::initialize, nb::arg("uniqueId"),
           nb::arg("timeoutSec") = 30)
      .def("initialize", (void (TcpBootstrap::*)(const std::string&, int64_t)) & TcpBootstrap::initialize,
           nb::arg("ifIpPortTrio"), nb::arg("timeoutSec") = 30);

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
      .def(nb::init_implicit<Transport>(), nb::arg("transport"))
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
      .def("write", &Connection::write, nb::arg("dst"), nb::arg("dstOffset"), nb::arg("src"), nb::arg("srcOffset"),
           nb::arg("size"))
      .def(
          "update_and_sync",
          [](Connection* self, RegisteredMemory dst, uint64_t dstOffset, uintptr_t src, uint64_t newValue) {
            self->updateAndSync(dst, dstOffset, (uint64_t*)src, newValue);
          },
          nb::arg("dst"), nb::arg("dstOffset"), nb::arg("src"), nb::arg("newValue"))
      .def("flush", &Connection::flush, nb::arg("timeoutUsec") = (int64_t)3e7)
      .def("remote_rank", &Connection::remoteRank)
      .def("tag", &Connection::tag)
      .def("transport", &Connection::transport)
      .def("remote_transport", &Connection::remoteTransport);

  def_nonblocking_future<RegisteredMemory>(m, "RegisteredMemory");

  nb::class_<Communicator>(m, "Communicator")
      .def(nb::init<std::shared_ptr<Bootstrap>>(), nb::arg("bootstrap"))
      .def("bootstrap", &Communicator::bootstrap)
      .def(
          "register_memory",
          [](Communicator* self, uintptr_t ptr, size_t size, TransportFlags transports) {
            return self->registerMemory((void*)ptr, size, transports);
          },
          nb::arg("ptr"), nb::arg("size"), nb::arg("transports"))
      .def("send_memory_on_setup", &Communicator::sendMemoryOnSetup, nb::arg("memory"), nb::arg("remoteRank"),
           nb::arg("tag"))
      .def("recv_memory_on_setup", &Communicator::recvMemoryOnSetup, nb::arg("remoteRank"), nb::arg("tag"))
      .def("connect_on_setup", &Communicator::connectOnSetup, nb::arg("remoteRank"), nb::arg("tag"),
           nb::arg("transport"), nb::arg("ibMaxCqSize") = 1024, nb::arg("ibMaxCqPollNum") = 1,
           nb::arg("ibMaxSendWr") = 8192, nb::arg("ibMaxWrPerSend") = 64)
      .def("setup", &Communicator::setup);
}

NB_MODULE(_mscclpp, m) {
  register_error(m);
  register_proxy_channel(m);
  register_sm_channel(m);
  register_fifo(m);
  register_semaphore(m);
  register_utils(m);
  register_core(m);
  register_numa(m);
}
