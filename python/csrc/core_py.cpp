// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mscclpp/core.hpp>
#include <sstream>

namespace nb = nanobind;
using namespace mscclpp;

extern void register_env(nb::module_& m);
extern void register_error(nb::module_& m);
extern void register_port_channel(nb::module_& m);
extern void register_memory_channel(nb::module_& m);
extern void register_fifo(nb::module_& m);
extern void register_semaphore(nb::module_& m);
extern void register_utils(nb::module_& m);
extern void register_numa(nb::module_& m);
extern void register_nvls(nb::module_& m);
extern void register_executor(nb::module_& m);
extern void register_npkit(nb::module_& m);
extern void register_gpu_utils(nb::module_& m);
extern void register_algorithm(nb::module_& m);

// ext
extern void register_collective(nb::module_& m);

template <typename T>
void def_shared_future(nb::handle& m, const std::string& typestr) {
  std::string pyclass_name = std::string("shared_future_") + typestr;
  nb::class_<std::shared_future<T>>(m, pyclass_name.c_str()).def("get", &std::shared_future<T>::get);
}

void register_core(nb::module_& m) {
  m.def("version", &version);

  nb::enum_<DataType>(m, "DataType")
      .value("int32", DataType::INT32)
      .value("uint32", DataType::UINT32)
      .value("float16", DataType::FLOAT16)
      .value("float32", DataType::FLOAT32)
      .value("bfloat16", DataType::BFLOAT16);

  nb::class_<Bootstrap>(m, "Bootstrap")
      .def("get_rank", &Bootstrap::getRank)
      .def("get_n_ranks", &Bootstrap::getNranks)
      .def("get_n_ranks_per_node", &Bootstrap::getNranksPerNode)
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
      .def("send", static_cast<void (Bootstrap::*)(const std::vector<char>&, int, int)>(&Bootstrap::send),
           nb::arg("data"), nb::arg("peer"), nb::arg("tag"))
      .def("recv", static_cast<void (Bootstrap::*)(std::vector<char>&, int, int)>(&Bootstrap::recv), nb::arg("data"),
           nb::arg("peer"), nb::arg("tag"));

  nb::class_<UniqueId>(m, "UniqueId");

  nb::class_<TcpBootstrap, Bootstrap>(m, "TcpBootstrap")
      .def(nb::init<int, int>(), "Do not use this constructor. Use create instead.")
      .def_static(
          "create", [](int rank, int nRanks) { return std::make_shared<TcpBootstrap>(rank, nRanks); }, nb::arg("rank"),
          nb::arg("nRanks"))
      .def_static("create_unique_id", &TcpBootstrap::createUniqueId)
      .def("get_unique_id", &TcpBootstrap::getUniqueId)
      .def("initialize", static_cast<void (TcpBootstrap::*)(UniqueId, int64_t)>(&TcpBootstrap::initialize),
           nb::call_guard<nb::gil_scoped_release>(), nb::arg("unique_id"), nb::arg("timeout_sec") = 30)
      .def("initialize", static_cast<void (TcpBootstrap::*)(const std::string&, int64_t)>(&TcpBootstrap::initialize),
           nb::call_guard<nb::gil_scoped_release>(), nb::arg("if_ip_port_trio"), nb::arg("timeout_sec") = 30);

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
      .def(nb::self | nb::self)
      .def(nb::self | Transport())
      .def(nb::self & nb::self)
      .def(nb::self & Transport())
      .def(nb::self ^ nb::self)
      .def(nb::self ^ Transport())
      .def(
          "__ior__", [](TransportFlags& lhs, const TransportFlags& rhs) { return lhs |= rhs; }, nb::is_operator())
      .def(
          "__iand__", [](TransportFlags& lhs, const TransportFlags& rhs) { return lhs &= rhs; }, nb::is_operator())
      .def(
          "__ixor__", [](TransportFlags& lhs, const TransportFlags& rhs) { return lhs ^= rhs; }, nb::is_operator())
      .def(~nb::self)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self);

  nb::enum_<DeviceType>(m, "DeviceType")
      .value("Unknown", DeviceType::Unknown)
      .value("CPU", DeviceType::CPU)
      .value("GPU", DeviceType::GPU);

  nb::class_<Device>(m, "Device")
      .def(nb::init<>())
      .def(nb::init_implicit<DeviceType>(), nb::arg("type"))
      .def(nb::init<DeviceType, int>(), nb::arg("type"), nb::arg("id") = -1)
      .def_rw("type", &Device::type)
      .def_rw("id", &Device::id)
      .def("__str__", [](const Device& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      });

  nb::class_<EndpointConfig::Ib>(m, "EndpointConfigIb")
      .def(nb::init<>())
      .def(nb::init<int, int, int, int, int, int, int>(), nb::arg("device_index") = -1,
           nb::arg("port") = EndpointConfig::Ib::DefaultPort,
           nb::arg("gid_index") = EndpointConfig::Ib::DefaultGidIndex,
           nb::arg("max_cq_size") = EndpointConfig::Ib::DefaultMaxCqSize,
           nb::arg("max_cq_poll_num") = EndpointConfig::Ib::DefaultMaxCqPollNum,
           nb::arg("max_send_wr") = EndpointConfig::Ib::DefaultMaxSendWr,
           nb::arg("max_wr_per_send") = EndpointConfig::Ib::DefaultMaxWrPerSend)
      .def_rw("device_index", &EndpointConfig::Ib::deviceIndex)
      .def_rw("port", &EndpointConfig::Ib::port)
      .def_rw("gid_index", &EndpointConfig::Ib::gidIndex)
      .def_rw("max_cq_size", &EndpointConfig::Ib::maxCqSize)
      .def_rw("max_cq_poll_num", &EndpointConfig::Ib::maxCqPollNum)
      .def_rw("max_send_wr", &EndpointConfig::Ib::maxSendWr)
      .def_rw("max_wr_per_send", &EndpointConfig::Ib::maxWrPerSend);

  nb::class_<RegisteredMemory>(m, "RegisteredMemory")
      .def(nb::init<>())
      .def("data", [](RegisteredMemory& self) { return reinterpret_cast<uintptr_t>(self.data()); })
      .def("size", &RegisteredMemory::size)
      .def("transports", &RegisteredMemory::transports)
      .def("serialize", &RegisteredMemory::serialize)
      .def_static("deserialize", &RegisteredMemory::deserialize, nb::arg("data"));

  nb::class_<Endpoint>(m, "Endpoint")
      .def("config", &Endpoint::config)
      .def("transport", &Endpoint::transport)
      .def("device", &Endpoint::device)
      .def("max_write_queue_size", &Endpoint::maxWriteQueueSize)
      .def("serialize", &Endpoint::serialize)
      .def_static("deserialize", &Endpoint::deserialize, nb::arg("data"));

  nb::class_<Connection>(m, "Connection")
      .def("write", &Connection::write, nb::arg("dst"), nb::arg("dstOffset"), nb::arg("src"), nb::arg("srcOffset"),
           nb::arg("size"))
      .def(
          "update_and_sync",
          [](Connection* self, RegisteredMemory dst, uint64_t dstOffset, uintptr_t src, uint64_t newValue) {
            self->updateAndSync(dst, dstOffset, (uint64_t*)src, newValue);
          },
          nb::arg("dst"), nb::arg("dst_offset"), nb::arg("src"), nb::arg("new_value"))
      .def("flush", &Connection::flush, nb::call_guard<nb::gil_scoped_release>(),
           nb::arg("timeout_usec") = (int64_t)3e7)
      .def("transport", &Connection::transport)
      .def("remote_transport", &Connection::remoteTransport)
      .def("context", &Connection::context)
      .def("local_device", &Connection::localDevice)
      .def("get_max_write_queue_size", &Connection::getMaxWriteQueueSize);

  nb::class_<EndpointConfig>(m, "EndpointConfig")
      .def(nb::init<>())
      .def(nb::init_implicit<Transport>(), nb::arg("transport"))
      .def(nb::init<Transport, Device, int, EndpointConfig::Ib>(), nb::arg("transport"), nb::arg("device"),
           nb::arg("max_write_queue_size") = -1, nb::arg("ib") = EndpointConfig::Ib{})
      .def_rw("transport", &EndpointConfig::transport)
      .def_rw("device", &EndpointConfig::device)
      .def_rw("ib", &EndpointConfig::ib)
      .def_prop_rw(
          "ib_device_index", [](EndpointConfig& self) { return self.ib.deviceIndex; },
          [](EndpointConfig& self, int v) { self.ib.deviceIndex = v; })
      .def_prop_rw(
          "ib_port", [](EndpointConfig& self) { return self.ib.port; },
          [](EndpointConfig& self, int v) { self.ib.port = v; })
      .def_prop_rw(
          "ib_gid_index", [](EndpointConfig& self) { return self.ib.gidIndex; },
          [](EndpointConfig& self, int v) { self.ib.gidIndex = v; })
      .def_prop_rw(
          "ib_max_cq_size", [](EndpointConfig& self) { return self.ib.maxCqSize; },
          [](EndpointConfig& self, int v) { self.ib.maxCqSize = v; })
      .def_prop_rw(
          "ib_max_cq_poll_num", [](EndpointConfig& self) { return self.ib.maxCqPollNum; },
          [](EndpointConfig& self, int v) { self.ib.maxCqPollNum = v; })
      .def_prop_rw(
          "ib_max_send_wr", [](EndpointConfig& self) { return self.ib.maxSendWr; },
          [](EndpointConfig& self, int v) { self.ib.maxSendWr = v; })
      .def_prop_rw(
          "ib_max_wr_per_send", [](EndpointConfig& self) { return self.ib.maxWrPerSend; },
          [](EndpointConfig& self, int v) { self.ib.maxWrPerSend = v; })
      .def_rw("max_write_queue_size", &EndpointConfig::maxWriteQueueSize);

  nb::class_<Context>(m, "Context")
      .def_static("create", &Context::create)
      .def(
          "register_memory",
          [](Context* self, uintptr_t ptr, size_t size, TransportFlags transports) {
            return self->registerMemory((void*)ptr, size, transports);
          },
          nb::arg("ptr"), nb::arg("size"), nb::arg("transports"))
      .def("create_endpoint", &Context::createEndpoint, nb::arg("config"))
      .def("connect", &Context::connect, nb::arg("local_endpoint"), nb::arg("remote_endpoint"));

  nb::class_<SemaphoreStub>(m, "SemaphoreStub")
      .def(nb::init<const Connection&>(), nb::arg("connection"))
      .def("memory", &SemaphoreStub::memory)
      .def("serialize", &SemaphoreStub::serialize)
      .def_static("deserialize", &SemaphoreStub::deserialize, nb::arg("data"));

  nb::class_<Semaphore>(m, "Semaphore")
      .def(nb::init<>())
      .def(nb::init<const SemaphoreStub&, const SemaphoreStub&>(), nb::arg("local_stub"), nb::arg("remote_stub"))
      .def("connection", &Semaphore::connection)
      .def("local_memory", &Semaphore::localMemory)
      .def("remote_memory", &Semaphore::remoteMemory);

  def_shared_future<RegisteredMemory>(m, "RegisteredMemory");
  def_shared_future<Connection>(m, "Connection");
  def_shared_future<Semaphore>(m, "Semaphore");

  nb::class_<Communicator>(m, "Communicator")
      .def(nb::init<std::shared_ptr<Bootstrap>, std::shared_ptr<Context>>(), nb::arg("bootstrap"),
           nb::arg("context") = nullptr)
      .def("bootstrap", &Communicator::bootstrap)
      .def("context", &Communicator::context)
      .def(
          "register_memory",
          [](Communicator* self, uintptr_t ptr, size_t size, TransportFlags transports) {
            return self->registerMemory((void*)ptr, size, transports);
          },
          nb::arg("ptr"), nb::arg("size"), nb::arg("transports"))
      .def("send_memory", &Communicator::sendMemory, nb::arg("memory"), nb::arg("remote_rank"), nb::arg("tag") = 0)
      .def("recv_memory", &Communicator::recvMemory, nb::arg("remote_rank"), nb::arg("tag") = 0)
      .def("connect",
           static_cast<std::shared_future<Connection> (Communicator::*)(const EndpointConfig&, int, int)>(
               &Communicator::connect),
           nb::arg("local_config"), nb::arg("remote_rank"), nb::arg("tag") = 0)
      .def(
          "connect_on_setup",
          [](Communicator* self, int remoteRank, int tag, EndpointConfig localConfig) {
            return self->connect(std::move(localConfig), remoteRank, tag);
          },
          nb::arg("remote_rank"), nb::arg("tag"), nb::arg("local_config"))
      .def("send_memory_on_setup", &Communicator::sendMemory, nb::arg("memory"), nb::arg("remote_rank"), nb::arg("tag"))
      .def("recv_memory_on_setup", &Communicator::recvMemory, nb::arg("remote_rank"), nb::arg("tag"))
      .def("build_semaphore", &Communicator::buildSemaphore, nb::arg("connection"), nb::arg("remote_rank"),
           nb::arg("tag") = 0)
      .def("remote_rank_of", &Communicator::remoteRankOf)
      .def("tag_of", &Communicator::tagOf)
      .def("setup", [](Communicator*) {});
}

NB_MODULE(_mscclpp, m) {
  register_env(m);
  register_error(m);
  register_port_channel(m);
  register_memory_channel(m);
  register_fifo(m);
  register_semaphore(m);
  register_utils(m);
  register_core(m);
  register_numa(m);
  register_nvls(m);
  register_executor(m);
  register_npkit(m);
  register_gpu_utils(m);
  register_algorithm(m);

  // ext
  register_collective(m);
}
