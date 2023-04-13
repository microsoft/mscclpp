#include <cuda_runtime.h>
#include <mscclpp.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// This is a poorman's substitute for std::format, which is a C++20 feature.
template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
// Shutup format warning.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"

  // Dry-run to the get the buffer size:
  // Extra space for '\0'
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }

  // allocate buffer
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);

  // actually format
  std::snprintf(buf.get(), size, format.c_str(), args...);

  // Bulid the return string.
  // We don't want the '\0' inside
  return std::string(buf.get(), buf.get() + size - 1);

#pragma GCC diagnostic pop
}

// Maybe return the value, maybe throw an exception.
template <typename... Args>
void checkResult(
    mscclppResult_t status, const std::string& format, Args... args) {
  switch (status) {
    case mscclppSuccess:
      return;

    case mscclppUnhandledCudaError:
    case mscclppSystemError:
    case mscclppInternalError:
    case mscclppRemoteError:
    case mscclppInProgress:
    case mscclppNumResults:
      throw std::runtime_error(
          string_format(format, args...) + " : " +
          std::string(mscclppGetErrorString(status)));

    case mscclppInvalidArgument:
    case mscclppInvalidUsage:
    default:
      throw std::invalid_argument(
          string_format(format, args...) + " : " +
          std::string(mscclppGetErrorString(status)));
  }
}

#define RETRY(C, ...)                   \
  {                                     \
    mscclppResult_t res;                \
    do {                                \
      res = (C);                        \
    } while (res == mscclppInProgress); \
    checkResult(res, __VA_ARGS__);      \
  }

// Maybe return the value, maybe throw an exception.
template <typename Val, typename... Args>
Val maybe(
    mscclppResult_t status, Val val, const std::string& format, Args... args) {
  checkResult(status, format, args...);
  return val;
}

// Wrapper around connection state.
struct _Comm {
  int _rank;
  int _world_size;
  mscclppComm_t _handle;
  bool _is_open;
  bool _proxies_running;

 public:
  _Comm(int rank, int world_size, mscclppComm_t handle)
      : _rank(rank),
        _world_size(world_size),
        _handle(handle),
        _is_open(true),
        _proxies_running(false) {}

  ~_Comm() { close(); }

  // Close should be safe to call on a closed handle.
  void close() {
    if (_is_open) {
      if (_proxies_running) {
        mscclppProxyStop(_handle);
        _proxies_running = false;
      }
      checkResult(mscclppCommDestroy(_handle), "Failed to close comm channel");
      _handle = NULL;
      _is_open = false;
      _rank = -1;
      _world_size = -1;
    }
  }

  void check_open() {
    if (!_is_open) {
      throw std::invalid_argument("_Comm is not open");
    }
  }
};

struct _P2PHandle {
  struct mscclppRegisteredMemoryP2P _rm;

  _P2PHandle() : _rm({0}) {}

  _P2PHandle(const mscclppRegisteredMemoryP2P& p2p) : _rm(p2p) {}

  mscclppTransport_t transport() const {
    if (_rm.remoteBuff == nullptr) {
      return mscclppTransport_t::mscclppTransportIB;
    } else {
      return mscclppTransport_t::mscclppTransportP2P;
    }
  }
};

nb::callable _log_callback;

void _LogHandler(const char* msg) {
  // if (_log_callback) {
  //   nb::gil_scoped_acquire guard;
  //   _log_callback(msg);
  // }
}

static const std::string DOC_MscclppUniqueId =
    "MSCCLPP Unique Id; used by the MPI Interface";

static const std::string DOC__Comm = "MSCCLPP Communications Handle";

static const std::string DOC__P2PHandle = "MSCCLPP P2P MR Handle";

NB_MODULE(_py_mscclpp, m) {
  m.doc() = "Python bindings for MSCCLPP: which is not NCCL";

  m.attr("MSCCLPP_UNIQUE_ID_BYTES") = MSCCLPP_UNIQUE_ID_BYTES;

  m.def("_bind_log_handler", [](nb::callable cb) -> void {
    _log_callback = nb::borrow<nb::callable>(cb);
    mscclppSetLogHandler(_LogHandler);
  });
  m.def("_release_log_handler", []() -> void {
    _log_callback.reset();
    mscclppSetLogHandler(mscclppDefaultLogHandler);
  });

  nb::enum_<mscclppTransport_t>(m, "TransportType")
      .value("P2P", mscclppTransport_t::mscclppTransportP2P)
      .value("SHM", mscclppTransport_t::mscclppTransportSHM)
      .value("IB", mscclppTransport_t::mscclppTransportIB);

  nb::class_<mscclppUniqueId>(m, "MscclppUniqueId")
      .def_ro_static("__doc__", &DOC_MscclppUniqueId)
      .def_static(
          "from_context",
          []() -> mscclppUniqueId {
            mscclppUniqueId uniqueId;
            return maybe(
                mscclppGetUniqueId(&uniqueId),
                uniqueId,
                "Failed to get MSCCLP Unique Id.");
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_static(
          "from_bytes",
          [](nb::bytes source) -> mscclppUniqueId {
            if (source.size() != MSCCLPP_UNIQUE_ID_BYTES) {
              throw std::invalid_argument(string_format(
                  "Requires exactly %d bytes; found %d",
                  MSCCLPP_UNIQUE_ID_BYTES,
                  source.size()));
            }

            mscclppUniqueId uniqueId;
            std::memcpy(
                uniqueId.internal, source.c_str(), sizeof(uniqueId.internal));
            return uniqueId;
          })
      .def("bytes", [](mscclppUniqueId id) {
        return nb::bytes(id.internal, sizeof(id.internal));
      });

  nb::class_<_P2PHandle>(m, "_P2PHandle")
      .def_ro_static("__doc__", &DOC__P2PHandle)
      .def(
          "transport",
          &_P2PHandle::transport,
          "Get the transport type of the handle")
      .def(
          "data_ptr",
          [](const _P2PHandle& self) -> uint64_t {
            if (self.transport() == mscclppTransport_t::mscclppTransportP2P) {
              return reinterpret_cast<uint64_t>(self._rm.remoteBuff);
            }
            throw std::invalid_argument(
                "IB transport does not have a local data ptr");
          },
          "Get the local data pointer, only for P2P handles");

  nb::class_<mscclppRegisteredMemory>(m, "_RegisteredMemory")
      .def(
          "handles",
          [](const mscclppRegisteredMemory& self) -> std::vector<_P2PHandle> {
            std::vector<_P2PHandle> handles;
            for (const auto& p2p : self.p2p) {
              handles.push_back(_P2PHandle(p2p));
            }
            return handles;
          },
          "Get the P2P handle for this memory")
      .def(
          "write_all",
          [](const mscclppRegisteredMemory& self,
             const _Comm& comm,
             mscclppRegisteredMemory& src_data,
             size_t size,
             uint32_t src_offset = 0,
             uint32_t dst_offset = 0,
             int64_t stream = 0) -> void {
            checkResult(
                mscclppRegisteredBufferWrite(
                    comm._handle,
                    const_cast<mscclppRegisteredMemory*>(&self),
                    &src_data,
                    size,
                    src_offset,
                    dst_offset,
                    stream),
                "Failed to write to registered memory");
          },
          "comm"_a,
          "src_data"_a,
          "size"_a,
          "src_offset"_a = 0,
          "dst_offset"_a = 0,
          "stream"_a = 0,
          "Write to all bound targets in the buffer");

  nb::class_<_Comm>(m, "_Comm")
      .def_ro_static("__doc__", &DOC__Comm)
      .def_static(
          "init_rank_from_address",
          [](const std::string& address, int rank, int world_size) -> _Comm* {
            mscclppComm_t handle;
            checkResult(
                mscclppCommInitRank(&handle, world_size, address.c_str(), rank),
                "Failed to initialize comms: %s rank=%d world_size=%d",
                address,
                rank,
                world_size);
            return new _Comm(rank, world_size, handle);
          },
          nb::rv_policy::take_ownership,
          nb::call_guard<nb::gil_scoped_release>(),
          "address"_a,
          "rank"_a,
          "world_size"_a,
          "Initialize comms given an IP address, rank, and world_size")
      .def_static(
          "init_rank_from_id",
          [](const mscclppUniqueId& id, int rank, int world_size) -> _Comm* {
            mscclppComm_t handle;
            checkResult(
                mscclppCommInitRankFromId(&handle, world_size, id, rank),
                "Failed to initialize comms: %02X%s rank=%d world_size=%d",
                id.internal,
                rank,
                world_size);
            return new _Comm(rank, world_size, handle);
          },
          nb::rv_policy::take_ownership,
          nb::call_guard<nb::gil_scoped_release>(),
          "id"_a,
          "rank"_a,
          "world_size"_a,
          "Initialize comms given UniqueID, rank, and world_size")
      .def(
          "opened",
          [](_Comm& self) -> bool { return self._is_open; },
          "Is this comm object opened?")
      .def(
          "closed",
          [](_Comm& self) -> bool { return !self._is_open; },
          "Is this comm object closed?")
      .def_ro("rank", &_Comm::_rank)
      .def_ro("world_size", &_Comm::_world_size)
      .def(
          "register_buffer",
          [](_Comm& self,
             uint64_t data_ptr,
             uint64_t size) -> mscclppRegisteredMemory {
            self.check_open();
            mscclppRegisteredMemory regMem;
            checkResult(
                mscclppRegisterBuffer(
                    self._handle,
                    reinterpret_cast<void*>(data_ptr),
                    size,
                    &regMem),
                "Registering buffer failed");
            return regMem;
            ;
          },
          "data_ptr"_a,
          "size"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Register a buffer for P2P transfers.")
      .def(
          "register_source_buffer",
          [](_Comm& self,
             uint64_t data_ptr,
             uint64_t size) -> mscclppRegisteredMemory {
            self.check_open();
            mscclppRegisteredMemory regMem;
            checkResult(
                mscclppRegisterSourceBuffer(
                    self._handle,
                    reinterpret_cast<void*>(data_ptr),
                    size,
                    &regMem),
                "Registering buffer failed");
            return regMem;
            ;
          },
          "data_ptr"_a,
          "size"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Register a buffer for P2P transfers.")
      .def(
          "connect",
          [](_Comm& self,
             int remote_rank,
             int tag,
             uint64_t data_ptr,
             uint64_t size,
             mscclppTransport_t transport_type,
             const char* ib_dev) -> void {
            self.check_open();
            RETRY(
                mscclppConnect(
                    self._handle,
                    remote_rank,
                    tag,
                    reinterpret_cast<void*>(data_ptr),
                    size,
                    transport_type,
                    ib_dev),
                "Connect failed");
          },
          "remote_rank"_a,
          "tag"_a,
          "data_ptr"_a,
          "size"_a,
          "transport_type"_a,
          "ib_dev"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Attach a local buffer to a remote connection.")
      .def(
          "connection_setup",
          [](_Comm& self) -> void {
            self.check_open();
            RETRY(
                mscclppConnectionSetup(self._handle),
                "Failed to setup MSCCLPP connection");
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "Run connection setup for MSCCLPP.")
      .def(
          "launch_proxies",
          [](_Comm& self) -> void {
            self.check_open();
            if (self._proxies_running) {
              throw std::invalid_argument("Proxy Threads Already Running");
            }
            checkResult(
                mscclppProxyLaunch(self._handle),
                "Failed to launch MSCCLPP proxy");
            self._proxies_running = true;
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "Start the MSCCLPP proxy.")
      .def(
          "stop_proxies",
          [](_Comm& self) -> void {
            self.check_open();
            if (self._proxies_running) {
              checkResult(
                  mscclppProxyStop(self._handle),
                  "Failed to stop MSCCLPP proxy");
              self._proxies_running = false;
            }
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "Start the MSCCLPP proxy.")
      .def("close", &_Comm::close, nb::call_guard<nb::gil_scoped_release>())
      .def("__del__", &_Comm::close, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "bootstrap_all_gather_int",
          [](_Comm& self, int val) -> std::vector<int> {
            std::vector<int> buf(self._world_size);
            buf[self._rank] = val;
            mscclppBootstrapAllGather(self._handle, buf.data(), sizeof(int));
            return buf;
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "val"_a,
          "all-gather ints over the bootstrap connection.")
      .def(
          "all_gather_bytes",
          [](_Comm& self, nb::bytes& item) -> std::vector<nb::bytes> {
            // First, all-gather the sizes of all bytes.
            std::vector<size_t> sizes(self._world_size);
            sizes[self._rank] = item.size();
            checkResult(
                mscclppBootstrapAllGather(
                    self._handle, sizes.data(), sizeof(size_t)),
                "bootstrapAllGather failed.");

            // Next, find the largest message to send.
            size_t max_size = *std::max_element(sizes.begin(), sizes.end());

            // Allocate an all-gather buffer large enough for max * world_size.
            std::shared_ptr<char[]> data_buf(
                new char[max_size * self._world_size]);

            // Copy the local item into the buffer.
            std::memcpy(
                &data_buf[self._rank * max_size], item.c_str(), item.size());

            // all-gather the data buffer.
            checkResult(
                mscclppBootstrapAllGather(
                    self._handle, data_buf.get(), max_size),
                "bootstrapAllGather failed.");

            // Build a response vector.
            std::vector<nb::bytes> ret;
            for (int i = 0; i < self._world_size; ++i) {
              // Copy out the relevant range of each item.
              ret.push_back(nb::bytes(&data_buf[i * max_size], sizes[i]));
            }
            return ret;
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "item"_a,
          "all-gather bytes over the bootstrap connection; sizes do not need "
          "to match.");
}
