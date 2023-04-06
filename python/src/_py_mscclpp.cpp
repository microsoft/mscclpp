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

 public:
  _Comm(int rank, int world_size, mscclppComm_t handle)
      : _rank(rank), _world_size(world_size), _handle(handle), _is_open(true) {}

  ~_Comm() { close(); }

  // Close should be safe to call on a closed handle.
  void close() {
    if (_is_open) {
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

nb::callable _log_callback;

void _LogHandler(const char* msg) {
  if (_log_callback) {
    nb::gil_scoped_acquire guard;
    _log_callback(msg);
  }
}

static const std::string DOC_MscclppUniqueId =
    "MSCCLPP Unique Id; used by the MPI Interface";

static const std::string DOC__Comm = "MSCCLPP Communications Handle";

NB_MODULE(_py_mscclpp, m) {
  m.doc() = "Python bindings for MSCCLPP: which is not NCCL";

  m.attr("MSCCLPP_UNIQUE_ID_BYTES") = MSCCLPP_UNIQUE_ID_BYTES;

  m.def("_bind_log_handler", [](nb::callable cb) {
    _log_callback = nb::borrow<nb::callable>(cb);
    mscclppSetLogHandler(_LogHandler);
  });
  m.def("_release_log_handler", []() {
    _log_callback.reset();
    mscclppSetLogHandler(mscclppDefaultLogHandler);
  });

  nb::class_<mscclppUniqueId>(m, "MscclppUniqueId")
      .def_ro_static("__doc__", &DOC_MscclppUniqueId)
      .def_static(
          "from_context",
          []() {
            mscclppUniqueId uniqueId;
            return maybe(
                mscclppGetUniqueId(&uniqueId),
                uniqueId,
                "Failed to get MSCCLP Unique Id.");
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_static(
          "from_bytes",
          [](nb::bytes source) {
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

  nb::class_<_Comm>(m, "_Comm")
      .def_ro_static("__doc__", &DOC__Comm)
      .def_static(
          "init_rank_from_address",
          [](const std::string& address, int rank, int world_size) {
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
          [](const mscclppUniqueId& id, int rank, int world_size) {
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
          "Initialize comms given u UniqueID, rank, and world_size")
      .def(
          "opened",
          [](_Comm& comm) { return comm._is_open; },
          "Is this comm object opened?")
      .def(
          "closed",
          [](_Comm& comm) { return !comm._is_open; },
          "Is this comm object closed?")
      .def_ro("rank", &_Comm::_rank)
      .def_ro("world_size", &_Comm::_world_size)
      .def(
          "connection_setup",
          [](_Comm& comm) {
            comm.check_open();
            return maybe(
                mscclppConnectionSetup(comm._handle),
                true,
                "Failed to settup MSCCLPP connection");
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "Run connection setup for MSCCLPP.")
      .def(
          "launch_proxy",
          [](_Comm& comm) {
            comm.check_open();
            return maybe(
                mscclppProxyLaunch(comm._handle),
                true,
                "Failed to launch MSCCLPP proxy");
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "Start the MSCCLPP proxy.")
      .def(
          "stop_proxy",
          [](_Comm& comm) {
            comm.check_open();
            return maybe(
                mscclppProxyStop(comm._handle),
                true,
                "Failed to stop MSCCLPP proxy");
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "Start the MSCCLPP proxy.")
      .def("close", &_Comm::close, nb::call_guard<nb::gil_scoped_release>())
      .def("__del__", &_Comm::close, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "connection_setup",
          [](_Comm& comm) -> void {
            comm.check_open();
            checkResult(
                mscclppConnectionSetup(comm._handle),
                "Connection Setup Failed");
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "bootstrap_all_gather_int",
          [](_Comm& comm, int val) -> std::vector<int> {
            std::vector<int> buf(comm._world_size);
            buf[comm._rank] = val;
            mscclppBootstrapAllGather(comm._handle, buf.data(), sizeof(int));
            return buf;
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "val"_a,
          "all-gather ints over the bootstrap connection.")
      .def(
          "all_gather_bytes",
          [](_Comm& comm, nb::bytes& item) {
            // First, all-gather the sizes of all bytes.
            std::vector<size_t> sizes(comm._world_size);
            sizes[comm._rank] = item.size();
            checkResult(
                mscclppBootstrapAllGather(
                    comm._handle, sizes.data(), sizeof(size_t)),
                "bootstrapAllGather failed.");

            // Next, find the largest message to send.
            size_t max_size = *std::max_element(sizes.begin(), sizes.end());

            // Allocate an all-gather buffer large enough for max * world_size.
            std::shared_ptr<char[]> data_buf(
                new char[max_size * comm._world_size]);

            // Copy the local item into the buffer.
            std::memcpy(
                &data_buf[comm._rank * max_size], item.c_str(), item.size());

            // all-gather the data buffer.
            checkResult(
                mscclppBootstrapAllGather(
                    comm._handle, data_buf.get(), max_size),
                "bootstrapAllGather failed.");

            // Build a response vector.
            std::vector<nb::bytes> ret;
            for (int i = 0; i < comm._world_size; ++i) {
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
