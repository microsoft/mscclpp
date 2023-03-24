#include <mscclpp.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// This is a poorman's substitute for std::format, which is a C++20 feature.
template <typename... Args>
std::string string_format(const std::string &format, Args... args) {
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
    mscclppResult_t status, const std::string &format, Args... args) {
  switch (status) {
    case mscclppSuccess:
      return;

    case mscclppUnhandledCudaError:
    case mscclppSystemError:
    case mscclppInternalError:
    case mscclppRemoteError:
    case mscclppInProgress:
    case mscclppNumResults:
      throw std::runtime_error(string_format(format, args...));

    case mscclppInvalidArgument:
    case mscclppInvalidUsage:
    default:
      throw std::invalid_argument(string_format(format, args...));
  }
}

// Maybe return the value, maybe throw an exception.
template <typename Val, typename... Args>
Val maybe(
    mscclppResult_t status, Val val, const std::string &format, Args... args) {
  checkResult(status, format, args...);
  return val;
}

// Wrapper around connection state.
struct MscclppComm {
  mscclppComm_t _handle;
  bool _is_open = false;

 public:
  ~MscclppComm() { close(); }

  // Close should be safe to call on a closed handle.
  void close() {
    if (_is_open) {
      checkResult(mscclppCommDestroy(_handle), "Failed to close comm channel");
      _handle = 0;
      _is_open = false;
    }
  }

  void check_open() {
    if (!_is_open) {
      throw std::invalid_argument("MscclppComm is not open");
    }
  }
};

static const std::string DOC_MscclppUniqueId =
    "MSCCLPP Unique Id; used by the MPI Interface";

static const std::string DOC_MscclppComm = "MSCCLPP Communications Handle";

NB_MODULE(_py_mscclpp, m) {
  m.doc() = "Python bindings for MSCCLPP: which is not NCCL";

  m.attr("MSCCLPP_UNIQUE_ID_BYTES") = MSCCLPP_UNIQUE_ID_BYTES;

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

  nb::class_<MscclppComm>(m, "MscclppComm")
      .def_ro_static("__doc__", &DOC_MscclppComm)
      .def_static(
          "init_rank_from_address",
          [](const std::string &address, int rank, int world_size) {
            MscclppComm comm = {0};
            comm._is_open = true;
            return maybe(
                mscclppCommInitRank(
                    &comm._handle, world_size, rank, address.c_str()),
                comm,
                "Failed to initialize comms: %s rank=%d world_size=%d",
                address,
                rank,
                world_size);
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "address"_a,
          "rank"_a,
          "world_size"_a,
          "Initialize comms given an IP address, rank, and world_size")
      .def_static(
          "init_rank_from_id",
          [](const mscclppUniqueId &id, int rank, int world_size) {
            MscclppComm comm = {0};
            comm._is_open = true;
            return maybe(
                mscclppCommInitRankFromId(&comm._handle, world_size, id, rank),
                comm,
                "Failed to initialize comms: %02X%s rank=%d world_size=%d",
                id.internal,
                rank,
                world_size);
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "id"_a,
          "rank"_a,
          "world_size"_a,
          "Initialize comms given u UniqueID, rank, and world_size")
      .def(
          "opened",
          [](MscclppComm &comm) { return comm._is_open; },
          "Is this comm object opened?")
      .def(
          "closed",
          [](MscclppComm &comm) { return !comm._is_open; },
          "Is this comm object closed?")
      .def(
          "rank",
          [](MscclppComm &comm) {
            comm.check_open();
            int rank;
            return maybe(
                mscclppCommRank(comm._handle, &rank),
                rank,
                "Failed to retrieve MSCCLPP rank");
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "The rank of this node.")
      .def(
          "size",
          [](MscclppComm &comm) {
            comm.check_open();
            int size;
            return maybe(
                mscclppCommSize(comm._handle, &size),
                size,
                "Failed to retrieve MSCCLPP world size");
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "The world size of this node.")
      .def(
          "connection_setup",
          [](MscclppComm &comm) {
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
          [](MscclppComm &comm) {
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
          [](MscclppComm &comm) {
            comm.check_open();
            return maybe(
                mscclppProxyStop(comm._handle),
                true,
                "Failed to stop MSCCLPP proxy");
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "Start the MSCCLPP proxy.")
      .def(
          "close",
          &MscclppComm::close,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "__del__",
          &MscclppComm::close,
          nb::call_guard<nb::gil_scoped_release>());
}
