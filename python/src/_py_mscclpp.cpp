#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <mscclpp.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

// This is a poorman's substitute for std::format, which is a C++20 feature.
template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

template<typename Val, typename ... Args>
Val maybe(mscclppResult_t status, Val val, const std::string& format, Args ... args) {
    switch (status) {
        case mscclppSuccess:
	    return val;

	case mscclppUnhandledCudaError:
	case mscclppSystemError:
	case mscclppInternalError:
	case mscclppRemoteError:
	case mscclppInProgress:
	case mscclppNumResults:
	    throw std::runtime_error(string_format(format, args ...));

	case mscclppInvalidArgument:
	case mscclppInvalidUsage:
	default:
	    throw std::invalid_argument(string_format(format, args ...));
    }
}

struct MscclppComm {
  mscclppComm_t internal;
};


NB_MODULE(_py_mscclpp, m) {
    m.doc() = "Python bindings for MSCCLPP";

    m.attr("MSCCLPP_UNIQUE_ID_BYTES") = MSCCLPP_UNIQUE_ID_BYTES;

    nb::class_<mscclppUniqueId>(m, "MscclppUniqueId")
	.def_static("from_context", []() {
   	    mscclppUniqueId uniqueId;
	    return maybe(
   	        mscclppGetUniqueId(&uniqueId),
		uniqueId,
		"Failed to get MSCCLP Unique Id."
            );
	})
	.def_static("from_bytes", [](nb::bytes source) {
	    if (source.size() != MSCCLPP_UNIQUE_ID_BYTES) {
	        throw std::invalid_argument(
	            string_format(
	                "Requires exactly %d bytes; found %d",
                        MSCCLPP_UNIQUE_ID_BYTES,
	                source.size()
	            )
	        );
	    }

   	    mscclppUniqueId uniqueId;
	    std::memcpy(uniqueId.internal, source.c_str(), sizeof(uniqueId.internal));
	    return uniqueId;
	})
	.def("bytes", [](mscclppUniqueId id){ 
	    return nb::bytes(id.internal, sizeof(id.internal));
	});

  nb::class_<MscclppComm>(m, "MscclppComm")
       .def_static(
           "init_rank_from_address",
           [](const std::string &address, int rank, int world_size) {
               MscclppComm comm = { 0 };
               return maybe(
                   mscclppCommInitRank(&comm.internal, world_size, rank, address.c_str()),
                   comm,
                   "Failed to initialize comms: %s rank=%d world_size=%d",
                   address,
                   rank,
                   world_size);
           },
	   "address"_a, "rank"_a, "world_size"_a,
	   "Initialize comms given an IP address, rank, and world_size"
	   )
       .def_static("init_rank_from_id", [](const mscclppUniqueId &id, int rank, int world_size) {
	   MscclppComm comm = { 0 };
	   return maybe(
               mscclppCommInitRankFromId(&comm.internal, world_size, id, rank),
	       comm,
	       "Failed to initialize comms: %02X%s rank=%d world_size=%d",
	       id.internal,
	       rank,
	       world_size);
       })
       .def("close", [](MscclppComm &comm) {
           maybe(
               mscclppCommDestroy(comm.internal),
	       nb::none(),
	       "Failed to close comm channel"
	   );
	   comm.internal = 0;
       })
       .def("__del__", [](MscclppComm &comm) {
           maybe(
               mscclppCommDestroy(comm.internal),
	       nb::none(),
	       "Failed to close comm channel"
	   );
	   comm.internal = 0;
       });

}

