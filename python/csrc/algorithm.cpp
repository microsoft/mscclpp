// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <mscclpp/algorithm.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_algorithm(nb::module_& m) {
  nb::enum_<CollectiveBufferMode>(m, "CollectiveBufferMode")
      .value("ANY", CollectiveBufferMode::Any)
      .value("IN_PLACE", CollectiveBufferMode::InPlace)
      .value("OUT_OF_PLACE", CollectiveBufferMode::OutOfPlace);

  nb::enum_<AlgorithmType>(m, "AlgorithmType").value("NATIVE", AlgorithmType::Native).value("DSL", AlgorithmType::DSL);

  nb::enum_<CommResult>(m, "CommResult")
      .value("COMM_SUCCESS", CommResult::CommSuccess)
      .value("COMM_UNHANDLED_CUDA_ERROR", CommResult::CommUnhandledCudaError)
      .value("COMM_SYSTEM_ERROR", CommResult::CommSystemError)
      .value("COMM_INTERNAL_ERROR", CommResult::CommInternalError)
      .value("COMM_INVALID_ARGUMENT", CommResult::CommInvalidArgument)
      .value("COMM_INVALID_USAGE", CommResult::CommInvalidUsage)
      .value("COMM_REMOTE_ERROR", CommResult::CommRemoteError)
      .value("COMM_IN_PROGRESS", CommResult::CommInProgress)
      .value("COMM_NUM_RESULTS", CommResult::CommNumResults);

  nb::enum_<ReduceOp>(m, "ReduceOp")
      .value("SUM", ReduceOp::SUM)
      .value("MIN", ReduceOp::MIN)
      .value("NOP", ReduceOp::NOP);

  auto algorithmClass =
      nb::class_<Algorithm>(m, "Algorithm")
          .def_static(
              "from_native_capsule",
              [](nb::capsule cap) {
                const char* name = cap.name();
                if (name == nullptr || std::strcmp(name, ALGORITHM_NATIVE_CAPSULE_NAME) != 0) {
                  throw nb::type_error("Invalid capsule: expected 'mscclpp::AlgorithmPtr'");
                }
                void* data = cap.data();
                if (data == nullptr) {
                  throw nb::value_error("Failed to get pointer from capsule");
                }
                return *static_cast<std::shared_ptr<Algorithm>*>(data);
              },
              nb::arg("capsule"))
          .def_prop_ro("name", &Algorithm::name)
          .def_prop_ro("collective", &Algorithm::collective)
          .def_prop_ro("message_range", &Algorithm::messageRange)
          .def_prop_ro("tags", &Algorithm::tags)
          .def_prop_ro("buffer_mode", &Algorithm::bufferMode)
          .def_prop_ro("constraint", &Algorithm::constraint)
          .def_prop_ro("type", &Algorithm::type)
          .def(
              "execute",
              [](Algorithm& self, std::shared_ptr<Communicator> comm, uintptr_t input, uintptr_t output,
                 size_t inputSize, size_t outputSize, DataType dtype, ReduceOp op, uintptr_t stream,
                 std::shared_ptr<Executor> executor, int nBlocks, int nThreadsPerBlock,
                 std::unordered_map<std::string, uintptr_t> extras) {
                return self.execute(comm, reinterpret_cast<const void*>(input), reinterpret_cast<void*>(output),
                                    inputSize, outputSize, dtype, op, reinterpret_cast<cudaStream_t>(stream), executor,
                                    nBlocks, nThreadsPerBlock, extras);
              },
              nb::arg("comm"), nb::arg("input"), nb::arg("output"), nb::arg("input_size"), nb::arg("output_size"),
              nb::arg("dtype"), nb::arg("op") = ReduceOp::NOP, nb::arg("stream") = 0, nb::arg("executor") = nullptr,
              nb::arg("n_blocks") = 0, nb::arg("n_threads_per_block") = 0,
              nb::arg("extras") = std::unordered_map<std::string, uintptr_t>())
            .def("reset", &Algorithm::reset);

  nb::class_<Algorithm::Constraint>(algorithmClass, "Constraint")
      .def(nb::init<>())
      .def(nb::init<int, int>(), nb::arg("world_size"), nb::arg("n_ranks_per_node"))
      .def_rw("world_size", &Algorithm::Constraint::worldSize)
      .def_rw("n_ranks_per_node", &Algorithm::Constraint::nRanksPerNode);

  nb::class_<AlgorithmBuilder>(m, "AlgorithmBuilder").def("build", &AlgorithmBuilder::build);

  nb::class_<DslAlgorithm, Algorithm>(m, "DslAlgorithm")
      .def(nb::init<std::string, ExecutionPlan, std::unordered_map<std::string, uint64_t>, Algorithm::Constraint>(),
           nb::arg("id"), nb::arg("plan"), nb::arg("tags") = std::unordered_map<std::string, uint64_t>(),
           nb::arg("constraint") = Algorithm::Constraint())
      .def("build", &DslAlgorithm::build);

  nb::class_<AlgorithmCollection>(m, "AlgorithmCollection")
      .def("register_algorithm", &AlgorithmCollection::registerAlgorithm, nb::arg("collective"), nb::arg("algo_name"),
           nb::arg("algorithm"))
      .def("get_algorithms_by_collective", &AlgorithmCollection::getAlgorithmsByCollective, nb::arg("collective"))
      .def("to_list", &AlgorithmCollection::getAllAlgorithms);

  nb::class_<CollectiveRequest>(m, "CollectiveRequest")
      .def_ro("world_size", &CollectiveRequest::worldSize)
      .def_ro("n_ranks_per_node", &CollectiveRequest::nRanksPerNode)
      .def_ro("rank", &CollectiveRequest::rank)
      .def_prop_ro("input_buffer",
                   [](const CollectiveRequest& self) { return reinterpret_cast<uintptr_t>(self.inputBuffer); })
      .def_prop_ro("output_buffer",
                   [](const CollectiveRequest& self) { return reinterpret_cast<uintptr_t>(self.outputBuffer); })
      .def_ro("message_size", &CollectiveRequest::messageSize)
      .def_prop_ro("collective", [](const CollectiveRequest& self) { return self.collective; })
      .def_ro("dtype", &CollectiveRequest::dtype)
      .def_prop_ro("hints", [](const CollectiveRequest& self) { return self.hints; })
      .def("buffer_mode", &CollectiveRequest::bufferMode);
}