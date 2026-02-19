// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mpi.h>
#include <unistd.h>

#include <filesystem>
#include <mscclpp/env.hpp>
#include <mscclpp/npkit/npkit.hpp>

#include "mp_unit_tests.hpp"

namespace {
std::string getExecutablePath() {
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  if (count == -1) {
    throw std::runtime_error("Failed to get executable path");
  }
  return std::string(result, count);
}
}  // namespace

void ExecutorTest::SetUp() {
  if (gEnv->worldSize != 2 || gEnv->nRanksPerNode != 2) {
    SKIP_TEST() << "This test requires world size to be 2 and ranks per node to be 2";
  }
  MultiProcessTest::SetUp();

  MSCCLPP_CUDATHROW(cudaSetDevice(rankToLocalRank(gEnv->rank)));
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  mscclpp::UniqueId id;
  bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, gEnv->worldSize);
  if (gEnv->rank == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  std::shared_ptr<mscclpp::Communicator> communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
  executor = std::make_shared<mscclpp::Executor>(communicator);
  npkitDumpDir = mscclpp::env()->npkitDumpDir;
  if (npkitDumpDir != "") {
    NpKit::Init(gEnv->rank);
  }
}

void ExecutorTest::TearDown() {
  if (npkitDumpDir != "") {
    NpKit::Dump(npkitDumpDir);
    NpKit::Shutdown();
  }
  MultiProcessTest::TearDown();
}

TEST_F(ExecutorTest, TwoNodesAllreduce) {
  std::string executablePath = getExecutablePath();
  std::filesystem::path path = executablePath;
  std::filesystem::path executionFilesPath =
      path.parent_path().parent_path().parent_path() / "test/execution-files/allreduce.json";
  mscclpp::ExecutionPlan plan(executionFilesPath.string(), gEnv->rank);
  const int bufferSize = 1024 * 1024;
  std::shared_ptr<char> sendbuff = mscclpp::GpuBuffer(bufferSize).memory();
  mscclpp::CudaStreamWithFlags stream(cudaStreamNonBlocking);
  executor->execute(gEnv->rank, sendbuff.get(), sendbuff.get(), bufferSize, bufferSize, mscclpp::DataType::FLOAT16,
                    plan, stream);
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
}
