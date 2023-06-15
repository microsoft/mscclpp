// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "mp_unit_tests.hpp"

#include <mpi.h>

#include <sstream>

const char gDefaultIpPort[] = "127.0.0.1:50053";
MultiProcessTestEnv* gEnv = nullptr;

mscclpp::Transport ibIdToTransport(int id) {
  mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                              mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                              mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  return IBs[id];
}

MultiProcessTestEnv::MultiProcessTestEnv(int argc, const char** argv) : argc(argc), argv(argv) {}

static std::unordered_map<std::string, std::string> parseArgs(int argc, const char* argv[]) {
  auto printUsage = [](const char* prog) {
    std::stringstream ss;
    ss << "Usage: " << prog << " [-ip_port IP:PORT]\n";
    std::cout << ss.str();
  };

  std::unordered_map<std::string, std::string> options;

  // Default values
  options["ip_port"] = gDefaultIpPort;

  // Parse the command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-ip_port") {
      if (i + 1 < argc) {
        options["ip_port"] = argv[++i];
      } else {
        throw std::invalid_argument("Error: -ip_port option requires an argument.\n");
      }
    } else if (arg == "-help" || arg == "-h") {
      printUsage(argv[0]);
      exit(0);
    } else {
      throw std::invalid_argument("Error: Unknown option " + std::string(argv[i]) + "\n");
    }
  }
  return options;
}

void MultiProcessTestEnv::SetUp() {
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  // get the local number of nodes with MPI
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  int shmrank;
  MPI_Comm_size(shmcomm, &shmrank);
  nRanksPerNode = shmrank;
  MPI_Comm_free(&shmcomm);

  // parse the command line arguments
  args = parseArgs(argc, argv);
}

void MultiProcessTestEnv::TearDown() { MPI_Finalize(); }

void MultiProcessTest::TearDown() {
  // Wait for all ranks to finish the previous test
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gEnv = new MultiProcessTestEnv(argc, (const char**)argv);
  ::testing::AddGlobalTestEnvironment(gEnv);
  return RUN_ALL_TESTS();
}

TEST_F(MultiProcessTest, Prelim) {
  // Test to make sure the MPI environment is set up correctly
  ASSERT_GE(gEnv->worldSize, 2);
}
