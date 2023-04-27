#include "mscclpp.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mpi.h>

#define CUDATHROW(cmd)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t err = cmd;                                                                                             \
    if (err != cudaSuccess) {                                                                                          \
      throw std::runtime_error(std::string("Cuda failure '") + cudaGetErrorString(err) + "'");                         \
    }                                                                                                                  \
  } while (false)

mscclpp::Transport findIb(int localRank)
{
  mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                              mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                              mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  return IBs[localRank];
}

void test_communicator(int rank, int worldSize, int nranksPerNode)
{
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(rank, worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0)
    id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);

  auto communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
  if (bootstrap->getRank() == 0)
    std::cout << "Communicator initialization passed" << std::endl;

  auto myIbDevice = findIb(rank % nranksPerNode);
  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
      if (i / nranksPerNode == rank / nranksPerNode) {
        auto connect = communicator->connect(i, 0, mscclpp::Transport::CudaIpc);
      } else {
        auto connect = communicator->connect(i, 0, myIbDevice);
      }
    }
  }
  communicator->connectionSetup();

  if (bootstrap->getRank() == 0)
    std::cout << "Connection setup passed" << std::endl;

  int* devicePtr;
  int size = 1024;
  CUDATHROW(cudaMalloc(&devicePtr, size));
  auto registeredMemory = communicator->registerMemory(devicePtr, size, mscclpp::Transport::CudaIpc | myIbDevice);

  if (bootstrap->getRank() == 0)
    std::cout << "Memory registeration passed" << std::endl;

  if (bootstrap->getRank() == 0)
    std::cout << "--- MSCCLPP::Communicator tests passed! ---" << std::endl;
}

int main(int argc, char** argv)
{
  int rank, worldSize;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  int shmWorldSize;
  MPI_Comm_size(shmcomm, &shmWorldSize);
  int nranksPerNode = shmWorldSize;
  MPI_Comm_free(&shmcomm);

  test_communicator(rank, worldSize, nranksPerNode);

  MPI_Finalize();
  return 0;
}