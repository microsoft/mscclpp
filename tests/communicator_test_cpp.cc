#include "mscclpp.hpp"

#include <memory>
#include <cassert>
#include <iostream>
#include <mpi.h>

void test_communicator(int rank, int worldSize, int nranksPerNode){
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(rank, worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0)
    id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);

  auto communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
  for (int i = 0; i < worldSize; i++){
    if (i != rank){
      if (i % nranksPerNode == rank % nranksPerNode)
        auto connect = communicator->connect(i, 0, mscclpp::TransportCudaIpc);
      else
        auto connect = communicator->connect(i, 0, mscclpp::TransportAllIB);
    }
  }

  if (bootstrap->getRank() == 0)
    std::cout << "--- MSCCLPP::Communicator tests passed! ---" << std::endl;
}


int main(int argc, char **argv)
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