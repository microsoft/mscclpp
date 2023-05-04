#include "mscclpp.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <mpi.h>

void test_allgather(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap)
{
  std::vector<int> tmp(bootstrap->getNranks(), 0);
  tmp[bootstrap->getRank()] = bootstrap->getRank() + 1;
  bootstrap->allGather(tmp.data(), sizeof(int));
  for (int i = 0; i < bootstrap->getNranks(); i++) {
    assert(tmp[i] == i + 1);
  }
  if (bootstrap->getRank() == 0)
    std::cout << "AllGather test passed!" << std::endl;
}

void test_barrier(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap)
{
  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "Barrier test passed!" << std::endl;
}

void test_sendrecv(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap)
{
  for (int i = 0; i < bootstrap->getNranks(); i++) {
    if (bootstrap->getRank() == i)
      continue;
    int msg1 = (bootstrap->getRank() + 1) * 3;
    int msg2 = (bootstrap->getRank() + 1) * 3 + 1;
    int msg3 = (bootstrap->getRank() + 1) * 3 + 2;
    bootstrap->send(&msg1, sizeof(int), i, 0);
    bootstrap->send(&msg2, sizeof(int), i, 1);
    bootstrap->send(&msg3, sizeof(int), i, 2);
  }

  for (int i = 0; i < bootstrap->getNranks(); i++) {
    if (bootstrap->getRank() == i)
      continue;
    int msg1 = 0;
    int msg2 = 0;
    int msg3 = 0;
    // recv them in the opposite order to check correctness
    bootstrap->recv(&msg2, sizeof(int), i, 1);
    bootstrap->recv(&msg3, sizeof(int), i, 2);
    bootstrap->recv(&msg1, sizeof(int), i, 0);
    assert(msg1 == (i + 1) * 3);
    assert(msg2 == (i + 1) * 3 + 1);
    assert(msg3 == (i + 1) * 3 + 2);
  }
  if (bootstrap->getRank() == 0)
    std::cout << "Send/Recv test passed!" << std::endl;
}

void test_all(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap)
{
  test_allgather(bootstrap);
  test_barrier(bootstrap);
  test_sendrecv(bootstrap);
}

void test_mscclpp_bootstrap_with_id(int rank, int worldSize)
{
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(rank, worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0)
    id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);

  test_all(bootstrap);
  if (bootstrap->getRank() == 0)
    std::cout << "--- MSCCLPP::Bootstrap test with unique id passed! ---" << std::endl;
}

void test_mscclpp_bootstrap_with_ip_port_pair(int rank, int worldSize, char* ipPortPiar)
{
  std::shared_ptr<mscclpp::Bootstrap> bootstrap(new mscclpp::Bootstrap(rank, worldSize));
  bootstrap->initialize(ipPortPiar);

  test_all(bootstrap);
  if (bootstrap->getRank() == 0)
    std::cout << "--- MSCCLPP::Bootstrap test with ip_port pair passed! ---" << std::endl;
}

class MPIBootstrap : public mscclpp::BaseBootstrap
{
public:
  MPIBootstrap() : BaseBootstrap()
  {
  }
  int getRank() override
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }
  int getNranks() override
  {
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    return worldSize;
  }
  void allGather(void* sendbuf, int size) override
  {
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, sendbuf, size, MPI_BYTE, MPI_COMM_WORLD);
  }
  void barrier() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
  }
  void send(void* sendbuf, int size, int dest, int tag) override
  {
    MPI_Send(sendbuf, size, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
  }
  void recv(void* recvbuf, int size, int source, int tag) override
  {
    MPI_Recv(recvbuf, size, MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
};

void test_mpi_bootstrap()
{
  std::shared_ptr<mscclpp::BaseBootstrap> bootstrap(new MPIBootstrap());
  test_all(bootstrap);
  if (bootstrap->getRank() == 0)
    std::cout << "--- MPI Bootstrap test passed! ---" << std::endl;
}

int main(int argc, char** argv)
{
  int rank, worldSize;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (argc > 2) {
    if (rank == 0)
      std::cout << "Usage: " << argv[0] << " [ip:port]" << std::endl;
    MPI_Finalize();
    return 0;
  }
  test_mscclpp_bootstrap_with_id(rank, worldSize);
  if (argc == 2)
    test_mscclpp_bootstrap_with_ip_port_pair(rank, worldSize, argv[1]);
  test_mpi_bootstrap();

  MPI_Finalize();
  return 0;
}