#include "mscclpp.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <unordered_map>

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

  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> connections;
  auto myIbDevice = findIb(rank % nranksPerNode);
  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
      std::shared_ptr<mscclpp::Connection> conn;
      if (i / nranksPerNode == rank / nranksPerNode) {
        conn = communicator->connect(i, 0, mscclpp::Transport::CudaIpc);
      } else {
        conn = communicator->connect(i, 0, myIbDevice);
      }
      connections[i] = conn;
    }
  }
  communicator->connectionSetup();

  if (bootstrap->getRank() == 0)
    std::cout << "Connection setup passed" << std::endl;

  int* devicePtr;
  int size = 1024;
  CUDATHROW(cudaMalloc(&devicePtr, size));
  auto registeredMemory = communicator->registerMemory(devicePtr, size, mscclpp::Transport::CudaIpc | myIbDevice);

  for (int i = 0; i < worldSize; i++) {
    if (i != rank){
      auto serialized = registeredMemory.serialize();
      int serializedSize = serialized.size();
      bootstrap->send(&serializedSize, sizeof(int), i, 0);
      bootstrap->send(serialized.data(), serializedSize, i, 1);
    }
  }
  std::unordered_map<int, mscclpp::RegisteredMemory> registeredMemories;
  for (int i = 0; i < worldSize; i++) {
    if (i != rank){
      int deserializedSize;
      bootstrap->recv(&deserializedSize, sizeof(int), i, 0);
      std::vector<char> deserialized(deserializedSize);
      bootstrap->recv(deserialized.data(), deserializedSize, i, 1);
      auto deserializedRegisteredMemory = mscclpp::RegisteredMemory::deserialize(deserialized);
      registeredMemories.insert({deserializedRegisteredMemory.rank(), deserializedRegisteredMemory});
    }
  }

  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "Memory registration passed" << std::endl;

  assert((size / sizeof(int)) % worldSize == 0);
  size_t writeSize = size / worldSize;
  size_t dataCount = size / sizeof(int);
  // std::vector<int> hostBuffer(dataCount, 0);
  std::shared_ptr<int[]> hostBuffer(new int[dataCount]);
  for (int i = 0; i < dataCount; i++) {
    hostBuffer[i] = rank;
  }
  CUDATHROW(cudaMemcpy(devicePtr, hostBuffer.get(), size, cudaMemcpyHostToDevice));
  CUDATHROW(cudaDeviceSynchronize());

  bootstrap->barrier();
  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
      auto& conn = connections.at(i);
      auto& peerMemory = registeredMemories.at(i);
      // printf("write to rank: %d, rank is %d\n", peerMemory.rank(), rank);
      conn->write(peerMemory, rank * writeSize, registeredMemory, rank * writeSize, writeSize);
      conn->flush();
    }
  }
  bootstrap->barrier();
  // polling until it becomes ready
  bool ready = false;
  int niter = 0;
  do {
    ready = true;
    CUDATHROW(cudaMemcpy(hostBuffer.get(), devicePtr, size, cudaMemcpyDeviceToHost));
    size_t dataPerRank = writeSize / sizeof(int);
    for (int i = 0; i < dataCount; i++) {
      if (hostBuffer[i] != i / dataPerRank) {
        ready = false;
      }
    }
    if (niter == 10000){
      throw std::runtime_error("Polling is stuck.");
    }
    niter++;
  } while (!ready);

  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "Connection write passed" << std::endl;

  CUDATHROW(cudaFree(devicePtr));
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