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

void register_all_memories(std::unique_ptr<mscclpp::Communicator>& communicator, int rank, int worldSize, void* devicePtr, size_t deviceBufferSize, mscclpp::Transport myIbDevice, mscclpp::RegisteredMemory& localMemory, std::unordered_map<int, mscclpp::RegisteredMemory>& remoteMemory){
  localMemory = communicator->registerMemory(devicePtr, deviceBufferSize, mscclpp::Transport::CudaIpc | myIbDevice);
  int serializedSize = 0;
  for (int i = 0; i < worldSize; i++) {
    if (i != rank){
      auto serialized = localMemory.serialize();
      serializedSize = serialized.size();
      communicator->bootstrapper()->send(serialized.data(), serializedSize, i, 0);
    }
  }
  if (serializedSize == 0) {
    throw std::runtime_error("Serialized size should have been set to a non-zero value.");
  }
  for (int i = 0; i < worldSize; i++) {
    if (i != rank){
      std::vector<char> deserialized(serializedSize);
      communicator->bootstrapper()->recv(deserialized.data(), serializedSize, i, 0);
      auto remote = mscclpp::RegisteredMemory::deserialize(deserialized);
      remoteMemory[i] = remote;
    }
  }
}

void make_connections(std::unique_ptr<mscclpp::Communicator>& communicator, int rank, int worldSize, int nRanksPerNode, mscclpp::Transport myIbDevice, std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections){
  for (int i = 0; i < worldSize; i++) {
    if (i != rank){
      if (i / nRanksPerNode == rank / nRanksPerNode) {
        connections[i] = communicator->connect(i, 0, mscclpp::Transport::CudaIpc);
      } else {
        connections[i] = communicator->connect(i, 0, myIbDevice);
      }
    }
  }
  communicator->connectionSetup();
}

void write_remote(int rank, int worldSize, std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections, std::unordered_map<int, mscclpp::RegisteredMemory>& remoteRegisteredMemories, mscclpp::RegisteredMemory& registeredMemory, int writeSize){
  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
      auto& conn = connections.at(i);
      auto& peerMemory = remoteRegisteredMemories.at(i);
      // printf("write to rank: %d, rank is %d\n", peerMemory.rank(), rank);
      conn->write(peerMemory, rank * writeSize, registeredMemory, rank * writeSize, writeSize);
      conn->flush();
    }
  }

}

void test_communicator(int rank, int worldSize, int nranksPerNode)
{
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(rank, worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0)
    id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);

  auto communicator = std::make_unique<mscclpp::Communicator>(bootstrap);
  if (bootstrap->getRank() == 0)
    std::cout << "Communicator initialization passed" << std::endl;

  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> connections;
  auto myIbDevice = findIb(rank % nranksPerNode);

  make_connections(communicator, rank, worldSize, nranksPerNode, myIbDevice, connections);
  if (bootstrap->getRank() == 0)
    std::cout << "Connection setup passed" << std::endl;

  int numBuffers = 1000;
  std::vector<int*> devicePtr(numBuffers);
  int deviceBufferSize = 1024*1024;
  
  std::vector<mscclpp::RegisteredMemory> localMemory(numBuffers);
  std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>> remoteMemory(numBuffers);

  for (int n = 0; n < numBuffers; n++) {
    if (n % 100 == 0)
      std::cout << "Registering memory for " << std::to_string(n) << " buffers" << std::endl;
    CUDATHROW(cudaMalloc(&devicePtr[n], deviceBufferSize));
    register_all_memories(communicator, rank, worldSize, devicePtr[n], deviceBufferSize, myIbDevice, localMemory[n], remoteMemory[n]);
  }
  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "Memory registration for " << std::to_string(numBuffers) << " buffers passed" << std::endl;


  assert((deviceBufferSize / sizeof(int)) % worldSize == 0);
  size_t writeSize = deviceBufferSize / worldSize;
  size_t dataCount = deviceBufferSize / sizeof(int);
  for (int n = 0; n < numBuffers; n++){
    std::vector<int> hostBuffer(dataCount, 0);
    for (int i = 0; i < dataCount; i++) {
      hostBuffer[i] = rank + n * worldSize;
    }
    CUDATHROW(cudaMemcpy(devicePtr[n], hostBuffer.data(), deviceBufferSize, cudaMemcpyHostToDevice));
  }
  CUDATHROW(cudaDeviceSynchronize());

  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "CUDA memory initialization passed" << std::endl;
  
  for (int n = 0; n < numBuffers; n++){
    write_remote(rank, worldSize, connections, remoteMemory[n], localMemory[n], writeSize);
  }
  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "RDMA write for " << std::to_string(numBuffers) << " buffers passed" << std::endl;

  for (int n = 0; n < numBuffers; n++){
    // polling until it becomes ready
    bool ready = false;
    int niter = 0;
    std::vector<int> hostBuffer(dataCount, 0);
    do {
      ready = true;
      CUDATHROW(cudaMemcpy(hostBuffer.data(), devicePtr[n], deviceBufferSize, cudaMemcpyDeviceToHost));
      for (int i = 0; i < worldSize; i++) {
        for (int j = i*writeSize/sizeof(int); j < (i+1)*writeSize/sizeof(int); j++) {
          if (hostBuffer[j] != i + n * worldSize) {
            ready = false;
          }
        }
      }
      if (niter == 10000){
        throw std::runtime_error("Polling is stuck.");
      }
      niter++;
    } while (!ready);
  }

  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "Polling for " << std::to_string(numBuffers) << " buffers passed" << std::endl;

  if (bootstrap->getRank() == 0)
    std::cout << "--- MSCCLPP::Communicator tests passed! ---" << std::endl;

  for (int n = 0; n < numBuffers; n++){
    CUDATHROW(cudaFree(devicePtr[n]));
  }
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