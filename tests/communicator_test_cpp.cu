#include "mscclpp.hpp"
#include "epoch.hpp"

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

void register_all_memories(mscclpp::Communicator& communicator, int rank, int worldSize, void* devicePtr, size_t deviceBufferSize, mscclpp::Transport myIbDevice, mscclpp::RegisteredMemory& localMemory, std::unordered_map<int, mscclpp::RegisteredMemory>& remoteMemory){
  localMemory = communicator.registerMemory(devicePtr, deviceBufferSize, mscclpp::Transport::CudaIpc | myIbDevice);
  std::unordered_map<int, mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> futureRemoteMemory;
  for (int i = 0; i < worldSize; i++) {
    if (i != rank){
      communicator.sendMemoryOnSetup(localMemory, i, 0);
      futureRemoteMemory[i] = communicator.recvMemoryOnSetup(i, 0);
    }
  }
  communicator.setup();
  for (int i = 0; i < worldSize; i++) {
    if (i != rank){
      remoteMemory[i] = futureRemoteMemory[i].get();
    }
  }


  // auto serialized = localMemory.serialize();
  // int serializedSize = serialized.size();
  // for (int i = 0; i < worldSize; i++) {
  //   if (i != rank){
  //     communicator.bootstrapper()->send(serialized.data(), serializedSize, i, 0);
  //   }
  // }
  // for (int i = 0; i < worldSize; i++) {
  //   if (i != rank){
  //     std::vector<char> deserialized(serializedSize);
  //     communicator.bootstrapper()->recv(deserialized.data(), serializedSize, i, 0);
  //     auto remote = mscclpp::RegisteredMemory::deserialize(deserialized);
  //     remoteMemory[i] = remote;
  //   }
  // }
}

void make_connections(mscclpp::Communicator& communicator, int rank, int worldSize, int nRanksPerNode, mscclpp::Transport myIbDevice, std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections){
  for (int i = 0; i < worldSize; i++) {
    if (i != rank){
      if (i / nRanksPerNode == rank / nRanksPerNode) {
        connections[i] = communicator.connectOnSetup(i, 0, mscclpp::Transport::CudaIpc);
      } else {
        connections[i] = communicator.connectOnSetup(i, 0, myIbDevice);
      }
    }
  }
  communicator.setup();
}

void write_remote(int rank, int worldSize, std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections, 
                  std::unordered_map<int, mscclpp::RegisteredMemory>& remoteRegisteredMemories, mscclpp::RegisteredMemory& registeredMemory, int dataCountPerRank){
  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
      auto& conn = connections.at(i);
      auto& peerMemory = remoteRegisteredMemories.at(i);
      conn->write(peerMemory, rank * dataCountPerRank * sizeof(int), registeredMemory, rank * dataCountPerRank*sizeof(int), dataCountPerRank*sizeof(int));
      conn->flush();
    }
  }
}

void device_buffer_init(int rank, int worldSize, int dataCount, std::vector<int*>& devicePtr){
  for (int n = 0; n < (int)devicePtr.size(); n++){
    std::vector<int> hostBuffer(dataCount, 0);
    for (int i = 0; i < dataCount; i++) {
      hostBuffer[i] = rank + n * worldSize;
    }
    CUDATHROW(cudaMemcpy(devicePtr[n], hostBuffer.data(), dataCount*sizeof(int), cudaMemcpyHostToDevice));
  }
  CUDATHROW(cudaDeviceSynchronize());
}

bool test_device_buffer_write_correctness(int worldSize, int dataCount, std::vector<int*>& devicePtr){
  for (int n = 0; n < (int)devicePtr.size(); n++){
    std::vector<int> hostBuffer(dataCount, 0);
    CUDATHROW(cudaMemcpy(hostBuffer.data(), devicePtr[n], dataCount*sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < worldSize; i++) {
      for (int j = i*dataCount/worldSize; j < (i+1)*dataCount/worldSize; j++) {
        if (hostBuffer[j] != i + n * worldSize) {
          return false;
        }
      }
    }
  }
  return true;
}

void test_write(int rank, int worldSize, int deviceBufferSize, std::shared_ptr<mscclpp::BaseBootstrap> bootstrap, std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections, 
                std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>>& remoteMemory, std::vector<mscclpp::RegisteredMemory>& localMemory, std::vector<int*>& devicePtr, int numBuffers){

  assert((deviceBufferSize / sizeof(int)) % worldSize == 0);
  size_t dataCount = deviceBufferSize / sizeof(int);

  device_buffer_init(rank, worldSize, dataCount, devicePtr);
  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "CUDA memory initialization passed" << std::endl;
  
  for (int n = 0; n < numBuffers; n++){
    write_remote(rank, worldSize, connections, remoteMemory[n], localMemory[n], dataCount / worldSize);
  }
  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "RDMA write for " << std::to_string(numBuffers) << " buffers passed" << std::endl;

  // polling until it becomes ready
  bool ready = false;
  int niter = 0;
  do {
    ready = test_device_buffer_write_correctness(worldSize, dataCount, devicePtr);
    niter++;
    if (niter == 10000){
      throw std::runtime_error("Polling is stuck.");
    }
  } while (!ready);

  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "Polling for " << std::to_string(numBuffers) << " buffers passed" << std::endl;
}

__global__ void increament_epochs(mscclpp::DeviceEpoch* deviceEpochs, int rank, int worldSize){
  int tid = threadIdx.x;
  if (tid != rank && tid < worldSize){
    deviceEpochs[tid].epochIncrement();
  }
}

__global__ void wait_epochs(mscclpp::DeviceEpoch* deviceEpochs, int rank, int worldSize){
  int tid = threadIdx.x;
  if (tid != rank && tid < worldSize){
    deviceEpochs[tid].wait();
  }
}

void test_write_with_epochs(int rank, int worldSize, int deviceBufferSize, std::shared_ptr<mscclpp::BaseBootstrap> bootstrap, std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections, 
                std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>>& remoteMemory, std::vector<mscclpp::RegisteredMemory>& localMemory, std::vector<int*>& devicePtr, std::unordered_map<int, std::shared_ptr<mscclpp::Epoch>> epochs, int numBuffers){

  assert((deviceBufferSize / sizeof(int)) % worldSize == 0);
  size_t dataCount = deviceBufferSize / sizeof(int);

  device_buffer_init(rank, worldSize, dataCount, devicePtr);
  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "CUDA memory initialization passed" << std::endl;

  mscclpp::DeviceEpoch* deviceEpochs;
  CUDATHROW(cudaMalloc(&deviceEpochs, sizeof(mscclpp::DeviceEpoch) * worldSize));
  for (int i = 0; i < worldSize; i++){
    if (i != rank){
      mscclpp::DeviceEpoch deviceEpoch = epochs[i]->deviceEpoch();
      CUDATHROW(cudaMemcpy(&deviceEpochs[i], &deviceEpoch, sizeof(mscclpp::DeviceEpoch), cudaMemcpyHostToDevice));
    }
  }
  CUDATHROW(cudaDeviceSynchronize());

  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "CUDA device epochs are created" << std::endl;

  
  for (int n = 0; n < numBuffers; n++){
    write_remote(rank, worldSize, connections, remoteMemory[n], localMemory[n], dataCount / worldSize);
  }

  increament_epochs<<<1, worldSize>>>(deviceEpochs, rank, worldSize);
  CUDATHROW(cudaDeviceSynchronize());

  for (int i = 0; i < worldSize; i++){
    if (i != rank){
      epochs[i]->signal();
    }
  }

  wait_epochs<<<1, worldSize>>>(deviceEpochs, rank, worldSize);
  CUDATHROW(cudaDeviceSynchronize());

  if (!test_device_buffer_write_correctness(worldSize, dataCount, devicePtr)){
    throw std::runtime_error("unexpected result.");
  }

  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "--- Testing writes with singal for " << std::to_string(numBuffers) << " buffers passed ---" << std::endl;
}

void test_communicator(int rank, int worldSize, int nranksPerNode)
{
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(rank, worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0)
    id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);

  mscclpp::Communicator communicator(bootstrap);
  if (bootstrap->getRank() == 0)
    std::cout << "Communicator initialization passed" << std::endl;

  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> connections;
  auto myIbDevice = findIb(rank % nranksPerNode);

  make_connections(communicator, rank, worldSize, nranksPerNode, myIbDevice, connections);
  if (bootstrap->getRank() == 0)
    std::cout << "Connection setup passed" << std::endl;

  int numBuffers = 10;
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

  test_write(rank, worldSize, deviceBufferSize, bootstrap, connections, remoteMemory, localMemory, devicePtr, numBuffers);
  if (bootstrap->getRank() == 0)
    std::cout << "--- Testing vanialla writes passed ---" << std::endl;

  std::unordered_map<int, std::shared_ptr<mscclpp::Epoch>> epochs;
  for (auto entry : connections) {
    auto& conn = entry.second;
    epochs.insert({entry.first, std::make_shared<mscclpp::Epoch>(communicator, conn)});
  }
  communicator.setup();
  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "Epochs are created" << std::endl;

  test_write_with_epochs(rank, worldSize, deviceBufferSize, bootstrap, connections, remoteMemory, localMemory, devicePtr, epochs, numBuffers);

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