#include <cuda_runtime.h>
#include <mpi.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/epoch.hpp>
#include <unordered_map>

#define CUDATHROW(cmd)                                                                         \
  do {                                                                                         \
    cudaError_t err = cmd;                                                                     \
    if (err != cudaSuccess) {                                                                  \
      throw std::runtime_error(std::string("Cuda failure '") + cudaGetErrorString(err) + "'"); \
    }                                                                                          \
  } while (false)

mscclpp::Transport findIb(int localRank) {
  mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                              mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                              mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  return IBs[localRank];
}

void register_all_memories(mscclpp::Communicator& communicator, int rank, int worldSize, void* devicePtr,
                           size_t deviceBufferSize, mscclpp::Transport myIbDevice,
                           mscclpp::RegisteredMemory& localMemory,
                           std::unordered_map<int, mscclpp::RegisteredMemory>& remoteMemory) {
  localMemory = communicator.registerMemory(devicePtr, deviceBufferSize, mscclpp::Transport::CudaIpc | myIbDevice);
  std::unordered_map<int, mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> futureRemoteMemory;
  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
      communicator.sendMemoryOnSetup(localMemory, i, 0);
      futureRemoteMemory[i] = communicator.recvMemoryOnSetup(i, 0);
    }
  }
  communicator.setup();
  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
      remoteMemory[i] = futureRemoteMemory[i].get();
    }
  }
}

void make_connections(mscclpp::Communicator& communicator, int rank, int worldSize, int nRanksPerNode,
                      mscclpp::Transport myIbDevice,
                      std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections) {
  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
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
                  std::unordered_map<int, mscclpp::RegisteredMemory>& remoteRegisteredMemories,
                  mscclpp::RegisteredMemory& registeredMemory, int dataCountPerRank) {
  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
      auto& conn = connections.at(i);
      auto& peerMemory = remoteRegisteredMemories.at(i);
      conn->write(peerMemory, rank * dataCountPerRank * sizeof(int), registeredMemory,
                  rank * dataCountPerRank * sizeof(int), dataCountPerRank * sizeof(int));
      conn->flush();
    }
  }
}

void device_buffer_init(int rank, int worldSize, int dataCount, std::vector<int*>& devicePtr) {
  for (int n = 0; n < (int)devicePtr.size(); n++) {
    std::vector<int> hostBuffer(dataCount, 0);
    for (int i = 0; i < dataCount; i++) {
      hostBuffer[i] = rank + n * worldSize;
    }
    CUDATHROW(cudaMemcpy(devicePtr[n], hostBuffer.data(), dataCount * sizeof(int), cudaMemcpyHostToDevice));
  }
  CUDATHROW(cudaDeviceSynchronize());
}

bool test_device_buffer_write_correctness(int rank, int worldSize, int nRanksPerNode, int dataCount,
                                          std::vector<int*>& devicePtr, bool skipLocal = false) {
  for (int n = 0; n < (int)devicePtr.size(); n++) {
    std::vector<int> hostBuffer(dataCount, 0);
    CUDATHROW(cudaMemcpy(hostBuffer.data(), devicePtr[n], dataCount * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < worldSize; i++) {
      if (i / nRanksPerNode == rank / nRanksPerNode && skipLocal) {
        continue;
      }
      for (int j = i * dataCount / worldSize; j < (i + 1) * dataCount / worldSize; j++) {
        if (hostBuffer[j] != i + n * worldSize) {
          return false;
        }
      }
    }
  }
  return true;
}

void test_write(int rank, int worldSize, int nRanksPerNode, int deviceBufferSize,
                std::shared_ptr<mscclpp::BaseBootstrap> bootstrap,
                std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections,
                std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>>& remoteMemory,
                std::vector<mscclpp::RegisteredMemory>& localMemory, std::vector<int*>& devicePtr, int numBuffers) {
  assert((deviceBufferSize / sizeof(int)) % worldSize == 0);
  size_t dataCount = deviceBufferSize / sizeof(int);

  device_buffer_init(rank, worldSize, dataCount, devicePtr);
  bootstrap->barrier();
  if (bootstrap->getRank() == 0) std::cout << "CUDA memory initialization passed" << std::endl;

  for (int n = 0; n < numBuffers; n++) {
    write_remote(rank, worldSize, connections, remoteMemory[n], localMemory[n], dataCount / worldSize);
  }
  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "RDMA write for " << std::to_string(numBuffers) << " buffers passed" << std::endl;

  // polling until it becomes ready
  bool ready = false;
  int niter = 0;
  do {
    ready = test_device_buffer_write_correctness(rank, worldSize, nRanksPerNode, dataCount, devicePtr);
    niter++;
    if (niter == 10000) {
      throw std::runtime_error("Polling is stuck.");
    }
  } while (!ready);

  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "Polling for " << std::to_string(numBuffers) << " buffers passed" << std::endl;

  if (bootstrap->getRank() == 0) std::cout << "--- Testing vanialla writes passed ---" << std::endl;
}

__global__ void increament_epochs(mscclpp::DeviceEpoch::DeviceHandle* deviceEpochs, int rank, int worldSize) {
  int tid = threadIdx.x;
  if (tid != rank && tid < worldSize) {
    deviceEpochs[tid].epochIncrement();
  }
}

__global__ void wait_epochs(mscclpp::DeviceEpoch::DeviceHandle* deviceEpochs, int rank, int worldSize) {
  int tid = threadIdx.x;
  if (tid != rank && tid < worldSize) {
    deviceEpochs[tid].wait();
  }
}

void test_write_with_device_epochs(int rank, int worldSize, int nRanksPerNode, int deviceBufferSize,
                                   mscclpp::Communicator& communicator,
                                   std::shared_ptr<mscclpp::BaseBootstrap> bootstrap,
                                   std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections,
                                   std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>>& remoteMemory,
                                   std::vector<mscclpp::RegisteredMemory>& localMemory, std::vector<int*>& devicePtr,
                                   int numBuffers) {
  std::unordered_map<int, std::shared_ptr<mscclpp::DeviceEpoch>> epochs;
  for (auto entry : connections) {
    auto& conn = entry.second;
    epochs.insert({entry.first, std::make_shared<mscclpp::DeviceEpoch>(communicator, conn)});
  }
  communicator.setup();
  bootstrap->barrier();
  if (bootstrap->getRank() == 0) std::cout << "Epochs are created" << std::endl;

  assert((deviceBufferSize / sizeof(int)) % worldSize == 0);
  size_t dataCount = deviceBufferSize / sizeof(int);

  device_buffer_init(rank, worldSize, dataCount, devicePtr);
  bootstrap->barrier();
  if (bootstrap->getRank() == 0) std::cout << "CUDA memory initialization passed" << std::endl;

  mscclpp::DeviceEpoch::DeviceHandle* deviceEpochHandles;
  CUDATHROW(cudaMalloc(&deviceEpochHandles, sizeof(mscclpp::DeviceEpoch::DeviceHandle) * worldSize));
  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
      mscclpp::DeviceEpoch::DeviceHandle deviceHandle = epochs[i]->deviceHandle();
      CUDATHROW(cudaMemcpy(&deviceEpochHandles[i], &deviceHandle, sizeof(mscclpp::DeviceEpoch::DeviceHandle),
                           cudaMemcpyHostToDevice));
    }
  }
  CUDATHROW(cudaDeviceSynchronize());

  bootstrap->barrier();
  if (bootstrap->getRank() == 0) std::cout << "CUDA device epochs are created" << std::endl;

  for (int n = 0; n < numBuffers; n++) {
    write_remote(rank, worldSize, connections, remoteMemory[n], localMemory[n], dataCount / worldSize);
  }

  increament_epochs<<<1, worldSize>>>(deviceEpochHandles, rank, worldSize);
  CUDATHROW(cudaDeviceSynchronize());

  for (int i = 0; i < worldSize; i++) {
    if (i != rank) {
      epochs[i]->signal();
    }
  }

  wait_epochs<<<1, worldSize>>>(deviceEpochHandles, rank, worldSize);
  CUDATHROW(cudaDeviceSynchronize());

  if (!test_device_buffer_write_correctness(rank, worldSize, nRanksPerNode, dataCount, devicePtr)) {
    throw std::runtime_error("unexpected result.");
  }

  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "--- Testing writes with device epochs for " << std::to_string(numBuffers) << " buffers passed ---"
              << std::endl;
}

void test_write_with_host_epochs(int rank, int worldSize, int nRanksPerNode, int deviceBufferSize,
                                 mscclpp::Communicator& communicator, std::shared_ptr<mscclpp::BaseBootstrap> bootstrap,
                                 std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>& connections,
                                 std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>>& remoteMemory,
                                 std::vector<mscclpp::RegisteredMemory>& localMemory, std::vector<int*>& devicePtr,
                                 int numBuffers) {
  std::unordered_map<int, std::shared_ptr<mscclpp::HostEpoch>> epochs;
  for (auto entry : connections) {
    auto& conn = entry.second;
    if (conn->transport() == mscclpp::Transport::CudaIpc) continue;
    epochs.insert({entry.first, std::make_shared<mscclpp::HostEpoch>(communicator, conn)});
  }
  communicator.setup();
  bootstrap->barrier();
  if (bootstrap->getRank() == 0) std::cout << "Epochs are created" << std::endl;

  assert((deviceBufferSize / sizeof(int)) % worldSize == 0);
  size_t dataCount = deviceBufferSize / sizeof(int);

  device_buffer_init(rank, worldSize, dataCount, devicePtr);
  bootstrap->barrier();
  if (bootstrap->getRank() == 0) std::cout << "CUDA memory initialization passed" << std::endl;

  bootstrap->barrier();
  if (bootstrap->getRank() == 0) std::cout << "Host epochs are created" << std::endl;

  for (int n = 0; n < numBuffers; n++) {
    write_remote(rank, worldSize, connections, remoteMemory[n], localMemory[n], dataCount / worldSize);
  }

  for (int i = 0; i < worldSize; i++) {
    if (i != rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      epochs[i]->incrementAndSignal();
    }
  }

  for (int i = 0; i < worldSize; i++) {
    if (i != rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      epochs[i]->wait();
    }
  }

  if (!test_device_buffer_write_correctness(rank, worldSize, nRanksPerNode, dataCount, devicePtr, true)) {
    throw std::runtime_error("unexpected result.");
  }

  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "--- Testing writes with host epochs for " << std::to_string(numBuffers) << " buffers passed ---"
              << std::endl;
}

void test_communicator(int rank, int worldSize, int nRanksPerNode) {
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(rank, worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);

  mscclpp::Communicator communicator(bootstrap);
  if (bootstrap->getRank() == 0) std::cout << "Communicator initialization passed" << std::endl;

  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> connections;
  auto myIbDevice = findIb(rank % nRanksPerNode);

  make_connections(communicator, rank, worldSize, nRanksPerNode, myIbDevice, connections);
  if (bootstrap->getRank() == 0) std::cout << "Connection setup passed" << std::endl;

  int numBuffers = 10;
  std::vector<int*> devicePtr(numBuffers);
  int deviceBufferSize = 1024 * 1024;

  std::vector<mscclpp::RegisteredMemory> localMemory(numBuffers);
  std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>> remoteMemory(numBuffers);

  for (int n = 0; n < numBuffers; n++) {
    if (n % 100 == 0) std::cout << "Registering memory for " << std::to_string(n) << " buffers" << std::endl;
    CUDATHROW(cudaMalloc(&devicePtr[n], deviceBufferSize));
    register_all_memories(communicator, rank, worldSize, devicePtr[n], deviceBufferSize, myIbDevice, localMemory[n],
                          remoteMemory[n]);
  }
  bootstrap->barrier();
  if (bootstrap->getRank() == 0)
    std::cout << "Memory registration for " << std::to_string(numBuffers) << " buffers passed" << std::endl;

  test_write(rank, worldSize, nRanksPerNode, deviceBufferSize, bootstrap, connections, remoteMemory, localMemory,
             devicePtr, numBuffers);

  test_write_with_device_epochs(rank, worldSize, nRanksPerNode, deviceBufferSize, communicator, bootstrap, connections,
                                remoteMemory, localMemory, devicePtr, numBuffers);

  test_write_with_host_epochs(rank, worldSize, nRanksPerNode, deviceBufferSize, communicator, bootstrap, connections,
                              remoteMemory, localMemory, devicePtr, numBuffers);

  if (bootstrap->getRank() == 0) std::cout << "--- MSCCLPP::Communicator tests passed! ---" << std::endl;

  for (int n = 0; n < numBuffers; n++) {
    CUDATHROW(cudaFree(devicePtr[n]));
  }
}

int main(int argc, char** argv) {
  int rank, worldSize;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  int shmWorldSize;
  MPI_Comm_size(shmcomm, &shmWorldSize);
  int nRanksPerNode = shmWorldSize;
  MPI_Comm_free(&shmcomm);

  test_communicator(rank, worldSize, nRanksPerNode);

  MPI_Finalize();
  return 0;
}
