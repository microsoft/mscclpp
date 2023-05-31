#include <gtest/gtest.h>
#include <mpi.h>

#include <iostream>
#include <mscclpp/core.hpp>
#include <mscclpp/cuda_utils.hpp>
#include <mscclpp/epoch.hpp>
#include <sstream>

static const char gDefaultIpPort[] = "127.0.0.1:50053";

class MultiProcessTestEnv : public ::testing::Environment {
 public:
  MultiProcessTestEnv(int argc, const char** argv) : argc(argc), argv(argv) {}

  // Override this to define how to set up the environment.
  void SetUp() {
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

  // Override this to define how to tear down the environment.
  void TearDown() { MPI_Finalize(); }

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

  const int argc;
  const char** argv;
  int rank;
  int worldSize;
  int nRanksPerNode;
  std::unordered_map<std::string, std::string> args;
};

MultiProcessTestEnv* gEnv = nullptr;

class MultiProcessTest : public ::testing::Test {};

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bootstrap tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class BootstrapTest : public MultiProcessTest {
};

void bootstrapTestAllGather(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap) {
  std::vector<int> tmp(bootstrap->getNranks(), 0);
  tmp[bootstrap->getRank()] = bootstrap->getRank() + 1;
  bootstrap->allGather(tmp.data(), sizeof(int));
  for (int i = 0; i < bootstrap->getNranks(); ++i) {
    EXPECT_EQ(tmp[i], i + 1);
  }
}

void bootstrapTestBarrier(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap) { bootstrap->barrier(); }

void bootstrapTestSendRecv(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap) {
  for (int i = 0; i < bootstrap->getNranks(); i++) {
    if (bootstrap->getRank() == i) continue;
    int msg1 = (bootstrap->getRank() + 1) * 3;
    int msg2 = (bootstrap->getRank() + 1) * 3 + 1;
    int msg3 = (bootstrap->getRank() + 1) * 3 + 2;
    bootstrap->send(&msg1, sizeof(int), i, 0);
    bootstrap->send(&msg2, sizeof(int), i, 1);
    bootstrap->send(&msg3, sizeof(int), i, 2);
  }

  for (int i = 0; i < bootstrap->getNranks(); i++) {
    if (bootstrap->getRank() == i) continue;
    int msg1 = 0;
    int msg2 = 0;
    int msg3 = 0;
    // recv them in the opposite order to check correctness
    bootstrap->recv(&msg2, sizeof(int), i, 1);
    bootstrap->recv(&msg3, sizeof(int), i, 2);
    bootstrap->recv(&msg1, sizeof(int), i, 0);
    EXPECT_EQ(msg1, (i + 1) * 3);
    EXPECT_EQ(msg2, (i + 1) * 3 + 1);
    EXPECT_EQ(msg3, (i + 1) * 3 + 2);
  }
}

void bootstrapTestAll(std::shared_ptr<mscclpp::BaseBootstrap> bootstrap) {
  bootstrapTestAllGather(bootstrap);
  bootstrapTestBarrier(bootstrap);
  bootstrapTestSendRecv(bootstrap);
}

TEST_F(BootstrapTest, WithId) {
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, gEnv->worldSize);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  bootstrapTestAll(bootstrap);
}

TEST_F(BootstrapTest, WithIpPortPair) {
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, gEnv->worldSize);
  bootstrap->initialize(gEnv->args["ip_port"]);
  bootstrapTestAll(bootstrap);
}

class MPIBootstrap : public mscclpp::BaseBootstrap {
 public:
  MPIBootstrap() : BaseBootstrap() {}
  int getRank() override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }
  int getNranks() override {
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    return worldSize;
  }
  void allGather(void* sendbuf, int size) override {
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, sendbuf, size, MPI_BYTE, MPI_COMM_WORLD);
  }
  void barrier() override { MPI_Barrier(MPI_COMM_WORLD); }
  void send(void* sendbuf, int size, int dest, int tag) override {
    MPI_Send(sendbuf, size, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
  }
  void recv(void* recvbuf, int size, int source, int tag) override {
    MPI_Recv(recvbuf, size, MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
};

TEST_F(BootstrapTest, MPIBootstrap) {
  auto bootstrap = std::make_shared<MPIBootstrap>();
  bootstrapTestAll(bootstrap);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Communicator tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CommunicatorTest : public MultiProcessTest {
protected:
  void SetUp() override {
    MultiProcessTest::SetUp();

    ASSERT_EQ((deviceBufferSize / sizeof(int)) % gEnv->worldSize, 0);

    communicator = std::make_shared<mscclpp::Communicator>(createBootstrap());
    myIbDevice = findIb(gEnv->rank % gEnv->nRanksPerNode);

    makeConnections();

    devicePtr.resize(numBuffers);
    localMemory.resize(numBuffers);
    remoteMemory.resize(numBuffers);

    for (int n = 0; n < numBuffers; n++) {
      devicePtr[n] = mscclpp::allocSharedCuda<int>(deviceBufferSize / sizeof(int));
      registerAllMemories(devicePtr[n], localMemory[n], remoteMemory[n]);
    }
  }

  void TearDown() override {
    remoteMemory.clear();
    localMemory.clear();
    devicePtr.clear();
    connections.clear();
    communicator.reset();
    MultiProcessTest::TearDown();
  }

  std::shared_ptr<mscclpp::Bootstrap> createBootstrap() const {
    auto bootstrap = std::make_shared<mscclpp::Bootstrap>(gEnv->rank, gEnv->worldSize);
    mscclpp::UniqueId id;
    if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    bootstrap->initialize(id);
    return bootstrap;
  }

  mscclpp::Transport findIb(int localRank) const {
    mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                                mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                                mscclpp::Transport::IB6, mscclpp::Transport::IB7};
    return IBs[localRank];
  }

  void makeConnections() {
    for (int i = 0; i < gEnv->worldSize; i++) {
      if (i != gEnv->rank) {
        if (i / gEnv->nRanksPerNode == gEnv->rank / gEnv->nRanksPerNode) {
          connections[i] = communicator->connectOnSetup(i, 0, mscclpp::Transport::CudaIpc);
        } else {
          connections[i] = communicator->connectOnSetup(i, 0, myIbDevice);
        }
      }
    }
    communicator->setup();
  }

  void registerAllMemories(std::shared_ptr<int> devicePtr, mscclpp::RegisteredMemory& localMem,
                           std::unordered_map<int, mscclpp::RegisteredMemory>& remoteMem) {
    localMem = communicator->registerMemory(devicePtr.get(), deviceBufferSize, mscclpp::Transport::CudaIpc | myIbDevice);
    std::unordered_map<int, mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> futureRemoteMemory;
    for (int i = 0; i < gEnv->worldSize; i++) {
      if (i != gEnv->rank) {
        communicator->sendMemoryOnSetup(localMem, i, 0);
        futureRemoteMemory[i] = communicator->recvMemoryOnSetup(i, 0);
      }
    }
    communicator->setup();
    for (int i = 0; i < gEnv->worldSize; i++) {
      if (i != gEnv->rank) {
        remoteMem[i] = futureRemoteMemory[i].get();
      }
    }
  }

  void deviceBufferInit() {
    size_t dataCount = deviceBufferSize / sizeof(int);
    for (int n = 0; n < (int)devicePtr.size(); n++) {
      std::vector<int> hostBuffer(dataCount, 0);
      for (int i = 0; i < dataCount; i++) {
        hostBuffer[i] = gEnv->rank + n * gEnv->worldSize;
      }
      mscclpp::memcpyCuda<int>(devicePtr[n].get(), hostBuffer.data(), dataCount, cudaMemcpyHostToDevice);
    }
  }

  void writeToRemote(int dataCountPerRank)
  {
    for (int n = 0; n < numBuffers; n++) {
      for (int i = 0; i < gEnv->worldSize; i++) {
        if (i != gEnv->rank) {
          auto& conn = connections.at(i);
          auto& peerMemory = remoteMemory[n].at(i);
          conn->write(peerMemory, gEnv->rank * dataCountPerRank * sizeof(int), localMemory[n],
                      gEnv->rank * dataCountPerRank * sizeof(int), dataCountPerRank * sizeof(int));
          conn->flush();
        }
      }
    }
  }

  bool testWriteCorrectness(bool skipLocal = false)
  {
    size_t dataCount = deviceBufferSize / sizeof(int);
    for (int n = 0; n < (int)devicePtr.size(); n++) {
      std::vector<int> hostBuffer(dataCount, 0);
      mscclpp::memcpyCuda<int>(hostBuffer.data(), devicePtr[n].get(), dataCount, cudaMemcpyDeviceToHost);
      for (int i = 0; i < gEnv->worldSize; i++) {
        if (((i / gEnv->nRanksPerNode) == (gEnv->rank / gEnv->nRanksPerNode)) && skipLocal) {
          continue;
        }
        for (int j = i * dataCount / gEnv->worldSize; j < (i + 1) * dataCount / gEnv->worldSize; j++) {
          if (hostBuffer[j] != i + n * gEnv->worldSize) {
            return false;
          }
        }
      }
    }
    return true;
  }

  std::shared_ptr<mscclpp::Communicator> communicator;
  std::unordered_map<int, std::shared_ptr<mscclpp::Connection>> connections;
  mscclpp::Transport myIbDevice;

  const size_t numBuffers = 10;
  const int deviceBufferSize = 1024 * 1024;
  std::vector<std::shared_ptr<int>> devicePtr;
  std::vector<mscclpp::RegisteredMemory> localMemory;
  std::vector<std::unordered_map<int, mscclpp::RegisteredMemory>> remoteMemory;
};

TEST_F(CommunicatorTest, BasicWrite) {
  deviceBufferInit();
  communicator->bootstrapper()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);
  communicator->bootstrapper()->barrier();

  // polling until it becomes ready
  bool ready = false;
  int niter = 0;
  do {
    ready = testWriteCorrectness();
    niter++;
    if (niter == 10000) {
      FAIL() << "Polling is stuck.";
    }
  } while (!ready);
  communicator->bootstrapper()->barrier();
}

__global__ void kernelIncEpochs(mscclpp::DeviceEpoch::DeviceHandle* deviceEpochs, int rank, int worldSize)
{
  int tid = threadIdx.x;
  if (tid != rank && tid < worldSize) {
    deviceEpochs[tid].epochIncrement();
  }
}

__global__ void kernelWaitEpochs(mscclpp::DeviceEpoch::DeviceHandle* deviceEpochs, int rank, int worldSize)
{
  int tid = threadIdx.x;
  if (tid != rank && tid < worldSize) {
    deviceEpochs[tid].wait();
  }
}

TEST_F(CommunicatorTest, WriteWithDeviceEpochs) {
  std::unordered_map<int, std::shared_ptr<mscclpp::DeviceEpoch>> epochs;
  for (auto entry : connections) {
    auto& conn = entry.second;
    epochs.insert({entry.first, std::make_shared<mscclpp::DeviceEpoch>(*communicator.get(), conn)});
  }
  communicator->setup();
  communicator->bootstrapper()->barrier();

  deviceBufferInit();
  communicator->bootstrapper()->barrier();

  auto deviceEpochHandles = mscclpp::allocSharedCuda<mscclpp::DeviceEpoch::DeviceHandle>(gEnv->worldSize);
  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank) {
      mscclpp::DeviceEpoch::DeviceHandle deviceHandle = epochs[i]->deviceHandle();
      mscclpp::memcpyCuda<mscclpp::DeviceEpoch::DeviceHandle>(deviceEpochHandles.get() + i, &deviceHandle, 1, cudaMemcpyHostToDevice);
    }
  }
  communicator->bootstrapper()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);

  kernelIncEpochs<<<1, gEnv->worldSize>>>(deviceEpochHandles.get(), gEnv->rank, gEnv->worldSize);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank) {
      epochs[i]->signal();
    }
  }

  kernelWaitEpochs<<<1, gEnv->worldSize>>>(deviceEpochHandles.get(), gEnv->rank, gEnv->worldSize);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  ASSERT_TRUE(testWriteCorrectness());
  communicator->bootstrapper()->barrier();
}

TEST_F(CommunicatorTest, WriteWithHostEpochs) {
  std::unordered_map<int, std::shared_ptr<mscclpp::HostEpoch>> epochs;
  for (auto entry : connections) {
    auto& conn = entry.second;
    // HostEpoch cannot be used with CudaIpc transport
    if (conn->transport() == mscclpp::Transport::CudaIpc)
      continue;
    epochs.insert({entry.first, std::make_shared<mscclpp::HostEpoch>(*communicator.get(), conn)});
  }
  communicator->setup();
  communicator->bootstrapper()->barrier();

  deviceBufferInit();
  communicator->bootstrapper()->barrier();

  writeToRemote(deviceBufferSize / sizeof(int) / gEnv->worldSize);

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      epochs[i]->incrementAndSignal();
    }
  }

  for (int i = 0; i < gEnv->worldSize; i++) {
    if (i != gEnv->rank && connections[i]->transport() != mscclpp::Transport::CudaIpc) {
      epochs[i]->wait();
    }
  }

  ASSERT_TRUE(testWriteCorrectness());
  communicator->bootstrapper()->barrier();
}
