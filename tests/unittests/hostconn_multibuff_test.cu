#include "mscclpp.h"

#include "../common.h"
#include "comm.h"

#include "mpi.h"
#include <vector>

namespace {
std::vector<std::pair<void*, size_t>> buffers;
std::vector<std::vector<mscclppBufferHandle_t>> bufferHandles;

int is_main_thread = 0;
} // namespace

inline void getChunkOffsetAndSize(int rank, int wordSize, size_t dataSize, size_t& offset, size_t& chunkSize)
{
  size_t rest = dataSize % wordSize;
  size_t chunkSizePerRank = (dataSize - rest) / wordSize;
  offset = chunkSizePerRank * rank;
  chunkSize = chunkSizePerRank;
  if (rank == wordSize - 1) {
    chunkSize += rest;
  }
}

testResult_t allocateBuffer(const std::vector<size_t>& bufferSize, const void* hostBuffer, size_t maxBufferSize,
                            int rank)
{
  for (size_t i = 0; i < maxBufferSize; i++) {
    ((uint8_t*)hostBuffer)[i] = rank;
  }

  buffers.resize(bufferSize.size());
  for (size_t i = 0; i < bufferSize.size(); i++) {
    CUDACHECK(cudaMalloc(&buffers[i].first, bufferSize[i]));
    buffers[i].second = bufferSize[i];
    CUDACHECK(cudaMemcpy(buffers[i].first, hostBuffer, bufferSize[i], cudaMemcpyHostToDevice));
  }
  return testSuccess;
}

// This is only works for single node
testResult_t setupMscclppConnections(int rank, int worldSize, mscclppComm_t comm)
{
  for (int r = 0; r < worldSize; ++r) {
    if (r == rank)
      continue;

    // Connect with all other ranks
    MSCCLPPCHECK(mscclppConnectWithoutBuffer(comm, r, 0, mscclppTransportP2P, nullptr));
    std::vector<mscclppBufferHandle_t> handles(buffers.size(), 0);
    for (size_t i = 0; i < buffers.size(); i++) {
      void* data = buffers[i].first;
      int connIdx = comm->nConns - 1;
      size_t dataSize = buffers[i].second;
      MSCCLPPCHECK(mscclppRegisterBufferForConnection(comm, connIdx, data, dataSize, &handles[i]));
    }
    bufferHandles.push_back(std::move(handles));
  }

  MSCCLPPCHECK(mscclppConnectionSetup(comm));
  return testSuccess;
}

mscclppResult_t runMultiBufferTest(mscclppComm_t comm, int rank, int wordSize)
{
  int nConns = comm->nConns;
  mscclppConn* hostConns = comm->conns;
  for (int i = 0; i < nConns; i++) {
    mscclppConn* conn = &hostConns[i];
    for (size_t j = 0; j < bufferHandles[i].size(); j++) {
      mscclppBufferHandle_t handle = bufferHandles[i][j];
      size_t dataSize = conn->bufferRegistrations[handle].size;
      size_t offset, chunkSize;
      getChunkOffsetAndSize(rank, wordSize, dataSize, offset, chunkSize);
      conn->hostConn->put(handle, offset, handle, offset, chunkSize);
    }
    conn->hostConn->signal();
    conn->hostConn->flush();
  }
  return mscclppSuccess;
}

void initExpectedData(void* hostBuffer, int wordSize, size_t dataSize)
{
  size_t offset, chunkSize;
  for (int rank = 0; rank < wordSize; rank++) {
    getChunkOffsetAndSize(rank, wordSize, dataSize, offset, chunkSize);
    for (size_t i = 0; i < chunkSize; i++) {
      ((uint8_t*)hostBuffer)[offset + i] = rank;
    }
  }
}

testResult_t checkData(mscclppComm_t comm, void* hostBuffer, int worldSize)
{
  int nConns = comm->nConns;
  mscclppConn* hostConns = comm->conns;
  for (int i = 0; i < nConns; i++) {
    mscclppConn* conn = &hostConns[i];
    for (size_t j = 0; j < bufferHandles[i].size(); j++) {
      mscclppBufferHandle_t handle = bufferHandles[i][j];
      void* data = conn->bufferRegistrations[handle].data;
      size_t size = conn->bufferRegistrations[handle].size;
      uint8_t* hostData = new uint8_t[size];
      CUDACHECK(cudaMemcpy(hostData, data, size, cudaMemcpyDeviceToHost));
      initExpectedData(hostBuffer, worldSize, size);
      if (memcmp(hostData, hostBuffer, size) != 0) {
        printf("Data mismatch for conn %d\n", i);
        return testInternalError;
      }
      free(hostData);
    }
  }
  return testSuccess;
}

int main(int argc, const char* argv[])
{
  int rank;
  int worldSize;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  mscclppUniqueId mscclppId;
  if (rank == 0) {
    MSCCLPPCHECK(mscclppGetUniqueId(&mscclppId));
  }
  MPI_Bcast((void*)&mscclppId, sizeof(mscclppId), MPI_BYTE, 0, MPI_COMM_WORLD);

  CUDACHECK(cudaSetDevice(rank));

  is_main_thread = (rank == 0) ? 1 : 0;
  PRINT("Initializing MSCCL++\n");
  mscclppComm_t comm;
  MSCCLPPCHECK(mscclppCommInitRankFromId(&comm, worldSize, mscclppId, rank));

  PRINT("Initializing data for testing\n");
  std::vector<size_t> bufferSizes(4, 0);
  for (size_t i = 0; i < bufferSizes.size(); i++) {
    bufferSizes[i] = (i + 1) * 32;
  }
  size_t maxBufferSize = bufferSizes.back();
  uint8_t* hostBuffer = new uint8_t[maxBufferSize];
  TESTCHECK(allocateBuffer(bufferSizes, hostBuffer, maxBufferSize, rank));

  PRINT("Setting up the connection in MSCCL++\n");
  TESTCHECK(setupMscclppConnections(rank, worldSize, comm));

  PRINT("Launching MSCCL++ proxy threads\n");
  MSCCLPPCHECK(mscclppProxyLaunch(comm));

  PRINT("Starting to run multi buffer test for host conn\n");
  MSCCLPPCHECK(runMultiBufferTest(comm, rank, worldSize));
  CUDACHECK(cudaDeviceSynchronize());
  int tmp[16];
  // A simple barrier
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));

  TESTCHECK(checkData(comm, hostBuffer, worldSize));
  PRINT("Correctness check passed\n");

  PRINT("Stopping MSCCL++ proxy threads\n");
  MSCCLPPCHECK(mscclppProxyStop(comm));

  PRINT("Destroying MSCCL++ communicator\n");
  MSCCLPPCHECK(mscclppCommDestroy(comm));

  free(hostBuffer);
  MPI_Finalize();
  return 0;
}
