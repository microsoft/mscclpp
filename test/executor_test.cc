#include <mpi.h>

#include <mscclpp/executor.hpp>
#include <mscclpp/gpu_utils.hpp>

// Check CUDA RT calls
#define CUDACHECK(cmd)                                                                  \
  do {                                                                                  \
    cudaError_t err = cmd;                                                              \
    if (err != cudaSuccess) {                                                           \
      printf("%s:%d Cuda failure '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  } while (false)

const std::string MSCCLPP_ROOT_PATH = "/root/mscclpp";

int main() {
  int rank;
  int world_size;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
  mscclpp::UniqueId id;
  if (rank == 0) {
    id = bootstrap->createUniqueId();
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  // sleep 20s
  // std::this_thread::sleep_for(std::chrono::seconds(20));
  auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);
  CUDACHECK(cudaSetDevice(rank));

  std::shared_ptr<mscclpp::Executor> executor = std::make_shared<mscclpp::Executor>(comm, 8 /*nranksPerNode*/);
  mscclpp::ExecutionPlan plan(MSCCLPP_ROOT_PATH + "/test/execution-files/allreduce.json");
  const int bufferSize = 1024 * 1024;
  std::shared_ptr<char> sendbuff = mscclpp::allocExtSharedCuda<char>(bufferSize);
  mscclpp::CudaStreamWithFlags stream(cudaStreamNonBlocking);
  executor->execute(rank, sendbuff.get(), sendbuff.get(), bufferSize, bufferSize, mscclpp::DataType::FLOAT16, 512, plan,
                    stream);
  CUDACHECK(cudaStreamSynchronize(stream));

  MPI_Finalize();
  return 0;
}
