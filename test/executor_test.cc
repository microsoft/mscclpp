#include <mpi.h>
#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <mscclpp/executor.hpp>
#include <mscclpp/utils.hpp>

namespace {
std::string getExecutablePath() {
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  if (count == -1) {
    throw std::runtime_error("Failed to get executable path");
  }
  return std::string(result, count);
}
}  // namespace

double benchTime(int rank, std::shared_ptr<mscclpp::Bootstrap> bootstrap, std::shared_ptr<mscclpp::Executor> executor,
                 const mscclpp::ExecutionPlan& plan, std::shared_ptr<char> sendbuff, size_t bufferSize, int niters,
                 int ngrapthIters) {
  mscclpp::CudaStreamWithFlags stream(cudaStreamNonBlocking);
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  mscclpp::Timer timer;
  MSCCLPP_CUDATHROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < niters; i++) {
    executor->execute(rank, sendbuff.get(), sendbuff.get(), bufferSize, bufferSize, mscclpp::DataType::FLOAT16, 1024,
                      plan, stream, mscclpp::PacketType::LL8);
  }
  MSCCLPP_CUDATHROW(cudaStreamEndCapture(stream, &graph));
  MSCCLPP_CUDATHROW(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  MSCCLPP_CUDATHROW(cudaGraphLaunch(graphExec, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  bootstrap->barrier();

  timer.reset();
  for (int i = 0; i < ngrapthIters; i++) {
    MSCCLPP_CUDATHROW(cudaGraphLaunch(graphExec, stream));
  }
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  double deltaSec = timer.elapsed() * 1.e-6;
  deltaSec = deltaSec / (niters) / (ngrapthIters);
  MSCCLPP_CUDATHROW(cudaGraphExecDestroy(graphExec));
  MSCCLPP_CUDATHROW(cudaGraphDestroy(graph));
  return deltaSec;
}

int main() {
  int rank;
  int worldSize;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MSCCLPP_CUDATHROW(cudaSetDevice(rank));

  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  mscclpp::UniqueId id;
  bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, worldSize);
  if (rank == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  std::shared_ptr<mscclpp::Communicator> communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
  std::shared_ptr<mscclpp::Executor> executor = std::make_shared<mscclpp::Executor>(communicator);

  std::string executablePath = getExecutablePath();
  std::filesystem::path path = executablePath;
  std::filesystem::path executionFilesPath =
      path.parent_path().parent_path().parent_path() / "test/execution-files/allreduce1.json";
  mscclpp::ExecutionPlan plan("allreduce_pairs", executionFilesPath.string());
  const int bufferSize = 1024 * 1024 * 64;
  std::shared_ptr<char> sendbuff = mscclpp::allocExtSharedCuda<char>(bufferSize);
  double deltaSec = benchTime(rank, bootstrap, executor, plan, sendbuff, bufferSize, 200, 50);
  std::cout << "Rank " << rank << " delta micro second: " << deltaSec * 1.e6 << std::endl;
  MPI_Finalize();
  return 0;
}
