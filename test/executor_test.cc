#include <mpi.h>
#include <unistd.h>

#include <iostream>
#include <mscclpp/executor.hpp>
#include <mscclpp/npkit/npkit.hpp>
#include <mscclpp/utils.hpp>
#include <sstream>

double parseSize(const char* value) {
  std::string valueStr(value);
  std::istringstream iss(valueStr);
  long long int units;
  double size;
  char size_lit = 0;

  if (iss >> size) {
    iss >> std::ws;  // eat whitespace
    iss >> size_lit;
  } else {
    return -1.0;
  }

  if (size_lit != 0 && !std::isspace(size_lit)) {
    switch (size_lit) {
      case 'G':
      case 'g':
        units = 1024 * 1024 * 1024;
        break;
      case 'M':
      case 'm':
        units = 1024 * 1024;
        break;
      case 'K':
      case 'k':
        units = 1024;
        break;
      default:
        return -1.0;
    };
  } else {
    units = 1;
  }
  return size * units;
}

double benchTime(int rank, std::shared_ptr<mscclpp::Bootstrap> bootstrap, std::shared_ptr<mscclpp::Executor> executor,
                 const mscclpp::ExecutionPlan& plan, std::shared_ptr<char> sendbuff, size_t bufferSize,
                 int nthreadsPerBlock, int niters, int ngrapthIters) {
  mscclpp::CudaStreamWithFlags stream(cudaStreamNonBlocking);
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  mscclpp::Timer timer;
  MSCCLPP_CUDATHROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < niters; i++) {
    executor->execute(rank, sendbuff.get(), sendbuff.get(), bufferSize, bufferSize, mscclpp::DataType::FLOAT16,
                      nthreadsPerBlock, plan, stream, mscclpp::PacketType::LL16);
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

int main(int argc, char* argv[]) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0] << " <buffer size>"
              << " <execution plan name>"
              << " <execution plan path>"
              << " <nthreads per block>"
              << " <number of iterations>"
              << " <number of graph iterations>"
              << " <enable npkit>" << std::endl;
    return 1;
  }

  int rank;
  int worldSize;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MSCCLPP_CUDATHROW(cudaSetDevice(rank));

  const size_t bufferSize = parseSize(argv[1]);
  const std::string executionPlanName = argv[2];
  const std::string executionPlanPath = argv[3];
  const int nthreadsPerBlock = std::stoi(argv[4]);
  const int niters = std::stoi(argv[5]);
  const int ngraphIters = std::stoi(argv[6]);
  const int enableNpKit = std::stoi(argv[7]);

  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  mscclpp::UniqueId id;
  bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, worldSize);
  if (rank == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  std::shared_ptr<mscclpp::Communicator> communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
  std::shared_ptr<mscclpp::Executor> executor = std::make_shared<mscclpp::Executor>(communicator);

  if (enableNpKit) {
    NpKit::Init(rank);
  }

  mscclpp::ExecutionPlan plan(executionPlanName, executionPlanPath);
  std::shared_ptr<char> sendbuff = mscclpp::allocExtSharedCuda<char>(bufferSize);
  std::vector<int> dataHost(bufferSize / sizeof(int), rank);
  MSCCLPP_CUDATHROW(cudaMemcpy(sendbuff.get(), dataHost.data(), bufferSize, cudaMemcpyHostToDevice));
  double deltaSec =
      benchTime(rank, bootstrap, executor, plan, sendbuff, bufferSize, nthreadsPerBlock, niters, ngraphIters);

  if (enableNpKit) {
    const char* npkitDumpDir = getenv("NPKIT_DUMP_DIR");
    if (npkitDumpDir == nullptr) {
      std::cerr << "NPKIT_DUMP_DIR is empty" << std::endl;
    } else {
      NpKit::Dump(npkitDumpDir);
    }
    NpKit::Shutdown();
  }

  std::cout << "Rank " << rank << ": " << bufferSize << " bytes " << deltaSec * 1.e6 << " us" << std::endl;
  MPI_Finalize();
  return 0;
}
