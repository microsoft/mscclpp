#include <mpi.h>

#include <mscclpp/executor.hpp>

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
  // sleep 10s
  // std::this_thread::sleep_for(std::chrono::seconds(20));
  auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);
  std::shared_ptr<mscclpp::Executor> executor = std::make_shared<mscclpp::Executor>(comm, 8 /*nranksPerNode*/);

  mscclpp::ExecutionPlan plan(MSCCLPP_ROOT_PATH + "/test/execution-files/allreduce.json");
  std::shared_ptr<char> sendbuff = mscclpp::allocExtSharedCuda<char>(1024);
  std::shared_ptr<char> recvbuff = mscclpp::allocExtSharedCuda<char>(1024);
  executor->execute(rank, sendbuff.get(), recvbuff.get(), 1024, 1024, plan);

  MPI_Finalize();
  return 0;
}
