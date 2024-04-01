#include <mpi.h>

#include <fstream>
#include <mscclpp/executor.hpp>

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
  auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);
  std::shared_ptr<mscclpp::Executor> executor =
      std::make_shared<mscclpp::Executor>(comm, std::unordered_map<int, std::shared_ptr<mscclpp::Connection>>());
  std::ifstream file("execution_plan.json");
  mscclpp::ExecutionPlan plan(file);
  std::shared_ptr<char> sendbuff = mscclpp::allocExtSharedCuda<char>(1024);
  std::shared_ptr<char> recvbuff = mscclpp::allocExtSharedCuda<char>(1024);
  executor->execute(sendbuff.get(), recvbuff.get(), 1024, 1024, plan);

  MPI_Finalize();
  return 0;
}
