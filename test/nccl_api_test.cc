
#include <mpi.h>

#include "nccl.h"

int main(int argc, char** argv) {
  int rank, world_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  ncclUniqueId id;
  if (rank == 0) {
    ncclGetUniqueId(&id);
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  ncclCommInitRank(&comm, world_size, id, rank);

  MPI_Finalize();
  return 0;
}
