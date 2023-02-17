#include "mscclpp.h"
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include "mpi.h"
#endif // MSCCLPP_USE_MPI_FOR_TESTS
#include <stdio.h>
#include <stdlib.h>

void print_usage(const char *prog)
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  printf("usage: %s IP:PORT\n", prog);
#else
  printf("usage: %s IP:PORT rank nranks\n", prog);
#endif
}

int main(int argc, const char *argv[])
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (argc != 2) {
    print_usage(argv[0]);
    return -1;
  }
  MPI_Init(NULL, NULL);
  const char *ip_port = argv[1];
  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
  if (argc != 4) {
    print_usage(argv[0]);
    return -1;
  }
  const char *ip_port = argv[1];
  int rank = atoi(argv[2]);
  int world_size = atoi(argv[3]);
#endif

  mscclppComm_t comm;
  mscclppCommInitRank(&comm, world_size, rank, ip_port);

  int *buf = (int *)calloc(world_size, sizeof(int));
  if (buf == nullptr) {
    printf("calloc failed\n");
    return -1;
  }
  buf[rank] = rank;
  mscclppResult_t res = mscclppBootStrapAllGather(comm, buf, sizeof(int));
  if (res != mscclppSuccess) {
    printf("bootstrapAllGather failed\n");
    return -1;
  }

  for (int i = 0; i < world_size; ++i) {
    if (buf[i] != i) {
      printf("wrong data: %d, expected %d\n", buf[i], i);
      return -1;
    }
  }

  res = mscclppCommDestroy(comm);
  if (res != mscclppSuccess) {
    printf("mscclppDestroy failed\n");
    return -1;
  }

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Finalize();
#endif

  printf("Succeeded! %d\n", rank);
  return 0;
}
