#include "mscclpp.h"
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include <mpi.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MSCCLPPCHECK(call) do { \
  mscclppResult_t res = call; \
  if (res != mscclppSuccess && res != mscclppInProgress) { \
    /* Print the back trace*/ \
    printf("Failure at %s:%d -> %d\n", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

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
  MSCCLPPCHECK(mscclppCommInitRank(&comm, world_size, rank, ip_port));

  // allocate some test buffer
  int *buf = (int *)calloc(world_size, sizeof(int));
  if (buf == nullptr) {
    printf("calloc failed\n");
    return -1;
  }
  // each rank sets one element in the array
  buf[rank] = rank;

  MSCCLPPCHECK(mscclppBootStrapAllGather(comm, buf, sizeof(int)));

  // check the correctness of all elements in the output of AllGather
  for (int i = 0; i < world_size; ++i) {
    if (buf[i] != i) {
      printf("wrong data: %d, expected %d\n", buf[i], i);
      return -1;
    }
  }

  MSCCLPPCHECK(mscclppCommDestroy(comm));

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Finalize();
#endif

  printf("Rank %d Succeeded\n", rank);
  return 0;
}
