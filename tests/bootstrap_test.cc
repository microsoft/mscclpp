#include "mscclpp.h"
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include <mpi.h>
#define TEST_GET_UNIQUE_ID 0  // cannot test without MPI
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

#if (TEST_GET_UNIQUE_ID == 0)
  printf("usage: %s IP:PORT [rank nranks]\n", prog);
#else
  printf("usage: %s [IP:PORT rank nranks]\n", prog);
#endif

#else
  printf("usage: %s IP:PORT rank nranks\n", prog);
#endif
}

int main(int argc, const char *argv[])
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS

#if (TEST_GET_UNIQUE_ID == 0)
  if (argc != 2 && argc != 4) {
    print_usage(argv[0]);
    return -1;
  }
  const char *ip_port = argv[1];
#else
  if (argc != 1) {
    print_usage(argv[0]);
    return -1;
  }
#endif

  int rank;
  int world_size;
  if (argc == 4) {
    rank = atoi(argv[2]);
    world_size = atoi(argv[3]);
  } else {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  }
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

#if (TEST_GET_UNIQUE_ID == 1)
  mscclppUniqueId id;
  if (rank == 0) MSCCLPPCHECK(mscclppGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  MSCCLPPCHECK(mscclppCommInitRankFromId(&comm, world_size, id, rank));
#else
  MSCCLPPCHECK(mscclppCommInitRank(&comm, world_size, rank, ip_port));
#endif

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
  if (argc == 1 || argc == 2) {
    MPI_Finalize();
  }
#endif

  printf("Rank %d Succeeded\n", rank);
  return 0;
}
