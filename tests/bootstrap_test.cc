#include "mscclpp.h"
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include <mpi.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

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
  std::string st = "you are using MPI for this test\n";
  st += "tow possilbe usages are:\n";
  st += "> " + std::string(prog) + "\n";
  st += "or\n";
  st += "> " + std::string(prog) + " ip:port\n";
  printf("%s", st.c_str());
#else
  std::string st = "you are NOT using MPI for this test\n";
  st += "the only possible usage:\n";
  st += "> " + std::string(prog) + " ip:port rank world_size\n";
  printf("%s", st.c_str());
#endif
}

int main(int argc, const char *argv[])
{
  if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    print_usage(argv[0]);
    return 0;
  }
  int rank, world_size;
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (argc > 2) {
    print_usage(argv[0]);
    return -1;
  }
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const char *ip_port;
  if (argc == 2)
    ip_port = argv[1];
  else
    ip_port = NULL;
#else
  if (argc != 4) {
    print_usage(argv[0]);
    return -1;
  }
  const char *ip_port = argv[1];
  rank = atoi(argv[2]);
  world_size = atoi(argv[3]);
#endif

  mscclppComm_t comm;

  if (ip_port) {
    MSCCLPPCHECK(mscclppCommInitRank(&comm, world_size, rank, ip_port));
  } else {
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
    mscclppUniqueId id;
    if (rank == 0) MSCCLPPCHECK(mscclppGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    MSCCLPPCHECK(mscclppCommInitRankFromId(&comm, world_size, id, rank));
#else
    fprintf(stderr, "this should have not been possible!\n");
    return -1;
#endif
  }

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
