#ifndef MSCCLPP_TESTS_COMMON_H_
#define MSCCLPP_TESTS_COMMON_H_

#include <stdio.h>
#include <stdlib.h>

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include "mpi.h"
#endif // MSCCLPP_USE_MPI_FOR_TESTS

void print_usage(const char* prog)
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  printf("usage: %s IP:PORT [rank nranks]\n", prog);
#else
  printf("usage: %s IP:PORT rank nranks\n", prog);
#endif
}

void parse_arguments(int argc, const char* argv[], const char** ip_port, int* rank, int* world_size)
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (argc != 2 && argc != 4) {
    print_usage(argv[0]);
    exit(-1);
  }
  *ip_port = argv[1];
  if (argc == 4) {
    *rank = atoi(argv[2]);
    *world_size = atoi(argv[3]);
  } else {
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, world_size);
  }
#else
  if (argc != 4) {
    print_usage(argv[0]);
    exit(-1);
  }
  *ip_port = argv[1];
  *rank = atoi(argv[2]);
  *world_size = atoi(argv[3]);
#endif
}

#endif // MSCCLPP_TESTS_COMMON_H_