#include "bootstrap.h"
#include "alloc.h"
#include "mpi.h"
#include <stdio.h>

int main()
{
  MPI_Init(NULL, NULL);

  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  mscclppResult_t res = bootstrapNetInit();
  if (res != mscclppSuccess) {
    printf("bootstrapNetInit failed\n");
    return -1;
  }

  mscclppBootstrapHandle handle;
  if (rank == 0) {
    res = bootstrapGetUniqueId(&handle);
    if (res != mscclppSuccess) {
      printf("bootstrapGetUniqueId failed\n");
      return -1;
    }
  }

  MPI_Bcast(&handle, sizeof(mscclppBootstrapHandle), MPI_BYTE, 0, MPI_COMM_WORLD);

  mscclppComm *comm;
  res = mscclppCalloc(&comm, 1);
  if (res != mscclppSuccess) {
      printf("mscclppCalloc failed\n");
  }

  comm->magic = 0xdeadbeef;
  comm->rank = rank;
  comm->nRanks = world_size;
  res = mscclppCudaHostCalloc((uint32_t **)&comm->abortFlag, 1);
  if (res != mscclppSuccess) {
      printf("mscclppCudaHostCalloc failed\n");
  }

  res = bootstrapInit(&handle, comm);
  if (res != mscclppSuccess) {
    printf("bootstrapInit failed\n");
  }

  printf("bootstrapInit done\n");

  int *buf = (int *)calloc(world_size, sizeof(int));
  if (buf == nullptr) {
    printf("calloc failed\n");
  }
  buf[rank] = rank;
  res = bootstrapAllGather(comm->bootstrap, buf, sizeof(int));
  if (res != mscclppSuccess) {
    printf("bootstrapAllGather failed\n");
  }

  for (int i = 0; i < world_size; ++i) {
    if (buf[i] != i) {
      printf("wrong data: %d, expected %d\n", buf[i], i);
    }
  }

  res = bootstrapClose(comm->bootstrap);
  if (res != mscclppSuccess) {
    printf("bootstrapClose failed\n");
  }

  MPI_Finalize();

  printf("Succeeded!\n");
  return 0;
}
