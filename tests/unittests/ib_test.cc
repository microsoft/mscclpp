#include "alloc.h"
#include "checks.h"
#include "ib.h"
#include <set>
#include <string>

// Measure current time in second.
static double getTime(void)
{
  struct timespec tspec;
  if (clock_gettime(CLOCK_MONOTONIC, &tspec) == -1) {
    printf("clock_gettime failed\n");
    exit(EXIT_FAILURE);
  }
  return (tspec.tv_nsec / 1.0e9) + tspec.tv_sec;
}

// Example usage:
//   Receiver: ./build/bin/tests/unittests/ib_test 127.0.0.1:50000 0 0 0
//   Sender:   ./build/bin/tests/unittests/ib_test 127.0.0.1:50000 1 0 0
int main(int argc, const char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <ip:port> <0(recv)/1(send)> <gpu id> <ib id>\n", argv[0]);
    return 1;
  }
  const char* ip_port = argv[1];
  int is_send = atoi(argv[2]);
  int cudaDevId = atoi(argv[3]);
  std::string ibDevName = "mlx5_ib" + std::string(argv[4]);

  CUDACHECK(cudaSetDevice(cudaDevId));

  int* data;
  int nelem = 1;
  MSCCLPPCHECK(mscclppCudaCalloc(&data, nelem));

  mscclppComm_t comm;
  MSCCLPPCHECK(mscclppCommInitRank(&comm, 2, ip_port, is_send));

  struct mscclppIbContext* ctx;
  struct mscclppIbQp* qp;
  struct mscclppIbMr* mr;
  MSCCLPPCHECK(mscclppIbContextCreate(&ctx, ibDevName.c_str()));
  MSCCLPPCHECK(mscclppIbContextCreateQp(ctx, &qp));
  MSCCLPPCHECK(mscclppIbContextRegisterMr(ctx, data, sizeof(int) * nelem, &mr));

  struct mscclppIbQpInfo* qpInfo;
  MSCCLPPCHECK(mscclppCalloc(&qpInfo, 2));
  qpInfo[is_send] = qp->info;

  struct mscclppIbMrInfo* mrInfo;
  MSCCLPPCHECK(mscclppCalloc(&mrInfo, 2));
  mrInfo[is_send] = mr->info;

  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, qpInfo, sizeof(struct mscclppIbQpInfo)));
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, mrInfo, sizeof(struct mscclppIbMrInfo)));

  for (int i = 0; i < 2; ++i) {
    if (i == is_send)
      continue;
    qp->rtr(&qpInfo[i]);
    qp->rts();
    break;
  }

  printf("connection succeed\n");

  // A simple barrier
  int* tmp;
  MSCCLPPCHECK(mscclppCalloc(&tmp, 2));
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));

  if (is_send) {
    int maxIter = 100000;
    double start = getTime();
    for (int iter = 0; iter < maxIter; ++iter) {
      qp->stageSend(mr, &mrInfo[0], sizeof(int) * nelem, 0, 0, 0, true);
      if (qp->postSend() != 0) {
        WARN("postSend failed");
        return 1;
      }
      bool waiting = true;
      while (waiting) {
        int wcNum = qp->pollCq();
        if (wcNum < 0) {
          WARN("pollCq failed: errno %d", errno);
          return 1;
        }
        for (int i = 0; i < wcNum; ++i) {
          struct ibv_wc* wc = &qp->wcs[i];
          if (wc->status != IBV_WC_SUCCESS) {
            WARN("wc status %d", wc->status);
            return 1;
          }
          waiting = false;
          break;
        }
      }
    }
    // TODO(chhwang): print detailed stats such as avg, 99%p, etc.
    printf("%f us/iter\n", (getTime() - start) / maxIter * 1e6);
  }

  // A simple barrier
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));

  MSCCLPPCHECK(mscclppIbContextDestroy(ctx));
  MSCCLPPCHECK(mscclppCommDestroy(comm));

  return 0;
}
