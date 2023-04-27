#include "alloc.h"
#include "checks.h"
#include "ib.hpp"
#include "infiniband/verbs.h"
#include "mscclpp.hpp"
#include <string>
#include <array>

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
  const char* ipPortPair = argv[1];
  int isSend = atoi(argv[2]);
  int cudaDevId = atoi(argv[3]);
  std::string ibDevName = "mlx5_ib" + std::string(argv[4]);

  CUDACHECK(cudaSetDevice(cudaDevId));

  int* data;
  int nelem = 1;
  MSCCLPPCHECK(mscclppCudaCalloc(&data, nelem));

  std::shared_ptr<mscclpp::Bootstrap> bootstrap(new mscclpp::Bootstrap(isSend, 2));
  bootstrap->initialize(ipPortPair);

  mscclpp::IbCtx ctx(ibDevName);
  mscclpp::IbQp* qp = ctx.createQp();
  const mscclpp::IbMr* mr = ctx.registerMr(data, sizeof(int) * nelem);

  std::array<mscclpp::IbQpInfo, 2> qpInfo;
  qpInfo[isSend] = qp->getInfo();

  std::array<mscclpp::IbMrInfo, 2> mrInfo;
  mrInfo[isSend] = mr->getInfo();

  bootstrap->allGather(qpInfo.data(), sizeof(mscclpp::IbQpInfo));
  bootstrap->allGather(mrInfo.data(), sizeof(mscclpp::IbMrInfo));

  for (int i = 0; i < bootstrap->getNranks(); ++i) {
    if (i == isSend)
      continue;
    qp->rtr(qpInfo[i]);
    qp->rts();
    break;
  }

  printf("connection succeed\n");

  bootstrap->barrier();

  if (isSend) {
    int maxIter = 100000;
    double start = getTime();
    for (int iter = 0; iter < maxIter; ++iter) {
      qp->stageSend(mr, mrInfo[0], sizeof(int) * nelem, 0, 0, 0, true);
      qp->postSend();
      bool waiting = true;
      while (waiting) {
        int wcNum = qp->pollCq();
        if (wcNum < 0) {
          WARN("pollCq failed: errno %d", errno);
          return 1;
        }
        for (int i = 0; i < wcNum; ++i) {
          const struct ibv_wc* wc = reinterpret_cast<const struct ibv_wc*>(qp->getWc(i));
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
  bootstrap->barrier();

  return 0;
}
