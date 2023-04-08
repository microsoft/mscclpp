/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "cuda.h"
#include "mscclpp.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>

#include <getopt.h>
#include <libgen.h>

#define NUM_BLOCKS 32

int is_main_proc = 0;
thread_local int is_main_thread = 0;

namespace {
class timer
{
  std::uint64_t t0;

public:
  timer();
  double elapsed() const;
  double reset();
};

std::uint64_t now()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
}

// Command line parameter defaults
size_t minBytes = 32 * 1024 * 1024;
size_t maxBytes = 32 * 1024 * 1024;
size_t stepBytes = 1 * 1024 * 1024;
size_t stepFactor = 1;
int datacheck = 1;
int warmup_iters = 10;
int iters = 100;
int timeout = 0;
int report_cputime = 0;
// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
int average = 1;
int kernel_num = 0;
int cudaGraphLaunches = 15;

double parsesize(const char* value)
{
  long long int units;
  double size;
  char size_lit;

  int count = sscanf(value, "%lf %1s", &size, &size_lit);

  switch (count) {
  case 2:
    switch (size_lit) {
    case 'G':
    case 'g':
      units = 1024 * 1024 * 1024;
      break;
    case 'M':
    case 'm':
      units = 1024 * 1024;
      break;
    case 'K':
    case 'k':
      units = 1024;
      break;
    default:
      return -1.0;
    };
    break;
  case 1:
    units = 1;
    break;
  default:
    return -1.0;
  }

  return size * units;
}

inline testResult_t Barrier(struct testArgs* args)
{
  int tmp[16];
  // A simple barrier
  MSCCLPPCHECK(mscclppBootstrapAllGather(args->comm, tmp, sizeof(int)));
  return testSuccess;
}
} // namespace

timer::timer()
{
  t0 = now();
}

double timer::elapsed() const
{
  std::uint64_t t1 = now();
  return 1.e-9 * (t1 - t0);
}

double timer::reset()
{
  std::uint64_t t1 = now();
  double ans = 1.e-9 * (t1 - t0);
  t0 = t1;
  return ans;
}

testResult_t AllocateBuffs(void** sendbuff, size_t sendBytes, void** recvbuff, size_t recvBytes, void** expected,
                           size_t nbytes)
{
  CUDACHECK(cudaMalloc(sendbuff, nbytes));
  CUDACHECK(cudaMalloc(recvbuff, nbytes));
  if (datacheck)
    CUDACHECK(cudaMalloc(expected, recvBytes));
  return testSuccess;
}

testResult_t startColl(struct testArgs* args, int in_place, int iter)
{
  size_t count = args->nbytes;

  // Try to change offset for each iteration so that we avoid cache effects and catch race conditions in ptrExchange
  size_t totalnbytes = max(args->sendBytes, args->expectedBytes);
  size_t steps = totalnbytes ? args->maxbytes / totalnbytes : 1;
  size_t shift = totalnbytes * (iter % steps);

  int rank = args->proc;
  char* recvBuff = ((char*)args->recvbuff) + shift;
  char* sendBuff = ((char*)args->sendbuff) + shift;

  TESTCHECK(args->collTest->runColl((void*)(in_place ? recvBuff + args->sendInplaceOffset * rank : sendBuff),
                                    (void*)(in_place ? recvBuff + args->recvInplaceOffset * rank : recvBuff),
                                    args->nranksPerNode, count, args->comm, args->stream, args->kernel_num));
  return testSuccess;
}

testResult_t testStreamSynchronize(cudaStream_t stream)
{
  cudaError_t cudaErr;
  timer tim;

  while (true) {
    cudaErr = cudaStreamQuery(stream);
    if (cudaErr == cudaSuccess) {
      break;
    }

    if (cudaErr != cudaErrorNotReady)
      CUDACHECK(cudaErr);

    double delta = tim.elapsed();
    if (delta > timeout && timeout > 0) {
      char hostname[1024];
      getHostName(hostname, 1024);
      printf("%s: Test timeout (%ds) %s:%d\n", hostname, timeout, __FILE__, __LINE__);
      return testTimeout;
    }

    // We might want to let other threads (including MSCCLPP threads) use the CPU.
    sched_yield();
  }
  return testSuccess;
}

testResult_t completeColl(struct testArgs* args)
{
  TESTCHECK(testStreamSynchronize(args->stream));
  return testSuccess;
}

// Inter process barrier+allreduce. The quality of the return value
// for average=0 is just value itself.
// Inter process barrier+allreduce. The quality of the return value
// for average=0 is just value itself.
template <typename T> void Allreduce(struct testArgs* args, T* value, int average)
{
  T accumulator = *value;

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (average != 0) {
    static_assert(std::is_same<T, long long>::value || std::is_same<T, double>::value,
                  "Allreduce<T> only for T in {long long, double}");
    MPI_Datatype ty = std::is_same<T, long long>::value ? MPI_LONG_LONG
                      : std::is_same<T, double>::value  ? MPI_DOUBLE
                                                        : MPI_Datatype();
    MPI_Op op = average == 1   ? MPI_SUM
                : average == 2 ? MPI_MIN
                : average == 3 ? MPI_MAX
                : average == 4 ? MPI_SUM
                               : MPI_Op();
    MPI_Allreduce(MPI_IN_PLACE, (void*)&accumulator, 1, ty, op, MPI_COMM_WORLD);
  }
#endif

  if (average == 1)
    accumulator /= args->totalProcs;
  *value = accumulator;
}

testResult_t CheckData(struct testArgs* args, int in_place, int64_t* wrongElts)
{
  if (in_place == 0) {
    return testInternalError;
  }
  size_t count = args->expectedBytes / sizeof(int);

  int* dataHostRecv = new int[count];
  int* dataHostExpected = new int[count];
  CUDACHECK(cudaMemcpy(dataHostRecv, args->recvbuff, args->expectedBytes, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(dataHostExpected, args->expected, args->expectedBytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < count; i++) {
    if (dataHostRecv[i] != dataHostExpected[i]) {
      *wrongElts += 1;
    }
  }

  if (args->reportErrors && *wrongElts) {
    (args->error)++;
  }
  return testSuccess;
}

testResult_t BenchTime(struct testArgs* args, int in_place)
{
  size_t count = args->nbytes;

  TESTCHECK(args->collTest->initData(args, in_place));
  // Sync
  TESTCHECK(startColl(args, in_place, 0));
  TESTCHECK(completeColl(args));

  TESTCHECK(Barrier(args));

  // Performance Benchmark
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDACHECK(cudaStreamBeginCapture(args->stream, cudaStreamCaptureModeGlobal));
  timer tim;
  for (int iter = 0; iter < iters; iter++) {
    TESTCHECK(startColl(args, in_place, iter));
  }
  CUDACHECK(cudaStreamEndCapture(args->stream, &graph));
  CUDACHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  // Launch the graph
  TESTCHECK(Barrier(args));
  tim.reset();
  for (int l = 0; l < cudaGraphLaunches; ++l) {
    CUDACHECK(cudaGraphLaunch(graphExec, args->stream));
  }

  double cputimeSec = tim.elapsed() / (iters);
  TESTCHECK(completeColl(args));

  double deltaSec = tim.elapsed();
  deltaSec = deltaSec / (iters) / (cudaGraphLaunches);
  Allreduce(args, &deltaSec, average);

  CUDACHECK(cudaGraphExecDestroy(graphExec));
  CUDACHECK(cudaGraphDestroy(graph));

  double algBw, busBw;
  args->collTest->getBw(count, 1, deltaSec, &algBw, &busBw, args->totalProcs);
  TESTCHECK(Barrier(args));

  int64_t wrongElts = 0;
  if (datacheck) {
    // Initialize sendbuffs, recvbuffs and expected
    TESTCHECK(args->collTest->initData(args, in_place));
    // Begin cuda graph capture for data check
    CUDACHECK(cudaStreamBeginCapture(args->stream, cudaStreamCaptureModeGlobal));
    // test validation in single itertion, should ideally be included into the multi-iteration run
    TESTCHECK(startColl(args, in_place, 0));
    // End cuda graph capture
    CUDACHECK(cudaStreamEndCapture(args->stream, &graph));
    // Instantiate cuda graph
    CUDACHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    // Launch cuda graph
    CUDACHECK(cudaGraphLaunch(graphExec, args->stream));

    TESTCHECK(completeColl(args));

    // destroy cuda graph
    CUDACHECK(cudaGraphExecDestroy(graphExec));
    CUDACHECK(cudaGraphDestroy(graph));

    TESTCHECK(CheckData(args, in_place, &wrongElts));

    // aggregate delta from all threads and procs
    long long wrongElts1 = wrongElts;
    Allreduce(args, &wrongElts1, /*sum*/ 4);
    wrongElts = wrongElts1;
  }

  double timeUsec = (report_cputime ? cputimeSec : deltaSec) * 1.0E6;
  char timeStr[100];
  if (timeUsec >= 10000.0) {
    sprintf(timeStr, "%7.0f", timeUsec);
  } else if (timeUsec >= 100.0) {
    sprintf(timeStr, "%7.1f", timeUsec);
  } else {
    sprintf(timeStr, "%7.2f", timeUsec);
  }
  if (args->reportErrors) {
    PRINT("  %7s  %6.2f  %6.2f  %5g", timeStr, algBw, busBw, (double)wrongElts);
  } else {
    PRINT("  %7s  %6.2f  %6.2f  %5s", timeStr, algBw, busBw, "N/A");
  }

  args->bw += busBw;
  args->bw_count++;
  return testSuccess;
}

void setupArgs(size_t size, struct testArgs* args)
{
  int nranks = args->totalProcs;
  size_t count, sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset;

  // TODO: support more data types
  int typeSize = sizeof(int);
  count = size / typeSize;
  args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset, &recvInplaceOffset,
                                   (size_t)count, (size_t)nranks);

  args->nbytes = paramCount * typeSize;
  args->sendBytes = sendCount * typeSize;
  args->expectedBytes = recvCount * typeSize;
  args->sendInplaceOffset = sendInplaceOffset * typeSize;
  args->recvInplaceOffset = recvInplaceOffset * typeSize;
}

testResult_t TimeTest(struct testArgs* args)
{
  // Sync to avoid first-call timeout
  TESTCHECK(Barrier(args));

  // Warm-up for large size
  setupArgs(args->maxbytes, args);
  TESTCHECK(args->collTest->initData(args, 1));
  for (int iter = 0; iter < warmup_iters; iter++) {
    TESTCHECK(startColl(args, 1, iter));
  }
  TESTCHECK(completeColl(args));

  // Warm-up for small size
  setupArgs(args->minbytes, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    TESTCHECK(startColl(args, 1, iter));
  }
  TESTCHECK(completeColl(args));

  PRINT("#\n");
  PRINT("# %10s  %12s           in-place                       out-of-place          \n", "", "");
  PRINT("# %10s  %12s  %7s  %6s  %6s  %6s  %7s  %6s  %6s  %6s\n", "size", "count", "time", "algbw", "busbw", "#wrong",
        "time", "algbw", "busbw", "#wrong");
  PRINT("# %10s  %12s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "(us)", "(GB/s)", "(GB/s)", "",
        "(us)", "(GB/s)", "(GB/s)", "");
  // Benchmark
  for (size_t size = args->minbytes; size <= args->maxbytes;
       size = ((args->stepfactor > 1) ? size * args->stepfactor : size + args->stepbytes)) {
    setupArgs(size, args);
    PRINT("%12li  %12li", max(args->sendBytes, args->expectedBytes), args->nbytes / sizeof(int));
    // Don't support out-of-place for now
    // TESTCHECK(BenchTime(args, 0));
    TESTCHECK(BenchTime(args, 1));
    PRINT("\n");
  }
  return testSuccess;
}

testResult_t setupMscclppConnections(int rank, int worldSize, int ranksPerNode, mscclppComm_t comm, void* dataDst,
                                     size_t dataSize)
{
  int thisNode = rank / ranksPerNode;
  int localRank = rank % ranksPerNode;
  std::string ibDevStr = "mlx5_ib" + std::to_string(localRank);

  for (int r = 0; r < worldSize; ++r) {
    if (r == rank)
      continue;
    mscclppTransport_t transportType;
    const char* ibDev = ibDevStr.c_str();
    if (r / ranksPerNode == thisNode) {
      ibDev = NULL;
      transportType = mscclppTransportP2P;
    } else {
      transportType = mscclppTransportIB;
    }
    // Connect with all other ranks
    MSCCLPPCHECK(mscclppConnect(comm, r, 0, dataDst, dataSize, transportType, ibDev));
  }

  MSCCLPPCHECK(mscclppConnectionSetup(comm));

  return testSuccess;
}

testResult_t runTests(struct testArgs* args)
{
  PRINT("# Setting up the connection in MSCCL++\n");
  TESTCHECK(setupMscclppConnections(args->proc, args->totalProcs, args->nranksPerNode, args->comm, args->recvbuff,
                                    args->maxbytes));
  PRINT("# Launching MSCCL++ proxy threads\n");
  MSCCLPPCHECK(mscclppProxyLaunch(args->comm));
  TESTCHECK(mscclppTestEngine.runTest(args));
  PRINT("Stopping MSCCL++ proxy threads\n");
  MSCCLPPCHECK(mscclppProxyStop(args->comm));
  return testSuccess;
}

testResult_t run(); // Main function

int main(int argc, char* argv[])
{
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  // Parse args
  double parsed;
  int longindex;
  static struct option longopts[] = {{"minbytes", required_argument, 0, 'b'},
                                     {"maxbytes", required_argument, 0, 'e'},
                                     {"stepbytes", required_argument, 0, 'i'},
                                     {"stepfactor", required_argument, 0, 'f'},
                                     {"iters", required_argument, 0, 'n'},
                                     {"warmup_iters", required_argument, 0, 'w'},
                                     {"check", required_argument, 0, 'c'},
                                     {"timeout", required_argument, 0, 'T'},
                                     {"cudagraph", required_argument, 0, 'G'},
                                     {"report_cputime", required_argument, 0, 'C'},
                                     {"average", required_argument, 0, 'a'},
                                     {"kernel_num", required_argument, 0, 'k'},
                                     {"help", no_argument, 0, 'h'},
                                     {}};

  while (1) {
    int c;
    c = getopt_long(argc, argv, "b:e:i:f:n:w:c:T:G:C:a:P:k:h:", longopts, &longindex);

    if (c == -1)
      break;

    switch (c) {
    case 'b':
      parsed = parsesize(optarg);
      if (parsed < 0) {
        fprintf(stderr, "invalid size specified for 'minbytes'\n");
        return -1;
      }
      minBytes = (size_t)parsed;
      break;
    case 'e':
      parsed = parsesize(optarg);
      if (parsed < 0) {
        fprintf(stderr, "invalid size specified for 'maxbytes'\n");
        return -1;
      }
      maxBytes = (size_t)parsed;
      break;
    case 'i':
      stepBytes = strtol(optarg, NULL, 0);
      break;
    case 'f':
      stepFactor = strtol(optarg, NULL, 0);
      break;
    case 'n':
      iters = (int)strtol(optarg, NULL, 0);
      break;
    case 'w':
      warmup_iters = (int)strtol(optarg, NULL, 0);
      break;
    case 'c':
      datacheck = (int)strtol(optarg, NULL, 0);
      break;
    case 'T':
      timeout = strtol(optarg, NULL, 0);
      break;
    case 'G':
      cudaGraphLaunches = strtol(optarg, NULL, 0);
      if (cudaGraphLaunches <= 0) {
        fprintf(stderr, "invalid number for 'cudaGraphLaunches'\n");
        return -1;
      }
      break;
    case 'C':
      report_cputime = strtol(optarg, NULL, 0);
      break;
    case 'a':
      average = (int)strtol(optarg, NULL, 0);
      break;
    case 'k':
      kernel_num = (int)strtol(optarg, NULL, 0);
      break;
    case 'h':
    default:
      if (c != 'h')
        printf("invalid option '%c'\n", c);
      printf("USAGE: %s \n\t"
             "[-b,--minbytes <min size in bytes>] \n\t"
             "[-e,--maxbytes <max size in bytes>] \n\t"
             "[-i,--stepbytes <increment size>] \n\t"
             "[-f,--stepfactor <increment factor>] \n\t"
             "[-n,--iters <iteration count>] \n\t"
             "[-w,--warmup_iters <warmup iteration count>] \n\t"
             "[-c,--check <0/1>] \n\t"
             "[-T,--timeout <time in seconds>] \n\t"
             "[-G,--cudagraph <num graph launches>] \n\t"
             "[-C,--report_cputime <0/1>] \n\t"
             "[-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] \n\t"
             "[-k,--kernel_num <kernel number of commnication primitive>] \n\t"
             "[-h,--help]\n",
             basename(argv[0]));
      return 0;
    }
  }
  if (minBytes > maxBytes) {
    fprintf(stderr, "invalid sizes for 'minbytes' and 'maxbytes': %llu > %llu\n", (unsigned long long)minBytes,
            (unsigned long long)maxBytes);
    return -1;
  }
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Init(&argc, &argv);
#endif
  TESTCHECK(run());
  return 0;
}

testResult_t run()
{
  int totalProcs = 1, proc = 0;
  int nranksPerNode = 0, localRank = 0;
  char hostname[1024];
  getHostName(hostname, 1024);

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_size(shmcomm, &nranksPerNode);
  MPI_Comm_free(&shmcomm);
  localRank = proc % nranksPerNode;
#endif
  is_main_thread = is_main_proc = (proc == 0) ? 1 : 0;
  is_main_thread = is_main_proc = (proc == 0) ? 1 : 0;

  PRINT("# minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: %d validation: %d graph: %d, "
        "kernel num: %d\n",
        minBytes, maxBytes, (stepFactor > 1) ? stepFactor : stepBytes, (stepFactor > 1) ? "factor" : "bytes",
        warmup_iters, iters, datacheck, cudaGraphLaunches, kernel_num);
  PRINT("#\n");
  PRINT("# Using devices\n");

#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  size_t maxMem = ~0;

  int cudaDev = localRank;
  int rank = proc;
  cudaDeviceProp prop;
  char busIdChar[] = "00000000:00:00.0";
  CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
  CUDACHECK(cudaDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), cudaDev));
  len += snprintf(line + len, MAX_LINE - len, "#  Rank %2d Pid %6d on %10s device %2d [%s] %s\n", rank, getpid(),
                  hostname, cudaDev, busIdChar, prop.name);
  maxMem = std::min(maxMem, prop.totalGlobalMem);

#if MSCCLPP_USE_MPI_FOR_TESTS
  char* lines = (proc == 0) ? (char*)malloc(totalProcs * MAX_LINE) : NULL;
  // Gather all output in rank order to root (0)
  MPI_Gather(line, MAX_LINE, MPI_BYTE, lines, MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (proc == 0) {
    for (int p = 0; p < totalProcs; p++)
      PRINT("%s", lines + MAX_LINE * p);
    free(lines);
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
#else
  PRINT("%s", line);
#endif
  // We need sendbuff, recvbuff, expected (when datacheck enabled), plus 1G for the rest.
  size_t memMaxBytes = (maxMem - (1 << 30)) / (datacheck ? 3 : 2);
  if (maxBytes > memMaxBytes) {
    maxBytes = memMaxBytes;
    if (proc == 0)
      printf("#\n# Reducing maxBytes to %ld due to memory limitation\n", maxBytes);
  }

  cudaStream_t stream;
  void* sendbuff;
  void* recvbuff;
  void* expected;
  size_t sendBytes, recvBytes;

  mscclppTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)totalProcs);

  CUDACHECK(cudaSetDevice(cudaDev));
  TESTCHECK(AllocateBuffs(&sendbuff, sendBytes, &recvbuff, recvBytes, &expected, (size_t)maxBytes));
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  PRINT("#\n");
  PRINT("# Initializing MSCCL++\n");

  mscclppUniqueId mscclppId;
  if (proc == 0) MSCCLPPCHECK(mscclppGetUniqueId(&mscclppId));
  MPI_Bcast((void*)&mscclppId, sizeof(mscclppId), MPI_BYTE, 0, MPI_COMM_WORLD);
  mscclppComm_t comm;
  MSCCLPPCHECK(mscclppCommInitRankFromId(&comm, totalProcs, mscclppId, rank));

  double* delta;
  CUDACHECK(cudaHostAlloc(&delta, sizeof(double) * NUM_BLOCKS, cudaHostAllocPortable | cudaHostAllocMapped));

  fflush(stdout);

  struct testWorker worker;

  worker.args.minbytes = minBytes;
  worker.args.maxbytes = maxBytes;
  worker.args.stepbytes = stepBytes;
  worker.args.stepfactor = stepFactor;
  worker.args.localRank = localRank;
  worker.args.nranksPerNode = nranksPerNode;

  worker.args.totalProcs = totalProcs;
  worker.args.proc = proc;
  worker.args.gpuNum = cudaDev;
  worker.args.kernel_num = kernel_num;
  worker.args.sendbuff = sendbuff;
  worker.args.recvbuff = recvbuff;
  worker.args.expected = expected;
  worker.args.comm = comm;
  worker.args.stream = stream;

  worker.args.error = 0;
  worker.args.bw = 0.0;
  worker.args.bw_count = 0;

  worker.args.reportErrors = datacheck;

  worker.func = runTests;
  TESTCHECK(worker.func(&worker.args));

  MSCCLPPCHECK(mscclppCommDestroy(comm));

  // Free off CUDA allocated memory
  if (sendbuff)
    CUDACHECK(cudaFree((char*)sendbuff));
  if (recvbuff)
    CUDACHECK(cudaFree((char*)recvbuff));
  if (datacheck)
    CUDACHECK(cudaFree(expected));
  CUDACHECK(cudaFreeHost(delta));

  int error = worker.args.error;
  PRINT("# Out of bounds values : %d %s\n", error, error ? "FAILED" : "OK");
  PRINT("#\n");

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Finalize();
#endif
  return testSuccess;
}