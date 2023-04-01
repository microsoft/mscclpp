/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define MSCCLPP_USE_MPI_FOR_TESTS 1

#include "common.h"
#include "cuda.h"
#include "mscclpp.h"
#include "timer.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>

#include <getopt.h>
#include <libgen.h>
#include <pthread.h>

int is_main_proc = 0;
thread_local int is_main_thread = 0;

// Command line parameter defaults
static size_t minBytes = 32*1024*1024;
static size_t maxBytes = 32*1024*1024;
static size_t stepBytes = 1*1024*1024;
static size_t stepFactor = 1;
static int datacheck = 1;
static int warmup_iters = 5;
static int iters = 20;
static int timeout = 0;
static int report_cputime = 0;
// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
static int average = 1;
static std::string ip_port;

#define NUM_BLOCKS 32

static double parsesize(const char *value) {
    long long int units;
    double size;
    char size_lit;

    int count = sscanf(value, "%lf %1s", &size, &size_lit);

    switch (count) {
    case 2:
      switch (size_lit) {
      case 'G':
      case 'g':
        units = 1024*1024*1024;
        break;
      case 'M':
      case 'm':
        units = 1024*1024;
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

void Barrier(struct threadArgs *args) {
  thread_local int epoch = 0;
  static pthread_mutex_t lock[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
  static pthread_cond_t cond[2] = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
  static int counter[2] = {0, 0};

  pthread_mutex_lock(&lock[epoch]);
  if(++counter[epoch] == args->nThreads)
    pthread_cond_broadcast(&cond[epoch]);

  if(args->thread+1 == args->nThreads) {
    while(counter[epoch] != args->nThreads)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);
    #ifdef MSCCLPP_USE_MPI_FOR_TESTS
      MPI_Barrier(MPI_COMM_WORLD);
    #endif
    counter[epoch] = 0;
    pthread_cond_broadcast(&cond[epoch]);
  }
  else {
    while(counter[epoch] != 0)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);
  }
  pthread_mutex_unlock(&lock[epoch]);
  epoch ^= 1;
}

testResult_t AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, size_t nbytes) {
    CUDACHECK(cudaMalloc(sendbuff, nbytes));
    CUDACHECK(cudaMalloc(recvbuff, nbytes));
    if (datacheck) CUDACHECK(cudaMalloc(expected, recvBytes));
    return testSuccess;
}

testResult_t startColl(struct threadArgs* args, int in_place, int iter) {
  size_t count = args->nbytes;

  // Try to change offset for each iteration so that we avoid cache effects and catch race conditions in ptrExchange
  size_t totalnbytes = max(args->sendBytes, args->expectedBytes);
  size_t steps = totalnbytes ? args->maxbytes / totalnbytes : 1;
  size_t shift = totalnbytes * (iter % steps);

  for (int i = 0; i < args->nGpus; i++) {
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    char* recvBuff = ((char*)args->recvbuffs[i]) + shift;
    char* sendBuff = ((char*)args->sendbuffs[i]) + shift;

    TESTCHECK(args->collTest->runColl((void*)(in_place ? recvBuff + args->sendInplaceOffset * rank : sendBuff),
                                      (void*)(in_place ? recvBuff + args->recvInplaceOffset * rank : recvBuff), count,
                                      args->comms[0], args->streams[i]));
  }
  return testSuccess;
}

testResult_t testStreamSynchronize(int ngpus, cudaStream_t* streams)
{
  cudaError_t cudaErr;
  int remaining = ngpus;
  int* done = (int*)malloc(sizeof(int) * ngpus);
  memset(done, 0, sizeof(int) * ngpus);
  timer tim;

  while (remaining) {
    int idle = 1;
    for (int i = 0; i < ngpus; i++) {
      if (done[i])
        continue;

      cudaErr = cudaStreamQuery(streams[i]);
      if (cudaErr == cudaSuccess) {
        done[i] = 1;
        remaining--;
        idle = 0;
        continue;
      }

      if (cudaErr != cudaErrorNotReady)
        CUDACHECK(cudaErr);
    }

    // We might want to let other threads (including NCCL threads) use the CPU.
    if (idle)
      sched_yield();
  }
  free(done);
  return testSuccess;
}

testResult_t completeColl(struct threadArgs* args) {
  TESTCHECK(testStreamSynchronize(args->nGpus, args->streams));
  return testSuccess;
}

// Inter-thread/process barrier+allreduce. The quality of the return value
// for average=0 (which means broadcast from rank=0) is dubious. The returned
// value will actually be the result of process-local broadcast from the local thread=0.
template<typename T>
void Allreduce(struct threadArgs* args, T* value, int average)
{
  thread_local int epoch = 0;
  static pthread_mutex_t lock[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
  static pthread_cond_t cond[2] = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
  static T accumulator[2];
  static int counter[2] = {0, 0};

  pthread_mutex_lock(&lock[epoch]);
  if (counter[epoch] == 0) {
    if (average != 0 || args->thread == 0)
      accumulator[epoch] = *value;
  } else {
    switch (average) {
    case /*r0*/ 0:
      if (args->thread == 0)
        accumulator[epoch] = *value;
      break;
    case /*avg*/ 1:
      accumulator[epoch] += *value;
      break;
    case /*min*/ 2:
      accumulator[epoch] = std::min<T>(accumulator[epoch], *value);
      break;
    case /*max*/ 3:
      accumulator[epoch] = std::max<T>(accumulator[epoch], *value);
      break;
    case /*sum*/ 4:
      accumulator[epoch] += *value;
      break;
    }
  }

  if (++counter[epoch] == args->nThreads)
    pthread_cond_broadcast(&cond[epoch]);

  if (args->thread + 1 == args->nThreads) {
    while (counter[epoch] != args->nThreads)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);

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
      MPI_Allreduce(MPI_IN_PLACE, (void*)&accumulator[epoch], 1, ty, op, MPI_COMM_WORLD);
    }
#endif

    if (average == 1)
      accumulator[epoch] /= args->totalProcs * args->nThreads;
    counter[epoch] = 0;
    pthread_cond_broadcast(&cond[epoch]);
  } else {
    while (counter[epoch] != 0)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);
  }
  pthread_mutex_unlock(&lock[epoch]);

  *value = accumulator[epoch];
  epoch ^= 1;
}

testResult_t BenchTime(struct threadArgs* args, int in_place) {
  size_t count = args->nbytes;

  TESTCHECK(args->collTest->initData(args, in_place));
  // Sync
  TESTCHECK(startColl(args, in_place, 0));
  TESTCHECK(completeColl(args));

  Barrier(args);

  // Performance Benchmark
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDACHECK(cudaStreamBeginCapture(args->streams[0], cudaStreamCaptureModeGlobal));
  timer tim;
  for (int iter = 0; iter < iters; iter++) {
    TESTCHECK(startColl(args, in_place, iter));
  }
  CUDACHECK(cudaStreamEndCapture(args->streams[0], &graph));
  CUDACHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  // Launch the graph
  Barrier(args);
  tim.reset();
  CUDACHECK(cudaGraphLaunch(graphExec, args->streams[0]));

  double cputimeSec = tim.elapsed()/(iters);
  TESTCHECK(completeColl(args));

  double deltaSec = tim.elapsed();
  deltaSec = deltaSec/(iters);
  Allreduce(args, &deltaSec, average);

  CUDACHECK(cudaGraphExecDestroy(graphExec));
  CUDACHECK(cudaGraphDestroy(graph));

  double algBw, busBw;
  args->collTest->getBw(count, 1, deltaSec, &algBw, &busBw, args->totalProcs * args->nThreads * args->nGpus);

  Barrier(args);

  double timeUsec = (report_cputime ? cputimeSec : deltaSec)*1.0E6;
  char timeStr[100];
  if (timeUsec >= 10000.0) {
    sprintf(timeStr, "%7.0f", timeUsec);
  } else if (timeUsec >= 100.0) {
    sprintf(timeStr, "%7.1f", timeUsec);
  } else {
    sprintf(timeStr, "%7.2f", timeUsec);
  }
  PRINT("  %7s  %6.2f  %6.2f  %5s", timeStr, algBw, busBw, "N/A");

  args->bw[0] += busBw;
  args->bw_count[0]++;
  return testSuccess;
}

void setupArgs(size_t size, struct threadArgs* args) {
  int nranks = args->totalProcs*args->nGpus*args->nThreads;
  size_t count, sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset;

  // TODO: support more data types
  int typeSize = sizeof(char);
  count = size / typeSize;
  args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset, &recvInplaceOffset, (size_t)count, (size_t)nranks);

  args->nbytes = paramCount * typeSize;
  args->sendBytes = sendCount * typeSize;
  args->expectedBytes = recvCount * typeSize;
  args->sendInplaceOffset = sendInplaceOffset * typeSize;
  args->recvInplaceOffset = recvInplaceOffset * typeSize;
}

testResult_t TimeTest(struct threadArgs* args) {
  // Sync to avoid first-call timeout
  Barrier(args);

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
  PRINT("# %10s  %12s           in-place                       out-of-place          \n", "",
        "");
  PRINT("# %10s  %12s  %7s  %6s  %6s  %6s  %7s  %6s  %6s  %6s\n", "size", "count", "time", "algbw", "busbw", "#wrong",
        "time", "algbw", "busbw", "#wrong");
  PRINT("# %10s  %12s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "(us)", "(GB/s)", "(GB/s)", "",
        "(us)", "(GB/s)", "(GB/s)", "");
  // Benchmark
  for (size_t size = args->minbytes; size<=args->maxbytes; size = ((args->stepfactor > 1) ? size*args->stepfactor : size+args->stepbytes)) {
      setupArgs(size, args);
      PRINT("%12li  %12li", max(args->sendBytes, args->expectedBytes), args->nbytes);
      // Don't support out-of-place for now
      // TESTCHECK(BenchTime(args, 0));
      TESTCHECK(BenchTime(args, 1));
      PRINT("\n");
  }
  return testSuccess;
}

testResult_t setupMscclppConnections(int rank, int ranksPerNode, int worldSize, mscclppComm_t comm, void* dataDst,
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

testResult_t threadRunTests(struct threadArgs* args)
{
  TESTCHECK(setupMscclppConnections(args->proc, args->totalProcs, args->ranksPerNode, args->comms[0],
                                    args->recvbuffs[0], args->expectedBytes));
  PRINT("Setting up the connection in MSCCL++\n");
  MSCCLPPCHECK(mscclppProxyLaunch(args->comms[0]));
  TESTCHECK(mscclppTestEngine.runTest(args));
  PRINT("Stopping MSCCL++ proxy threads\n");
  MSCCLPPCHECK(mscclppProxyStop(args->comms[0]));
  return testSuccess;
}

testResult_t run(); // Main function

int main(int argc, char* argv[]) {
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  // Parse args
  double parsed;
  int longindex;
  static struct option longopts[] = {
    {"minbytes", required_argument, 0, 'b'},
    {"maxbytes", required_argument, 0, 'e'},
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"iters", required_argument, 0, 'n'},
    {"warmup_iters", required_argument, 0, 'w'},
    {"check", required_argument, 0, 'c'},
    {"timeout", required_argument, 0, 'T'},
    {"report_cputime", required_argument, 0, 'C'},
    {"average", required_argument, 0, 'a'},
    {"ip_port", required_argument, 0, 'P'},
    {"help", no_argument, 0, 'h'},
    {}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "b:e:i:f:n:w:c:T:C:a:P:h:", longopts, &longindex);

    if (c == -1)
      break;

    switch(c) {
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
      case 'C':
        report_cputime = strtol(optarg, NULL, 0);
        break;
      case 'a':
        average = (int)strtol(optarg, NULL, 0);
        break;
      case 'P':
        ip_port = optarg;
        break;
      case 'h':
      default:
        if (c != 'h') printf("invalid option '%c'\n", c);
        printf("USAGE: %s \n\t"
            "[-b,--minbytes <min size in bytes>] \n\t"
            "[-e,--maxbytes <max size in bytes>] \n\t"
            "[-i,--stepbytes <increment size>] \n\t"
            "[-f,--stepfactor <increment factor>] \n\t"
            "[-n,--iters <iteration count>] \n\t"
            "[-w,--warmup_iters <warmup iteration count>] \n\t"
            "[-c,--check <0/1>] \n\t"
            "[-T,--timeout <time in seconds>] \n\t"
            "[-C,--report_cputime <0/1>] \n\t"
            "[-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] \n\t"
            "[-P,--ip_port <ip port for bootstrap>] \n\t"
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
  if (ip_port.empty()) {
    fprintf(stderr, "--ip_port is required'\n");
    return -1;
  }
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Init(&argc, &argv);
#endif
  TESTCHECK(run());
  return 0;
}

testResult_t run() {
  int totalProcs = 1, proc = 0;
  int ranksPerNode = 0, localRank = 0;
  char hostname[1024];
  getHostName(hostname, 1024);

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_size(shmcomm, &ranksPerNode);
  MPI_Comm_free(&shmcomm);
  localRank = proc % ranksPerNode;
#endif
  is_main_thread = is_main_proc = (proc == 0) ? 1 : 0;
  is_main_thread = is_main_proc = (proc == 0) ? 1 : 0;

  PRINT("# minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: %d validation: %d ip port: %s\n", minBytes,
        maxBytes, (stepFactor > 1) ? stepFactor : stepBytes, (stepFactor > 1) ? "factor" : "bytes", warmup_iters, iters,
        datacheck, ip_port.c_str());
  PRINT("#\n");
  PRINT("# Using devices\n");

#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  size_t maxMem = ~0;

  int cudaDev = localRank;
  int rank = proc;
  cudaDeviceProp prop;
  CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
  len += snprintf(line + len, MAX_LINE - len, "#  Rank %2d Pid %6d on %10s device %2d [0x%02x] %s\n", rank, getpid(),
                  hostname, cudaDev, prop.pciBusID, prop.name);
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
  size_t memMaxBytes = (maxMem - (1<<30)) / (datacheck ? 3 : 2);
  if (maxBytes > memMaxBytes) {
    maxBytes = memMaxBytes;
    if (proc == 0) printf("#\n# Reducing maxBytes to %ld due to memory limitation\n", maxBytes);
  }

  int gpu = cudaDev;
  cudaStream_t stream;
  void* sendbuff;
  void* recvbuff;
  void* expected;
  size_t sendBytes, recvBytes;

  mscclppTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)totalProcs);

  CUDACHECK(cudaSetDevice(gpu));
  TESTCHECK(AllocateBuffs(&sendbuff, sendBytes, &recvbuff, recvBytes, &expected, (size_t)maxBytes));
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  PRINT("#\n");
  PRINT("# Initializing MSCCL++\n");

  mscclppComm_t comms;
  MSCCLPPCHECK(mscclppCommInitRank(&comms, totalProcs, ip_port.c_str(), localRank));

  int error = 0;
  double bw = 0.0;
  double* delta;
  CUDACHECK(cudaHostAlloc(&delta, sizeof(double) * NUM_BLOCKS, cudaHostAllocPortable | cudaHostAllocMapped));
  int bw_count = 0;

  fflush(stdout);

  struct testThread thread = {0};

  thread.args.minbytes = minBytes;
  thread.args.maxbytes = maxBytes;
  thread.args.stepbytes = stepBytes;
  thread.args.stepfactor = stepFactor;
  thread.args.localRank = localRank;
  thread.args.ranksPerNode = ranksPerNode;

  thread.args.totalProcs = totalProcs;
  thread.args.proc = proc;
  thread.args.nThreads = 1;
  thread.args.thread = 0;
  thread.args.nGpus = 1;
  thread.args.gpus = &gpu;
  thread.args.sendbuffs = &sendbuff;
  thread.args.recvbuffs = &recvbuff;
  thread.args.expected = &expected;
  thread.args.comms = &comms;
  thread.args.streams = &stream;

  thread.args.errors = &error;
  thread.args.bw = &bw;
  thread.args.bw_count = &bw_count;

  thread.args.reportErrors = datacheck;

  thread.func = threadRunTests;
  TESTCHECK(thread.func(&thread.args));

  // Wait for other threads and accumulate stats and errors
  TESTCHECK(thread.ret);
  MSCCLPPCHECK(mscclppCommDestroy(comms));

  // Free off CUDA allocated memory
  if (sendbuff)
    CUDACHECK(cudaFree((char*)sendbuff));
  if (recvbuff)
    CUDACHECK(cudaFree((char*)recvbuff));
  if (datacheck)
    CUDACHECK(cudaFree(expected));
  CUDACHECK(cudaFreeHost(delta));

  bw /= bw_count;

  PRINT("# Out of bounds values : %d %s\n", error, error ? "FAILED" : "OK");
  PRINT("#\n");

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Finalize();
#endif
  return testSuccess;
}