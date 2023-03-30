/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define MSCCLPP_USE_MPI_FOR_TESTS 1

#include "common.h"
#include "cuda.h"
#include "mscclpp.h"

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
static int nThreads = 1;
static int nGpus = 1;
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
    #ifdef MPI_SUPPORT
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

testResult_t startColl(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t opIndex, int root, int in_place, int iter) {
  size_t count = args->nbytes / wordSize(type);

  // Try to change offset for each iteration so that we avoid cache effects and catch race conditions in ptrExchange
  size_t totalnbytes = max(args->sendBytes, args->expectedBytes);
  size_t steps = totalnbytes ? args->maxbytes / totalnbytes : 1;
  size_t shift = totalnbytes * (iter % steps);

  if (args->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < args->nGpus; i++) {
#ifndef NCCL_MAJOR
    CUDACHECK(cudaSetDevice(args->gpus[i]));
#endif
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    char* recvBuff = ((char*)args->recvbuffs[i]) + shift;
    char* sendBuff = ((char*)args->sendbuffs[i]) + shift;
    ncclRedOp_t op;

    if(opIndex < ncclNumOps) {
      op = opIndex;
    }
    #if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
    else {
      union {
        int8_t i8; uint8_t u8; int32_t i32; uint32_t u32; int64_t i64; uint64_t u64;
        half f16; float f32; double f64;
        #if defined(__CUDA_BF16_TYPES_EXIST__)
        __nv_bfloat16 bf16;
        #endif
      };
      switch(type) {
      case ncclInt8: i8 = ncclVerifiablePremulScalar<int8_t>(rank); break;
      case ncclUint8: u8 = ncclVerifiablePremulScalar<uint8_t>(rank); break;
      case ncclInt32: i32 = ncclVerifiablePremulScalar<int32_t>(rank); break;
      case ncclUint32: u32 = ncclVerifiablePremulScalar<uint32_t>(rank); break;
      case ncclInt64: i64 = ncclVerifiablePremulScalar<int64_t>(rank); break;
      case ncclUint64: u64 = ncclVerifiablePremulScalar<uint64_t>(rank); break;
      case ncclFloat16: f16 = ncclVerifiablePremulScalar<half>(rank); break;
      case ncclFloat32: f32 = ncclVerifiablePremulScalar<float>(rank); break;
      case ncclFloat64: f64 = ncclVerifiablePremulScalar<double>(rank); break;
      #if defined(__CUDA_BF16_TYPES_EXIST__)
      case ncclBfloat16: bf16 = ncclVerifiablePremulScalar<__nv_bfloat16>(rank); break;
      #endif
      }
      NCCLCHECK(ncclRedOpCreatePreMulSum(&op, &u64, type, ncclScalarHostImmediate, args->comms[i]));
    }
    #endif

    TESTCHECK(args->collTest->runColl(
          (void*)(in_place ? recvBuff + args->sendInplaceOffset*rank : sendBuff),
          (void*)(in_place ? recvBuff + args->recvInplaceOffset*rank : recvBuff),
        count, type, op, root, args->comms[i], args->streams[i]));

    #if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
    if(opIndex >= ncclNumOps) {
      NCCLCHECK(ncclRedOpDestroy(op, args->comms[i]));
    }
    #endif
  }
  if (args->nGpus > 1) NCCLCHECK(ncclGroupEnd());

  if (blocking_coll) {
    // Complete op before returning
    TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
  }
  if (blocking_coll) Barrier(args);
  return testSuccess;
}

testResult_t completeColl(struct threadArgs* args) {
  if (blocking_coll) return testSuccess;

  TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
  return testSuccess;
}

testResult_t BenchTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place) {
  size_t count = args->nbytes / wordSize(type);
  if (datacheck) {
    // Initialize sendbuffs, recvbuffs and expected
    TESTCHECK(args->collTest->initData(args, type, op, root, 99, in_place));
  }

  // Sync
  TESTCHECK(startColl(args, type, op, root, in_place, 0));
  TESTCHECK(completeColl(args));

  Barrier(args);

#if CUDART_VERSION >= 11030
  cudaGraph_t graphs[args->nGpus];
  cudaGraphExec_t graphExec[args->nGpus];
  if (cudaGraphLaunches >= 1) {
    // Begin cuda graph capture
    for (int i=0; i<args->nGpus; i++) {
      // Thread local mdoe is needed for:
      // - Multi-thread mode: where graph capture and instantiation can happen concurrently across threads
      // - P2P pre-connect: when there is no warm-up, P2P pre-connect is done during graph capture.
      //   Since pre-connect calls cudaMalloc, we cannot use global capture mode
      CUDACHECK(cudaStreamBeginCapture(args->streams[i], cudaStreamCaptureModeThreadLocal));
    }
  }
#endif

  // Performance Benchmark
  timer tim;
  for (int iter = 0; iter < iters; iter++) {
    if (agg_iters>1) NCCLCHECK(ncclGroupStart());
    for (int aiter = 0; aiter < agg_iters; aiter++) {
      TESTCHECK(startColl(args, type, op, root, in_place, iter*agg_iters+aiter));
    }
    if (agg_iters>1) NCCLCHECK(ncclGroupEnd());
  }

#if CUDART_VERSION >= 11030
  if (cudaGraphLaunches >= 1) {
    // End cuda graph capture
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs+i));
    }
    // Instantiate cuda graph
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaGraphInstantiate(graphExec+i, graphs[i], NULL, NULL, 0));
    }
    // Resync CPU, restart timing, launch cuda graph
    Barrier(args);
    tim.reset();
    for (int l=0; l<cudaGraphLaunches; l++) {
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
      }
    }
  }
#endif

  double cputimeSec = tim.elapsed()/(iters*agg_iters);
  TESTCHECK(completeColl(args));

  double deltaSec = tim.elapsed();
  deltaSec = deltaSec/(iters*agg_iters);
  if (cudaGraphLaunches >= 1) deltaSec = deltaSec/cudaGraphLaunches;
  Allreduce(args, &deltaSec, average);

#if CUDART_VERSION >= 11030
  if (cudaGraphLaunches >= 1) {
    //destroy cuda graph
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
      CUDACHECK(cudaGraphDestroy(graphs[i]));
    }
  }
#endif

  double algBw, busBw;
  args->collTest->getBw(count, wordSize(type), deltaSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus);

  Barrier(args);

  int64_t wrongElts = 0;
  static __thread int rep = 0;
  rep++;
  if (datacheck) {
      // Initialize sendbuffs, recvbuffs and expected
      TESTCHECK(args->collTest->initData(args, type, op, root, rep, in_place));

#if CUDART_VERSION >= 11030
      if (cudaGraphLaunches >= 1) {
        // Begin cuda graph capture for data check
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaStreamBeginCapture(args->streams[i], args->nThreads > 1 ? cudaStreamCaptureModeThreadLocal : cudaStreamCaptureModeGlobal));
        }
      }
#endif

      //test validation in single itertion, should ideally be included into the multi-iteration run
      TESTCHECK(startColl(args, type, op, root, in_place, 0));

#if CUDART_VERSION >= 11030
      if (cudaGraphLaunches >= 1) {
        // End cuda graph capture
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs+i));
        }
        // Instantiate cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphInstantiate(graphExec+i, graphs[i], NULL, NULL, 0));
        }
        // Launch cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
        }
      }
#endif

      TESTCHECK(completeColl(args));

#if CUDART_VERSION >= 11030
      if (cudaGraphLaunches >= 1) {
        //destroy cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
          CUDACHECK(cudaGraphDestroy(graphs[i]));
        }
      }
#endif

      TESTCHECK(CheckData(args, type, op, root, in_place, &wrongElts));

      //aggregate delta from all threads and procs
      long long wrongElts1 = wrongElts;
      Allreduce(args, &wrongElts1, /*sum*/4);
      wrongElts = wrongElts1;
  }

  double timeUsec = (report_cputime ? cputimeSec : deltaSec)*1.0E6;
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

  args->bw[0] += busBw;
  args->bw_count[0]++;
  return testSuccess;
}

void setupArgs(size_t size, ncclDataType_t type, struct threadArgs* args) {
  int nranks = args->nProcs*args->nGpus*args->nThreads;
  size_t count, sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset;

  count = size / wordSize(type);
  args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset, &recvInplaceOffset, (size_t)count, (size_t)nranks);

  args->nbytes = paramCount * wordSize(type);
  args->sendBytes = sendCount * wordSize(type);
  args->expectedBytes = recvCount * wordSize(type);
  args->sendInplaceOffset = sendInplaceOffset * wordSize(type);
  args->recvInplaceOffset = recvInplaceOffset * wordSize(type);
}

testResult_t TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName, int root) {
  // Sync to avoid first-call timeout
  Barrier(args);

  // Warm-up for large size
  setupArgs(args->maxbytes, type, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    TESTCHECK(startColl(args, type, op, root, 0, iter));
  }
  TESTCHECK(completeColl(args));

  // Warm-up for small size
  setupArgs(args->minbytes, type, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    TESTCHECK(startColl(args, type, op, root, 0, iter));
  }
  TESTCHECK(completeColl(args));

  // Benchmark
  for (size_t size = args->minbytes; size<=args->maxbytes; size = ((args->stepfactor > 1) ? size*args->stepfactor : size+args->stepbytes)) {
      setupArgs(size, type, args);
      char rootName[100];
      sprintf(rootName, "%6i", root);
      PRINT("%12li  %12li  %8s  %6s  %6s", max(args->sendBytes, args->expectedBytes), args->nbytes / wordSize(type), typeName, opName, rootName);
      TESTCHECK(BenchTime(args, type, op, root, 0));
      TESTCHECK(BenchTime(args, type, op, root, 1));
      PRINT("\n");
  }
  return testSuccess;
}

testResult_t threadRunTests(struct threadArgs* args) {
  TESTCHECK(mscclppTestEngine.runTest(args, test_typenames[nccltype], (ncclRedOp_t)ncclop, test_opnames[ncclop]));
  return testSuccess;
}

void* threadLauncher(void* thread_) {
  struct testThread* thread = (struct testThread*)thread_;
  thread->ret = thread->func(&thread->args);
  return NULL;
}
testResult_t threadLaunch(struct testThread* thread) {
  pthread_create(&thread->thread, NULL, threadLauncher, thread);
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
    {"nthreads", required_argument, 0, 't'},
    {"ngpus", required_argument, 0, 'g'},
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
    c = getopt_long(argc, argv, "t:g:b:e:i:f:n:w:c:T:C:a:P:h:", longopts, &longindex);

    if (c == -1)
      break;

    switch(c) {
      case 't':
        nThreads = strtol(optarg, NULL, 0);
        break;
      case 'g':
        nGpus = strtol(optarg, NULL, 0);
        break;
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
            "[-t,--nthreads <num threads>] \n\t"
            "[-g,--ngpus <gpus per thread>] \n\t"
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

  PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: %d validation: %d ip "
        "port: %s\n",
        nThreads, nGpus, minBytes, maxBytes, (stepFactor > 1) ? stepFactor : stepBytes,
        (stepFactor > 1) ? "factor" : "bytes", warmup_iters, iters, datacheck, ip_port.c_str());
  PRINT("#\n");
  PRINT("# Using devices\n");

#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  size_t maxMem = ~0;
  for (int i = 0; i < nThreads * nGpus; i++) {
    int cudaDev = (localRank * nThreads * nGpus) + i;
    int rank = proc * nThreads * nGpus + i;
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    len += snprintf(line + len, MAX_LINE - len, "#  Rank %2d Pid %6d on %10s device %2d [0x%02x] %s\n", rank, getpid(),
                    hostname, cudaDev, prop.pciBusID, prop.name);
    maxMem = std::min(maxMem, prop.totalGlobalMem);
  }

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

  int gpus[nGpus*nThreads];
  cudaStream_t streams[nGpus*nThreads];
  void* sendbuffs[nGpus*nThreads];
  void* recvbuffs[nGpus*nThreads];
  void* expected[nGpus*nThreads];
  size_t sendBytes = 10, recvBytes = 10;

  // mscclppTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)totalProcs*nGpus*nThreads);

  for (int i = 0; i < nGpus * nThreads; i++) {
    gpus[i] = (localRank * nThreads * nGpus) + i;
    CUDACHECK(cudaSetDevice(gpus[i]));
    TESTCHECK(AllocateBuffs(sendbuffs + i, sendBytes, recvbuffs + i, recvBytes, expected + i, (size_t)maxBytes));
    CUDACHECK(cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking));
  }
  PRINT("#\n");
  PRINT("# Initializing MSCCL++\n");

  mscclppComm_t* comms = (mscclppComm_t*)malloc(sizeof(mscclppComm_t)*nThreads*nGpus);
  for (int i = 0; i < nGpus * nThreads; i++) {
    MSCCLPPCHECK(
      mscclppCommInitRank(comms + i, totalProcs * nGpus * nThreads, ip_port.c_str(), localRank * nThreads * nGpus + i));
  }

  int errors[nThreads];
  double bw[nThreads];
  double* delta;
  CUDACHECK(cudaHostAlloc(&delta, sizeof(double) * nThreads * NUM_BLOCKS, cudaHostAllocPortable | cudaHostAllocMapped));
  int bw_count[nThreads];
  for (int t = 0; t < nThreads; t++) {
    bw[t] = 0.0;
    errors[t] = bw_count[t] = 0;
  }

  fflush(stdout);

  const char* timeStr = report_cputime ? "cputime" : "time";
  PRINT("#\n");
  PRINT("# %10s  %12s  %8s  %6s  %6s           out-of-place                       in-place          \n", "", "", "", "",
        "");
  PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s %6s  %7s  %6s  %6s %6s\n", "size", "count", "type", "redop", "root",
        timeStr, "algbw", "busbw", "#wrong", timeStr, "algbw", "busbw", "#wrong");
  PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "", "", "",
        "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "");

  struct testThread threads[nThreads];
  memset(threads, 0, sizeof(struct testThread) * nThreads);

  for (int t = nThreads - 1; t >= 0; t--) {
    threads[t].args.minbytes = minBytes;
    threads[t].args.maxbytes = maxBytes;
    threads[t].args.stepbytes = stepBytes;
    threads[t].args.stepfactor = stepFactor;
    threads[t].args.localRank = localRank;

    threads[t].args.totalProcs = totalProcs;
    threads[t].args.proc = proc;
    threads[t].args.nThreads = nThreads;
    threads[t].args.thread = t;
    threads[t].args.nGpus = nGpus;
    threads[t].args.gpus = gpus + t * nGpus;
    threads[t].args.sendbuffs = sendbuffs + t * nGpus;
    threads[t].args.recvbuffs = recvbuffs + t * nGpus;
    threads[t].args.expected = expected + t * nGpus;
    threads[t].args.comms = comms + t * nGpus;
    threads[t].args.streams = streams + t * nGpus;

    threads[t].args.errors = errors + t;
    threads[t].args.bw = bw + t;
    threads[t].args.bw_count = bw_count + t;

    threads[t].args.reportErrors = datacheck;

    threads[t].func = threadRunTests;
    if (t) {
        TESTCHECK(threadLaunch(threads + t));
    } else {
        TESTCHECK(threads[t].func(&threads[t].args));
    }
  }

  // Wait for other threads and accumulate stats and errors
  for (int t = nThreads - 1; t >= 0; t--) {
    if (t)
        pthread_join(threads[t].thread, NULL);
    TESTCHECK(threads[t].ret);
    if (t) {
        errors[0] += errors[t];
        bw[0] += bw[t];
        bw_count[0] += bw_count[t];
    }
  }

  for (int i = 0; i < nGpus * nThreads; i++) {
    MSCCLPPCHECK(mscclppCommDestroy(comms[i]));
  }
  free(comms);

  // Free off CUDA allocated memory
  for (int i=0; i<nGpus*nThreads; i++) {
    if (sendbuffs[i]) CUDACHECK(cudaFree((char*)sendbuffs[i]));
    if (recvbuffs[i]) CUDACHECK(cudaFree((char*)recvbuffs[i]));
    if (datacheck) CUDACHECK(cudaFree(expected[i]));
  }
  CUDACHECK(cudaFreeHost(delta));

  bw[0] /= bw_count[0];

  PRINT("# Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
  PRINT("#\n");

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Finalize();
#endif
  return testSuccess;
}