#include <cuda.h>
#include <getopt.h>
#include <libgen.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>

#include "common.hpp"

int is_main_proc = 0;

mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                            mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                            mscclpp::Transport::IB6, mscclpp::Transport::IB7};

namespace {

// Command line parameter defaults
size_t minBytes = 32 * 1024 * 1024;
size_t maxBytes = 32 * 1024 * 1024;
size_t stepBytes = 1 * 1024 * 1024;
size_t stepFactor = 1;
int datacheck = 1;
int warmup_iters = 10;
int iters = 20;
// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
int average = 1;
int kernel_num = 0;
int cudaGraphLaunches = 15;

class timer {
  std::uint64_t t0;

 public:
  timer();
  double elapsed() const;
  double reset();
};

std::uint64_t now() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
}

double parseSize(const char* value) {
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

double allreduceTime(int worldSize, double value, int average)
{
  double accumulator = value;

  if (average != 0) {
    MPI_Op op = average == 1   ? MPI_SUM
                : average == 2 ? MPI_MIN
                : average == 3 ? MPI_MAX
                : average == 4 ? MPI_SUM
                               : MPI_Op();
    MPI_Allreduce(MPI_IN_PLACE, (void*)&accumulator, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
  }

  if (average == 1)
    accumulator /= worldSize;
  return accumulator;
}
}  // namespace

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


double BaseTestEngine::benchTime() {
  // Performance Benchmark
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDATHROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
  timer tim;
  for (int iter = 0; iter < iters; iter++) {
    coll_->runColl(args_, stream_);
  }
  CUDATHROW(cudaStreamEndCapture(stream_, &graph));
  CUDATHROW(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  this->barrier();
  tim.reset();
  for (int l = 0; l < cudaGraphLaunches; ++l) {
    CUDATHROW(cudaGraphLaunch(graphExec, stream_));
  }
  CUDATHROW(cudaStreamSynchronize(stream_));
  double deltaSec = tim.elapsed();
  deltaSec = deltaSec / (iters) / (cudaGraphLaunches);
  // all-reduce to get the average time
  allreduceTime(args_.totalRanks, deltaSec, average);
  CUDATHROW(cudaGraphExecDestroy(graphExec));
  CUDATHROW(cudaGraphDestroy(graph));
  return deltaSec;
}

void BaseTestEngine::barrier() {
  this->comm_->bootstrapper()->barrier();
}

void BaseTestEngine::runTest()
{
  // warm-up for large size
  this->coll_->setupCollTest(args_.maxBytes, sizeof(int));
  this->barrier();
  for (int iter = 0; iter < warmup_iters; iter++) {
    this->coll_->runColl(args_, stream_);
  }
  CUDATHROW(cudaDeviceSynchronize());

  // warm-up for small size
  this->coll_->setupCollTest(args_.minBytes, sizeof(int));
  this->barrier();
  for (int iter = 0; iter < warmup_iters; iter++) {
    this->coll_->runColl(args_, stream_);
  }
  CUDATHROW(cudaDeviceSynchronize());

  PRINT("#\n");
  PRINT("# %10s  %12s           in-place                       out-of-place          \n", "", "");
  PRINT("# %10s  %12s  %7s  %6s  %6s  %6s  %7s  %6s  %6s  %6s\n", "size", "count", "time", "algbw", "busbw", "#wrong",
        "time", "algbw", "busbw", "#wrong");
  PRINT("# %10s  %12s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "(us)", "(GB/s)", "(GB/s)", "",
        "(us)", "(GB/s)", "(GB/s)", "");

  // Benchmark
  for (size_t size = args_.minBytes; size <= args_.maxBytes;
       size = ((args_.stepFactor > 1) ? size * args_.stepFactor : size + args_.stepBytes)) {
    coll_->setupCollTest(size, sizeof(int));
    this->coll_->initData(this->args_, this->getSendBuff(), this->getExpectedBuff());
    PRINT("%12li  %12li", max(coll_->getSendBytes(), coll_->getExpectedBytes()), coll_->getParamBytes() / sizeof(int));
    double deltaSec = benchTime();

    size_t nErrors = 0;
    if (args_.reportErrors) {
      this->coll_->setupCollTest(size, sizeof(int));
      this->coll_->initData(this->args_, this->getSendBuff(), this->getExpectedBuff());
      this->barrier();
      this->coll_->runColl(args_, stream_);
      CUDATHROW(cudaDeviceSynchronize());

      nErrors = this->checkData();
      if (nErrors > 0) {
        this->error_++;
      }
      MPI_Allreduce(MPI_IN_PLACE, &nErrors, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    }

    double timeUsec = deltaSec * 1e6;
    char timeStr[100];
    if (timeUsec >= 10000.0) {
      sprintf(timeStr, "%7.0f", timeUsec);
    } else if (timeUsec >= 100.0) {
      sprintf(timeStr, "%7.1f", timeUsec);
    } else {
      sprintf(timeStr, "%7.2f", timeUsec);
    }
    double algBw, busBw;
    this->coll_->getBw(deltaSec, algBw, busBw);
    if (!this->inPlace_) {
      PRINT("                                 ");
    }
    if (args_.reportErrors) {
      PRINT("  %7s  %6.2f  %6.2f  %5g", timeStr, algBw, busBw, (double)nErrors);
    } else {
      PRINT("  %7s  %6.2f  %6.2f  %5s", timeStr, algBw, busBw, "N/A");
    }
    PRINT("\n");
  }
  PRINT("\n");
}

void BaseTestEngine::bootstrap(const TestArgs& args) {
  this->args_ = args;
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(args_.rank, args_.totalRanks);
  mscclpp::UniqueId id;
  if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  comm_ = std::make_shared<mscclpp::Communicator>(bootstrap);
  chanService_ = std::make_shared<mscclpp::channel::DeviceChannelService>(*comm_);
}

void BaseTestEngine::setupTest() {
  this->coll_ = testColl;
  CUDATHROW(cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking));
  this->setupConnections();
  this->chanService_->startProxy();
}

void BaseTestEngine::teardownTest() {
  this->chanService_->stopProxy();
  this->teardown();
}

size_t BaseTestEngine::checkData() {
  size_t nErrors = 0;
  void* recvBuff = this->getRecvBuff();
  void* expectedBuff = this->getExpectedBuff();

  size_t recvBytes = this->coll_->getRecvBytes();
  std::vector<int> recvData(recvBytes / sizeof(int), 0);
  CUDATHROW(cudaMemcpy(recvData.data(), recvBuff, recvBytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < recvData.size(); i++) {
    if (recvData[i] != ((int*)expectedBuff)[i]) {
      nErrors++;
    }
  }
  return nErrors;
}

void run(int argc, char* argv[]);
int main(int argc, char* argv[]) {
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  // Parse args
  double parsed;
  int longindex;
  static option longopts[] = {{"minbytes", required_argument, 0, 'b'},
                              {"maxbytes", required_argument, 0, 'e'},
                              {"stepbytes", required_argument, 0, 'i'},
                              {"stepfactor", required_argument, 0, 'f'},
                              {"iters", required_argument, 0, 'n'},
                              {"warmup_iters", required_argument, 0, 'w'},
                              {"check", required_argument, 0, 'c'},
                              {"cudagraph", required_argument, 0, 'G'},
                              {"average", required_argument, 0, 'a'},
                              {"kernel_num", required_argument, 0, 'k'},
                              {"help", no_argument, 0, 'h'},
                              {}};

  while (1) {
    int c;
    c = getopt_long(argc, argv, "b:e:i:f:n:w:c:G:a:k:h:", longopts, &longindex);

    if (c == -1) break;

    switch (c) {
      case 'b':
        parsed = parseSize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'minbytes'\n");
          return -1;
        }
        minBytes = (size_t)parsed;
        break;
      case 'e':
        parsed = parseSize(optarg);
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
      case 'G':
        cudaGraphLaunches = strtol(optarg, NULL, 0);
        if (cudaGraphLaunches <= 0) {
          fprintf(stderr, "invalid number for 'cudaGraphLaunches'\n");
          return -1;
        }
        break;
      case 'a':
        average = (int)strtol(optarg, NULL, 0);
        break;
      case 'k':
        kernel_num = (int)strtol(optarg, NULL, 0);
        break;
      case 'h':
      default:
        if (c != 'h') printf("invalid option '%c'\n", c);
        printf(
            "USAGE: %s \n\t"
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
    std::cerr << "invalid sizes for 'minbytes' and 'maxbytes': " << minBytes << " > " << maxBytes << std::endl;
    return -1;
  }
  run(argc, argv);
  return 0;
}

void run(int argc, char* argv[]) {
  int totalRanks = 1, rank = 0;
  int nRanksPerNode = 0, localRank = 0;
  char hostname[1024];
  getHostName(hostname, 1024);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_size(shmcomm, &nRanksPerNode);
  MPI_Comm_free(&shmcomm);
  localRank = rank % nRanksPerNode;
  is_main_proc = (rank == 0) ? 1 : 0;

  PRINT(
      "# minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: %d validation: %d graph: %d, "
      "kernel num: %d\n",
      minBytes, maxBytes, (stepFactor > 1) ? stepFactor : stepBytes, (stepFactor > 1) ? "factor" : "bytes",
      warmup_iters, iters, datacheck, cudaGraphLaunches, kernel_num);
  PRINT("#\n");
  PRINT("# Using devices\n");

  constexpr int MAX_LINE = 2048;
  char line[MAX_LINE];
  int len = 0;
  size_t maxMem = ~0;

  int cudaDev = localRank;
  cudaDeviceProp prop;
  char busIdChar[] = "00000000:00:00.0";
  CUDATHROW(cudaGetDeviceProperties(&prop, cudaDev));
  CUDATHROW(cudaDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), cudaDev));
  len += snprintf(line + len, MAX_LINE - len, "#  Rank %2d Pid %6d on %10s device %2d [%s] %s\n", rank, getpid(),
                  hostname, cudaDev, busIdChar, prop.name);
  maxMem = std::min(maxMem, prop.totalGlobalMem);

  std::shared_ptr<char[]> lines(new char[totalRanks * MAX_LINE]);
  // Gather all output in rank order to root (0)
  MPI_Gather(line, MAX_LINE, MPI_BYTE, lines.get(), MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int r = 0; r < totalRanks; r++) PRINT("%s", &lines[MAX_LINE * r]);
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);

  // We need sendbuff, recvbuff, expected (when datacheck enabled), plus 1G for the rest.
  size_t memMaxBytes = (maxMem - (1 << 30)) / (datacheck ? 3 : 2);
  if (maxBytes > memMaxBytes) {
    maxBytes = memMaxBytes;
    PRINT("#\n# Reducing maxBytes to %ld due to memory limitation\n", maxBytes);
  }

  CUDATHROW(cudaSetDevice(cudaDev));
  TestArgs args = {minBytes, maxBytes,  stepBytes,     stepFactor, totalRanks, rank,
                   cudaDev,  localRank, nRanksPerNode, kernel_num, datacheck};
  PRINT("#\n");
  PRINT("# Initializing MSCCL++\n");
  testEngine->bootstrap(args);
  testEngine->allocateBuffer();
  PRINT("# Setting up the connection in MSCCL++\n");
  testEngine->setupTest();
  testEngine->barrier();
  testEngine->runTest();
  testEngine->teardownTest();

  fflush(stdout);

  int error = testEngine->getTestErrors();
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  PRINT("# Out of bounds values : %d %s\n", error, error ? "FAILED" : "OK");
  PRINT("#\n");

  MPI_Finalize();
}
