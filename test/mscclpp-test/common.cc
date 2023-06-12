#include "common.hpp"

#include <cuda.h>
#include <getopt.h>
#include <mpi.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mscclpp/utils.hpp>
#include <sstream>
#include <string>
#include <type_traits>

int isMainProc = 0;

mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                            mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                            mscclpp::Transport::IB6, mscclpp::Transport::IB7};

#define PRINT(__message)                    \
  do {                                      \
    if (isMainProc) std::cout << __message; \
  } while (0);

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

double allreduceTime(int worldSize, double value, int average) {
  double accumulator = value;

  if (average != 0) {
    MPI_Op op;
    if (average == 1) {
      op = MPI_SUM;
    } else if (average == 2) {
      op = MPI_MIN;
    } else if (average == 3) {
      op = MPI_MAX;
    } else if (average == 4) {
      op = MPI_SUM;
    }
    MPI_Allreduce(MPI_IN_PLACE, (void*)&accumulator, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
  }

  if (average == 1) accumulator /= worldSize;
  return accumulator;
}
}  // namespace

BaseTestEngine::BaseTestEngine(bool inPlace) : error_(0), inPlace_(inPlace) {
  this->coll_ = getTestColl();
  CUDATHROW(cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking));
}

BaseTestEngine::~BaseTestEngine() { cudaStreamDestroy(stream_); }

void BaseTestColl::setupCollTest(const TestArgs& args, size_t size) {
  this->worldSize_ = args.totalRanks;
  this->typeSize_ = sizeof(int);
  this->setupCollTest(size);
}

double BaseTestEngine::benchTime() {
  // Performance Benchmark
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDATHROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
  mscclpp::Timer timer;
  for (int iter = 0; iter < iters; iter++) {
    coll_->runColl(args_, stream_);
  }
  CUDATHROW(cudaStreamEndCapture(stream_, &graph));
  CUDATHROW(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  this->barrier();
  timer.reset();
  for (int l = 0; l < cudaGraphLaunches; ++l) {
    CUDATHROW(cudaGraphLaunch(graphExec, stream_));
  }
  CUDATHROW(cudaStreamSynchronize(stream_));
  double deltaSec = timer.elapsed() * 1.e-6;
  deltaSec = deltaSec / (iters) / (cudaGraphLaunches);
  // all-reduce to get the average time
  allreduceTime(args_.totalRanks, deltaSec, average);
  CUDATHROW(cudaGraphExecDestroy(graphExec));
  CUDATHROW(cudaGraphDestroy(graph));
  return deltaSec;
}

void BaseTestEngine::barrier() { this->comm_->bootstrapper()->barrier(); }

void BaseTestEngine::runTest() {
  // warm-up for large size
  this->coll_->setupCollTest(args_, args_.maxBytes);
  this->barrier();
  for (int iter = 0; iter < warmup_iters; iter++) {
    this->coll_->runColl(args_, stream_);
  }
  CUDATHROW(cudaDeviceSynchronize());

  // warm-up for small size
  this->coll_->setupCollTest(args_, args_.minBytes);
  this->barrier();
  for (int iter = 0; iter < warmup_iters; iter++) {
    this->coll_->runColl(args_, stream_);
  }
  CUDATHROW(cudaDeviceSynchronize());

  std::stringstream ss;
  ss << "#\n";
  ss << "#                                        in-place                       out-of-place\n";
  ss << "#       size         count     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong\n";
  ss << "#        (B)    (elements)     (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)\n";
  PRINT(ss.str());

  ss.clear();

  // Benchmark
  for (size_t size = args_.minBytes; size <= args_.maxBytes;
       size = ((args_.stepFactor > 1) ? size * args_.stepFactor : size + args_.stepBytes)) {
    coll_->setupCollTest(args_, size);
    this->coll_->initData(this->args_, this->getSendBuff(), this->getExpectedBuff());

    ss << std::setw(12) << std::max(coll_->getSendBytes(), coll_->getExpectedBytes()) << "  " << std::setw(12)
       << coll_->getParamBytes() / sizeof(int);

    double deltaSec = benchTime();

    size_t nErrors = 0;
    if (args_.reportErrors) {
      this->coll_->setupCollTest(args_, size);
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
      ss << "                                 ";
    }
    if (args_.reportErrors) {
      ss << "  " << std::setw(7) << timeStr << "  " << std::setw(6) << algBw << "  " << std::setw(6) << busBw << "  "
         << std::setw(5) << nErrors;
    } else {
      ss << "  " << std::setw(7) << timeStr << "  " << std::setw(6) << algBw << "  " << std::setw(6) << busBw;
    }
    ss << "\n";
    PRINT(ss.str());
    ss.clear();
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
  this->setupConnections();
  this->chanService_->startProxy();
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

// Create mesh connections between all ranks. If recvBuff is nullptr, assume in-place.
// TODO(saemal): retrun the actual vector instead of void
void BaseTestEngine::setupMeshConnections(std::vector<mscclpp::channel::SimpleDeviceChannel>& devChannels,
                                          void* inputBuff, size_t inputBuffBytes, void* outputBuff,
                                          size_t outputBuffBytes) {
  const int worldSize = args_.totalRanks;
  const int rank = args_.rank;
  const int nRanksPerNode = args_.nRanksPerNode;
  const int thisNode = rank / nRanksPerNode;
  const mscclpp::Transport ibTransport = IBs[args_.gpuNum];
  const bool isOutPlace = (outputBuff != nullptr);

  std::vector<mscclpp::channel::ChannelId> channelIds;
  mscclpp::RegisteredMemory inputBufRegMem =
      comm_->registerMemory(inputBuff, inputBuffBytes, mscclpp::Transport::CudaIpc | ibTransport);
  mscclpp::RegisteredMemory outputBufRegMem;
  if (isOutPlace) {
    outputBufRegMem = comm_->registerMemory(outputBuff, outputBuffBytes, mscclpp::Transport::CudaIpc | ibTransport);
  }
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;

  auto rankToNode = [&](int rank) { return rank / nRanksPerNode; };
  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    mscclpp::Transport transport;
    if (rankToNode(r) == thisNode) {
      transport = mscclpp::Transport::CudaIpc;
    } else {
      transport = ibTransport;
    }
    // Connect with all other ranks
    auto connection = comm_->connectOnSetup(r, 0, transport);
    channelIds.push_back(chanService_->addChannel(connection));

    if (isOutPlace) {
      comm_->sendMemoryOnSetup(outputBufRegMem, r, 0);
    } else {
      comm_->sendMemoryOnSetup(inputBufRegMem, r, 0);
    }
    auto remoteMemory = comm_->recvMemoryOnSetup(r, 0);
    remoteRegMemories.push_back(remoteMemory);
  }
  comm_->setup();

  for (size_t i = 0; i < channelIds.size(); ++i) {
    devChannels.push_back(mscclpp::channel::SimpleDeviceChannel(chanService_->deviceChannel(channelIds[i]),
                                                                chanService_->addMemory(remoteRegMemories[i].get()),
                                                                chanService_->addMemory(inputBufRegMem)));
  }
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
  std::string hostname = mscclpp::getHostName(1024, '.');

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_size(shmcomm, &nRanksPerNode);
  MPI_Comm_free(&shmcomm);
  localRank = rank % nRanksPerNode;
  isMainProc = (rank == 0) ? 1 : 0;

  std::stringstream ss;
  ss << "# minBytes " << minBytes << " maxBytes " << maxBytes
     << " step: " << ((stepFactor > 1) ? stepFactor : stepBytes) << "(" << ((stepFactor > 1) ? "factor" : "bytes")
     << ") warmup iters: " << warmup_iters << " iters: " << iters << " validation: " << datacheck
     << " graph: " << cudaGraphLaunches << " kernel num: " << kernel_num << "\n";
  ss << "#\n# Using devices\n";
  PRINT(ss.str());
  ss.clear();

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
                  hostname.c_str(), cudaDev, busIdChar, prop.name);
  maxMem = std::min(maxMem, prop.totalGlobalMem);

  std::shared_ptr<char[]> lines(new char[totalRanks * MAX_LINE]);
  // Gather all output in rank order to root (0)
  MPI_Gather(line, MAX_LINE, MPI_BYTE, lines.get(), MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int r = 0; r < totalRanks; r++) {
      ss << &lines[MAX_LINE * r];
    }
    PRINT(ss.str());
    ss.clear();
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);

  // We need sendbuff, recvbuff, expected (when datacheck enabled), plus 1G for the rest.
  size_t memMaxBytes = (maxMem - (1 << 30)) / (datacheck ? 3 : 2);
  if (maxBytes > memMaxBytes) {
    maxBytes = memMaxBytes;
    ss << "#\n# Reducing maxBytes to " << maxBytes << " due to memory limitation\n";
    PRINT(ss.str());
    ss.clear();
  }

  CUDATHROW(cudaSetDevice(cudaDev));
  TestArgs args = {minBytes, maxBytes,  stepBytes,     stepFactor, totalRanks, rank,
                   cudaDev,  localRank, nRanksPerNode, kernel_num, datacheck};

  PRINT("#\n# Initializing MSCCL++\n");

  auto testEngine = getTestEngine();
  testEngine->bootstrap(args);
  testEngine->allocateBuffer();
  int* inputBuff = (int*)testEngine->getSendBuff()[0];
  int* scratchBuff = (int*)testEngine->getScratchBuff();
  args.inputBuff = inputBuff;
  args.scratchBuff = scratchBuff;
  PRINT("# Setting up the connection in MSCCL++\n");
  testEngine->setupTest();
  testEngine->barrier();
  testEngine->runTest();

  fflush(stdout);

  int error = testEngine->getTestErrors();
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  ss << "# Out of bounds values : " << error << " " << (error ? "FAILED" : "OK") << "\n#\n";
  PRINT(ss.str());

  MPI_Finalize();
}
