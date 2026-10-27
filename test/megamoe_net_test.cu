// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Multi-process test of the MegaMoE inter-rack (IB) exchange in megamoe_net.cuh on any CUDA GPU.
//
// It runs the production send kernel, relay, result return, and result collection over real IB port channels set up
// like MegaMoeContext, and replaces the SM100 routed GEMM with a kernel that consumes the relayed LL8 routing packets
// exactly as the routed kernel's received-route path does, checks every received token row, entry, and weight against
// the source's deterministic data, and writes a deterministic result row per route. Sources then check every partial
// output row. Ranks and token counts vary per launch, including empty launches, to exercise epochs and buffer reuse.
//
// Launch one process per GPU on one node, e.g.:
//   mpirun -np 8 ./megamoe_net_test --ranks-per-rack 2 --iterations 30
// Options: --ranks-per-rack, --iterations, --max-tokens, --hidden, --top-k, --experts-per-rank, --static-fp8.
// Environment: OMPI_COMM_WORLD_{RANK,SIZE,LOCAL_RANK} (or RANK/WORLD_SIZE/LOCAL_RANK) and MEGAMOE_NET_TEST_PORT
// (bootstrap port on 127.0.0.1).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/utils.hpp>
#include <string>
#include <vector>

#include "megamoe_net.cuh"

using namespace mscclpp;
using namespace mscclpp::megamoe;
using namespace mscclpp::megamoe::detail;

namespace {

constexpr int RouteChunk = 384;
constexpr int Sentinel = 0xEE;

struct Problem {
  int worldSize;
  int maxTokens;
  int topK;
  int numExperts;
  int localExperts;
  int hidden;
  int scaleStride;
  int outBytes;
  uint32_t seed;
};

__host__ __device__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__host__ __device__ uint32_t key(const Problem& p, int a, int b, int c, int d) {
  return mix(p.seed ^ mix(uint32_t(a) * 0x9e3779b9U ^
                          mix(uint32_t(b) * 0x85ebca6bU ^ mix(uint32_t(c) * 0xc2b2ae35U ^ uint32_t(d)))));
}

__host__ __device__ int tokensOf(const Problem& p, int rank, int iteration) {
  uint32_t h = key(p, 1, rank, iteration, 0);
  if (h % 6 == 0) return 0;
  if (h % 6 == 1) return p.maxTokens;
  return int(h / 6 % (p.maxTokens + 1));
}

__host__ __device__ int idOf(const Problem& p, int rank, int iteration, int token, int slot) {
  uint32_t h = key(p, 2, rank, iteration, token * 64 + slot);
  return h % 13 == 0 ? -1 : int(h / 13 % p.numExperts);
}

__host__ __device__ float weightOf(const Problem& p, int rank, int iteration, int token, int slot) {
  return float(key(p, 3, rank, iteration, token * 64 + slot) % 4096) / 1024.0f;
}

__host__ __device__ uint8_t inputByte(const Problem& p, int rank, int iteration, int token, int byte) {
  return uint8_t(key(p, 4, rank, iteration, token) + uint32_t(byte) * 131U);
}

__host__ __device__ uint8_t scaleByte(const Problem& p, int rank, int iteration, int token, int byte) {
  return uint8_t(key(p, 5, rank, iteration, token) + uint32_t(byte) * 29U);
}

__host__ __device__ uint8_t resultByte(const Problem& p, int owner, int iteration, int route, int source, int byte) {
  return uint8_t(key(p, 6, owner * 1024 + source, iteration, route) + uint32_t(byte) * 7U);
}

__global__ void fillSource(Problem p, int rank, int iteration, int nTokens, int32_t* ids, float* weights,
                           uint8_t* input, uint8_t* scales) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.maxTokens * p.topK; i += gridDim.x * blockDim.x) {
    int token = i / p.topK;
    ids[i] = token < nTokens ? idOf(p, rank, iteration, token, i % p.topK) : 12345678;
    weights[i] = weightOf(p, rank, iteration, token, i % p.topK);
  }
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size_t(nTokens) * p.hidden; i += gridDim.x * blockDim.x)
    input[i] = inputByte(p, rank, iteration, int(i / p.hidden), int(i % p.hidden));
  if (p.scaleStride)
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size_t(nTokens) * p.scaleStride;
         i += gridDim.x * blockDim.x)
      scales[i] = scaleByte(p, rank, iteration, int(i / p.scaleStride), int(i % p.scaleStride));
}

__global__ void relayKernel(NetDevice net, NetRouting routing, uint64_t epoch) {
  for (int index = blockIdx.x; index < net.nNet; index += gridDim.x)
    netRelaySource<RouteChunk>(net, routing, index, epoch);
}

// Stand-in for the routed GEMM: consume the relayed packets like processReceivedRouteChunks, verify the received data,
// and stage one result row per route like the FC2 epilogue.
__global__ void ownerKernel(NetDevice net, NetRouting routing, Problem p, int rank, int iteration, uint64_t epoch,
                            int* routeCounts, int* errors) {
  const size_t routesPerRank = size_t(p.maxTokens) * p.topK;
  for (int index = blockIdx.x; index < net.nNet; index += gridDim.x) {
    const int source = net.rankOf(index);
    const int nTokens = int(routing.tokenCount[source].read(uint32_t(epoch), NetSpinLimit));
    if (nTokens != tokensOf(p, source, iteration)) atomicAdd(errors, 1);
    const int nChunks = (nTokens * p.topK + RouteChunk - 1) / RouteChunk;
    for (int chunk = 0; chunk < nChunks; ++chunk) {
      const int count =
          int(routing.headers[size_t(source) * routing.chunksPerRank + chunk].read(uint32_t(epoch), NetSpinLimit));
      for (int item = threadIdx.x; item < count; item += blockDim.x) {
        uint32_t data = routing.packets[size_t(source) * routesPerRank + size_t(chunk) * RouteChunk + item].read(
            uint32_t(epoch), NetSpinLimit);
        int expert = int(data >> 16);
        int route = chunk * RouteChunk + int(data & 0xffff);
        int token = route / p.topK;
        int slot = route % p.topK;
        int bad = 0;
        bad += token >= nTokens;
        bad += idOf(p, source, iteration, token, slot) != rank * p.localExperts + expert;
        NetEntry entry = netRouteEntry(net, source, token, slot, p.topK, routesPerRank);
        bad += entry.route != route || entry.expert != expert;
        bad += entry.weight != weightOf(p, source, iteration, token, slot);
        const uint8_t* row = netRecvRow(net, source, entry.row, p.maxTokens);
        for (int b = 0; b < p.hidden; ++b) bad += row[b] != inputByte(p, source, iteration, token, b);
        for (int b = 0; b < p.scaleStride; ++b) bad += row[p.hidden + b] != scaleByte(p, source, iteration, token, b);
        uint8_t* out = netCombineRow(net, source, token, slot, p.topK, routesPerRank);
        for (int b = 0; b < p.outBytes; ++b) out[b] = resultByte(p, rank, iteration, route, source, b);
        if (bad) atomicAdd(errors, 1);
        atomicAdd(routeCounts + index, 1);
      }
    }
  }
}

__global__ void returnKernel(NetDevice net, uint64_t epoch) {
  for (int index = threadIdx.x; index < net.nNet; index += blockDim.x) netReturnResults(net, index, epoch);
}

__global__ void collectKernel(NetDevice net, uint8_t* partial, int nRoutes, uint64_t epoch) {
  netCollectResults(net, partial, nRoutes, epoch);
}

__global__ void verifySource(NetDevice net, Problem p, int rank, int iteration, int nTokens, const uint8_t* partial,
                             int* errors) {
  for (int route = blockIdx.x; route < nTokens * p.topK; route += gridDim.x) {
    int id = idOf(p, rank, iteration, route / p.topK, route % p.topK);
    int owner = id >= 0 ? id / p.localExperts : -1;
    bool networked = owner >= 0 && net.isNet(owner);
    int bad = 0;
    for (int b = threadIdx.x; b < p.outBytes; b += blockDim.x) {
      uint8_t expected = networked ? resultByte(p, owner, iteration, route, rank, b) : uint8_t(Sentinel);
      bad += partial[size_t(route) * p.outBytes + b] != expected;
    }
    if (bad) atomicAdd(errors, 1);
  }
}

int envInt(const char* a, const char* b, int fallback) {
  const char* value = getenv(a);
  if (!value) value = getenv(b);
  return value ? atoi(value) : fallback;
}

}  // namespace

int main(int argc, char** argv) {
  int ranksPerRack = 1;
  int iterations = 20;
  int maxTokens = 160;
  int hidden = 4096;
  int topK = 8;
  int expertsPerRank = 4;
  bool staticFp8 = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&] { return atoi(argv[++i]); };
    if (arg == "--ranks-per-rack")
      ranksPerRack = next();
    else if (arg == "--iterations")
      iterations = next();
    else if (arg == "--max-tokens")
      maxTokens = next();
    else if (arg == "--hidden")
      hidden = next();
    else if (arg == "--top-k")
      topK = next();
    else if (arg == "--experts-per-rank")
      expertsPerRank = next();
    else if (arg == "--static-fp8")
      staticFp8 = true;
  }
  const int rank = envInt("OMPI_COMM_WORLD_RANK", "RANK", 0);
  const int worldSize = envInt("OMPI_COMM_WORLD_SIZE", "WORLD_SIZE", 1);
  const int localRank = envInt("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK", 0);
  const int port = envInt("MEGAMOE_NET_TEST_PORT", "MEGAMOE_NET_TEST_PORT", 50531);
  MSCCLPP_CUDATHROW(cudaSetDevice(localRank));

  auto bootstrap = std::make_shared<TcpBootstrap>(rank, worldSize);
  bootstrap->initialize("127.0.0.1:" + std::to_string(port));
  auto comm = std::make_shared<Communicator>(bootstrap);

  NativeConfig c{};
  c.rank = rank;
  c.worldSize = worldSize;
  c.maxTokens = maxTokens;
  c.hidden = hidden;
  c.topK = topK;
  c.numExperts = expertsPerRank * worldSize;
  c.activationDtype = staticFp8 ? ActivationDtype::Fp8Static : ActivationDtype::Mxfp8;
  const int tag = 17920;
  const NetLayout layout = getNetLayout(c, ranksPerRack);
  const int rackBegin = rank / ranksPerRack * ranksPerRack;
  auto isNet = [&](int peer) { return peer < rackBegin || peer >= rackBegin + ranksPerRack; };
  if (!layout.bytes) throw std::runtime_error("the test requires more than one rack");

  // Setup as in MegaMoeContext.
  static constexpr Transport IBs[] = {Transport::IB0, Transport::IB1, Transport::IB2, Transport::IB3,
                                      Transport::IB4, Transport::IB5, Transport::IB6, Transport::IB7};
  const Transport ib = IBs[localRank % std::min(getIBDeviceCount(), 8)];
  GpuBuffer<char> buffer(layout.bytes);
  RegisteredMemory local = comm->registerMemory(buffer.data(), layout.mainBytes, ib);
  std::vector<RegisteredMemory> slots;
  for (int index = 0; index < worldSize - ranksPerRack; ++index)
    slots.push_back(comm->registerMemory(buffer.data() + layout.combineOut + size_t(index) * layout.slotBytes,
                                         layout.slotBytes, ib));
  std::vector<std::shared_future<Connection>> connectionFutures(worldSize);
  std::vector<std::shared_future<RegisteredMemory>> memoryFutures(worldSize);
  for (int peer = 0; peer < worldSize; ++peer) {
    if (!isNet(peer)) continue;
    EndpointConfig endpoint;
    endpoint.transport = ib;
    connectionFutures[peer] = comm->connect(endpoint, peer, tag + 2);
    comm->sendMemory(local, peer, tag + 3);
    memoryFutures[peer] = comm->recvMemory(peer, tag + 3);
  }
  std::vector<Connection> connections;
  std::vector<RegisteredMemory> remotes(worldSize);
  std::vector<std::shared_future<Semaphore>> semaphoreFutures(worldSize);
  for (int peer = 0; peer < worldSize; ++peer) {
    if (!isNet(peer)) continue;
    connections.push_back(connectionFutures[peer].get());
    remotes[peer] = memoryFutures[peer].get();
    semaphoreFutures[peer] = comm->buildSemaphore(connections.back(), peer, tag + 4);
  }
  auto proxy = std::make_shared<ProxyService>();
  const MemoryId localId = proxy->addMemory(local);
  std::vector<PortChannelDeviceHandle> handles(2 * worldSize);
  for (int peer = 0; peer < worldSize; ++peer) {
    if (!isNet(peer)) continue;
    const MemoryId remoteId = proxy->addMemory(remotes[peer]);
    const MemoryId slotId = proxy->addMemory(slots[peer < rackBegin ? peer : peer - ranksPerRack]);
    const SemaphoreId semaphore = proxy->addSemaphore(semaphoreFutures[peer].get());
    handles[peer] = proxy->portChannel(semaphore, remoteId, localId).deviceHandle();
    handles[worldSize + peer] = proxy->portChannel(semaphore, remoteId, slotId).deviceHandle();
  }
  auto channels = mscclpp::detail::gpuCallocShared<PortChannelDeviceHandle>(handles.size());
  MSCCLPP_CUDATHROW(cudaMemcpy(channels.get(), handles.data(), handles.size() * sizeof(PortChannelDeviceHandle),
                               cudaMemcpyHostToDevice));
  proxy->startProxy();
  const NetDevice net = makeNetDevice(c, NetPlan{ranksPerRack, buffer.data(), channels.get()});

  // Routing slots of the symmetric workspace.
  const size_t routesPerRank = size_t(maxTokens) * topK;
  const int chunksPerRank = int((routesPerRank + RouteChunk - 1) / RouteChunk);
  auto tokenCount = mscclpp::detail::gpuCallocShared<LL8Packet>(worldSize);
  auto headers = mscclpp::detail::gpuCallocShared<LL8Packet>(size_t(worldSize) * chunksPerRank);
  auto packets = mscclpp::detail::gpuCallocShared<LL8Packet>(size_t(worldSize) * routesPerRank);
  const NetRouting routing{tokenCount.get(), headers.get(), packets.get(), chunksPerRank, maxTokens, topK};

  Problem p{worldSize,
            maxTokens,
            topK,
            c.numExperts,
            expertsPerRank,
            hidden,
            staticFp8 ? 0 : (hidden / 32 + 15) / 16 * 16,
            int(layout.outBytes),
            0x5eedU};
  auto ids = mscclpp::detail::gpuCallocShared<int32_t>(routesPerRank);
  auto weights = mscclpp::detail::gpuCallocShared<float>(routesPerRank);
  auto input = mscclpp::detail::gpuCallocShared<uint8_t>(size_t(maxTokens) * hidden);
  auto scales = mscclpp::detail::gpuCallocShared<uint8_t>(size_t(maxTokens) * (p.scaleStride + 16));
  auto partial = mscclpp::detail::gpuCallocShared<uint8_t>(routesPerRank * layout.outBytes);
  auto routeCounts = mscclpp::detail::gpuCallocShared<int>(net.nNet);
  auto errors = mscclpp::detail::gpuCallocShared<int>(2);

  cudaStream_t stream;
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int failures = 0;
  long long checkedRoutes = 0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const uint64_t epoch = uint64_t(iteration) + 1;
    const int nTokens = tokensOf(p, rank, iteration);
    MSCCLPP_CUDATHROW(cudaMemsetAsync(partial.get(), Sentinel, routesPerRank * layout.outBytes, stream));
    MSCCLPP_CUDATHROW(cudaMemsetAsync(routeCounts.get(), 0, net.nNet * sizeof(int), stream));
    MSCCLPP_CUDATHROW(cudaMemsetAsync(errors.get(), 0, 2 * sizeof(int), stream));
    fillSource<<<64, 256, 0, stream>>>(p, rank, iteration, nTokens, ids.get(), weights.get(), input.get(),
                                       scales.get());
    NetSendArguments arguments{rank,    maxTokens, topK,          expertsPerRank, hidden,      p.scaleStride,
                               nTokens, ids.get(), weights.get(), input.get(),    scales.get()};
    netSendKernel<<<net.nNet, NetSendThreads, 0, stream>>>(net, arguments);
    relayKernel<<<std::min(net.nNet, 3), RouteChunk, 0, stream>>>(net, routing, epoch);
    ownerKernel<<<net.nNet, 128, 0, stream>>>(net, routing, p, rank, iteration, epoch, routeCounts.get(), errors.get());
    returnKernel<<<1, 64, 0, stream>>>(net, epoch);
    collectKernel<<<5, 256, 0, stream>>>(net, partial.get(), nTokens * topK, epoch);
    verifySource<<<64, 256, 0, stream>>>(net, p, rank, iteration, nTokens, partial.get(), errors.get() + 1);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

    int hostErrors[2];
    std::vector<int> counts(net.nNet);
    MSCCLPP_CUDATHROW(cudaMemcpy(hostErrors, errors.get(), sizeof(hostErrors), cudaMemcpyDeviceToHost));
    MSCCLPP_CUDATHROW(
        cudaMemcpy(counts.data(), routeCounts.get(), counts.size() * sizeof(int), cudaMemcpyDeviceToHost));
    int countErrors = 0;
    for (int index = 0; index < net.nNet; ++index) {
      const int source = net.rankOf(index);
      int expected = 0;
      for (int token = 0; token < tokensOf(p, source, iteration); ++token)
        for (int slot = 0; slot < topK; ++slot) {
          int id = idOf(p, source, iteration, token, slot);
          expected += id >= 0 && id / expertsPerRank == rank;
        }
      countErrors += counts[index] != expected;
      checkedRoutes += counts[index];
    }
    if (hostErrors[0] || hostErrors[1] || countErrors) {
      ++failures;
      fprintf(stderr, "rank %d iteration %d: %d owner errors, %d source errors, %d route-count errors\n", rank,
              iteration, hostErrors[0], hostErrors[1], countErrors);
    }
  }
  bootstrap->barrier();
  proxy->stopProxy();
  printf(
      "rank %d/%d rack %d (%d per rack, %d network peers, %s, %.2f GB workspace): %d/%d iterations passed, %lld "
      "routes received\n",
      rank, worldSize, rank / ranksPerRack, ranksPerRack, net.nNet, staticFp8 ? "fp8-static" : "mxfp8",
      layout.bytes / 1e9, iterations - failures, iterations, checkedRoutes);
  return failures ? 1 : 0;
}
