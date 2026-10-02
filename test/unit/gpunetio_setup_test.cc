// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <endian.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <map>
#include <mscclpp/port_channel_gpunetio_device.hpp>
#include <mutex>
#include <optional>
#include <thread>

#include "gpu_net_io_binding.hpp"
#include "gpu_net_io_qp_table.hpp"

using namespace mscclpp;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
thread_local int currentRank = 0;
thread_local std::vector<void*> allocations;
std::string fault;
int failedRank;
std::array<int, 4> created{};
void step(const char* label) {
  if (currentRank == failedRank && fault == label) throw std::runtime_error(label);
}
struct Guard {
  explicit Guard(int) { step("device selection"); }
};
#define CudaDeviceGuard Guard
struct Environment {
  int ibGidIndex = 3;
};
Environment environment;
Environment* env() { return &environment; }
struct Mr {
  uint32_t getLkey() const { return 17; }
  struct Info {
    uint32_t rkey = 19;
  };
  Info getInfo() const {
    step("rkey query");
    return {};
  }
};
struct IbCtx {
  explicit IbCtx(const std::string&) { step("HCA open"); }
  bool supportsRdmaAtomics() { return !(currentRank == failedRank && fault == "unsupported atomics"); }
  std::unique_ptr<const Mr> registerMr(void*, size_t) {
    step("registration");
    return std::make_unique<Mr>();
  }
  ibv_pd* getPd() { return nullptr; }
};
int fakeDevOpen(ibv_pd*, doca_dev_t**) {
  step("device open");
  return 0;
}
int fakeGpu(char*, doca_gpu_t**) {
  step("GPU create");
  return 0;
}
void fakePci(char* output, size_t, int) {
  step("PCI query");
  std::strcpy(output, "gpu");
}
template <class Value>
void fakeMalloc(Value** output, size_t bytes) {
  step("CUDA allocation");
  *output = static_cast<Value*>(::operator new(bytes));
  allocations.push_back(*output);
}
void fakeMemset(void* output, int value, size_t bytes) {
  step("CUDA memset");
  std::memset(output, value, bytes);
}
void fakeMemcpy(void* output, const void* input, size_t bytes, int) {
  step("CUDA copy");
  std::memcpy(output, input, bytes);
}
void fakeSync() { step("device sync"); }
int fakeQpn(doca_verbs_qp_t*, uint32_t* output) {
  step("QPN query");
  *output = 100 + currentRank;
  return 0;
}
namespace mscclpp::detail {
int createDirectGpuNetIoQp(doca_gpu_t*, doca_dev_t*, ibv_pd*, doca_gpu_verbs_qp_hl** output) {
  step("QP creation");
  ++created[currentRank];
  auto* descriptor = new doca_gpu_dev_verbs_qp{};
  auto* verbs = new doca_gpu_verbs_qp{};
  auto* qp = new doca_gpu_verbs_qp_hl{};
  verbs->qp_cpu = descriptor;
  qp->qp_gverbs = verbs;
  *output = qp;
  return 0;
}
}  // namespace mscclpp::detail
#define doca_verbs_dev_open fakeDevOpen
#define doca_gpu_create fakeGpu
#define doca_verbs_qp_get_qpn fakeQpn
#define cudaDeviceGetPCIBusId fakePci
#define cudaMalloc fakeMalloc
#define cudaMemset fakeMemset
#define cudaMemcpy fakeMemcpy
#define cudaDeviceSynchronize fakeSync
#define cudaMemcpyHostToDevice 0
#define MSCCLPP_CUDA_THROW(call) call
#define MSCCLPP_DOCA_THROW(call) require((call) == 0, "DOCA failure")

struct Network {
  std::mutex mutex;
  std::condition_variable changed;
  int arrived = 0, generation = 0;
  size_t width = 0;
  std::vector<char> gathered;
  std::map<std::pair<int, int>, std::vector<char>> messages;
};
class TestBootstrap : public Bootstrap {
  Network& network;
  int rank;
  void barrierLocked(std::unique_lock<std::mutex>& lock) {
    const int generation = network.generation;
    if (++network.arrived == 4) {
      network.arrived = 0;
      ++network.generation;
      network.changed.notify_all();
    } else if (!network.changed.wait_for(lock, std::chrono::seconds(5),
                                         [&] { return network.generation != generation; })) {
      throw std::runtime_error("collective timeout");
    }
  }

 public:
  TestBootstrap(Network& network, int rank) : network(network), rank(rank) {}
  int getRank() const override { return rank; }
  int getNranks() const override { return 4; }
  int getNranksPerNode() const override { return 4; }
  void allGather(void* data, int bytes) override {
    std::unique_lock<std::mutex> lock(network.mutex);
    if (network.arrived == 0) {
      network.width = bytes;
      network.gathered.resize(4 * bytes);
    }
    require(network.width == static_cast<size_t>(bytes), "collective order");
    std::memcpy(network.gathered.data() + rank * bytes, static_cast<char*>(data) + rank * bytes, bytes);
    barrierLocked(lock);
    std::memcpy(data, network.gathered.data(), 4 * bytes);
    barrierLocked(lock);
  }
  void send(void* data, int bytes, int peer, int) override {
    std::unique_lock<std::mutex> lock(network.mutex);
    network.messages[{rank, peer}] = std::vector<char>(static_cast<char*>(data), static_cast<char*>(data) + bytes);
    network.changed.notify_all();
  }
  void recv(void* data, int bytes, int peer, int) override {
    std::unique_lock<std::mutex> lock(network.mutex);
    const auto key = std::make_pair(peer, rank);
    if (!network.changed.wait_for(lock, std::chrono::seconds(5), [&] { return network.messages.count(key) > 0; }))
      throw std::runtime_error("peer timeout");
    require(network.messages[key].size() == static_cast<size_t>(bytes), "peer size");
    std::memcpy(data, network.messages[key].data(), bytes);
    network.messages.erase(key);
  }
  void barrier() override {
    std::unique_lock<std::mutex> lock(network.mutex);
    barrierLocked(lock);
  }
};

#include "gpunetio_setup_metadata.inc"

struct State {
  std::shared_ptr<TestBootstrap> bootstrap;
  int rank, worldSize = 4, cudaDeviceId = 0, numQpsPerPeer = 1, portNum = 1, gidIndex = -1;
  bool didSetup = false, deviceSelected = false, setupComplete = false;
  std::string ibDeviceName = "fake";
  std::vector<detail::GpuNetIoSetupStatus> setupStatuses = std::vector<detail::GpuNetIoSetupStatus>(4);
  std::vector<int> peerQpOffsets;
  std::unique_ptr<IbCtx> ibCtx;
  std::unique_ptr<const Mr> mr, atomicResultMr;
  doca_dev_t* netDev = nullptr;
  doca_gpu_t* gpuDev = nullptr;
  uint64_t* atomicResultsGpu = nullptr;
  std::vector<doca_gpu_verbs_qp_hl*> qpHl;
  doca_gpu_dev_verbs_qp* qpFlatGpu = nullptr;
  int* peerQpOffsetsGpu = nullptr;
  uint32_t* rkeysGpu = nullptr;
  uintptr_t* peerBaseGpu = nullptr;
  GpuNetIoDeviceContext* ctxGpu = nullptr;
  QpExchangeInfo localPortInfo{};
  void validateLocalPort() {
    step("port validation");
    localPortInfo.gidIndex = gidIndex;
  }
  void connectQp(doca_gpu_verbs_qp_hl*, const QpExchangeInfo&) { step("QP transition"); }
  void setupImpl(void*, size_t, const std::vector<int>&, int, bool);
  ~State() {
    for (auto* qp : qpHl)
      if (qp) {
        delete qp->qp_gverbs->qp_cpu;
        delete qp->qp_gverbs;
        delete qp;
      }
    for (auto* pointer : allocations) ::operator delete(pointer);
    allocations.clear();
  }
};

#include "gpunetio_setup_body.inc"

int main() {
  int cases = 0;
  for (const char* label : {"none", "device selection", "HCA open", "unsupported atomics", "port validation",
                            "registration", "device open", "CUDA allocation", "CUDA memset", "PCI query", "GPU create",
                            "QP creation", "QPN query", "QP transition", "CUDA copy", "rkey query", "device sync"}) {
    for (failedRank = 0; failedRank < 4; ++failedRank) {
      fault = label;
      created = {};
      Network network;
      std::vector<std::thread> threads;
      std::array<std::string, 4> errors;
      std::array<bool, 4> completed{};
      for (int rank = 0; rank < 4; ++rank)
        threads.emplace_back([&, rank] {
          currentRank = rank;
          State state;
          state.rank = rank;
          state.bootstrap = std::make_shared<TestBootstrap>(network, rank);
          std::vector<int> counts(4, 1);
          counts[rank] = 0;
          try {
            state.setupImpl(reinterpret_cast<void*>(0x1000), 4096, counts, 97, false);
          } catch (const Error& error) {
            errors[rank] = error.what();
          } catch (const std::exception& error) {
            errors[rank] = std::string("UNCOORDINATED ") + error.what();
          }
          completed[rank] = state.setupComplete;
        });
      for (auto& thread : threads) thread.join();
      for (int rank = 0; rank < 4; ++rank) {
        if (fault == "none")
          require(completed[rank] && errors[rank].empty(), "success not published");
        else {
          require(!completed[rank] &&
                      errors[rank].find("failed on rank " + std::to_string(failedRank)) != std::string::npos,
                  "failure not collective");
          require(errors[rank] == errors[0], "peer outcomes differ");
        }
      }
      if (fault == "unsupported atomics")
        for (int value : created) require(value == 0, "unsupported atomics created QPs");
      ++cases;
    }
  }
  std::cout << "Actual setupImpl CPU integration: " << cases << " four-rank fault cases passed\n";
}