// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(TEST_QP_TABLE)
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <map>
#include <mutex>
#include <thread>

#include "gpu_net_io_policy.hpp"
#include "gpu_net_io_qp_table.hpp"

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int qpCreationCalls = 0;
int qpFailureCase = 0;
doca_gpu_verbs_qp_hl policyQp{};
doca_gpu_verbs_qp policyVerbs{};
doca_gpu_dev_verbs_qp policyDeviceQp{};

extern "C" doca_error_t doca_gpu_verbs_create_qp_hl(doca_gpu_verbs_qp_init_attr_hl* attributes,
                                                    doca_gpu_verbs_qp_hl** output) {
  ++qpCreationCalls;
  require(attributes->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB &&
              attributes->send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR &&
              attributes->cq_type == DOCA_GPUNETIO_VERBS_CQ_64B && !attributes->cq_collapsed &&
              !attributes->enable_umem_cpu && attributes->flags == 0 && attributes->comp_channel == nullptr &&
              attributes->sq_nwqe == 1024 && attributes->ordering_semantic == DOCA_VERBS_QP_ORDERING_SEMANTIC_IBTA,
          "unsafe attributes reached upstream QP creation");
  require(*output == nullptr, "QP output not cleared before create");
  if (qpFailureCase == 1) return DOCA_ERROR_DRIVER;
  if (qpFailureCase == 2) return DOCA_SUCCESS;
  policyQp = {};
  policyVerbs = {};
  policyDeviceQp = {};
  policyQp.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
  policyQp.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
  policyQp.qp_gverbs = &policyVerbs;
  policyVerbs.qp_cpu = &policyDeviceQp;
  policyVerbs.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
  policyVerbs.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B;
  policyDeviceQp.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
  policyDeviceQp.mem_type = DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU;
  switch (qpFailureCase) {
    case 3:
      policyQp.qp_gverbs = nullptr;
      break;
    case 4:
      policyVerbs.qp_cpu = nullptr;
      break;
    case 5:
      policyQp.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
      break;
    case 6:
      policyQp.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED;
      break;
    case 7:
      policyVerbs.cpu_proxy = true;
      break;
    case 8:
      policyVerbs.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED;
      break;
    case 9:
      policyVerbs.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST;
      break;
    case 10:
      policyDeviceQp.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
      break;
    case 11:
      policyDeviceQp.mem_type = DOCA_GPUNETIO_VERBS_MEM_TYPE_HOST;
      break;
    default:
      break;
  }
  *output = &policyQp;
  return DOCA_SUCCESS;
}

int main() {
  doca_gpu_t gpu{};
  doca_dev_t device{};
  int portCases = 0;
  for (const int link : {IBV_LINK_LAYER_ETHERNET, IBV_LINK_LAYER_INFINIBAND}) {
    for (const bool grh : {false, true}) {
      for (const int gidIndex : {0, 3, 255}) {
        for (int failure = 0; failure < 10; ++failure) {
          int portQueries = 0;
          int gidQueries = 0;
          const auto queryPort = [&](ibv_context*, uint8_t port, ibv_port_attr* attributes) {
            require(port == 1, "wrong GPUNetIO port queried");
            ++portQueries;
            attributes->state = failure == 1 ? IBV_PORT_DOWN : IBV_PORT_ACTIVE;
            attributes->link_layer = failure == 2 ? IBV_LINK_LAYER_UNSPECIFIED : link;
            attributes->active_mtu = failure == 3 ? static_cast<ibv_mtu>(IBV_MTU_4096 + 1) : IBV_MTU_4096;
            attributes->gid_tbl_len = failure == 4 ? gidIndex : 256;
            attributes->flags = grh ? IBV_QPF_GRH_REQUIRED : 0;
            return failure == 5 ? -1 : 0;
          };
          const auto queryGid = [&](ibv_context*, uint8_t port, int selected, ibv_gid* gid) {
            require(port == 1 && selected == gidIndex, "configured GID index replaced or truncated");
            ++gidQueries;
            gid->raw[15] = failure == 7 ? 0 : 42;
            return failure == 6 ? -1 : 0;
          };
          const int selected = failure == 8 ? -1 : failure == 9 ? 256 : gidIndex;
          bool rejected = false;
          try {
            const auto info = mscclpp::detail::queryGpuNetIoPort(nullptr, 1, selected, queryPort, queryGid);
            require(info.port.active_mtu == IBV_MTU_4096 && info.gid.raw[15] == (failure == 7 ? 0 : 42),
                    "validated port/GID was not retained");
          } catch (const std::invalid_argument&) {
            rejected = true;
          } catch (const std::runtime_error& error) {
            require(failure == 5 || failure == 6, error.what());
            require(std::string(error.what()) ==
                        (failure == 5 ? "ibv_query_port failed for GPUNetIO" : "ibv_query_gid failed for GPUNetIO"),
                    "unexpected failure masked by query-error test");
            rejected = true;
          }
          const bool valid = failure == 0 || (failure == 7 && link == IBV_LINK_LAYER_INFINIBAND && !grh);
          require(rejected != valid, "port/GID admission mismatch");
          require(portQueries == (failure >= 8 ? 0 : 1), "invalid index reached port query");
          require(gidQueries == (failure == 0 || failure == 6 || failure == 7 ? 1 : 0),
                  "invalid port/index reached GID query");
          require(qpCreationCalls == 0, "port validation created QPs");
          ++portCases;
        }
      }
    }
  }
  for (const int invalidPort : {-1, 0, 256}) {
    int queries = 0;
    bool rejected = false;
    try {
      (void)mscclpp::detail::queryGpuNetIoPort(
          nullptr, invalidPort, 0,
          [&](ibv_context*, uint8_t, ibv_port_attr*) {
            ++queries;
            return 0;
          },
          [&](ibv_context*, uint8_t, int, ibv_gid*) {
            ++queries;
            return 0;
          });
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected && queries == 0, "invalid port reached verbs queries");
    ++portCases;
  }
  std::cout << "GPUNetIO port/GID preflight: " << portCases << " cases passed\n";
  int sparsePlans = 0;
  for (const int ranks : {1, 2, 4, 8}) {
    for (const int pattern : {0, 1, 2}) {
      std::vector<int> plans(static_cast<size_t>(ranks) * ranks);
      for (int first = 0; first < ranks; ++first) {
        for (int second = first + 1; second < ranks; ++second) {
          const int count = pattern == 0   ? 0
                            : pattern == 1 ? (second == first + 1 || (first == 0 && second == ranks - 1))
                                           : (first == 0 ? second : 0);
          plans[first * ranks + second] = plans[second * ranks + first] = count;
        }
      }
      for (int rank = 0; rank < ranks; ++rank) {
        const auto offsets = mscclpp::detail::gpuNetIoQpOffsets(plans, rank, ranks);
        int total = 0;
        for (int peer = 0; peer < ranks; ++peer) {
          require(offsets[peer] == total, "sparse peer offset");
          total += plans[rank * ranks + peer];
        }
        require(offsets.back() == total, "sparse allocation includes unrequested queues");
        struct Transport {
          int rank;
          const std::vector<int>& offsets;
          int calls = 0;
          void send(const void* data, int bytes, int peer, int tag) {
            require(peer != rank && bytes == (offsets[peer + 1] - offsets[peer]) * static_cast<int>(sizeof(int)) &&
                        tag == 91 && calls % 2 == (rank < peer ? 0 : 1),
                    "sparse send scope/order");
            require(*static_cast<const int*>(data) == offsets[peer], "wrong send segment");
            ++calls;
          }
          void recv(void* data, int bytes, int peer, int tag) {
            require(peer != rank && bytes == (offsets[peer + 1] - offsets[peer]) * static_cast<int>(sizeof(int)) &&
                        tag == 91 && calls % 2 == (rank < peer ? 1 : 0),
                    "sparse recv scope/order");
            auto* values = static_cast<int*>(data);
            for (int queue = 0; queue < bytes / static_cast<int>(sizeof(int)); ++queue)
              values[queue] = peer * 100 + queue;
            ++calls;
          }
        } transport{rank, offsets};
        std::vector<int> local(total);
        for (int index = 0; index < total; ++index) local[index] = index;
        const auto remote = mscclpp::detail::exchangeGpuNetIoQpMetadata(transport, local, offsets, rank, 91);
        int edges = 0;
        for (int peer = 0; peer < ranks; ++peer) {
          if (plans[rank * ranks + peer] > 0) ++edges;
          for (int queue = 0; queue < plans[rank * ranks + peer]; ++queue)
            require(remote[offsets[peer] + queue] == peer * 100 + queue, "remote queue mismatch");
        }
        require(transport.calls == 2 * edges, "unrequested peer contacted");
        ++sparsePlans;
      }
      for (const int invalid : {-1, 1, 65}) {
        auto bad = plans;
        bad[0] = invalid;
        bool rejected = false;
        try {
          (void)mscclpp::detail::gpuNetIoQpOffsets(bad, 0, ranks);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        require(rejected, "invalid self plan accepted");
      }
      if (ranks > 1) {
        auto bad = plans;
        ++bad[1];
        bool rejected = false;
        try {
          (void)mscclpp::detail::gpuNetIoQpOffsets(bad, 0, ranks);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        require(rejected, "asymmetric plan accepted");
        for (const int count : {-1, 65}) {
          bad = plans;
          bad[1] = bad[ranks] = count;
          rejected = false;
          try {
            (void)mscclpp::detail::gpuNetIoQpOffsets(bad, 0, ranks);
          } catch (const std::invalid_argument&) {
            rejected = true;
          }
          require(rejected, "out-of-range reciprocal counts accepted");
        }
      }
    }
  }
  require(mscclpp::detail::buildGpuNetIoQpTable({}).empty(), "idle rank QP table must be empty");
  for (const int count : {1, 4, 65}) {
    std::vector<doca_gpu_dev_verbs_qp> descriptors(count);
    std::vector<doca_gpu_verbs_qp> verbs(count);
    std::vector<doca_gpu_verbs_qp_hl> handles(count);
    std::vector<doca_gpu_verbs_qp_hl*> compact(count);
    for (int index = 0; index < count; ++index) {
      descriptors[index].sq_num = 9000 + index;
      descriptors[index].sq_rsvd_index = 100 + index;
      verbs[index].qp_cpu = &descriptors[index];
      handles[index].qp_gverbs = &verbs[index];
      compact[index] = &handles[index];
    }
    const auto table = mscclpp::detail::buildGpuNetIoQpTable(compact);
    require(table.size() == static_cast<size_t>(count) &&
                std::memcmp(table.data(), descriptors.data(), count * sizeof(descriptors[0])) == 0,
            "compact descriptors changed or extra slots allocated");
    for (const int failure : {0, 1, 2}) {
      compact[0] = failure == 0 ? nullptr : &handles[0];
      handles[0].qp_gverbs = failure == 1 ? nullptr : &verbs[0];
      verbs[0].qp_cpu = failure == 2 ? nullptr : &descriptors[0];
      bool rejected = false;
      try {
        (void)mscclpp::detail::buildGpuNetIoQpTable(compact);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "missing compact descriptor accepted");
    }
  }
  for (const int pattern : {0, 1, 2}) {
    constexpr int ranks = 8;
    std::vector<int> plan(ranks * ranks);
    for (int first = 0; first < ranks - 1; ++first) {
      for (int second = first + 1; second < ranks - 1; ++second) {
        const int count = pattern == 0   ? 0
                          : pattern == 1 ? (second == first + 1 || (first == 0 && second == ranks - 2))
                                         : (first == 0 ? second : 0);
        plan[first * ranks + second] = plan[second * ranks + first] = count;
      }
    }
    struct Network {
      std::mutex mutex;
      std::condition_variable changed;
      std::map<std::pair<int, int>, std::vector<int>> messages;
    } network;
    struct Endpoint {
      Network& network;
      int rank;
      void send(void* data, int bytes, int peer, int tag) {
        require(tag == 19, "unexpected QP tag");
        std::unique_lock<std::mutex> lock(network.mutex);
        const auto key = std::make_pair(rank, peer);
        const auto* values = static_cast<int*>(data);
        network.messages[key] = std::vector<int>(values, values + bytes / sizeof(int));
        network.changed.notify_all();
        require(
            network.changed.wait_for(lock, std::chrono::seconds(5), [&] { return network.messages.count(key) == 0; }),
            "sparse send deadlock");
      }
      void recv(void* data, int bytes, int peer, int tag) {
        require(tag == 19, "unexpected QP tag");
        std::unique_lock<std::mutex> lock(network.mutex);
        const auto key = std::make_pair(peer, rank);
        require(
            network.changed.wait_for(lock, std::chrono::seconds(5), [&] { return network.messages.count(key) != 0; }),
            "sparse receive deadlock");
        require(network.messages.at(key).size() * sizeof(int) == static_cast<size_t>(bytes), "QP count mismatch");
        std::memcpy(data, network.messages.at(key).data(), bytes);
        network.messages.erase(key);
        network.changed.notify_all();
      }
    };
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> failures(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
      threads.emplace_back([&, rank] {
        try {
          Endpoint endpoint{network, rank};
          const auto offsets = mscclpp::detail::gpuNetIoQpOffsets(plan, rank, ranks);
          std::vector<int> local(offsets.back(), rank);
          const auto remote = mscclpp::detail::exchangeGpuNetIoQpMetadata(endpoint, local, offsets, rank, 19);
          for (int peer = 0; peer < ranks; ++peer)
            for (int index = offsets[peer]; index < offsets[peer + 1]; ++index)
              require(remote[index] == peer, "parallel QP metadata crossed peers");
        } catch (...) {
          failures[rank] = std::current_exception();
        }
      });
    }
    for (auto& thread : threads) thread.join();
    for (const auto& failure : failures)
      if (failure) std::rethrow_exception(failure);
    require(network.messages.empty(), "unconsumed QP metadata");
  }
  std::cout << "Sparse plans and requested-peer metadata: " << sparsePlans << " cases passed\n";
  for (qpFailureCase = 0; qpFailureCase < 12; ++qpFailureCase) {
    doca_gpu_verbs_qp_hl* output = &policyQp;
    const auto status = mscclpp::detail::createDirectGpuNetIoQp(&gpu, &device, nullptr, &output);
    const auto expected = qpFailureCase == 0   ? DOCA_SUCCESS
                          : qpFailureCase == 1 ? DOCA_ERROR_DRIVER
                                               : DOCA_ERROR_NOT_SUPPORTED;
    require(status == expected, "unsupported fallback accepted or error lost");
    require(qpCreationCalls == qpFailureCase + 1, "unsafe fallback retry");
    if (qpFailureCase >= 3) require(output == &policyQp, "rejected QP unavailable for owner cleanup");
  }
  require(mscclpp::detail::createDirectGpuNetIoQp(&gpu, &device, nullptr, nullptr) == DOCA_ERROR_INVALID_VALUE &&
              qpCreationCalls == 12,
          "invalid output reached upstream create");
  std::cout << "Direct GPU QP policy: 13 admission cases passed\n";
  std::cout << "Upstream qp_gverbs offset: 0x" << std::hex << offsetof(doca_gpu_verbs_qp_hl, qp_gverbs) << std::dec
            << '\n';
  int layouts = 0;
  int rejectedInputs = 0;
  for (const int peers : {1, 2, 4}) {
    for (const int queues : {1, 2, 4, 64}) {
      for (int rank = 0; rank < peers; ++rank) {
        const size_t count = static_cast<size_t>(peers) * queues;
        std::vector<doca_gpu_dev_verbs_qp> descriptors(count);
        std::vector<doca_gpu_verbs_qp> verbs(count);
        std::vector<doca_gpu_verbs_qp_hl> handles(count);
        std::vector<doca_gpu_verbs_qp_hl*> qps(count, nullptr);
        std::vector<uint64_t> doorbells(count);
        for (size_t index = 0; index < count; ++index) {
          auto& descriptor = descriptors[index];
          descriptor.sq_num = static_cast<uint32_t>(1000 + index);
          descriptor.sq_rsvd_index = 70000 + index;
          descriptor.sq_ready_index = 60000 + index;
          descriptor.sq_wqe_pi = 50000 + index;
          descriptor.sq_db = &doorbells[index];
          descriptor.cq_sq.cqe_ci = 40000 + index;
          descriptor.cq_sq.cq_num = static_cast<uint32_t>(2000 + index);
          descriptor.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
          descriptor.mem_type = DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU;
          verbs[index].qp_cpu = &descriptor;
          handles[index].qp_gverbs = &verbs[index];
          if (index / queues != static_cast<size_t>(rank)) qps[index] = &handles[index];
        }
        const auto table = mscclpp::detail::buildGpuNetIoQpTable(qps, rank, queues);
        require(table.size() == count, "peer-major QP table was compacted");
        const doca_gpu_dev_verbs_qp empty{};
        for (size_t index = 0; index < count; ++index) {
          const bool self = index / queues == static_cast<size_t>(rank);
          const auto& expected = self ? empty : descriptors[index];
          require(std::memcmp(&table[index], &expected, sizeof(expected)) == 0,
                  "self slot not zero or remote QP moved/modified");
          require(qps[index] == (self ? nullptr : &handles[index]), "source QP list changed");
        }
        const auto expectRejected = [&](int testRank, int testQueues) {
          bool rejected = false;
          try {
            (void)mscclpp::detail::buildGpuNetIoQpTable(qps, testRank, testQueues);
          } catch (const std::invalid_argument&) {
            rejected = true;
          }
          require(rejected, "invalid QP table accepted");
          ++rejectedInputs;
        };
        expectRejected(-1, queues);
        expectRejected(peers, queues);
        expectRejected(rank, 0);
        expectRejected(rank, -1);
        expectRejected(rank, 65);
        const size_t selfIndex = static_cast<size_t>(rank) * queues;
        qps[selfIndex] = &handles[selfIndex];
        expectRejected(rank, queues);
        qps[selfIndex] = nullptr;
        if (peers > 1) {
          const size_t remoteIndex = static_cast<size_t>((rank + 1) % peers) * queues;
          qps[remoteIndex] = nullptr;
          expectRejected(rank, queues);
          qps[remoteIndex] = &handles[remoteIndex];
          handles[remoteIndex].qp_gverbs = nullptr;
          expectRejected(rank, queues);
          handles[remoteIndex].qp_gverbs = &verbs[remoteIndex];
          verbs[remoteIndex].qp_cpu = nullptr;
          expectRejected(rank, queues);
          verbs[remoteIndex].qp_cpu = &descriptors[remoteIndex];
        }
        qps.clear();
        expectRejected(rank, queues);
        qps.resize(3, nullptr);
        expectRejected(0, 2);
        ++layouts;
      }
    }
  }
  std::cout << "GPUNetIO sparse QP table: " << layouts << " layouts passed, " << rejectedInputs
            << " invalid inputs rejected\n";
}

#elif defined(TEST_DEVICE_QPS)
#include <cerrno>
#define MSCCLPP_DEVICE_HPP_
#define MSCCLPP_DEVICE_COMPILE
#define MSCCLPP_DEVICE_INLINE inline
#define MSCCLPP_USE_GPUNETIO
#define MSCCLPP_ASSERT_DEVICE_HPP_
#define MSCCLPP_ASSERT_DEVICE(condition, message) require(condition, message)
#define DOCA_GPUNETIO_DEVICE_H

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void __trap() { throw std::runtime_error("CQ error"); }
using __be32 = uint32_t;
using doca_gpu_dev_verbs_ticket_t = uint64_t;
constexpr int DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU = 0;
constexpr int DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD = 1;
uint32_t __byte_perm(uint32_t value, int, int) { return __builtin_bswap32(value); }
struct doca_gpu_dev_verbs_qp {
  uint64_t sq_rsvd_index = 7;
};
struct doca_gpu_dev_verbs_addr {
  uintptr_t addr;
  uint32_t key;
};
doca_gpu_dev_verbs_qp* selectedQp;
doca_gpu_dev_verbs_addr destination, source, signalDestination, signalResult;
uint64_t transferredBytes, signalValue, polledTicket;
int polls = 0;
int completionStatus = 0;

template <int Mode>
void doca_gpu_dev_verbs_put(doca_gpu_dev_verbs_qp* qp, doca_gpu_dev_verbs_addr remote, doca_gpu_dev_verbs_addr local,
                            uint64_t bytes, doca_gpu_dev_verbs_ticket_t* ticket) {
  selectedQp = qp;
  destination = remote;
  source = local;
  transferredBytes = bytes;
  *ticket = 6;
}
template <int Operation, int Mode>
void doca_gpu_dev_verbs_put_signal(doca_gpu_dev_verbs_qp* qp, doca_gpu_dev_verbs_addr remote,
                                   doca_gpu_dev_verbs_addr local, uint64_t bytes, doca_gpu_dev_verbs_addr signalRemote,
                                   doca_gpu_dev_verbs_addr signalLocal, uint64_t value,
                                   doca_gpu_dev_verbs_ticket_t* ticket) {
  doca_gpu_dev_verbs_put<Mode>(qp, remote, local, bytes, ticket);
  signalDestination = signalRemote;
  signalResult = signalLocal;
  signalValue = value;
  *reinterpret_cast<uint64_t*>(signalLocal.addr) = 99;
}
template <typename Value, int Mode>
Value doca_gpu_dev_verbs_atomic_read(Value* pointer) {
  return *pointer;
}
template <int Mode>
int doca_gpu_dev_verbs_poll_one_cq_at(doca_gpu_dev_verbs_qp* qp, uint64_t ticket) {
  selectedQp = qp;
  polledTicket = ticket;
  ++polls;
  return completionStatus;
}

#include <mscclpp/port_channel_gpunetio_device.hpp>

#define MSCCLPP_FIFO_DEVICE_HPP_
#define MSCCLPP_SEMAPHORE_DEVICE_HPP_
#define MSCCLPP_INLINE inline
#define MSCCLPP_HOST_DEVICE_INLINE inline
#define MSCCLPP_DEVICE_CUDA
#define POLL_MAYBE_JAILBREAK(condition, budget) require(!(condition), "unexpected proxy polling")
namespace mscclpp {
enum { scopeSystem, memoryOrderAcquire };
enum { TriggerPut, TriggerSignal, TriggerPutWithSignal, TriggerPutWithSignalAndFlush, TriggerFlush, TriggerAccumulate };
constexpr int TriggerBitsSize = 32;
template <typename Value, int Scope>
Value atomicLoad(Value* pointer, int) {
  return *pointer;
}
struct ProxyTrigger {
  int type;
  uint32_t dst;
  uint64_t dstOffset;
  uint32_t src;
  uint64_t srcOffset;
  uint64_t bytes;
  uint32_t sem;
};
struct FifoDeviceHandle {
  uint64_t push(ProxyTrigger) { throw std::runtime_error("unexpected proxy operation"); }
};
struct Host2DeviceSemaphoreDeviceHandle {
  uint64_t* inboundToken;
  uint64_t* expectedInboundToken;
  bool poll() {
    if (*inboundToken <= *expectedInboundToken) return false;
    ++*expectedInboundToken;
    return true;
  }
  void wait(int64_t) { require(poll(), "missing signal"); }
};
}  // namespace mscclpp
#include <mscclpp/port_channel_device.hpp>

int main() {
  constexpr int peers = 4;
  const uint32_t keys[peers] = {11, 12, 13, 14};
  const uintptr_t bases[peers] = {1024, 2048, 3072, 4096};
  uint64_t payload[8] = {123};
  for (const int queues : {1, 2, 4, 8, 64}) {
    std::vector<doca_gpu_dev_verbs_qp> qps(peers * queues);
    std::vector<uint64_t> scratch(peers * queues, 0);
    mscclpp::GpuNetIoDeviceContext context{};
    context.qps = qps.data();
    context.rkeys = keys;
    context.peerBase = bases;
    context.localBase = reinterpret_cast<uintptr_t>(payload);
    context.lkey = 0x01020304;
    context.numPeers = peers;
    context.numQpsPerPeer = queues;
    context.atomicResultBase = reinterpret_cast<uintptr_t>(scratch.data());
    context.atomicResultLkey = 0x05060708;
    for (int peer = 0; peer < peers; ++peer) {
      for (int queue = 0; queue < queues; ++queue) {
        const int index = peer * queues + queue;
        auto* expectedQp = &qps[index];
        context.put(peer, 16, 24, 32, queue);
        require(selectedQp == expectedQp && destination.addr == bases[peer] + 16 && destination.key == keys[peer] &&
                    source.addr == context.localBase + 24 && source.key == 0x04030201 && transferredBytes == 32,
                "multi-QP put addressing");
        context.putWithSignal(peer, 16, 24, 32, 64, 5, queue);
        require(selectedQp == expectedQp && signalDestination.addr == bases[peer] + 64 &&
                    signalResult.addr == reinterpret_cast<uintptr_t>(&scratch[index]) &&
                    signalResult.key == 0x08070605 && signalValue == 5,
                "multi-QP signal scratch");
        context.atomicAdd(peer, 128, -7, queue);
        require(selectedQp == expectedQp && signalDestination.addr == bases[peer] + 128 &&
                    signalValue == static_cast<uint64_t>(-7) && transferredBytes == 0 && payload[0] == 123,
                "multi-QP atomic isolation");
        context.flush(peer, queue);
        require(selectedQp == expectedQp && polledTicket == 6, "multi-QP completion queue");
        completionStatus = EBUSY;
        polls = 0;
        require(context.tryFlush(peer, 3, queue) == EBUSY && polls == 3 && selectedQp == expectedQp,
                "bounded queue-local completion");
        completionStatus = -EIO;
        require(context.tryFlush(peer, 3, queue) == -EIO, "CQ error propagation");
        completionStatus = 0;
        require(context.tryFlush(peer, 3, queue) == 0, "CQ success");
        expectedQp->sq_rsvd_index = 0;
        polls = 0;
        context.flush(peer, queue);
        require(context.tryFlush(peer, 0, queue) == 0 && polls == 0, "empty queue completion");
      }
    }
    for (const auto result : scratch) require(result == 99, "per-QP scratch coverage");
    context.put(1, 0, 0, 8);
    require(selectedQp == &qps[queues], "legacy default QP zero");
    bool rejected = false;
    try {
      context.put(1, 0, 0, 8, queues);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "out-of-range QP accepted");
  }
  for (const int count : {1, 2, 64}) {
    const int offsets[] = {0, 0, 1, 1, 1 + count};
    std::vector<doca_gpu_dev_verbs_qp> sparseQps(1 + count);
    std::vector<uint64_t> sparseScratch(1 + count);
    mscclpp::GpuNetIoDeviceContext sparse{};
    sparse.qps = sparseQps.data();
    sparse.numPeers = 4;
    sparse.numQpsPerPeer = count;
    sparse.peerQpOffsets = offsets;
    sparse.atomicResultBase = reinterpret_cast<uintptr_t>(sparseScratch.data());
    sparse.atomicResultLkey = 7;
    for (const int peer : {1, 3}) {
      for (int queue = 0; queue < offsets[peer + 1] - offsets[peer]; ++queue) {
        sparse.atomicAddRegistered(peer, queue, {0x1000, 256, 11, peer}, 8, 1);
        const int index = offsets[peer] + queue;
        require(
            selectedQp == &sparseQps[index] && signalResult.addr == reinterpret_cast<uintptr_t>(&sparseScratch[index]),
            "sparse QP and scratch lookup disagree");
        sparse.flush(peer, queue);
        require(selectedQp == &sparseQps[index], "sparse flush chose wrong QP");
      }
    }
    for (const int peer : {0, 1, 2, 3}) {
      selectedQp = nullptr;
      bool rejected = false;
      try {
        sparse.flush(peer, offsets[peer + 1] - offsets[peer]);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      require(rejected && selectedQp == nullptr, "unrequested QP reached transport");
    }
  }
  std::vector<doca_gpu_dev_verbs_qp> boundQps(4);
  uint64_t scratch[4]{};
  uint64_t counters[4]{};
  mscclpp::GpuNetIoDeviceContext bound{};
  bound.qps = boundQps.data();
  bound.numPeers = 2;
  bound.numQpsPerPeer = 2;
  bound.atomicResultBase = reinterpret_cast<uintptr_t>(scratch);
  bound.atomicResultLkey = 0x12345678;
  mscclpp::GpuNetIoMemoryDeviceHandle memories[4] = {{0x1000, 256, 0x11223344, 1},
                                                     {0x2000, 512, 0x22334455, 0},
                                                     {0x3000, 128, 0x33445566, 1},
                                                     {0x4000, 384, 0x44556677, 0}};
  for (int queue = 0; queue < 2; ++queue) {
    const mscclpp::GpuNetIoMemoryDeviceHandle signal{static_cast<uintptr_t>(0x8000 + queue * 64), 8, 0x55667788U, 1};
    mscclpp::BasePortChannelDeviceHandle base(&bound, 1, queue, 0, signal, &counters[queue * 2],
                                              &counters[queue * 2 + 1], memories, 4);
    mscclpp::PortChannelDeviceHandle channel(base, queue * 2, queue * 2 + 1);
    auto* expectedQp = &boundQps[2 + queue];
    channel.put(8, 16, 32);
    require(selectedQp == expectedQp && destination.addr == memories[queue * 2].base + 8 &&
                destination.key == __builtin_bswap32(memories[queue * 2].key) &&
                source.addr == memories[queue * 2 + 1].base + 16 &&
                source.key == __builtin_bswap32(memories[queue * 2 + 1].key),
            "channel memory/QP binding");
    channel.putWithSignalAndFlush(0, 16, 8, 7);
    require(selectedQp == expectedQp && signalDestination.addr == signal.base &&
                signalResult.addr == reinterpret_cast<uintptr_t>(&scratch[2 + queue]),
            "channel signal/QP binding");
    channel.signal();
    require(selectedQp == expectedQp && signalDestination.addr == signal.base, "standalone signal changed QP");
    channel.accumulate(64, -1);
    require(selectedQp == expectedQp && signalDestination.addr == memories[queue * 2].base + 64, "accumulate binding");
    channel.flush(-1);
    require(selectedQp == expectedQp, "flush changed QP");
    base.put(queue * 2, 8, queue * 2 + 1, 16, 8);
    require(destination.addr == memories[queue * 2].base + 8, "base memory IDs ignored");
    require(!channel.poll(), "semaphore falsely ready");
    counters[queue * 2] = 1;
    channel.wait();
    require(counters[queue * 2 + 1] == 1, "channel expected counter");
    for (int invalid = 0; invalid < 4; ++invalid) {
      bool rejected = false;
      selectedQp = nullptr;
      try {
        if (invalid == 0) base.put(4, 0, 1, 0, 8);
        if (invalid == 1) base.put(1, 0, 0, 0, 8);
        if (invalid == 2) channel.put(UINT64_MAX, 0, 8);
        if (invalid == 3) channel.accumulate(1, 1);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      require(rejected && selectedQp == nullptr, "invalid bound access reached DOCA");
    }
  }
  std::cout << "GPUNetIO multi-QP and two-channel binding checks passed\n";
}

#elif defined(TEST_HOST_API)
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_net_io_service.hpp>
#include <mscclpp/port_channel.hpp>

#include "gpu_net_io_binding.hpp"

static_assert(std::is_trivially_default_constructible_v<mscclpp::PortChannelDeviceHandle>);
static_assert(std::is_trivially_copyable_v<mscclpp::PortChannelDeviceHandle>);
static_assert(!std::is_constructible_v<mscclpp::PortChannel, mscclpp::GpuNetIoDeviceContext*, int, uint64_t, uint64_t*,
                                       uint64_t*>);
static_assert(!std::is_constructible_v<mscclpp::BasePortChannel, mscclpp::GpuNetIoDeviceContext*, int, uint64_t,
                                       uint64_t*, uint64_t*>);

#if defined(TEST_GPUNETIO_ENABLED)
class PeerTestBootstrap : public mscclpp::Bootstrap {
 public:
  PeerTestBootstrap(int rank, int worldSize) : rank_(rank), worldSize_(worldSize) {}
  int getRank() const override { return rank_; }
  int getNranks() const override { return worldSize_; }
  int getNranksPerNode() const override { return worldSize_; }
  void send(void*, int, int, int) override { throw std::runtime_error("unexpected send"); }
  void recv(void*, int, int, int) override { throw std::runtime_error("unexpected recv"); }
  void allGather(void*, int) override { throw std::runtime_error("unexpected allGather"); }
  void barrier() override { throw std::runtime_error("unexpected barrier"); }

 private:
  int rank_;
  int worldSize_;
};

class ExchangeTestBootstrap : public PeerTestBootstrap {
 public:
  ExchangeTestBootstrap(int rank, mscclpp::detail::GpuNetIoMemoryExchange response)
      : PeerTestBootstrap(rank, 2), response_(response) {}
  std::string order;
  mscclpp::detail::GpuNetIoMemoryExchange sent{};
  void send(void* data, int bytes, int peer, int tag) override {
    if (bytes != sizeof(sent) || peer != 1 - getRank() || tag != 123) throw std::runtime_error("invalid send envelope");
    sent = *static_cast<mscclpp::detail::GpuNetIoMemoryExchange*>(data);
    order += 'S';
  }
  void recv(void* data, int bytes, int peer, int tag) override {
    if (bytes != sizeof(response_) || peer != 1 - getRank() || tag != 123)
      throw std::runtime_error("invalid recv envelope");
    *static_cast<mscclpp::detail::GpuNetIoMemoryExchange*>(data) = response_;
    order += 'R';
  }

 private:
  mscclpp::detail::GpuNetIoMemoryExchange response_;
};
#endif

#if defined(__NVCC__)
__global__ void compileGpuNetIoChannel(mscclpp::PortChannelDeviceHandle channel) {
  channel.put(0, 8);
  channel.signal();
  channel.putWithSignal(0, 8);
  channel.putWithSignalAndFlush(0, 8, 8, 100);
  channel.accumulate(128, -1);
  channel.flush();
  channel.wait();
}
#endif

int main() {
#if defined(TEST_GPUNETIO_ENABLED)
  for (int rank = 0; rank < 2; ++rank) {
    for (const uint32_t kind : {1U, 2U}) {
      for (const int queue : {0, 1, 63}) {
        const mscclpp::detail::GpuNetIoMemoryExchange local{0x1000, 512, 17, kind, rank, 1 - rank, queue, 1};
        const mscclpp::detail::GpuNetIoMemoryExchange remote{0x2000, 1024, 29, kind, 1 - rank, rank, queue, 1};
        ExchangeTestBootstrap bootstrap(rank, remote);
        const auto result = mscclpp::detail::exchangeGpuNetIoMemory(bootstrap, local, 123);
        if (result.base != remote.base || result.bytes != remote.bytes || result.rkey != remote.rkey ||
            bootstrap.sent.base != local.base || bootstrap.sent.rkey != local.rkey ||
            bootstrap.order != (rank == 0 ? "SR" : "RS"))
          return 4;
        for (int failure = 0; failure < 8; ++failure) {
          auto bad = remote;
          switch (failure) {
            case 0:
              bad.version = 0;
              break;
            case 1:
              bad.kind = 3 - kind;
              break;
            case 2:
              bad.rank = rank;
              break;
            case 3:
              bad.peer = 1 - rank;
              break;
            case 4:
              bad.qpIndex = queue + 1;
              break;
            case 5:
              bad.base = 0;
              break;
            case 6:
              bad.bytes = 0;
              break;
            case 7:
              bad.bytes = UINT64_MAX;
              break;
          }
          ExchangeTestBootstrap mismatch(rank, bad);
          bool rejected = false;
          try {
            (void)mscclpp::detail::exchangeGpuNetIoMemory(mismatch, local, 123);
          } catch (const mscclpp::Error&) {
            rejected = true;
          }
          if (!rejected || mismatch.order != (rank == 0 ? "SR" : "RS")) return 5;
        }
      }
    }
  }
  for (const int worldSize : {1, 2, 4}) {
    for (int rank = 0; rank < worldSize; ++rank) {
      auto bootstrap = std::make_shared<PeerTestBootstrap>(rank, worldSize);
      mscclpp::GpuNetIoService service(bootstrap, "test-device", 0);
      for (const int peer : {-1, rank, worldSize, std::numeric_limits<int>::max(), (rank + 1) % worldSize}) {
        const bool invalid = peer < 0 || peer >= worldSize || peer == rank;
        for (const int queue : {-1, 0, 64}) {
          bool rejected = false;
          try {
            (void)service.connect(peer, queue);
          } catch (const mscclpp::Error& error) {
            const std::string message = error.what();
            rejected =
                message.find(invalid ? "remote bootstrap rank" : "successful service setup") != std::string::npos;
          }
          if (!rejected) return 1;
        }
      }
    }
  }
  for (const bool base : {false, true}) {
    bool rejected = false;
    try {
      if (base) {
        mscclpp::BasePortChannel channel(mscclpp::GpuNetIoSemaphore{});
      } else {
        mscclpp::PortChannel channel(mscclpp::GpuNetIoSemaphore{}, {}, {});
      }
    } catch (const mscclpp::Error&) {
      rejected = true;
    }
    if (!rejected) return 3;
  }
#else
  mscclpp::PortChannelDeviceHandle handle(0, {}, {}, 0, 0, nullptr);
  if (handle.backend_ != mscclpp::PortChannelBackend::Proxy) return 2;
#endif
  std::cout << "PortChannel linked host API checks passed\n";
}

#else
#define MSCCLPP_FIFO_DEVICE_HPP_
#define MSCCLPP_SEMAPHORE_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_
#define MSCCLPP_DEVICE_COMPILE
#define MSCCLPP_INLINE inline
#define MSCCLPP_DEVICE_INLINE inline
#define MSCCLPP_HOST_DEVICE_INLINE inline
#ifdef TEST_DEVICE_ASSERTS
#define MSCCLPP_ASSERT_DEVICE(condition, message) require(condition, message)
#else
#define MSCCLPP_ASSERT_DEVICE(condition, message)
#endif
#define POLL_MAYBE_JAILBREAK(condition, budget) require(!(condition), "unexpected polling")

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

namespace mscclpp {
enum { scopeSystem, memoryOrderAcquire };
enum { TriggerPut, TriggerSignal, TriggerPutWithSignal, TriggerPutWithSignalAndFlush, TriggerFlush, TriggerAccumulate };
constexpr int TriggerBitsSize = 32;

template <typename Value, int Scope>
Value atomicLoad(Value* pointer, int) {
  return *pointer;
}

struct ProxyTrigger {
  int type;
  uint32_t dstId;
  uint64_t dstOffset;
  uint32_t srcId;
  uint64_t srcOffset;
  uint64_t size;
  uint32_t semaphoreId;
};

struct FifoDeviceHandle {
  std::vector<ProxyTrigger>* calls = nullptr;
  uint64_t push(ProxyTrigger trigger) {
    require(calls != nullptr, "GDAKI touched proxy FIFO");
    calls->push_back(trigger);
    return calls->size() - 1;
  }
};

struct Host2DeviceSemaphoreDeviceHandle {
  uint64_t* inboundToken;
  uint64_t* expectedInboundToken;
  bool poll() {
    if (*inboundToken <= *expectedInboundToken) return false;
    ++*expectedInboundToken;
    return true;
  }
  void wait(int64_t) { require(poll(), "signal not received"); }
};

struct GpuNetIoMemoryDeviceHandle {
  uintptr_t base;
  uint64_t bytes;
  uint32_t key;
  int rank;
};
struct GpuNetIoDeviceContext {
  int numPeers = 4;
  int numQpsPerPeer = 2;
  int queue = -1;
  int peer = -1;
  int puts = 0;
  int atomics = 0;
  int flushes = 0;
  int boundedFlushes = 0;
  int status = 0;
  uint64_t destination = 0;
  uint64_t source = 0;
  uint64_t bytes = 0;
  uint64_t signalOffset = 0;
  int64_t value = 0;
  uint64_t budget = 0;
  void putRegistered(int remotePeer, int qp, GpuNetIoMemoryDeviceHandle dstMemory, uint64_t dst,
                     GpuNetIoMemoryDeviceHandle srcMemory, uint64_t src, uint64_t size) {
    queue = qp;
    peer = remotePeer;
    destination = dstMemory.base + dst;
    source = srcMemory.base + src;
    bytes = size;
    ++puts;
  }
  void putRegisteredWithSignal(int remotePeer, int qp, GpuNetIoMemoryDeviceHandle dstMemory, uint64_t dst,
                               GpuNetIoMemoryDeviceHandle srcMemory, uint64_t src, uint64_t size,
                               GpuNetIoMemoryDeviceHandle signal) {
    putRegistered(remotePeer, qp, dstMemory, dst, srcMemory, src, size);
    signalOffset = signal.base;
    value = 1;
  }
  void atomicAddRegistered(int remotePeer, int qp, GpuNetIoMemoryDeviceHandle memory, uint64_t dst, int64_t add) {
    queue = qp;
    peer = remotePeer;
    destination = memory.base + dst;
    value = add;
    ++atomics;
  }
  void flush(int remotePeer, int qp) {
    queue = qp;
    peer = remotePeer;
    ++flushes;
  }
  int tryFlush(int remotePeer, uint64_t spins, int qp) {
    queue = qp;
    peer = remotePeer;
    budget = spins;
    ++boundedFlushes;
    return status;
  }
};
}  // namespace mscclpp

#include <mscclpp/port_channel_device.hpp>

int main() {
  using namespace mscclpp;
  uint64_t inbound = 0;
  uint64_t expected = 0;
  uint64_t flushDone = UINT64_MAX;
  std::vector<ProxyTrigger> triggers;
  PortChannelDeviceHandle proxy(12, {&inbound, &expected}, {&triggers}, 9, 7, &flushDone);
  require(proxy.backend_ == PortChannelBackend::Proxy, "proxy is not the default");
  proxy.put(8, 16, 32);
  proxy.signal();
  proxy.putWithSignal(24, 40, 64);
  proxy.putWithSignalAndFlush(32, 48, 128, 17);
  proxy.flush();
  const int64_t operand = -((int64_t{1} << 40) + 3);
  proxy.accumulate(56, operand);
  require(triggers.size() == 6, "proxy calls changed");
  const int types[] = {TriggerPut,   TriggerSignal,    TriggerPutWithSignal, TriggerPutWithSignalAndFlush,
                       TriggerFlush, TriggerAccumulate};
  for (size_t index = 0; index < triggers.size(); ++index) {
    require(triggers[index].type == types[index] && triggers[index].semaphoreId == 12, "proxy opcode or semaphore");
  }
  require(triggers[0].dstId == 9 && triggers[0].srcId == 7 && triggers[0].dstOffset == 8 &&
              triggers[0].srcOffset == 16 && triggers[0].size == 32,
          "proxy addressing");
  require((triggers[5].srcOffset << TriggerBitsSize | triggers[5].size) == static_cast<uint64_t>(operand),
          "proxy truncated signed 64-bit accumulate");

  GpuNetIoDeviceContext context;
  const GpuNetIoMemoryDeviceHandle memories[] = {{1024, 512, 11, 3}, {2048, 512, 12, 0}};
  BasePortChannelDeviceHandle base(&context, 3, 1, 0, {64, 8, 13, 3}, &inbound, &expected, memories, 2);
  PortChannelDeviceHandle network(base, 0, 1);
  require(network.backend_ == PortChannelBackend::GpuNetIo, "GDAKI selection");
  network.put(8, 16, 32);
  require(context.peer == 3 && context.queue == 1 && context.destination == 1032 && context.source == 2064 &&
              context.bytes == 32,
          "GDAKI put");
  network.signal();
  require(context.peer == 3 && context.destination == 64 && context.value == 1, "GDAKI signal address");
  network.putWithSignal(24, 40, 64);
  require(context.signalOffset == 64 && context.value == 1, "GDAKI fused signal");
  network.putWithSignalAndFlush(32, 48, 128, 17);
  require(context.boundedFlushes == 1 && context.budget == 17, "bounded flush removed in release build");
  network.flush(0);
  require(context.boundedFlushes == 2 && context.budget == 0, "zero-budget flush");
  network.flush(-1);
  require(context.flushes == 1, "unbounded flush");
  network.accumulate(56, operand);
  require(context.destination == 1080 && context.value == operand, "GDAKI accumulate");
  base.put(0, 8, 1, 16, 32);
  require(context.peer == 3 && context.queue == 1, "memory ID used as GDAKI peer/QP");
  require(!network.poll(), "unsignaled counter");
  inbound = 2;
  require(network.poll(), "signal polling");
  network.wait();
  require(expected == 2 && !network.poll(), "signal consumption");
#ifdef TEST_DEVICE_ASSERTS
  context.status = 16;
  bool rejected = false;
  try {
    network.flush(7);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected && context.budget == 7, "flush failure ignored");
#endif
  std::cout << "PortChannel proxy/GDAKI routing checks passed\n";
}
#endif