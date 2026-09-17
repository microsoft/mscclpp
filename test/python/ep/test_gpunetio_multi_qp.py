# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU checks for the multi-QP and combine pipeline ports; not GPU/NIC correctness evidence."""

import argparse
import ast
from contextlib import nullcontext, redirect_stderr
import io
from itertools import product
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_gpunetio_feature_port import HOST_PREAMBLE, ROOT, block, code, function, source, structure

DISPATCH = "src/ext/ep/dispatch/common.cuh"
COMBINE = "src/ext/ep/combine/common.cuh"
SERVICE = "src/gpunetio/host/gpu_net_io_service.cpp"
HEADER = "include/mscclpp/port_channel_gpunetio_device.hpp"
IMPL = "include/mscclpp/internal/port_channel_gpunetio_device_impl.hpp"


class MultiQpTests(unittest.TestCase):
    def test_actual_plural_hca_selection_and_list_parsing(self):
        service = source(SERVICE)
        native = HOST_PREAMBLE + "\n#include <limits>\n#include <cstdio>\n"
        native += "enum class ErrorCode { InternalError, InvalidUsage };\n"
        native += (
            "struct Error : std::runtime_error { Error(const char* text, ErrorCode) : std::runtime_error(text) {} };\n"
        )
        native += source("src/gpunetio/host/gpu_net_io_topology.hpp")
        native += "\nnamespace detail = mscclpp::detail;\n"
        native += "using mscclpp::detail::gpunetio::HcaTopology;\n"
        native += "using mscclpp::detail::gpunetio::pciPathDistance;\n"
        native += structure(service, "TopologyExchangeInfo")
        native += "std::vector<HcaTopology> available;\n"
        native += "std::vector<HcaTopology> discoverActiveHcas() { return available; }\n"
        native += "std::string canonicalPath(const std::string& path) { return path; }\n"
        for name in ("selectAutomaticHcas", "splitIbDeviceNames"):
            native += function(service, name)
        native += r"""
int main() {
  require(splitIbDeviceNames("  nic0, nic1\t, ,nic2,") == std::vector<std::string>{"nic0","nic1","nic2"}, "list order/trim");
  for (const std::string spec : {"", " , \t", "nic0,nic0"}) {
    bool rejected = false;
    try { splitIbDeviceNames(spec); } catch (const Error&) { rejected = true; }
    require(rejected, "empty/duplicate HCA list accepted");
  }
  require(pciPathDistance("", "") == 1024, "unknown PCI path penalty");
  require(pciPathDistance("/root/a/gpu", "/root/a/nic") < pciPathDistance("/root/a/gpu", "/other/b/nic"), "PCI affinity");
  for (int gpus : {1,2,4,8}) for (int hcas : {1,2,4,8}) {
    available.clear();
    for (int index=0; index<hcas; ++index) available.push_back({"nic"+std::to_string(index), "/sys/bus/pci/devices/nic"+std::to_string(index),0});
    std::vector<TopologyExchangeInfo> topology(gpus+1);
    for (int rank=0; rank<=gpus; ++rank) {
      topology[rank].hostHash = rank==gpus ? 2 : 1;
      topology[rank].gpuNumaNode = 0;
      std::snprintf(topology[rank].gpuPciBusId,32,"gpu%d",rank);
    }
    std::vector<int> use(hcas);
    for (int rank=0; rank<gpus; ++rank) {
      auto selected=selectAutomaticHcas(topology,rank);
            require(selected.size()==static_cast<size_t>(hcas), "best-affinity set was divided among local GPUs");
      require(selected==selectAutomaticHcas(topology,rank), "nondeterministic selection");
      auto unique=selected;std::sort(unique.begin(),unique.end());
            require(selected==unique, "HCA set must be sorted by name");
      require(std::adjacent_find(unique.begin(),unique.end())==unique.end(), "duplicate HCA per GPU");
            std::reverse(available.begin(),available.end());
            require(selected==selectAutomaticHcas(topology,rank), "sysfs enumeration changed logical HCA indices");
      for (const auto& name:selected) ++use[std::stoi(name.substr(3))];
    }
        require(std::all_of(use.begin(),use.end(),[&](int count) { return count==gpus; }), "nearby GPUs must share every best-affinity HCA");
    topology[0].gpuNumaNode=1;
    available.push_back({"remote-numa", "/sys/bus/pci/devices/nic-local",1});
    require(selectAutomaticHcas(topology,0)==std::vector<std::string>{"remote-numa"}, "NUMA affinity ignored");
  }
}
"""
        self.run_native(native)

    def test_actual_collective_geometry_validation(self):
        setup = function(source(SERVICE), "GpuNetIoService::setup")
        validation = block(setup, r"for\s*\(const auto& config : configAll\)")
        native = HOST_PREAMBLE + "\nenum class ErrorCode { InvalidUsage };\n"
        native += (
            "struct Error : std::runtime_error { Error(const char* text, ErrorCode) : std::runtime_error(text) {} };\n"
        )
        native += structure(source(SERVICE), "ConfigExchangeInfo")
        native += "bool valid(int nHcas,int requestedQps,ConfigExchangeInfo remote) {\n"
        native += "std::vector<ConfigExchangeInfo> configAll={{static_cast<uint32_t>(nHcas),static_cast<uint32_t>(requestedQps)},remote};\n"
        native += (
            "try { for(const auto& config:configAll) "
            + validation
            + " } catch(const Error&) { return false; } return true; }\n"
        )
        native += r"""
int main() {
  for (int hcas : {0,1,2,3,4,8,64,65}) for (int queues : {0,1,2,3,4,8,12,64,65}) {
    const bool expected=hcas>=1 && queues>=hcas && queues<=64 && queues%hcas==0;
    require(valid(hcas,queues,{static_cast<uint32_t>(hcas),static_cast<uint32_t>(queues)})==expected,"geometry validation");
    require(!valid(hcas,queues,{static_cast<uint32_t>(hcas+1),static_cast<uint32_t>(queues)}),"HCA disagreement");
    require(!valid(hcas,queues,{static_cast<uint32_t>(hcas),static_cast<uint32_t>(queues+1)}),"QP disagreement");
  }
}
"""
        self.run_native(native)
        self.ordered(
            setup,
            "allGather(configAll.data()",
            "for (const auto& config : configAll)",
            "hca.ibCtx = std::make_unique<IbCtx>",
            "hca.mr = hca.ibCtx->registerMr",
            "s.qpHl.assign",
        )
        self.ordered(setup, "initAttr.ibpd = s.hcas[s.hcaIndex(qpIndex)].ibCtx->getPd()", "doca_gpu_verbs_create_qp_hl")
        self.assertIn("rkeysHost[static_cast<size_t>(hca)*s.worldSize+r]=htobe32(info.rkey)", code(setup))
        self.assertIn("ctxHost.lkeys=s.lkeysGpu", code(setup))
        self.assertIn("ctxHost.numHcas=nHcas", code(setup))
        self.assertIn("if(lkeysGpu)(void)cudaFree(lkeysGpu)", code(source(SERVICE)))
        initialize = function(source("src/ext/ep/latency.cc"), "LatencyContext::initialize")
        self.ordered(
            initialize,
            'std::getenv("MSCCLPP_EP_GPUNETIO_HCAS")',
            'std::getenv("MSCCLPP_EP_GPUNETIO_HCA")',
            "std::make_shared<mscclpp::GpuNetIoService>",
        )

    def test_all_qp_generations_and_fixed_stripe_markers(self):
        recv = function(source(DISPATCH), "dispatchRecvRankMajor")
        remote = block(
            recv, r"if\s*\(transport.gpuNetIo_\s*!=\s*nullptr\s*&&\s*!transport.isNvlinkPeer\(sourceRank\)\)"
        )
        self.ordered(
            remote,
            "dispatchArrivedBaseline_[sourceRank] + 1",
            "qpIndex < nQp",
            "while (flags[qpIndex] < target)",
            "__syncthreads()",
            "if (threadIdx.x == 0) workspaceView.dispatchArrivedBaseline_[sourceRank] = target",
            "return",
        )
        self.assertNotIn("nRankTokens", remote)
        for queues in (1, 4, 64):
            flags = [0] * queues
            baseline = 0
            for generation in range(1, 201):
                for queue in reversed(range(queues)):
                    flags[queue] = generation + (queue > 0)
                    self.assertEqual(all(value >= baseline + 1 for value in flags), queue == 0)
                baseline += 1
                self.assertEqual(baseline, generation)
            for hcas in (1, 2, 4, 8, 64):
                if queues % hcas:
                    continue
                for owner, rows in product((0, 1, 7, 63), (1, 2, 7, 133)):
                    used = [(owner % queues + stripe) % queues for stripe in range(hcas)]
                    self.assertEqual(len(set(queue % hcas for queue in used)), hcas)
                    intervals = [(rows * stripe // hcas, rows * (stripe + 1) // hcas) for stripe in range(hcas)]
                    self.assertEqual([row for begin, end in intervals for row in range(begin, end)], list(range(rows)))
                    self.assertEqual(len(intervals), hcas)

    def run_native(self, native):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ unavailable")
        with tempfile.TemporaryDirectory(prefix="ep-combine-pipeline-") as directory:
            binary = str(Path(directory) / "check")
            compiled = subprocess.run(
                [compiler, "-std=c++17", "-x", "c++", "-", "-o", binary],
                input=native,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([binary], text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_dense_landing_allocation_preserves_existing_offsets(self):
        path = "src/ext/ep/include/config.hpp"
        current = source(path)
        previous = subprocess.check_output(["git", "-C", str(ROOT), "show", "3bde321:" + path], text=True)
        native = HOST_PREAMBLE + r"""
#include <sys/mman.h>
using Bf16 = uint16_t;
using Fp8E4M3 = uint8_t;
enum class DispatchLayout { RANK_MAJOR, EXPERT_MAJOR, RANK_MAJOR_TOPK_EXPANDED };
enum class CombineMode { RANK_LOCAL_REDUCE, DIRECT_SEND };
"""
        native += "\n".join(line for line in current.splitlines() if line.startswith("inline constexpr int GpuNetIo"))
        native += "\ntemplate<typename DataType, typename ScaleType = void>\n" + structure(current, "PayloadView")
        for name in ("rankMajorTopkIdsOffset", "rankMajorTopkWeightsOffset", "rankMajorTokenOffset"):
            native += function(current, name)
        native += structure(previous, "LatencyStorageLayout").replace("LatencyStorageLayout", "PreviousLayout")
        native += structure(current, "LatencyStorageLayout")
        native += r"""
int main() {
  for (int ranks : {2, 8, 16, 32, 64}) for (int capacity : {8, 257, 1024})
  for (int hidden : {4096, 7168, 9216}) for (int topk : {1, 8, 32})
  for (auto layout : {DispatchLayout::RANK_MAJOR, DispatchLayout::EXPERT_MAJOR})
  for (auto mode : {CombineMode::RANK_LOCAL_REDUCE, CombineMode::DIRECT_SEND}) {
    LatencyStorageLayout sizing(nullptr, capacity, hidden, ranks, 256, topk, layout, mode);
    require(sizing.gpuNetIoCombineLandingBuffer_ == nullptr, "null layout must not publish a landing pointer");
    void* allocation = mmap(nullptr, sizing.totalBytes_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(allocation != MAP_FAILED, "virtual address reservation failed");
    LatencyStorageLayout current(allocation, capacity, hidden, ranks, 256, topk, layout, mode);
    PreviousLayout previous(allocation, capacity, hidden, ranks, 256, topk, layout, mode);
    require(current.dispatchRecvBuffer_ == previous.dispatchRecvBuffer_ &&
            current.combineRecvBuffer_ == previous.combineRecvBuffer_ &&
            current.dispatchOutputBuffer_ == previous.dispatchOutputBuffer_ &&
            current.rankMajorTopkIdsBuffer_ == previous.rankMajorTopkIdsBuffer_ &&
            current.rankMajorTopkWeightsBuffer_ == previous.rankMajorTopkWeightsBuffer_ &&
            current.gpuNetIoStagingBuffer_ == previous.gpuNetIoStagingBuffer_ &&
            current.gpuNetIoFlagsBuffer_ == previous.gpuNetIoFlagsBuffer_ &&
            current.gpuNetIoSlotStride_ == previous.gpuNetIoSlotStride_, "existing offsets changed");
    if (layout == DispatchLayout::RANK_MAJOR && mode == CombineMode::RANK_LOCAL_REDUCE)
      require(current.combineRecvBuffer_ == current.dispatchOutputBuffer_, "rank-major alias changed");
    auto* base = static_cast<uint8_t*>(allocation);
    auto* landing = static_cast<uint8_t*>(current.gpuNetIoCombineLandingBuffer_);
    const size_t bytes = static_cast<size_t>(ranks) * capacity * hidden * sizeof(Bf16);
        const size_t oldFlags = configAlign<size_t>(static_cast<size_t>(ranks) * sizeof(uint64_t), 128);
        const size_t flags = configAlign<size_t>(static_cast<size_t>(ranks) * 64 * sizeof(uint64_t), 128);
        require(current.gpuNetIoCombineFlagsBuffer_ == static_cast<uint8_t*>(current.gpuNetIoFlagsBuffer_) + flags,
            "dispatch flags overlap combine flags");
        require(landing == base + previous.totalBytes_ + flags - oldFlags, "landing must follow enlarged flags");
    require(reinterpret_cast<uintptr_t>(landing) % 128 == 0, "landing alignment");
    require(current.totalBytes_ == previous.totalBytes_ + flags - oldFlags + configAlign<size_t>(bytes, 128), "landing allocation size");
    for (int rank = 0; rank < ranks; ++rank) {
      const size_t offset = static_cast<size_t>(rank) * capacity * hidden * sizeof(Bf16);
      require(landing + offset + static_cast<size_t>(capacity) * hidden * sizeof(Bf16) <=
              base + current.totalBytes_, "bulk owner write exceeds landing storage");
    }
    require(munmap(allocation, sizing.totalBytes_) == 0, "virtual reservation cleanup");
  }
}
"""
        self.run_native(native)
        self.assertIn(
            "deviceContext_.gpuNetIoCombineLandingBuffer_=layout.gpuNetIoCombineLandingBuffer_;",
            code(source("src/ext/ep/latency.cc")),
        )
        self.assertIn(
            "gpuNetIoCombineLandingBuffer_(context->gpuNetIoCombineLandingBuffer_)",
            code(source("src/ext/ep/common/latency.cuh")),
        )

    def test_actual_owner_send_drain_and_independent_readiness(self):
        native = HOST_PREAMBLE + r"""
#define MSCCLPP_DEVICE_INLINE inline
#define __trap() throw std::runtime_error("invalid QP count")
#define EP_DEVICE_ASSERT(condition) require(condition, "device assertion")
using Bf16 = uint16_t;
constexpr int GpuNetIoMaxQpsPerPeer = 64;
struct Dim { unsigned int x = 0; } blockIdx, threadIdx;
namespace mscclpp {
constexpr int scopeDevice = 0, memoryOrderRelaxed = 0;
template<class Value, int Scope> void atomicStore(Value* target, Value value, int) { *target = value; }
}
struct Transfer { int owner; uint64_t destination, source, bytes, flag, value; int queue; };
struct Gin {
  int numQpsPerPeer = 1;
    int numHcas = 1;
  std::vector<Transfer> transfers;
  std::vector<std::pair<int, int>> drains;
  void putWithSignal(int owner, uint64_t destination, uint64_t source, uint64_t bytes,
                    uint64_t flag, uint64_t value, int queue) {
    transfers.push_back({owner, destination, source, bytes, flag, value, queue});
  }
  void flush(int owner, int queue) { drains.emplace_back(owner, queue); }
};
namespace mscclpp { using GpuNetIoDeviceContext = Gin; }
struct Channel {
    mutable int signals = 0, waits = 0;
    void relaxedSignal() const { ++signals; }
    void relaxedWait(int) const { require(signals > waits, "wait before signal"); ++waits; }
};
struct TransportView {
  int rank_; Gin* gpuNetIo_;
  uint8_t* base;
  void* gpuNetIoCombineLandingBuffer_; void* gpuNetIoCombineFlagsBuffer_;
    Channel baseMemoryChannels_[64];
  bool isSelf(int peer) const { return peer == rank_; }
  bool isNvlinkPeer(int peer) const { return peer / 2 == rank_ / 2; }
  uint64_t symmetricOffset(const void* ptr) const { return static_cast<const uint8_t*>(ptr) - base; }
};
struct WorkspaceView {
  int* dispatchRecvCounts_; int* rankMajorSendIndices_;
  uint64_t* combineArrivedBaseline_; uint32_t* combineRankReadyEpochs_;
};
"""
        combine = source(COMBINE)
        native += function(combine, "rankMajorSlotForDestination")
        native += function(combine, "rankMajorCombineStripeQp")
        native += function(combine, "signalRankMajorCombineLocalStart")
        native += function(combine, "publishRankMajorCombinePushReady")
        native += "template<int Hidden>\n" + function(combine, "sendRankMajorCombinePush")
        native += function(combine, "drainRankMajorCombinePush")
        native += r"""
int main() {
  constexpr size_t hiddenBytes = 8 * sizeof(Bf16);
  for (int ranks : {2, 8, 16, 32, 64}) for (int queues : {1, 4, 64})
    for (int capacity : {1, 8, 257, 1024}) for (int hcas : {1, 2, 4, 8, 64}) {
        if (hcas > queues || queues % hcas) continue;
    const size_t payloadBytes = static_cast<size_t>(ranks) * capacity * hiddenBytes;
    std::vector<uint64_t> storage((2 * payloadBytes) / 8 + ranks * 64);
    auto* base = reinterpret_cast<uint8_t*>(storage.data());
    auto* flags = reinterpret_cast<uint64_t*>(base + 2 * payloadBytes);
    std::vector<int> counts(ranks), slots(ranks, 0);
    std::vector<int64_t> routes(ranks);
    std::vector<uint64_t> baselines(ranks);
    std::vector<uint32_t> ready(ranks);
    WorkspaceView workspace{counts.data(), slots.data(), baselines.data(), ready.data()};
    Gin gin;
    gin.numQpsPerPeer = queues;
    gin.numHcas = hcas;
    TransportView transport{ranks - 1, &gin, base, base + payloadBytes, flags};
    size_t expectedTransfers = 0;
    for (int owner = 0; owner < ranks; ++owner) {
      counts[owner] = owner % 3 == 0 ? 0 : capacity;
    if (!transport.isNvlinkPeer(owner) && counts[owner] > 0) expectedTransfers += hcas;
    }
    for (unsigned int owner = 0; owner < static_cast<unsigned int>(ranks + 2); ++owner) {
      blockIdx.x = owner;
    for (threadIdx.x = 0; threadIdx.x < 65; ++threadIdx.x) {
        sendRankMajorCombinePush<8>(base, ranks, capacity, transport, workspace, 1);
        drainRankMajorCombinePush(ranks, transport, workspace, 1);
      }
    }
    require(gin.transfers.size() == expectedTransfers && gin.drains.size() == expectedTransfers,
            "must post and drain exactly once per HCA of each nonempty remote owner");
    for (size_t index = 0; index < gin.transfers.size(); ++index) {
      const auto& transfer = gin.transfers[index];
    const int stripe = index % hcas;
    const int queue = (transfer.owner % queues + stripe) % queues;
    const int begin = counts[transfer.owner] * stripe / hcas;
    const int end = counts[transfer.owner] * (stripe + 1) / hcas;
    require(transfer.source == (static_cast<size_t>(transfer.owner) * capacity + begin) * hiddenBytes, "source rows");
    require(transfer.destination == payloadBytes + (static_cast<size_t>(transport.rank_) * capacity + begin) * hiddenBytes,
              "landing keyed by expert-host rank");
    require(transfer.bytes == static_cast<size_t>(end - begin) * hiddenBytes, "stripe byte count incl zero-row marker");
      require(transfer.flag == 2 * payloadBytes + (transport.rank_ * 64 + queue) * sizeof(uint64_t) &&
              transfer.queue == queue && transfer.value == 1, "owner-QP marker");
      require(gin.drains[index] == std::make_pair(transfer.owner, queue), "drain wrong queue");
    }
    std::fill(baselines.begin(), baselines.end(), uint64_t{1} << 40);
    for (uint32_t epoch = 1; epoch <= 200; ++epoch) {
      std::fill(ready.begin(), ready.end(), epoch - 1);
      for (int peer = 0; peer < ranks; ++peer) routes[peer] = (epoch + peer) % 3 == 0 ? -1 : peer;
      blockIdx.x = 0;
            for (threadIdx.x = 0; threadIdx.x < static_cast<unsigned int>(ranks); ++threadIdx.x)
                signalRankMajorCombineLocalStart(transport, ranks);
      for (int peer = ranks - 1; peer >= 0; --peer) {
        threadIdx.x = peer;
        const bool incoming = !transport.isNvlinkPeer(peer) && routes[peer] >= 0;
        const uint64_t before = baselines[peer];
                if (incoming) for (int stripe = 0; stripe < hcas; ++stripe)
                    flags[peer * 64 + (transport.rank_ % queues + stripe) % queues] = before + 1;
        publishRankMajorCombinePushReady(routes.data(), ranks, 1, 1, ranks, epoch, transport, workspace);
        require(baselines[peer] == before + incoming && ready[peer] == epoch, "readiness/baseline generation");
        if (peer > 0) require(ready[peer - 1] == epoch - 1, "one peer must not publish another peer's readiness");
      }
      blockIdx.x = 1; threadIdx.x = 0;
      publishRankMajorCombinePushReady(nullptr, 0, 1, 1, ranks, epoch + 1, transport, workspace);
      require(ready[0] == epoch, "non-control block published readiness");
    }
  }
}
"""
        self.run_native(native)

    def test_graph_correctness_stress_controls(self):
        path = "test/python/ep/test_latency_multirank.py"
        tree = ast.parse(source(path), filename=path)
        definitions = [
            next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
            for name in ("parse_args", "_graph_capture", "_run_cuda_graph_correctness")
        ]
        events = []
        cuda = SimpleNamespace(
            CUDAGraph=lambda: SimpleNamespace(replay=lambda: events.append("replay")),
            Event=lambda **kwargs: SimpleNamespace(record=lambda: None),
            synchronize=lambda: events.append("sync"),
            graph=lambda graph: nullcontext(),
        )
        namespace = dict(
            argparse=argparse,
            torch=SimpleNamespace(cuda=cuda, empty_like=lambda value: value),
            dist=SimpleNamespace(barrier=lambda **kwargs: None),
            ep=SimpleNamespace(
                DispatchLayout=SimpleNamespace(RANK_MAJOR=1), CombineMode=SimpleNamespace(DIRECT_SEND=2)
            ),
            output_layout=1,
            combine_mode=1,
            dispatch_output_buffer="buffer",
            out="out",
            expected="expected",
            group=None,
            rank=0,
            x="x",
            topk_idx="ids",
            topk_weights="weights",
            moe_comm=SimpleNamespace(
                dispatch=lambda *args, **kwargs: events.append("dispatch") or ("tokens", "handle"),
                combine=lambda *args, **kwargs: events.append("combine") or "combined",
            ),
            stage_simulated_gemm_output=lambda output: events.append("stage") or "expert",
            validate_combine_output=lambda *args, **kwargs: events.append("validate") or (0, 0),
        )
        exec(compile(ast.Module(body=definitions, type_ignores=[]), path, "exec"), namespace)
        for options, pairs, replays in (([], 1, 1), (["--graph-pairs", "9", "--graph-replays", "50"], 9, 50)):
            events.clear()
            with patch("sys.argv", [path, *options]):
                namespace["args"] = namespace["parse_args"]()
            with patch("builtins.print"):
                namespace["_run_cuda_graph_correctness"]()
            self.assertEqual(
                [event for event in events if event in ("dispatch", "stage", "combine")],
                ["dispatch", "stage", "combine"] * pairs,
            )
            self.assertEqual(events[-(replays + 2) :], ["replay"] * replays + ["sync", "validate"])
        for option in ("--graph-pairs", "--graph-replays"):
            with patch("sys.argv", [path, option, "0"]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    namespace["parse_args"]()
            self.assertEqual(caught.exception.code, 2)

    def ordered(self, text, *fragments):
        text = code(text)
        offset = 0
        for fragment in fragments:
            needle = code(fragment)
            found = text.find(needle, offset)
            self.assertGreaterEqual(found, 0, fragment)
            offset = found + len(needle)

    def test_peer_major_device_and_bootstrap_indexing(self):
        for name in ("put", "putWithSignal", "atomicAdd", "get", "flush", "tryFlush", "putBatched3"):
            native = function(source(IMPL), "GpuNetIoDeviceContext::" + name)
            self.assertIn("detail::ginQp(qps,peer*numQpsPerPeer+qpIndex)", code(native))
            if name in ("put", "putWithSignal", "atomicAdd", "get", "putBatched3"):
                self.assertIn("detail::ginRemoteKey(*this,peer,qpIndex)", code(native))
                self.assertIn("detail::ginHtobe32(detail::ginLocalKey(*this,qpIndex))", code(native))
                self.assertNotIn("rkeys[peer]", code(native))
        native = code(function(source(SERVICE), "GpuNetIoService::setup"))
        self.assertIn("qpAll[static_cast<size_t>(r)*rowLen+static_cast<size_t>(s.rank)*nQp+qpIndex]", native)
        self.assertIn("ctxHost.numQpsPerPeer=s.numQpsPerPeer;", native)
        for ranks, queues in product((2, 8, 16, 32, 64), (1, 2, 4, 8, 64)):
            row = ranks * queues
            for rank, peer in ((0, ranks - 1), (ranks - 1, 0)):
                for queue in range(queues):
                    self.assertEqual(divmod(peer * row + rank * queues + queue, row), (peer, rank * queues + queue))

    def test_configuration_agreement_precedes_variable_sized_exchange(self):
        self.ordered(
            function(source(SERVICE), "GpuNetIoService::setup"),
            "int requestedQps = qpsEnv == nullptr ? nHcas : std::max(1, std::atoi(qpsEnv));",
            "s.bootstrap->allGather(configAll.data(), static_cast<int>(sizeof(ConfigExchangeInfo)));",
            "config.numQpsPerPeer > 64",
            "config.numQpsPerPeer % config.numHcas != 0",
            "throw Error(",
            "s.numQpsPerPeer = requestedQps;",
            "s.qpHl.assign(rowLen, nullptr)",
            "s.bootstrap->allGather(qpAll.data(), static_cast<int>(rowLen * sizeof(QpExchangeInfo)))",
        )

    def test_dispatch_posting_barrier_then_markers_and_parallel_drains(self):
        send = function(source(DISPATCH), "sendRankMajorGpuNetIo")
        self.ordered(
            send,
            "leaderExpert",
            "usedQpMask",
            "for (int vector = laneId",
            "__threadfence_system()",
            "gin->putBatched3(",
        )
        self.assertNotIn("putWithSignal", send)
        self.assertNotIn("flush(", send)
        self.ordered(
            function(source(DISPATCH), "dispatchBody"),
            "usedQpMask[peer] = 0",
            "dispatchSendRankMajor<Hidden>",
            "workspaceView.combineSyncer_->sync(gridDim.x)",
            "static_cast<int>(blockIdx.x) == nWorkerBlocks + 1",
            "gin->atomicAdd(peer, transport.symmetricOffset(flagsSelf + qpIndex), 1, qpIndex)",
            "__syncthreads()",
            "gin->flush(peer, qpIndex)",
            "__syncthreads()",
            "dispatchRecvRankMajor(",
        )
        self.assertNotIn("flush(", function(source(DISPATCH), "writeRankMajorCounts"))

    def test_private_staging_windows_drain_before_reuse(self):
        self.ordered(
            function(source(DISPATCH), "dispatchSendRankMajorBf16"),
            "slotsPerGroup = (GpuNetIoStagingSlots - nRanks) / (nPayloadBlocks * DispatchMaxNWarpGroups)",
            "availableTokens = slotsPerGroup / nTopk",
            "tokensSinceFlush * nTopk",
            "sendRankMajorGpuNetIo<Hidden>",
            "++tokensSinceFlush >= batchTokens",
            "flushAllCrossDomainAllQps(",
            "__syncwarp()",
            "tokensSinceFlush = 0",
        )
        for ranks, workers, topk in product((2, 8, 16, 32, 64), (64, 128), (1, 8, 32)):
            groups = workers * 2
            slots = (32768 - ranks) // groups
            batch = min(128, slots // topk)
            self.assertGreaterEqual(batch, 1)
            occupied = set()
            for group in range(groups):
                interval = set(range(ranks + group * slots, ranks + group * slots + batch * topk))
                self.assertFalse(occupied & interval)
                self.assertLess(max(interval), 32768)
                occupied.update(interval)
            live = set()
            for token in range(batch * 3 + 1):
                if token % batch == 0:
                    live.clear()
                interval = set(range((token % batch) * topk, (token % batch + 1) * topk))
                self.assertFalse(live & interval)
                live.update(interval)

    def test_combine_owner_write_marker_and_pipelined_readiness(self):
        send = function(source(COMBINE), "sendRankMajorCombinePush")
        self.ordered(
            send,
            "const int owner = static_cast<int>(blockIdx.x)",
            "workspaceView.dispatchRecvCounts_[owner] > 0",
            "const int qpIndex = rankMajorCombineStripeQp(gin, owner, stripe)",
            "gin->putWithSignal(owner, transport.symmetricOffset(landingSlot), srcRowOffset,",
            "static_cast<uint64_t>(rowEnd - rowBegin) * HiddenBytes, transport.symmetricOffset(remoteFlag), 1, qpIndex)",
        )
        self.assertNotIn("flush(", send)
        self.assertNotIn("combineSyncer_", send)
        self.ordered(
            function(source(COMBINE), "recvRankMajorCombinePush"),
            "if (!sendsToRank) continue",
            "combineArrivedBaseline_[destinationRank] + 1",
            "stripe < gin->numHcas",
            "const int qpIndex = rankMajorCombineStripeQp(gin, transport.rank_, stripe)",
            "while (flags[flagIndex] < target)",
            "combineArrivedBaseline_[destinationRank] = target",
            "combineSyncer_->sync(gridDim.x)",
        )
        self.ordered(
            function(source(COMBINE), "combineBody"),
            "signalRankMajorCombineLocalStart(transport, nRanks)",
            "synchronizeRankMajorCombine(transport, nRanks, epoch, workspaceView)",
            "sendRankMajorCombinePush<Hidden>",
            "if (nTopk <= RankMajorTmaMaxNTopk)",
            "publishRankMajorCombinePushReady(",
            "recvRankMajorRemotePartialsTma<Hidden, Mode>",
            "recvRankMajorCombinePush<Hidden>",
            "drainRankMajorCombinePush(",
            "workspaceView.combineSyncer_->sync(gridDim.x)",
            "synchronizeRankMajorCombine(transport, nRanks, epoch + 1",
        )
        self.ordered(
            function(source(COMBINE), "publishRankMajorCombinePushReady"),
            "if (blockIdx.x != 0) return",
            "threadIdx.x",
            "transport.baseMemoryChannels_[destinationRank].relaxedWait(-1)",
            "if (sendsToRank)",
            "stripe < gin->numHcas",
            "rankMajorCombineStripeQp(gin, transport.rank_, stripe)",
            "while (flags[flagIndex] < target)",
            "combineArrivedBaseline_[destinationRank] = target",
            "workspaceView.combineRankReadyEpochs_ + destinationRank",
            "epoch",
        )
        tma = function(source(COMBINE), "recvRankMajorRemotePartialsTma")
        self.ordered(
            tma,
            "combineRankReadyEpochs_ + destinationRank",
            "if (pending && ready)",
            "transport.gpuNetIo_ == nullptr || transport.isNvlinkPeer(destinationRank)",
            "IsDirectSend ? sourceRow * nTopk + laneId : sourceRow",
            "transport.gpuNetIoCombineLandingBuffer_",
            "mscclpp::bulkLoad(",
        )
        for name in ("sendRankMajorCombinePush", "recvRankMajorCombinePush"):
            self.assertNotIn("gpuNetIoStagingBuffer_", function(source(COMBINE), name))

    def test_allocation_and_no_later_optimizations(self):
        config = code(source("src/ext/ep/include/config.hpp"))
        self.assertIn("GpuNetIoMaxQpsPerPeer=64", config)
        self.assertIn("static_cast<size_t>(numRanks)*GpuNetIoMaxQpsPerPeer*sizeof(uint64_t)", config)
        self.assertIn(
            "gpuNetIoStagingBytes+gpuNetIoFlagsBytes+gpuNetIoCombineFlagsBytes+gpuNetIoCombineLandingBytes", config
        )
        for path in (DISPATCH, COMBINE, SERVICE):
            self.assertNotIn("putWarpRows", source(path))

    def test_actual_batched_wqes_and_default_api(self):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ unavailable")
        api_check = r"""
#include <mscclpp/port_channel_gpunetio_device.hpp>
void verify(mscclpp::GpuNetIoDeviceContext& context) {
 context.put(1,0,0,4); context.put(1,0,0,4,3);
 context.putWithSignal(1,0,0,4,8,1); context.putWithSignal(1,0,0,4,8,1,3);
 context.get(1,0,0,4); context.get(1,0,0,4,3);
 context.atomicAdd(1,0,1); context.atomicAdd(1,0,1,3);
 context.flush(1); context.flush(1,3); context.tryFlush(1,10); context.tryFlush(1,10,3);
 context.putBatched3(1,3,0,0,4,4,4,4,8,8,4);
}
"""
        checked = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-I" + str(ROOT / "include"),
                "-DMSCCLPP_DEVICE_COMPILE",
                "-DMSCCLPP_DEVICE_INLINE=inline",
                "-x",
                "c++",
                "-fsyntax-only",
                "-",
            ],
            input=api_check,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        preamble = r"""
#include <cstdint>
#include <stdexcept>
#include <vector>
#define MSCCLPP_DEVICE_INLINE inline
#define MSCCLPP_DEVICE_COMPILE
constexpr int DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU=0;
constexpr int DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT=0;
constexpr int DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD=0;
constexpr int DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO=0;
constexpr int DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE=8;
constexpr int DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE=2;
struct Wqe { uint64_t ticket,dst,src,bytes; uint32_t rkey,lkey; int flags; };
struct doca_gpu_dev_verbs_qp { uint64_t next=1023; Wqe entries[3]; };
doca_gpu_dev_verbs_qp* expected;
std::vector<int> events;
template<int Mode> uint64_t doca_gpu_dev_verbs_reserve_wq_slots(doca_gpu_dev_verbs_qp* qp,int count,int) {
 if(qp!=expected || count!=3) throw std::runtime_error("reservation");
 events.push_back(1); uint64_t base=qp->next; qp->next+=count; return base;
}
Wqe* doca_gpu_dev_verbs_get_wqe_ptr(doca_gpu_dev_verbs_qp* qp,uint64_t ticket){return &qp->entries[ticket%3];}
void doca_gpu_dev_verbs_wqe_prepare_write(doca_gpu_dev_verbs_qp*,Wqe* wqe,uint64_t ticket,int opcode,
 int flags,int,uint64_t dst,uint32_t rkey,uint64_t src,uint32_t lkey,uint64_t bytes){
 if(opcode!=8)throw std::runtime_error("opcode");
 *wqe={ticket,dst,src,bytes,rkey,lkey,flags};events.push_back(2);
}
template<int Mode> void doca_gpu_dev_verbs_mark_wqes_ready(doca_gpu_dev_verbs_qp* qp,uint64_t first,uint64_t last){
 if(qp!=expected || first!=1023 || last!=1025)throw std::runtime_error("mark");events.push_back(3);
}
template<int Mode,int Scope,int Handler> void doca_gpu_dev_verbs_submit(doca_gpu_dev_verbs_qp* qp,uint64_t end,int){
 if(qp!=expected || end!=1026)throw std::runtime_error("submit");events.push_back(4);
}
namespace mscclpp {
namespace detail {
doca_gpu_dev_verbs_qp* ginQp(void* ptr,int flat){return static_cast<doca_gpu_dev_verbs_qp*>(ptr)+flat;}
uint32_t ginHtobe32(uint32_t key){return __builtin_bswap32(key);}
}
"""
        native = structure(source(HEADER), "GpuNetIoDeviceContext") + "namespace detail {\n"
        for name in ("ginHcaIndex", "ginRemoteKey", "ginLocalKey"):
            native += function(source(IMPL), name)
        native += "}\n" + function(source(IMPL), "GpuNetIoDeviceContext::putBatched3")
        main = r"""
}
int main(){
 doca_gpu_dev_verbs_qp queues[8];expected=queues+7;
 uint32_t keys[8]={0,0x1234,0,0x5678,0,0x9abc,0,0xdef0};uintptr_t bases[2]={0,0x100000};
 uint32_t localKeys[4]={0x12345678,0x23456789,0x3456789a,0x456789ab};
 mscclpp::GpuNetIoDeviceContext context{queues,keys,bases,0x12345678,0x200000,2,4};
 for (int hcas : {1,2,4}) for (bool legacy : {false,true}) {
 if (legacy && hcas != 1) continue;
 context.numHcas=hcas;context.lkeys=legacy?nullptr:localKeys;
 expected->next=1023;events.clear();
 context.putBatched3(1,3,100,200,14336,300,400,32,500,600,32);
 if(events!=std::vector<int>{1,2,2,2,3,4})return 1;
 for(int index=0;index<3;++index){auto row=expected->entries[index];
  if(row.ticket!=1023+index || row.flags!=2 || row.rkey!=keys[(3%hcas)*2+1] ||
      row.lkey!=__builtin_bswap32(localKeys[3%hcas]))return 2;
  if(row.dst!=0x100000+100+index*200 || row.src!=0x200000+200+index*200)return 3;
  if(row.bytes!=(index?32:14336))return 4;
 }
 }
 return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="ep-multi-qp-") as directory:
            binary = str(Path(directory) / "batch")
            compiled = subprocess.run(
                [compiler, "-std=c++17", "-x", "c++", "-", "-o", binary],
                input=preamble + native + main,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([binary], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
