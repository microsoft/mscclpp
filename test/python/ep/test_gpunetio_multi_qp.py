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
enum class DispatchLayout { RANK_MAJOR, EXPERT_MAJOR };
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
            current.gpuNetIoCombineFlagsBuffer_ == previous.gpuNetIoCombineFlagsBuffer_ &&
            current.gpuNetIoSlotStride_ == previous.gpuNetIoSlotStride_, "existing offsets changed");
    if (layout == DispatchLayout::RANK_MAJOR && mode == CombineMode::RANK_LOCAL_REDUCE)
      require(current.combineRecvBuffer_ == current.dispatchOutputBuffer_, "rank-major alias changed");
    auto* base = static_cast<uint8_t*>(allocation);
    auto* landing = static_cast<uint8_t*>(current.gpuNetIoCombineLandingBuffer_);
    const size_t bytes = static_cast<size_t>(ranks) * capacity * hidden * sizeof(Bf16);
    require(landing == base + previous.totalBytes_, "landing must append after every existing region");
    require(reinterpret_cast<uintptr_t>(landing) % 128 == 0, "landing alignment");
    require(current.totalBytes_ == previous.totalBytes_ + configAlign<size_t>(bytes, 128), "landing allocation size");
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
  std::vector<Transfer> transfers;
  std::vector<std::pair<int, int>> drains;
  void putWithSignal(int owner, uint64_t destination, uint64_t source, uint64_t bytes,
                    uint64_t flag, uint64_t value, int queue) {
    transfers.push_back({owner, destination, source, bytes, flag, value, queue});
  }
  void flush(int owner, int queue) { drains.emplace_back(owner, queue); }
};
struct TransportView {
  int rank_; Gin* gpuNetIo_;
  uint8_t* base;
  void* gpuNetIoCombineLandingBuffer_; void* gpuNetIoCombineFlagsBuffer_;
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
        native += function(combine, "publishRankMajorCombinePushReady")
        native += "template<int Hidden>\n" + function(combine, "sendRankMajorCombinePush")
        native += function(combine, "drainRankMajorCombinePush")
        native += r"""
int main() {
  constexpr size_t hiddenBytes = 8 * sizeof(Bf16);
  for (int ranks : {2, 8, 16, 32, 64}) for (int queues : {1, 4, 64})
  for (int capacity : {1, 8, 257, 1024}) {
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
    TransportView transport{ranks - 1, &gin, base, base + payloadBytes, flags};
    size_t expectedTransfers = 0;
    for (int owner = 0; owner < ranks; ++owner) {
      counts[owner] = owner % 3 == 0 ? 0 : capacity;
      expectedTransfers += !transport.isNvlinkPeer(owner) && counts[owner] > 0;
    }
    for (unsigned int owner = 0; owner < static_cast<unsigned int>(ranks + 2); ++owner) {
      blockIdx.x = owner;
      for (threadIdx.x = 0; threadIdx.x < 4; ++threadIdx.x) {
        sendRankMajorCombinePush<8>(base, ranks, capacity, transport, workspace, 1);
        drainRankMajorCombinePush(ranks, transport, workspace, 1);
      }
    }
    require(gin.transfers.size() == expectedTransfers && gin.drains.size() == expectedTransfers,
            "must post and drain exactly once per nonempty remote owner");
    for (size_t index = 0; index < gin.transfers.size(); ++index) {
      const auto& transfer = gin.transfers[index];
      const int queue = transfer.owner % queues;
      require(transfer.source == static_cast<size_t>(transfer.owner) * capacity * hiddenBytes, "source rows");
      require(transfer.destination == payloadBytes + static_cast<size_t>(transport.rank_) * capacity * hiddenBytes,
              "landing keyed by expert-host rank");
      require(transfer.bytes == static_cast<size_t>(counts[transfer.owner]) * hiddenBytes, "contiguous byte count");
      require(transfer.flag == 2 * payloadBytes + (transport.rank_ * 64 + queue) * sizeof(uint64_t) &&
              transfer.queue == queue && transfer.value == 1, "owner-QP marker");
      require(gin.drains[index] == std::make_pair(transfer.owner, queue), "drain wrong queue");
    }
    std::fill(baselines.begin(), baselines.end(), uint64_t{1} << 40);
    for (uint32_t epoch = 1; epoch <= 200; ++epoch) {
      std::fill(ready.begin(), ready.end(), epoch - 1);
      for (int peer = 0; peer < ranks; ++peer) routes[peer] = (epoch + peer) % 3 == 0 ? -1 : peer;
      blockIdx.x = 0;
      for (int peer = ranks - 1; peer >= 0; --peer) {
        threadIdx.x = peer;
        const bool incoming = !transport.isNvlinkPeer(peer) && routes[peer] >= 0;
        const uint64_t before = baselines[peer];
        if (incoming) flags[peer * 64 + transport.rank_ % queues] = before + 1;
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
            "int requestedQps = 1;",
            "peerQps[s.rank] = requestedQps;",
            "s.bootstrap->allGather(peerQps.data(), sizeof(int));",
            "requestedQps > 64",
            "value == requestedQps",
            "throw Error(",
            "s.numQpsPerPeer = requestedQps;",
            "s.qpHl.assign(rowLen, nullptr)",
            "s.bootstrap->allGather(qpAll.data(), static_cast<int>(rowLen * sizeof(QpExchangeInfo)))",
        )

    def test_dispatch_drain_barrier_then_count_publication(self):
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
            "usedQpMask[peer] & (1ull << qpIndex)",
            "gin->flush(peer, qpIndex)",
            "workspaceView.combineSyncer_->sync(gridDim.x)",
            "sharedMem[peer] > 0",
            "gin->atomicAdd(peer, transport.symmetricOffset(flag), sharedMem[peer], qpIndex)",
            "flushAllCrossDomain(",
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
            "const int qpIndex = owner % nQp",
            "gin->putWithSignal(owner, transport.symmetricOffset(landingSlot), srcRowOffset,",
            "static_cast<uint64_t>(nRowsToOwner) * HiddenBytes, transport.symmetricOffset(remoteFlag), 1, qpIndex)",
        )
        self.assertNotIn("flush(", send)
        self.assertNotIn("combineSyncer_", send)
        self.ordered(
            function(source(COMBINE), "recvRankMajorCombinePush"),
            "if (!sendsToRank) continue",
            "combineArrivedBaseline_[destinationRank] + 1",
            "const int qpIndex = transport.rank_ % nQp",
            "while (flags[flagIndex] < target)",
            "combineArrivedBaseline_[destinationRank] = target",
            "combineSyncer_->sync(gridDim.x)",
        )
        self.ordered(
            function(source(COMBINE), "combineBody"),
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
            "if (sendsToRank)",
            "transport.rank_ % transport.gpuNetIo_->numQpsPerPeer",
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
            self.assertNotIn("numHcas", source(path))
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
        native = structure(source(HEADER), "GpuNetIoDeviceContext") + function(
            source(IMPL), "GpuNetIoDeviceContext::putBatched3"
        )
        main = r"""
}
int main(){
 doca_gpu_dev_verbs_qp queues[8];expected=queues+7;
 uint32_t keys[2]={0,0x1234};uintptr_t bases[2]={0,0x100000};
 mscclpp::GpuNetIoDeviceContext context{queues,keys,bases,0x12345678,0x200000,2,4};
 context.putBatched3(1,3,100,200,14336,300,400,32,500,600,32);
 if(events!=std::vector<int>{1,2,2,2,3,4})return 1;
 for(int index=0;index<3;++index){auto row=expected->entries[index];
  if(row.ticket!=1023+index || row.flags!=2 || row.rkey!=0x1234 || row.lkey!=0x78563412)return 2;
  if(row.dst!=0x100000+100+index*200 || row.src!=0x200000+200+index*200)return 3;
  if(row.bytes!=(index?32:14336))return 4;
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
