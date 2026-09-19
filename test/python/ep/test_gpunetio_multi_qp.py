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
    def test_cmake_gpunetio_prerequisites(self):
        cmake = source("CMakeLists.txt")
        guard = cmake[cmake.index("if(MSCCLPP_USE_GPUNETIO AND") : cmake.index("# Code coverage setup")]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "guard.cmake"
            script.write_text(guard)
            for enabled, cuda, rocm, ib in product((0, 1), repeat=4):
                result = subprocess.run(
                    [
                        "cmake",
                        f"-DMSCCLPP_USE_GPUNETIO={enabled}",
                        f"-DMSCCLPP_USE_CUDA={cuda}",
                        f"-DMSCCLPP_USE_ROCM={rocm}",
                        f"-DMSCCLPP_USE_IB={ib}",
                        "-P",
                        str(script),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, not enabled or bool(cuda and not rocm and ib), result.stderr)
                if result.returncode:
                    self.assertIn("requires CUDA and InfiniBand", result.stderr)

    def test_service_teardown_precedes_memory_release(self):
        context = source("src/ext/ep/latency.cc")
        destructor = context[
            context.index("LatencyContext::~LatencyContext()") : context.index("void LatencyContext::initialize()")
        ]
        native = HOST_PREAMBLE + "\n#include <memory>\n#define MSCCLPP_USE_GPUNETIO\n#define CUDA_CHECK(call) call\n"
        native += 'int stage=0;void cudaDeviceSynchronize(){require(stage++==0,"sync order");}'
        native += 'void cudaFree(void*){require(stage==2,"service still alive");}'
        native += 'namespace mscclpp::detail{void gpuFreePhysical(void*){require(stage++==2,"backing memory order");}}'
        native += 'struct Service{~Service(){require(stage++==1,"registration teardown order");}};'
        native += "struct LatencyContext{std::unique_ptr<Service> gpuNetIoService_=std::make_unique<Service>();"
        native += "struct{void* devicePtr_=(void*)1;}deviceContext_;void* peerMappedBufferBasesGpu_=(void*)2;"
        native += "void* workspace_=(void*)3;void* symmetricBuffer_=(void*)4;~LatencyContext()noexcept(false);};"
        native += destructor + 'int main(){{LatencyContext context;}require(stage==3,"all resources released");}'
        self.run_native(native)

    def test_qp_mtu_uses_both_active_ports(self):
        service = source(SERVICE)
        method = service[service.index("  doca_verbs_mtu_size pathMtu") : service.index("  // INIT -> RTR -> RTS")]
        native = HOST_PREAMBLE
        native += "enum{IBV_MTU_256=1,IBV_MTU_512,IBV_MTU_1024,IBV_MTU_2048,IBV_MTU_4096};"
        native += "enum doca_verbs_mtu_size{DOCA_VERBS_MTU_SIZE_256_BYTES=1,DOCA_VERBS_MTU_SIZE_512_BYTES,DOCA_VERBS_MTU_SIZE_1K_BYTES,DOCA_VERBS_MTU_SIZE_2K_BYTES,DOCA_VERBS_MTU_SIZE_4K_BYTES};"
        native += "enum class ErrorCode{InvalidUsage};struct Error:std::runtime_error{Error(const char* message,ErrorCode):std::runtime_error(message){}};"
        native += "struct QpExchangeInfo{uint8_t activeMtu;};struct Impl{" + method + "};"
        native += "int main(){Impl impl;for(uint8_t local=0;local<=6;++local)for(uint8_t remote=0;remote<=6;++remote){"
        native += "bool valid=local>=1&&local<=5&&remote>=1&&remote<=5;try{auto mtu=impl.pathMtu(local,{remote});"
        native += 'require(valid&&int(mtu)==std::min(local,remote),"MTU exceeds endpoint");}catch(const Error&){require(!valid,"valid MTU rejected");}}}'
        self.run_native(native)
        self.ordered(
            function(service, "connectQp"),
            "hcas[hcaIndex]",
            "ibv_query_port(",
            "pathMtu(",
            "doca_verbs_qp_attr_set_path_mtu(attr, mtu)",
        )

    def test_generic_channel_peer_and_signals(self):
        header = source("include/mscclpp/port_channel_device.hpp")
        native = HOST_PREAMBLE + r"""
#define MSCCLPP_DEVICE_COMPILE
#define MSCCLPP_INLINE
#define MSCCLPP_HOST_DEVICE_INLINE
#define MSCCLPP_DEVICE_INLINE
#define MSCCLPP_ASSERT_DEVICE(test,message) require(test,message)
using SemaphoreId=uint32_t;using MemoryId=uint32_t;
enum class PortChannelBackend{Proxy,GpuNetIo};
enum{TriggerData=1,TriggerFlag=2,TriggerSync=4};
struct ProxyTrigger{
 uint64_t fst=0,snd=0;struct{uint64_t dstOffset,dstMemoryId,type,semaphoreId;}fields;
 ProxyTrigger()=default;
 ProxyTrigger(int,uint32_t,uint64_t,uint32_t,uint64_t,uint64_t,uint32_t){}
};
struct FifoDeviceHandle{int count=0;uint64_t push(ProxyTrigger){return count++;}};
struct Host2DeviceSemaphoreDeviceHandle{
 uint64_t* inboundToken;uint64_t* expectedInboundToken;
 bool poll(){if(*inboundToken>*expectedInboundToken){++*expectedInboundToken;return true;}return false;}
 void wait(int64_t){require(poll(),"receive signal missing");}
};
namespace detail{void waitFlush(uint64_t*,uint64_t,int64_t){}}
struct GpuNetIoDeviceContext{
 int numPeers=4,puts=0,signals=0,flushes=0;uint64_t payload=99,counter=0;
 int status=0;uint64_t budget=0;
 void put(int peer,uint64_t,uint64_t,uint64_t){require(peer==3,"put peer");++puts;}
 void atomicAdd(int peer,uint64_t offset,int64_t value){
  require(peer==3,"atomic peer");
  if(offset==64){counter+=value;++signals;}
  else require(offset==128&&value==7,"atomic offset/value");
 }
 void putWithSignal(int peer,uint64_t dst,uint64_t src,uint64_t bytes,uint64_t offset,uint64_t value){
  put(peer,dst,src,bytes);atomicAdd(peer,offset,value);
 }
 void flush(int peer){require(peer==3,"flush peer");++flushes;}
 int tryFlush(int peer,uint64_t spins){budget=spins;flush(peer);return status;}
};
"""
        native += header[header.index("struct BasePortChannelDeviceHandle") : header.rindex("}  // namespace mscclpp")]
        native += r"""
int main(){
 GpuNetIoDeviceContext gin;uint64_t expected=0;
 BasePortChannelDeviceHandle channel(&gin,3,64,&gin.counter,&expected);
 channel.semaphoreId_=12;
 channel.put(0,0,8,0,4);channel.putWithSignal(0,0,8,0,4);
 require(channel.poll()&&!channel.poll(),"poll consumes exactly one signal");
 channel.signal();channel.wait();channel.putWithSignalAndFlush(0,0,8,0,4,100);channel.wait();
 channel.atomicAdd(0,128,7);channel.flush();
 require(gin.payload==99&&gin.puts==3&&gin.signals==3&&gin.flushes==2&&channel.fifo_.count==0,"network routing");
 PortChannelDeviceHandle derived(&gin,3,64,&gin.counter,&expected);
 derived.putWithSignalAndFlush(uint64_t(0),uint64_t(0),uint64_t(4),int64_t(100));derived.wait();
 require(gin.flushes==3,"derived fused flush");
 for(int status:{0,16,-5})for(uint64_t budget:{uint64_t(0),uint64_t(17)}){
  gin.status=status;bool rejected=false;
  try{channel.flush(budget);}catch(const std::runtime_error&){rejected=true;}
  require(rejected==(status!=0)&&gin.budget==budget,"finite flush status/budget");
 }
 gin.budget=123;channel.flush(-1);require(gin.budget==123,"negative flush uses blocking path");
 BasePortChannelDeviceHandle proxy(99,{&gin.counter,&expected},{},nullptr);
 proxy.put(0,0,8,0,4);proxy.signal();proxy.putWithSignalAndFlush(0,0,8,0,4,100);
 require(proxy.fifo_.count==3,"proxy fallback");
}
"""
        self.run_native(native)

    def test_storage_geometry_agreement_before_allocation(self):
        initialize = function(source("src/ext/ep/latency.cc"), "LatencyContext::initialize")
        begin = initialize.index("struct StorageConfig")
        end = initialize.index("EP_HOST_ASSERT(available_);", begin)
        agreement = initialize[begin:end]
        native = HOST_PREAMBLE + '\n#define EP_HOST_ASSERT(test) if(!(test)) throw std::runtime_error("mismatch")\n'
        native += "struct Bootstrap{int mismatch=0;template<class Config>void allGather(Config* values,int){"
        native += "values[1]=values[0];if(mismatch==1)values[1].bytes++;if(mismatch==2)values[1].ipcDomainSize++;if(mismatch==3)values[1].useGpuNetIo^=1;}};"
        native += "struct Communicator{Bootstrap state;Bootstrap* bootstrap(){return &state;}};"
        native += "void check(Communicator* communicator_,bool useGpuNetIo_){int rank_=0,numRanks_=2,numRanksPerIpcDomain_=1;uint64_t symmetricBufferBytes_=4096;"
        native += agreement + "}"
        native += "int main(){for(bool network:{false,true})for(int mismatch=0;mismatch<4;++mismatch){Communicator comm;comm.state.mismatch=mismatch;bool rejected=false;"
        native += 'try{check(&comm,network);}catch(const std::runtime_error&){rejected=true;}require(rejected==(mismatch!=0),"collective storage agreement");}}'
        self.run_native(native)
        self.ordered(initialize, "allGather(storageConfigs.data()", "config.bytes ==", "gpuCallocPhysical(")

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
    LatencyStorageLayout sizing(nullptr, capacity, hidden, ranks, 256, topk, layout, mode, true);
    require(sizing.gpuNetIoCombineLandingBuffer_ == nullptr, "null layout must not publish a landing pointer");
    void* allocation = mmap(nullptr, sizing.totalBytes_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(allocation != MAP_FAILED, "virtual address reservation failed");
    LatencyStorageLayout current(allocation, capacity, hidden, ranks, 256, topk, layout, mode, true);
    LatencyStorageLayout ipc(allocation, capacity, hidden, ranks, 256, topk, layout, mode);
    require(ipc.totalBytes_ < current.totalBytes_ && !ipc.gpuNetIoStagingBuffer_ && !ipc.gpuNetIoCombineLandingBuffer_, "IPC allocated network storage");
    require(ipc.dispatchOutputBuffer_ == current.dispatchOutputBuffer_ && ipc.combineRecvBuffer_ == current.combineRecvBuffer_, "IPC base offsets changed");
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
                DispatchLayout=SimpleNamespace(RANK_MAJOR=1, RANK_MAJOR_TOPK_EXPANDED=3),
                CombineMode=SimpleNamespace(DIRECT_SEND=2),
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
                if name == "atomicAdd":
                    self.assertIn("detail::ginAtomicResult(*this,peer,qpIndex)", code(native))
                else:
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
            "int requestedQps = qpsEnv == nullptr ? nHcas : std::atoi(qpsEnv);",
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


class ResourceLifetimeTests(unittest.TestCase):
    run_native = MultiQpTests.run_native

    def test_inline_size_limit_and_blueflame_source_lifetime(self):
        text = source("src/gpunetio/include/device/doca_gpunetio_dev_verbs_qp.cuh")
        symbol = "doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data"
        position = text.index("void " + symbol)
        definition = text[text.rindex("template <typename T>", 0, position) : text.index("\n}", position) + 2]
        preamble = HOST_PREAMBLE + r"""
#define __device__
#define __forceinline__ inline
struct doca_gpu_dev_verbs_qp{};struct doca_gpu_dev_verbs_wqe{};
struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg{uint32_t byte_count;};
struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg{uint64_t words[2];};
struct doca_gpunetio_ib_mlx5_wqe_raddr_seg{uint64_t words[2];};
constexpr uint32_t DOCA_GPUNETIO_IB_MLX5_INLINE_SEG=1u<<31;
uint32_t doca_gpu_dev_verbs_bswap32(uint32_t value){return __builtin_bswap32(value);}
"""
        for size in (1, 2, 4, 8, 16):
            invocation = (
                f"struct Value{{unsigned char data[{size}];}};int main(){{{symbol}(nullptr,nullptr,Value{{}});}}"
            )
            result = subprocess.run(
                ["g++", "-std=c++17", "-fsyntax-only", "-x", "c++", "-"],
                input=preamble + definition + invocation,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode == 0, size <= 8, result.stderr)
            if size > 8:
                self.assertIn("must not exceed 8 bytes", result.stderr)
        body = function(text, "doca_gpu_dev_verbs_ring_bf")
        offset = -1
        for instruction in (
            "fence.proxy.async.shared::cta",
            "cp.async.bulk.global.shared::cta.bulk_group",
            "cp.async.bulk.commit_group",
            "cp.async.bulk.wait_group.read 0",
        ):
            position = body.index(instruction)
            self.assertGreater(position, offset)
            offset = position
        self.assertIn("__cvta_generic_to_shared", body)

    def test_qp_export_failure_cleanup_and_null_outputs(self):
        text = source("src/gpunetio/src/doca_gpunetio_high_level.cpp")
        actual = text[
            text.index("doca_error_t doca_gpu_verbs_create_qp_hl(") : text.index(
                "doca_error_t doca_gpu_verbs_qp_flat_list_create_hl("
            )
        ]
        native = HOST_PREAMBLE + r"""
#include <cstdio>
template<class... Args>void log_error(int,const char*,Args... arguments){(void)sizeof...(arguments);}
#define DOCA_LOG log_error
constexpr int LOG_ERR=1;
enum doca_error_t{DOCA_SUCCESS,DOCA_ERROR_INVALID_VALUE,DOCA_ERROR_NO_MEMORY,DOCA_ERROR_DRIVER};
constexpr int DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED=9;
constexpr int DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO=0,DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY=1;
constexpr int DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB=2,DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF=3;
struct doca_gpu{bool support_gdrcopy=true;};struct ibv_pd{void* context=nullptr;};
struct doca_gpu_verbs_qp_init_attr_hl{
 doca_gpu* gpu_dev;ibv_pd* ibpd;uint32_t sq_nwqe=8;int send_dbr_mode_ext=0,nic_handler=0,mreg_type=0;bool cq_collapsed=false;
};
struct doca_gpu_verbs_qp_hl{
 doca_gpu* gpu_dev;int send_dbr_mode_ext,nic_handler;
 void* cq_sq_umem_gpu_ptr;void* cq_sq_umem;void* cq_sq_umem_dbr_gpu_ptr;void* cq_sq_umem_dbr;
 void* cq_sq;void* external_uar;void* qp_umem_gpu_ptr;void* qp_umem;void* qp_umem_dbr_gpu_ptr;void* qp_umem_dbr;
 void* qp;void* qp_gverbs;
};
struct doca_gpu_verbs_qp_group_hl{doca_gpu_verbs_qp_hl qp_main,qp_companion;};
int exportCount=0,failExport=1,cleaned=0;
uint32_t doca_internal_utils_next_power_of_two(uint32_t value){return value;}
template<class... Args>doca_error_t create_cq(Args...){return DOCA_SUCCESS;}
template<class... Args>doca_error_t create_uar(Args...){return DOCA_SUCCESS;}
template<class... Args>doca_error_t create_qp(Args...){return DOCA_SUCCESS;}
template<class... Args>doca_error_t doca_gpu_verbs_export_qp(Args...){return ++exportCount==failExport?DOCA_ERROR_DRIVER:DOCA_SUCCESS;}
void doca_gpu_verbs_destroy_qp_hl_internal(doca_gpu_verbs_qp_hl* qp){if(qp->gpu_dev)++cleaned;}
"""
        native += actual + r"""
int main(){
 doca_gpu gpu;ibv_pd pd;doca_gpu_verbs_qp_init_attr_hl attr{&gpu,&pd};
 require(doca_gpu_verbs_create_qp_hl(nullptr,nullptr)==DOCA_ERROR_INVALID_VALUE,"null QP output");
 require(doca_gpu_verbs_create_qp_group_hl(nullptr,nullptr)==DOCA_ERROR_INVALID_VALUE,"null group output");
 doca_gpu_verbs_qp_hl* qp=reinterpret_cast<doca_gpu_verbs_qp_hl*>(1);
 require(doca_gpu_verbs_create_qp_hl(&attr,&qp)==DOCA_ERROR_DRIVER&&!qp&&cleaned==1,"single export cleanup");
 for(int failure:{1,2,3}){
  exportCount=0;cleaned=0;failExport=failure;
  doca_gpu_verbs_qp_group_hl* group=reinterpret_cast<doca_gpu_verbs_qp_group_hl*>(1);
  auto status=doca_gpu_verbs_create_qp_group_hl(&attr,&group);
  if(failure<3)require(status==DOCA_ERROR_DRIVER&&!group&&cleaned==failure,"group export cleanup");
  else{require(status==DOCA_SUCCESS&&group,"group success");doca_gpu_verbs_destroy_qp_group_hl(group);require(cleaned==2,"group success cleanup");}
 }
}
"""
        self.run_native(native)

    def test_internal_uar_type_tracks_fallback(self):
        text = source("src/gpunetio/src/doca_verbs_qp.cpp")
        internal = block(text, r"if\s*\(m_init_attr.external_uar == nullptr\)")
        getter_start = text.index("enum doca_verbs_uar_allocation_type doca_verbs_qp::get_uar_mtype()")
        getter = text[getter_start : text.index("\n}", getter_start) + 2]
        native = HOST_PREAMBLE + r"""
#define DOCA_LOG(...) ((void)0)
enum doca_verbs_uar_allocation_type{DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME,DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE};
constexpr int DOCA_SUCCESS=0,DOCA_ERROR_DRIVER=1,MLX5DV_UAR_ALLOC_TYPE_BF=0,MLX5DV_UAR_ALLOC_TYPE_NC=1;
int failure=0,calls=0;struct Uar{void* reg_addr=nullptr;int page_id=3;}uar;
int doca_verbs_wrapper_mlx5dv_devx_alloc_uar(void*,int type,Uar** output){++calls;if(failure==2||(failure==1&&type==0))return 1;*output=&uar;return 0;}
struct External{doca_verbs_uar_allocation_type get_uar_mtype(){return DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE;}};
struct doca_verbs_qp{
 struct{External* external_uar=nullptr;}m_init_attr;
 doca_verbs_uar_allocation_type m_internal_uar_type=DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME;
 void* m_ibv_ctx=nullptr;Uar* m_uar_obj=nullptr;uint64_t* m_uar_db_reg=nullptr;
 doca_verbs_uar_allocation_type get_uar_mtype()const noexcept;
 void allocate(){uint32_t uar_id=0;
"""
        native += internal + "}};\n" + getter
        native += r"""
int main(){for(failure=0;failure<3;++failure){doca_verbs_qp qp;calls=0;bool rejected=false;
 try{qp.allocate();}catch(int){rejected=true;}
 require(rejected==(failure==2),"UAR allocation failure");
 if(!rejected)require(qp.get_uar_mtype()==failure&&calls==failure+1,"internal UAR allocation type");
 }External external;doca_verbs_qp qp;qp.m_init_attr.external_uar=&external;require(qp.get_uar_mtype()==1,"external UAR type");}
"""
        self.run_native(native)

    def test_unsupported_warp_verbs_fail_compilation(self):
        text = source("src/gpunetio/include/device/doca_gpunetio_dev_verbs_onesided.cuh")
        for name, arguments in (
            ("p", "nullptr,{},1,nullptr"),
            ("put_signal", "nullptr,{},{},0,{},{},1,nullptr"),
            ("signal", "nullptr,{},{},1,nullptr"),
        ):
            symbol = "doca_gpu_dev_verbs_" + name + "_warp"
            position = text.index("void " + symbol)
            definition = text[text.rindex("template <", 0, position) : text.index("\n}", position) + 2]
            preamble = "#include <cstdint>\n#include <cstddef>\n#define __device__\n#define __forceinline__ inline\n"
            preamble += "enum doca_gpu_dev_verbs_resource_sharing_mode{DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU};enum doca_gpu_dev_verbs_nic_handler{DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO};enum doca_gpu_dev_verbs_signal_op{ADD};"
            preamble += "constexpr int DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT=0;struct doca_gpu_dev_verbs_qp{};struct doca_gpu_dev_verbs_addr{};using doca_gpu_dev_verbs_ticket_t=uint64_t;"
            invocation = f"int main(){{{symbol}<{('int' if name == 'p' else 'ADD')}>({arguments});}}"
            compiled = subprocess.run(
                ["g++", "-std=c++17", "-fsyntax-only", "-x", "c++", "-"],
                input=preamble + definition + invocation,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(compiled.returncode, 0)
            self.assertIn("does not support warp scope", compiled.stderr)

    def test_host_allocations_and_failure_cleanup(self):
        text = source("src/gpunetio/src/doca_gpunetio.cpp")
        actual = text[text.index("doca_error_t doca_gpu_mem_alloc(") : text.index("doca_error_t doca_gpu_dmabuf_fd(")]
        native = HOST_PREAMBLE + r"""
#include <unordered_map>
#define DOCA_LOG(...) ((void)0)
#define DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(call) (call)
constexpr int GPU_PAGE_SIZE=65536,cudaSuccess=0,CUDA_SUCCESS=0,CU_POINTER_ATTRIBUTE_SYNC_MEMOPS=0;
constexpr int cudaHostRegisterPortable=1,cudaHostRegisterMapped=2;
using cudaError_t=int;using CUresult=int;using CUdeviceptr=uintptr_t;
enum doca_error_t{DOCA_SUCCESS,DOCA_ERROR_INVALID_VALUE,DOCA_ERROR_DRIVER,DOCA_ERROR_NO_MEMORY};
enum doca_gpu_mem_type{DOCA_GPU_MEM_TYPE_GPU,DOCA_GPU_MEM_TYPE_GPU_CPU,DOCA_GPU_MEM_TYPE_CPU_GPU};
struct doca_gpu_mtable{doca_gpu_mem_type mtype;size_t size,size_orig;uintptr_t base_addr,align_addr_gpu,align_addr_cpu;int gdr_mh;};
struct doca_gpu{bool support_gdrcopy=false;std::unordered_map<uint64_t,doca_gpu_mtable*>*mtable;};
int registered=0,gpuFrees=0,gpuAllocations=0,failure=0;size_t registeredBytes=0;
size_t priv_get_page_size(){return 4096;}
bool priv_is_power_of_two(size_t value){return value&&!(value&(value-1));}
const char* cudaGetErrorString(int){return "injected";}
int cudaMalloc(void**pointer,size_t bytes){*pointer=malloc(bytes);++gpuAllocations;return 0;}
int cudaFree(void*pointer){++gpuFrees;free(pointer);return 0;}
int cudaHostRegister(void*,size_t bytes,int){if(failure==1)return 1;++registered;registeredBytes=bytes;return 0;}
int cudaHostUnregister(void*){--registered;return 0;}
int cudaHostGetDevicePointer(void**device,void*host,int){if(failure==2)return 1;*device=host;return 0;}
int doca_verbs_wrapper_cuPointerSetAttribute(void*,int,uintptr_t){return failure==3;}
int doca_gpu_gdrcopy_create_mapping(void*device,size_t,int*,void**host){*host=device;return failure==4;}
void doca_gpu_gdrcopy_destroy_mapping(int,void*,size_t){}
"""
        native += actual + r"""
int main(){
 std::unordered_map<uint64_t,doca_gpu_mtable*> table;doca_gpu gpu{false,&table};
 for(auto type:{DOCA_GPU_MEM_TYPE_GPU_CPU,DOCA_GPU_MEM_TYPE_CPU_GPU})
 for(size_t alignment:{size_t(1),size_t(4096),size_t(65536)})for(failure=0;failure<3;++failure){
  void* device=nullptr;void* host=nullptr;
  auto result=doca_gpu_mem_alloc(&gpu,1048576,alignment,type,&device,&host);
  if(failure)require(result!=DOCA_SUCCESS&&!device&&!host&&registered==0&&table.empty(),"host failure cleanup");
  else{
   require(result==DOCA_SUCCESS&&device==host&&uintptr_t(host)%alignment==0&&registeredBytes==1048576,"host size/alignment");
   for(size_t offset=0;offset<1048576;++offset)require(static_cast<unsigned char*>(host)[offset]==0,"zero initialization");
   require(table.at(uintptr_t(device))->mtype==DOCA_GPU_MEM_TYPE_CPU_GPU,"effective backing type");
   require(doca_gpu_mem_free(&gpu,device)==DOCA_SUCCESS&&registered==0&&table.empty(),"host free");
  }
 }
 require(gpuFrees==0,"cudaFree received host allocation");gpu.support_gdrcopy=true;
 for(auto type:{DOCA_GPU_MEM_TYPE_GPU,DOCA_GPU_MEM_TYPE_GPU_CPU})for(int injected:{0,3,4}){
  failure=injected;void* device=nullptr;void* host=nullptr;
  auto result=doca_gpu_mem_alloc(&gpu,1024,4096,type,&device,&host);
  if(result==DOCA_SUCCESS)require(doca_gpu_mem_free(&gpu,device)==DOCA_SUCCESS,"GPU free");
  require(gpuAllocations==gpuFrees&&table.empty(),"partial GPU allocation leak");
 }
}
"""
        self.run_native(native)

    def test_atomic_results_never_alias_payload(self):
        text = source(IMPL)
        helper = text[
            text.index("MSCCLPP_DEVICE_INLINE doca_gpu_dev_verbs_addr ginAtomicResult") : text.index(
                "}  // namespace detail"
            )
        ]
        native = HOST_PREAMBLE + r"""
#include <set>
#define MSCCLPP_DEVICE_INLINE inline
#define MSCCLPP_ASSERT_DEVICE(test,message) require(test,message)
using __be32=uint32_t;
struct doca_gpu_dev_verbs_addr{uintptr_t addr;uint32_t key;};
struct GpuNetIoDeviceContext{
 int numPeers=8,numQpsPerPeer=4,numHcas=1;uintptr_t localBase,atomicResultBase;
 const uintptr_t* peerBase;const uint32_t* rkeys;const uint32_t* lkeys;uint32_t lkey;const uint32_t* atomicResultLkeys;
 void* qps=nullptr;
 void putWithSignal(int,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,int);
 void atomicAdd(int,uint64_t,int64_t,int);
};
namespace detail{
int ginHcaIndex(const GpuNetIoDeviceContext& context,int queue){return queue%context.numHcas;}
uint32_t ginHtobe32(uint32_t key){return __builtin_bswap32(key);}
uint32_t ginLocalKey(const GpuNetIoDeviceContext& context,int queue){return context.lkeys[queue%context.numHcas];}
uint32_t ginRemoteKey(const GpuNetIoDeviceContext& context,int peer,int queue){return context.rkeys[(queue%context.numHcas)*8+peer];}
int ginQp(void*,int flat){return flat;}
"""
        native += helper + r"""
}
using doca_gpu_dev_verbs_ticket_t=uint64_t;
constexpr int DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD=0,DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU=0;
uintptr_t scratchBase;uint32_t scratchKeys[4]={101,202,303,404};int hcas;
template<int,int>void doca_gpu_dev_verbs_put_signal(int flat,doca_gpu_dev_verbs_addr,doca_gpu_dev_verbs_addr,size_t,
 doca_gpu_dev_verbs_addr,doca_gpu_dev_verbs_addr result,uint64_t,doca_gpu_dev_verbs_ticket_t*){
 require(result.addr==scratchBase+flat*8&&result.key==__builtin_bswap32(scratchKeys[flat%hcas]),"atomic result MR");
 *reinterpret_cast<uint64_t*>(result.addr)=0xfeed;
}
"""
        native += function(text, "GpuNetIoDeviceContext::putWithSignal")
        native += function(text, "GpuNetIoDeviceContext::atomicAdd")
        native += r"""
int main(){
 uint64_t payload[16];std::fill_n(payload,16,0x1234);uint64_t scratch[8*64]{};
 uint32_t dataKeys[4]={11,22,33,44},remoteKeys[32]{};uintptr_t peers[8]{};
 scratchBase=reinterpret_cast<uintptr_t>(scratch);
 for(int count:{1,2,4})for(int queues:{4,8,64}){
  hcas=count;GpuNetIoDeviceContext context;context.numHcas=hcas;context.numQpsPerPeer=queues;
  context.localBase=reinterpret_cast<uintptr_t>(payload);context.atomicResultBase=scratchBase;
  context.atomicResultLkeys=scratchKeys;context.lkeys=dataKeys;context.rkeys=remoteKeys;context.peerBase=peers;
  std::set<uintptr_t> addresses;
  for(int peer=0;peer<8;++peer)for(int queue=0;queue<queues;++queue){
   require(addresses.insert(detail::ginAtomicResult(context,peer,queue).addr).second,"shared QP scratch");
   context.putWithSignal(peer,0,0,8,64,1,queue);context.atomicAdd(peer,64,1,queue);
  }
 }
 for(auto value:payload)require(value==0x1234,"atomic result corrupted payload");
}
"""
        self.run_native(native)
        service = source(SERVICE)
        setup = function(service, "GpuNetIoService::setup")
        self.assertLess(setup.index("CudaDeviceGuard deviceGuard(s.cudaDeviceId)"), setup.index("cudaMalloc("))
        self.assertIn("registerMr(s.atomicResultsGpu, atomicResultBytes)", setup)
        destructor = block(service, r"~Impl\(\)")
        self.assertLess(destructor.index("CudaDeviceGuard"), destructor.index("cudaFree("))
        self.assertLess(destructor.index("doca_gpu_verbs_destroy_qp_hl"), destructor.index("hcas.clear()"))
        self.assertLess(destructor.index("hcas.clear()"), destructor.index("cudaFree(atomicResultsGpu)"))

    def test_service_shutdown_with_continuous_progress(self):
        text = source("src/gpunetio/src/doca_gpunetio.cpp")
        native = HOST_PREAMBLE + r"""
#include <atomic>
#include <set>
#include <new>
#include <pthread.h>
#include <sched.h>
#define DOCA_LOG(...) ((void)0)
enum doca_error_t{DOCA_SUCCESS,DOCA_ERROR_INVALID_VALUE,DOCA_ERROR_NO_MEMORY,DOCA_ERROR_DRIVER};
struct doca_gpu_verbs_qp{};using doca_gpu_verbs_service_t=void*;
std::atomic<int> progress{0};
void doca_gpu_verbs_cpu_proxy_progress(doca_gpu_verbs_qp*,bool* advanced){++progress;*advanced=true;}
"""
        native += "struct doca_gpu_verbs_service " + block(text, r"struct doca_gpu_verbs_service(?=\s*\{)") + ";"
        native += text[
            text.index("static void *priv_service_mainloop") : text.index(
                "doca_error_t doca_gpu_verbs_query_last_error"
            )
        ]
        native += r"""
int main(){for(int round=0;round<100;++round){
 progress=0;void* handle=nullptr;doca_gpu_verbs_qp qp;
 require(doca_gpu_verbs_create_service(&handle)==DOCA_SUCCESS,"create service");
 require(doca_gpu_verbs_service_monitor_qp(handle,&qp)==DOCA_SUCCESS,"monitor QP");
 while(progress.load()==0)sched_yield();
 require(doca_gpu_verbs_destroy_service(handle)==DOCA_SUCCESS,"stop continuously progressing service");
}}
"""
        self.run_native(native)

    def test_public_c_headers_and_current_python_api(self):
        compiler = shutil.which("gcc")
        if compiler is None:
            self.skipTest("gcc unavailable")
        for header in ("doca_gpunetio.h", "doca_gpunetio_high_level.h", "doca_verbs.h"):
            result = subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-Werror",
                    "-fsyntax-only",
                    "-x",
                    "c",
                    "-I" + str(ROOT / "src/gpunetio/include"),
                    "-",
                ],
                input=f'#include "host/{header}"\nint main(void) {{ bool enabled = false; return enabled; }}',
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        tree = ast.parse(source("test/python/ep/test_low_latency_multirank.py"))
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "MoECommunicator"
        )
        self.assertEqual(next(keyword.value.attr for keyword in call.keywords if keyword.arg == "mode"), "LATENCY")
        self.assertTrue({"num_blocks", "combine_mode"} <= {keyword.arg for keyword in call.keywords})
        obsolete = {"combine_context", "get_expert_output_buffer", "LOW_LATENCY", "HIGH_THROUGHPUT"}
        self.assertFalse({node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} & obsolete)
        for module in ("low_latency.py", "high_throughput.py"):
            self.assertFalse((ROOT / "python/mscclpp/ep" / module).exists())


if __name__ == "__main__":
    unittest.main()
