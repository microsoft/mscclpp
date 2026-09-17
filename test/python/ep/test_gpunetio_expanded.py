# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU validation of the expanded feature/ep port, not GPU/NIC correctness proof."""

from itertools import product
import math
import os
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_gpunetio_benchmark_port as benchmark_tests
from test_gpunetio_benchmark_port import _api_types, _load, _args
from test_gpunetio_feature_port import HOST_PREAMBLE, block, code, function, source, structure
import test_gpunetio_multi_qp as native_tests

KERNEL = "src/ext/ep/topk_expanded.cu"
IPC_FAST = "src/ext/ep/topk_expanded_ipc.cuh"
NETWORK_FAST = "src/ext/ep/topk_expanded_gpunetio.cuh"


class Tensor:
    def __init__(self, shape, dtype="bf16", device=None, pointer=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or NS(type="cuda", index=0)
        self.pointer = id(self) if pointer is None else pointer

    def data_ptr(self):
        return self.pointer

    def dim(self):
        return len(self.shape)

    def size(self, axis):
        return self.shape[axis]

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 2 if self.dtype == "bf16" else 4

    def is_contiguous(self):
        return True


class ExpandedTests(unittest.TestCase):
    def runtime(self, **overrides):
        api = _api_types()
        device = NS(type="cuda", index=0)
        traces = []
        api.update(
            os=os,
            Context=object,
            Runtime=object,
            requires_initialized=lambda method: method,
            torch=NS(
                cuda=NS(current_device=lambda: 0),
                device=lambda *args: device,
                bfloat16="bf16",
                int32="int32",
                int64="int64",
                float32="fp32",
                float8_e4m3fn="fp8",
                empty=lambda shape, **kwargs: Tensor(shape, **kwargs),
            ),
            resolve_expert_placement=lambda **kwargs: (kwargs["num_experts"] // kwargs["world_size"], 0),
            resolve_dispatch_data_type=lambda quant: api["DispatchDataType"].BF16 if quant is None else quant.format,
            tensor_from_pointer=lambda pointer, shape, dtype, device, owner: (
                owner,
                Tensor(shape, dtype, device, pointer),
            ),
            cuda_stream_ptr=lambda stream: 17,
        )
        _load("python/mscclpp/ep/latency.py", ("LatencyContext", "LatencyRuntime"), api)
        if isinstance(overrides.get("combine_mode"), str):
            overrides["combine_mode"] = getattr(api["CombineMode"], overrides["combine_mode"])
        config = api["MoECommunicatorConfig"](
            **(
                dict(
                    comm=NS(my_rank=0, nranks=2),
                    num_experts=8,
                    hidden_size=4096,
                    topk=8,
                    max_tokens_per_rank=4,
                    output_layout=api["DispatchLayout"].RANK_MAJOR_TOPK_EXPANDED,
                )
                | overrides
            )
        )
        context = api["LatencyContext"](config)
        runtime = object.__new__(api["LatencyRuntime"])
        runtime.context = context
        runtime.cpp_runtime = NS(
            dispatch_output_buffer_ptr=lambda: 0x100000,
            output_topk_ids_buffer_ptr=lambda: 0x200000,
            output_topk_weights_buffer_ptr=lambda: 0x300000,
            combine_input_buffer_ptr=lambda: 0x100000,
            dispatch=lambda *args: traces.append(("dispatch", args)),
            combine=lambda *args: traces.append(("combine", args)),
        )
        runtime._bind_buffers()
        return api, runtime, traces

    def test_python_views_handle_weights_and_native_arguments(self):
        for topk, weighted in product((1, 8, 9), (False, True)):
            api, runtime, trace = self.runtime(topk=topk)
            context = runtime.context
            rows = 2 * 4 * topk
            self.assertEqual(context.dispatch_output_buffer.shape, (rows, 4096))
            self.assertEqual(context._output_topk_ids.shape, (rows,))
            self.assertIs(context.combine_input_buffer, context.dispatch_output_buffer)
            tokens = Tensor((2, 4096), device=context.device)
            ids = Tensor((2, topk), "int64", context.device)
            weights = Tensor((2, topk), "fp32", context.device) if weighted else None
            output, handle = runtime.dispatch(
                tokens,
                ids,
                weights,
                None,
                output_buffer=None,
                stream=None,
                previous_handle=None,
                runtime_max_tokens_per_rank=None,
            )
            self.assertIs(output.tokens, output.combine_input_buffer)
            self.assertIs(handle._context.topk_ids, ids)
            self.assertIs(handle._context.weights, weights)
            self.assertEqual(output.layout.num_tokens_per_rank.shape, (2,))
            result = Tensor((2, 4096), device=context.device, pointer=0x900000)
            self.assertIs(runtime.combine(output.tokens, handle, out=result, stream=None), result)
            self.assertEqual(trace[0][1][2], weights.data_ptr() if weighted else 0)
            self.assertEqual(trace[1][1][2], weights.data_ptr() if weighted else 0)
            self.assertEqual(trace[1][1][1], ids.data_ptr())
            self.assertEqual(trace[1][1][9], 4)
            self.assertEqual(trace[1][1][-2:], (128, 17))
            with self.assertRaises(ValueError):
                runtime._resolve_capacity(3)
            with self.assertRaises(ValueError):
                runtime._validate_dispatch(tokens, ids, weights, None, Tensor((rows, 4096)), 4)
            with self.assertRaises(ValueError):
                runtime.combine(Tensor((rows, 4096)), handle, out=result, stream=None)
            with self.assertRaises(ValueError):
                runtime.combine(output.tokens, handle, out=Tensor((2, 4096), pointer=0x100000), stream=None)

    def test_python_configuration_rejects_unsupported_cases(self):
        for options in ({"topk": 10}, {"enable_overlap": True}, {"max_tokens_per_rank": 1 << 30}):
            with self.assertRaises((ValueError, NotImplementedError)):
                self.runtime(**options)
        with self.assertRaises(ValueError):
            self.runtime(combine_mode="DIRECT_SEND")
        with self.assertRaises(NotImplementedError):
            self.runtime(quant=NS(format="fp8"))

    def test_benchmark_uses_unweighted_aliased_rows(self):
        helper = benchmark_tests.CpuPortTest()
        helper.setUp()
        try:
            ops, moe, namespace = helper.setup_benchmark(_args(ep_layout="rank_major_topk_expanded"))
            output, handle = ops["dispatch"]()
            output.combine_input_buffer = output.tokens
            with patch.dict(os.environ, {}, clear=True):
                ops["combine"]((output, handle))
            self.assertIs(moe.combined[-1][0], output.tokens)
            self.assertNotIn("normal", helper.trace)
            self.assertEqual(moe.config.output_layout.name, "RANK_MAJOR_TOPK_EXPANDED")
            ops["graph"]["dispatch"]()
            ops["graph"]["combine"]()
            self.assertIs(moe.combined[-1][0], moe.tokens)
        finally:
            helper.doCleanups()

    def test_actual_expanded_allocation_aliases_and_bounds(self):
        config = source("src/ext/ep/include/config.hpp")
        native = HOST_PREAMBLE + "\n#include <sys/mman.h>\nusing Bf16=uint16_t;using Fp8E4M3=uint8_t;\n"
        native += "enum class DispatchLayout { EXPERT_MAJOR,RANK_MAJOR,RANK_MAJOR_TOPK_EXPANDED };\n"
        native += "enum class CombineMode { RANK_LOCAL_REDUCE,DIRECT_SEND };\n"
        native += "\n".join(line for line in config.splitlines() if line.startswith("inline constexpr int GpuNetIo"))
        native += "\ntemplate<typename DataType,typename ScaleType=void>\n" + structure(config, "PayloadView")
        for name in ("rankMajorTopkIdsOffset", "rankMajorTopkWeightsOffset", "rankMajorTokenOffset"):
            native += function(config, name)
        native += structure(config, "LatencyStorageLayout")
        native += r"""
int main() {
  for (int ranks : {1,2,8,16,32,64}) for (int capacity : {1,8,133,32769}) for (int topk : {1,8,9}) {
    constexpr int hidden=4096;
    LatencyStorageLayout size(nullptr,capacity,hidden,ranks,ranks,topk,DispatchLayout::RANK_MAJOR_TOPK_EXPANDED,CombineMode::RANK_LOCAL_REDUCE);
    void* allocation=mmap(nullptr,size.totalBytes_,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    require(allocation!=MAP_FAILED,"address reservation");
    LatencyStorageLayout layout(allocation,capacity,hidden,ranks,ranks,topk,DispatchLayout::RANK_MAJOR_TOPK_EXPANDED,CombineMode::RANK_LOCAL_REDUCE);
    auto* base=static_cast<uint8_t*>(allocation);
    const size_t rows=static_cast<size_t>(ranks)*capacity*topk;
    require(layout.dispatchOutputBytes_==rows*hidden*2,"expanded payload size");
    require(layout.combineRecvBuffer_==layout.dispatchOutputBuffer_ && layout.combineRecvBufferBytes_==0,"payload alias");
    size_t end=0;
    auto region=[&](void* pointer,size_t bytes) {
      auto offset=static_cast<uint8_t*>(pointer)-base;
      require(offset%128==0 && static_cast<size_t>(offset)>=end,"misaligned/overlapping regions");
      end=offset+bytes;require(end<=layout.totalBytes_,"region exceeds allocation");
    };
    region(layout.rankMajorTopkIdsBuffer_,rows*sizeof(int));
    region(layout.rankMajorTopkWeightsBuffer_,rows*sizeof(float));
    region(layout.dispatchOutputBuffer_,rows*hidden*2);
    region(layout.gpuNetIoStagingBuffer_,static_cast<size_t>(std::max(capacity,32768))*layout.gpuNetIoSlotStride_);
    region(layout.gpuNetIoFlagsBuffer_,ranks*64*sizeof(uint64_t));
    region(layout.gpuNetIoCombineFlagsBuffer_,ranks*64*sizeof(uint64_t));
    region(layout.gpuNetIoCombineLandingBuffer_,rows*hidden*2);
    region(layout.expandedSendIds_,rows*sizeof(int));region(layout.expandedSendWeights_,rows*sizeof(float));
    region(layout.expandedSyncFlags_,ranks*sizeof(uint64_t));region(layout.expandedSyncEpoch_,sizeof(uint64_t));
    region(layout.expandedCounts_,ranks*sizeof(int));region(layout.expandedCountStaging_,ranks*sizeof(int));
    require(munmap(allocation,size.totalBytes_)==0,"release");
  }
}
"""
        native_tests.MultiQpTests.run_native(self, native)

    def test_generation_markers_and_final_ack_order(self):
        kernel = source(KERNEL)
        for name, baseline in (
            ("dispatchTopkExpandedKernel", "dispatchArrivedBaseline_"),
            ("combineTopkExpandedKernel", "combineArrivedBaseline_"),
        ):
            body = function(kernel, name)
            self.assertNotIn("work.epoch_", body)
            self.assertIn(baseline + "[context->rank_] + 1", body)
            self.assertGreater(body.index("finishCollective"), body.index("state.combineSyncer_->sync(gridDim.x)"))
            self.assertGreater(body.rindex("state.combineSyncer_->sync(gridDim.x)"), body.index("finishCollective"))
        post = function(kernel, "postRemoteDispatchMetadataAndMarkers")
        self.assertLess(post.index("gin->atomicAdd"), post.index("gin->flush"))
        self.assertIn("ranks * gin->numQpsPerPeer", post)
        push = function(kernel, "pushExpandedCombine")
        self.assertGreater(push.index("gin->atomicAdd"), push.index("gin->put"))
        self.assertIn("wgts[row] == 0.0f", push)
        self.assertIn("gin->numHcas", push)
        for name in ("recvRankMajorTopkExpandedRemotePartials", "recvRankMajorTopkExpandedRemotePartialsTma"):
            receive = function(kernel, name)
            self.assertIn("weight != 0.0f", receive)
            self.assertIn("fmaf(", receive)
            self.assertNotIn("isFirstLaneForRank", receive)
        self.assertEqual(function(kernel, "validExpert").count("expert < experts"), 1)


class FastPathTests(unittest.TestCase):
    def test_actual_metadata_batch_and_large_transfer_fallback(self):
        native = HOST_PREAMBLE + r"""
#define MSCCLPP_USE_GPUNETIO
#define __syncthreads() ((void)0)
constexpr size_t DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE=4096;
constexpr int GpuNetIoMaxQpsPerPeer=64;
struct Dim { int x; } threadIdx{0},blockDim{1};
struct Transfer {uint64_t destination,source,bytes;};
struct Gin {
 int numQpsPerPeer=4,batches=0;std::vector<Transfer> transfers;std::vector<std::string> events;
 void put(int peer,uint64_t dst,uint64_t src,uint64_t bytes,int queue) {
  require(peer==1 && queue==0,"metadata queue");transfers.push_back({dst,src,bytes});events.push_back("write");
 }
 void putBatched3(int peer,int queue,uint64_t dst0,uint64_t src0,uint64_t bytes0,uint64_t dst1,uint64_t src1,uint64_t bytes1,uint64_t dst2,uint64_t src2,uint64_t bytes2) {
  ++batches;require(bytes1<=4096 && bytes2<=4096,"oversized batched WQE");
  put(peer,dst0,src0,bytes0,queue);put(peer,dst1,src1,bytes1,queue);put(peer,dst2,src2,bytes2,queue);
 }
 void atomicAdd(int peer,uint64_t flag,int value,int queue) {
  require(peer==1 && queue==static_cast<int>(events.size())-3 && flag==90000+queue*8 && value==1,"ordered marker");events.push_back("marker");
 }
 void flush(int peer,int queue) {
  require(peer==1 && queue==static_cast<int>(events.size())-7,"drain before all markers");events.push_back("drain");
 }
};
struct TransportView { int rank_=0;Gin* gpuNetIo_;uint8_t* base;
 bool isNvlinkPeer(int peer) const {return peer==0;}
 uint64_t symmetricOffset(const void* ptr) const {return static_cast<const uint8_t*>(ptr)-base;}
};
struct LatencyStorageLayout {void* expandedCounts_;void* expandedCountStaging_;void* rankMajorTopkIdsBuffer_;void* expandedSendIds_;void* rankMajorTopkWeightsBuffer_;void* expandedSendWeights_;void* gpuNetIoFlagsBuffer_;};
struct Workload {int maxTokensPerRank_;int numTopk_;};
"""
        native += function(source(NETWORK_FAST), "postDispatch")
        native += r"""
int main() {
 std::vector<uint64_t> storage(16384);auto* base=reinterpret_cast<uint8_t*>(storage.data());
 LatencyStorageLayout layout{base,base+128,base+256,base+20000,base+40000,base+60000,base+90000};
 for(int capacity:{0,1,8,1024,1025}) {
  Gin gin;TransportView transport{0,&gin,base};Workload work{capacity,1};
  postDispatch(transport,layout,work,2);
  require(gin.batches==int(capacity<=1024) && gin.transfers.size()==3 && gin.events.size()==11,"fallback/batch cardinality");
  require(gin.transfers[0].destination==0 && gin.transfers[0].source==132 && gin.transfers[0].bytes==4,"count offset");
  require(gin.transfers[1].destination==256 && gin.transfers[1].source==20000+capacity*4 && gin.transfers[1].bytes==capacity*4,"ID offset/size");
  require(gin.transfers[2].destination==40000 && gin.transfers[2].source==60000+capacity*4 && gin.transfers[2].bytes==capacity*4,"weight offset/size");
 }
}
"""
        native_tests.MultiQpTests.run_native(self, native)

    def test_actual_sparse_wqe_batch_keys_and_wrap(self):
        native = HOST_PREAMBLE + r"""
#include <condition_variable>
#include <mutex>
#include <thread>
#define MSCCLPP_DEVICE_INLINE inline
#define MSCCLPP_DEVICE_COMPILE
class WarpBarrier {
  std::mutex mutex;std::condition_variable changed;int arrived=0,generation=0;
public:
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);const int current=generation;
    if(++arrived==32) {arrived=0;++generation;changed.notify_all();}
    else changed.wait(lock,[&]{return generation!=current;});
  }
} warpBarrier;
thread_local int laneId;
unsigned activeMask;
uint64_t broadcastBase;
int get_lane_id(){return laneId;}
unsigned __ballot_sync(unsigned,bool live) {
  require(live==((activeMask&(1u<<laneId))!=0),"live mask");warpBarrier.wait();return activeMask;
}
int __popc(unsigned mask){return __builtin_popcount(mask);}
uint64_t __shfl_sync(unsigned,uint64_t value,int) {
  if(laneId==0) broadcastBase=value;
  warpBarrier.wait();auto result=broadcastBase;warpBarrier.wait();return result;
}
void __syncwarp(){warpBarrier.wait();}
void __threadfence_system(){}
constexpr int DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU=0;
constexpr int DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT=0;
constexpr int DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD=0;
constexpr int DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO=0;
constexpr int DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE=8;
constexpr int DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE=2;
struct Wqe {uint64_t ticket=0,dst=0,src=0;uint32_t rkey=0,lkey=0;int flags=0;};
struct doca_gpu_dev_verbs_qp {uint64_t next=1023;Wqe entries[1024];};
doca_gpu_dev_verbs_qp* expected;
int reserved=0,marked=0,submitted=0;
template<int Mode> uint64_t doca_gpu_dev_verbs_reserve_wq_slots(doca_gpu_dev_verbs_qp* qp,int count,int) {
  require(qp==expected && laneId==0 && count==__popc(activeMask),"reservation");
  ++reserved;auto base=qp->next;qp->next+=count;return base;
}
Wqe* doca_gpu_dev_verbs_get_wqe_ptr(doca_gpu_dev_verbs_qp* qp,uint64_t ticket) {return &qp->entries[ticket%1024];}
void doca_gpu_dev_verbs_wqe_prepare_write(doca_gpu_dev_verbs_qp*,Wqe* wqe,uint64_t ticket,int opcode,int flags,int,
                                        uint64_t dst,uint32_t rkey,uint64_t src,uint32_t lkey,uint32_t bytes) {
  require(opcode==8 && flags==2 && bytes==8192,"opcode/signaled/bytes");*wqe={ticket,dst,src,rkey,lkey,flags};
}
template<int Mode> void doca_gpu_dev_verbs_mark_wqes_ready(doca_gpu_dev_verbs_qp* qp,uint64_t first,uint64_t last) {
  require(qp==expected && first==1023 && last==qp->next-1,"ready range");
  for(auto ticket=first;ticket<=last;++ticket) require(qp->entries[ticket%1024].ticket==ticket,"published unprepared WQE");
  ++marked;
}
template<int Mode,int Scope,int Handler> void doca_gpu_dev_verbs_submit(doca_gpu_dev_verbs_qp* qp,uint64_t end,int) {
  require(marked==1 && qp==expected && end==qp->next,"submit boundary/order");++submitted;
}
namespace mscclpp {
"""
        native += structure(source("include/mscclpp/port_channel_gpunetio_device.hpp"), "GpuNetIoDeviceContext")
        native += "namespace detail {\n"
        native += "doca_gpu_dev_verbs_qp* ginQp(void* ptr,int index){return static_cast<doca_gpu_dev_verbs_qp*>(ptr)+index;}\n"
        native += "uint32_t ginHtobe32(uint32_t key){return __builtin_bswap32(key);}\n"
        for name in ("ginHcaIndex", "ginRemoteKey", "ginLocalKey"):
            native += function(source("include/mscclpp/internal/port_channel_gpunetio_device_impl.hpp"), name)
        native += "}}\n" + function(source(NETWORK_FAST), "putWarpRows")
        native += r"""
int main() {
  std::vector<doca_gpu_dev_verbs_qp> queues(8);expected=&queues[7];
  uint32_t remote[8]={0,0x1234,0,0x5678,0,0x9abc,0,0xdef0};
  uint32_t local[4]={0x12345678,0x23456789,0x3456789a,0x456789ab};
  uintptr_t bases[2]={0,0x100000};
  mscclpp::GpuNetIoDeviceContext context{queues.data(),remote,bases,local[0],0x200000,2,4};
  for(int hcas:{1,2,4}) for(unsigned mask:{0u,1u,2u,0x80000000u,0xaaaaaaaa,0x80000001u,0xffffffffu}) {
    activeMask=mask;expected->next=1023;reserved=marked=submitted=0;
    context.numHcas=hcas;context.lkeys=hcas==1?nullptr:local;
    std::vector<std::thread> lanes;
    for(int lane=0;lane<32;++lane) lanes.emplace_back([&,lane]{
      laneId=lane;
      require(putWarpRows(&context,1,3,(mask&(1u<<lane))!=0,100+lane*8192,200+lane*8192,8192)==__popc(mask),"returned count");
    });
    for(auto& lane:lanes) lane.join();
    require(reserved==int(mask!=0) && marked==reserved && submitted==reserved,"batch count");
    uint64_t ticket=1023;
    for(int lane=0;lane<32;++lane) if(mask&(1u<<lane)) {
      const auto& entry=expected->entries[ticket%1024];
      require(entry.ticket==ticket++ && entry.dst==0x100000+100+lane*8192 && entry.src==0x200000+200+lane*8192,"sparse addresses/order");
      require(entry.rkey==remote[(3%hcas)*2+1] && entry.lkey==__builtin_bswap32(local[3%hcas]) && entry.flags==2,"HCA keys/signaled");
    }
    require(expected->next==ticket,"ticket end");
  }
}
"""
        native_tests.MultiQpTests.run_native(self, native)

    def test_actual_collective_default_and_opt_out_selection(self):
        host = function(source("src/ext/ep/latency.cc"), "LatencyContext::initialize")
        native = HOST_PREAMBLE + r"""
#include <cstdlib>
enum class DispatchLayout { RANK_MAJOR, RANK_MAJOR_TOPK_EXPANDED };
struct Device { void* gpuNetIo_=nullptr; bool expandedIpcFastPath_=false, expandedGpuNetIoFastPath_=false; };
struct Bootstrap {
  bool peerEnabled; int calls=0, local=-1;
  void allGather(int* values, size_t bytes) {
    require(bytes==sizeof(int), "wrong collective size"); ++calls; local=values[2];
    for(int peer=0;peer<4;++peer) if(peer!=2) values[peer]=peerEnabled;
  }
};
struct Communicator { Bootstrap* group; Bootstrap* bootstrap(){return group;} };
void check(bool expanded,int ipcDomainSize,bool mapped,bool network,const char* request,bool peerEnabled) {
  for (const char* name : {"MSCCLPP_EP_EXPANDED_IPC_FASTPATH","MSCCLPP_EP_EXPANDED_GPUNETIO_FASTPATH"}) {
    if(request) setenv(name,request,1); else unsetenv(name);
  }
  int token=0;
  const int numRanks_=4,rank_=2;
  auto outputLayout_=expanded?DispatchLayout::RANK_MAJOR_TOPK_EXPANDED:DispatchLayout::RANK_MAJOR;
  std::vector<void*> peerMappedBufferBases_(4,&token);
  if(!mapped) peerMappedBufferBases_[1]=nullptr;
  Device deviceContext_; if(network) deviceContext_.gpuNetIo_=&token;
  Bootstrap group{peerEnabled};Communicator communicator{&group};auto* communicator_=&communicator;
"""
        for relation in (">=", "<"):
            condition = (
                "outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED && ipcDomainSize " + relation + " numRanks_"
            )
            native += "if (" + condition + ") " + block(host, r"if\s*\(" + condition + r"\)")
        native += r"""
  const bool requested=request==nullptr || std::atoi(request)!=0;
  const bool ipc=expanded && ipcDomainSize>=numRanks_ && mapped && !network;
  const bool net=expanded && ipcDomainSize<numRanks_ && network;
  require(deviceContext_.expandedIpcFastPath_==(ipc&&requested&&peerEnabled),"IPC selection");
  require(deviceContext_.expandedGpuNetIoFastPath_==(net&&requested&&peerEnabled),"network selection");
  require(group.calls==static_cast<int>(expanded),"collective participation");
}
int main() {
for(bool expanded:{false,true}) for(int domain:{2,4}) for(bool mapped:{false,true})
for(bool network:{false,true}) for(bool peer:{false,true})
for(const char* request:{static_cast<const char*>(nullptr),"0","1","-1","garbage"})
  check(expanded,domain,mapped,network,request,peer);
}
"""
        native_tests.MultiQpTests.run_native(self, native)
        self.assertLess(host.index("svc->setup("), host.index("expandedIpcFastPath_"))
        self.assertLess(host.index("expandedGpuNetIoFastPath_"), host.index("deviceContext_.devicePtr_ ="))

    def test_actual_retirement_ack_and_cached_epoch_wrap(self):
        native = HOST_PREAMBLE + r"""
#define __syncthreads() ((void)0)
#define __threadfence_system() ((void)0)
struct Dim { unsigned x; } blockIdx{0},threadIdx{0},blockDim{1},gridDim{5};
bool retirePoll=false;
int ackCalls=0;
namespace mscclpp {
constexpr int scopeDevice=0,scopeSystem=1,memoryOrderRelease=1,memoryOrderAcquire=2;
template<class Value,int Scope> Value atomicFetchAdd(Value* ptr,Value amount,int order) {
require(order==memoryOrderRelease,"release order");auto previous=*ptr;*ptr+=amount;return previous;
}
template<class Value,int Scope> Value atomicLoad(Value* ptr,int order) {
require(order==memoryOrderAcquire,"acquire order");
if constexpr (Scope==scopeDevice) if(retirePoll && *ptr!=gridDim.x-1) throw std::runtime_error("pending retirement");
return *ptr;
}
}
struct TransportView { int rank_=0; void* mappedBuffer(void* ptr,int) const {return ptr;} };
struct LatencyStorageLayout {void* expandedSyncEpoch_;void* expandedSyncFlags_;};
struct WorkspaceView {int* dispatchNumRecvTasks_;uint32_t* combineRankReadyEpochs_;};
void finishCollective(const TransportView&,const LatencyStorageLayout&,int) {++ackCalls;}
"""
        for name in ("release", "wait", "ready", "retireAndAck"):
            native += function(source(IPC_FAST), name)
        native += function(source(NETWORK_FAST), "retireNetwork")
        native += r"""
int main() {
  int retired=0;uint64_t epoch=0,flag=0;uint32_t cache=0;
  WorkspaceView state{&retired,&cache};LatencyStorageLayout layout{&epoch,&flag};TransportView transport;
  for(bool network:{false,true}) for(int generation=0;generation<200;++generation) {
    auto run=[&](){if(network)retireNetwork(transport,layout,state,1);else retireAndAck(transport,layout,state,1);};
    const auto beforeEpoch=epoch;const int beforeAcks=ackCalls;
    blockIdx.x=0;retirePoll=true;bool blocked=false;
    try {run();} catch(const std::runtime_error&) {blocked=true;}
    require(blocked && epoch==beforeEpoch && ackCalls==beforeAcks,"ACK before all workers retired");
    for(blockIdx.x=gridDim.x-1;blockIdx.x>0;--blockIdx.x) {
      retirePoll=false;run();require(epoch==beforeEpoch && ackCalls==beforeAcks,"worker sent ACK");
    }
    require(retired==static_cast<int>(gridDim.x)-1,"missing/duplicate retirement");
    retirePoll=true;run();retirePoll=false;
    require(retired==0,"retirement reset");
    if(network)require(ackCalls==beforeAcks+1,"network collective ACK missing");
    else require(epoch==beforeEpoch+1 && flag==epoch,"IPC generation ACK missing");
  }
  for(uint64_t target:{uint64_t{1},(uint64_t{1}<<32)-1,uint64_t{1}<<32,(uint64_t{1}<<32)+1}) {
    cache=static_cast<uint32_t>(target-1);require(!ready(state,0,target),"stale cached readiness");
    cache=static_cast<uint32_t>(target);require(ready(state,0,target),"cached wrap projection");
  }
}
"""
        native_tests.MultiQpTests.run_native(self, native)

    def test_fastpath_launches_and_baseline_are_preserved(self):
        current = source(KERNEL)
        old = subprocess.check_output(["git", "-C", str(benchmark_tests.ROOT), "show", "65dc412:" + KERNEL], text=True)
        for name in (
            "dispatchTopkExpandedKernel",
            "combineTopkExpandedKernel",
            "finishCollective",
            "pushExpandedCombine",
            "dispatchSendRankMajorTopkExpandedBf16",
            "postRemoteDispatchMetadataAndMarkers",
        ):
            self.assertEqual(code(function(current, name)), code(function(old, name)))
        for name, kernel in (("launchDispatch", "dispatchKernel"), ("launchCombine", "combineKernel")):
            launch = function(current, name)
            self.assertLess(
                launch.index("context.expandedGpuNetIoFastPath_"), launch.index("context.expandedIpcFastPath_")
            )
            self.assertIn("gpunetio_fast::" + kernel, launch)
            self.assertIn("ipc::" + kernel, launch)
            self.assertEqual(launch.count("configureKernel("), 3)
            self.assertEqual(launch.count("context.devicePtr_"), 3)
        for path, retire in ((IPC_FAST, "retireAndAck"), (NETWORK_FAST, "retireNetwork")):
            text = source(path)
            for name in ("dispatchKernel", "combineKernel"):
                kernel = function(text, name)
                self.assertLess(kernel.index(retire), kernel.rindex("Baseline_"))
                self.assertNotIn("work.epoch_", kernel)
            self.assertNotIn("combineSyncer_", function(text, "combineKernel"))
            self.assertIn("__threadfence_system()", function(text, "send"))
        self.assertNotIn("combineSyncer_", function(source(IPC_FAST), "dispatchKernel"))
        dispatch = function(source(NETWORK_FAST), "dispatchKernel")
        self.assertLess(dispatch.index("combineSyncer_->sync"), dispatch.index("postDispatch("))
        self.assertLess(dispatch.index("postDispatch("), dispatch.index("retireNetwork("))
        self.assertIn("ipc::ready(*readyState", function(current, "recvRankMajorTopkExpandedRemotePartialsTma"))
        self.assertIn("ipc::ready(*readyState", function(current, "recvRankMajorTopkExpandedRemotePartials"))

    def test_sparse_stripes_and_aggregate_publication(self):
        send = function(source(NETWORK_FAST), "send")
        self.assertIn("++completedTokens", send)
        self.assertIn("state.dispatchRankPayloadCompletions_, completedTokens", send)
        self.assertNotIn("dispatchRankPayloadCompletions_ +", send)
        self.assertIn("!= work.numTokens_", function(source(NETWORK_FAST), "notify"))
        for workers, tokens in product((8, 64, 128), (0, 1, 8, 133, 257, 1024)):
            groups = 1 if tokens <= workers else 2
            assigned = [
                token
                for worker in range(workers)
                for group in range(groups)
                for token in range(worker * groups + group, tokens, workers * groups)
            ]
            self.assertEqual(sorted(assigned), list(range(tokens)))
        for rows, hcas, qps, owner in product((0, 1, 8, 133, 257), (1, 2, 4, 64), (1, 4, 64), (0, 7)):
            if hcas > qps or qps % hcas:
                continue
            live = [index % 7 not in (0, 3) for index in range(rows)]
            posted = []
            markers = []
            for warp in range(32):
                for stripe in range(warp, hcas, 32):
                    begin, end = rows * stripe // hcas, rows * (stripe + 1) // hcas
                    for tile in range(begin, end, 32):
                        posted.extend(row for row in range(tile, min(tile + 32, end)) if live[row])
                    markers.append((owner % qps + stripe) % qps)
            self.assertEqual(sorted(posted), [row for row in range(rows) if live[row]])
            self.assertEqual(len(set(markers)), hcas)
        post = function(source(NETWORK_FAST), "postDispatch")
        self.assertLess(post.index("DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE"), post.index("gin->putBatched3"))
        self.assertLess(post.index("gin->putBatched3"), post.index("gin->atomicAdd"))
        self.assertLess(post.index("gin->atomicAdd"), post.index("gin->flush"))
        batch = function(source(NETWORK_FAST), "putWarpRows")
        self.assertIn("DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE", batch)
        self.assertLess(batch.index("__threadfence_system"), batch.index("doca_gpu_dev_verbs_mark_wqes_ready"))
        self.assertIn("ginRemoteKey(*gin, peer, queue)", batch)
        self.assertIn("ginLocalKey(*gin, queue)", batch)


if __name__ == "__main__":
    unittest.main()
