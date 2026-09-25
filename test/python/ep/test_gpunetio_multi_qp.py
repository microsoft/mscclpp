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
    def test_actual_ep_service_setup_and_lifetime(self):
        adapter = source("src/ext/ep/gpu_net_io.cc")
        native = HOST_PREAMBLE + r"""
    #include <charconv>
    #include <cstdio>
    #include <cstdlib>
    #include <memory>
    #include <utility>
    #include <limits.h>
    struct DIR{};struct dirent{char d_name[32];};
    DIR* opendir(const char*){return nullptr;}int closedir(DIR*){return 0;}dirent* readdir(DIR*){return nullptr;}
    std::string canonicalPath(const std::string& path){return path;}
    int readNumber(const std::string&,int fallback){return fallback;}
    void cudaDeviceGetPCIBusId(char* output,size_t,int){std::strcpy(output,"0000:00:00.0");}
    void cudaDeviceSynchronize(){}
    constexpr int cudaMemcpyHostToDevice=0;
    void cudaMemcpy(void* destination,const void* source,size_t bytes,int){std::memcpy(destination,source,bytes);}
    #define MSCCLPP_CUDATHROW(call) call
    int liveServices=0,liveChannels=0,liveAllocations=0,createdServices=0,createdSemaphores=0,liveRegistrations=0;
    int registrationFailure=-1,registrationCalls=0,semaphoreFailure=-1;
    namespace mscclpp {
    enum class ErrorCode{SystemError,InvalidUsage};
    struct Error:std::runtime_error{Error(const std::string& message,ErrorCode):std::runtime_error(message){}};
    struct CudaDeviceGuard{explicit CudaDeviceGuard(int){}};
    struct Bootstrap {
     int rank=0,ranks=2,mismatch=0,barriers=0;
     int getRank(){return rank;}int getNranks(){return ranks;}
     template<typename Value>void allGather(Value* values,size_t){
      for(int peer=0;peer<ranks;++peer)values[peer]=values[rank];
      if constexpr(!std::is_same_v<Value,int>){
       const int peer=(rank+1)%ranks;
       if(mismatch==1)values[peer].bytes++;
       if(mismatch==2)values[peer].hcas++;
       if(mismatch==3)values[peer].queues++;
       if(mismatch==4)values[peer].valid=0;
      }else if(mismatch==5)values[(rank+1)%ranks]=0;
     }
     void barrier(){++barriers;}
    };
    struct Registration{Registration(){++liveRegistrations;}~Registration(){--liveRegistrations;}};
    struct ServiceState{int hca,queues;ServiceState(int hca,int queues):hca(hca),queues(queues){++liveServices;}~ServiceState(){--liveServices;}};
    struct GpuNetIoMemory{int peer=-1;std::shared_ptr<Registration> owner;};
    struct Connection{std::shared_ptr<ServiceState> service;int peer,queue;};
    struct Semaphore{Connection connection;};
    struct PortChannelDeviceHandle{int hca=-1,peer=-1,queue=-1;};
    struct GpuNetIoService {
     std::shared_ptr<ServiceState> state;
     GpuNetIoService(std::shared_ptr<Bootstrap>,const std::string& name,int,int queues){state=std::make_shared<ServiceState>(std::stoi(name.substr(3)),queues);++createdServices;}
     void setup(){}
     GpuNetIoMemory registerMemory(void*,size_t){if(registrationCalls++==registrationFailure)throw Error("register",ErrorCode::SystemError);return {-1,std::make_shared<Registration>()};}
     Connection connect(int peer,int queue){require(queue>=0&&queue<state->queues,"physical queue");return{state,peer,queue};}
     GpuNetIoMemory exchangeMemory(Connection connection,GpuNetIoMemory memory,int tag){require(tag>=22000&&tag%2==0,"memory tag");memory.peer=connection.peer;return memory;}
     Semaphore buildSemaphore(Connection connection,int tag){require(tag>=22001&&tag%2==1,"semaphore tag");if(createdSemaphores++==semaphoreFailure)throw Error("semaphore",ErrorCode::SystemError);return{connection};}
    };
    struct PortChannel {
     Semaphore semaphore;GpuNetIoMemory remote,local;
     PortChannel(Semaphore semaphore,GpuNetIoMemory remote,GpuNetIoMemory local):semaphore(semaphore),remote(remote),local(local){require(remote.peer==semaphore.connection.peer,"remote peer");++liveChannels;}
     PortChannel(const PortChannel& other):semaphore(other.semaphore),remote(other.remote),local(other.local){++liveChannels;}
     ~PortChannel(){--liveChannels;}
     PortChannelDeviceHandle deviceHandle(){return{semaphore.connection.service->hca,semaphore.connection.peer,semaphore.connection.queue};}
    };
    template<typename Value>struct GpuBuffer {
     std::shared_ptr<Value> owner;
     explicit GpuBuffer(size_t count):owner(new Value[count],[](Value* pointer){delete[] pointer;--liveAllocations;}){++liveAllocations;}
     std::shared_ptr<Value> memory(){return owner;}
    };
    namespace ep {
    struct EpGpuNetIoDeviceContext{PortChannelDeviceHandle* channels;int numPeers,numQpsPerPeer,numHcas;};
    """
        native += "namespace detail::gpunetio {struct HcaTopology{std::string name,path;int numa;};"
        native += "std::vector<std::string> selectClosestHcas(const std::string&,int,const std::vector<HcaTopology>&){return{};}}\n"
        native += function(adapter, "selectDevices")
        native += r"""
    class EpGpuNetIoService {
     public:EpGpuNetIoService(std::shared_ptr<Bootstrap>,const std::string&,int);~EpGpuNetIoService();
     void setup(void*,size_t);EpGpuNetIoDeviceContext* deviceContext()const;
     private:struct Impl;std::unique_ptr<Impl> impl_;
    };
    """
        native += adapter[adapter.index("struct EpGpuNetIoService::Impl") : adapter.rindex("#endif")]
        native += r"""
    }
    int main(){
     using namespace mscclpp;using namespace mscclpp::ep;
     require(selectDevices(" nic0, nic1\t, ,nic2,","gpu",0)==std::vector<std::string>{"nic0","nic1","nic2"},"list trim/order");
     for(const auto& list:{" , \t","nic0,nic0",""}){bool rejected=false;try{selectDevices(list,"gpu",0);}catch(const Error&){rejected=true;}require(rejected,"invalid HCA list");}
     for(int hcas:{1,2,4,8,64})for(int queues:{1,2,4,8,64})for(int rank:{0,1,3}){
      auto bootstrap=std::make_shared<Bootstrap>();bootstrap->rank=rank;bootstrap->ranks=4;
      std::string names;for(int hca=0;hca<hcas;++hca)names+=(hca?",":"")+std::string("nic")+std::to_string(hca);
      const auto requested=std::to_string(queues);setenv("MSCCLPP_EP_GPUNETIO_QPS_PER_PEER",requested.c_str(),1);
      const bool valid=queues>=hcas&&queues%hcas==0;createdServices=createdSemaphores=registrationCalls=0;
      {EpGpuNetIoService service(bootstrap,names,0);bool rejected=false;
       try{service.setup(reinterpret_cast<void*>(0x100000),4096);}catch(const Error&){rejected=true;}
       require(rejected!=valid,"geometry rejection");
       if(valid){auto* context=service.deviceContext();require(context&&context->numHcas==hcas&&context->numQpsPerPeer==queues&&bootstrap->barriers==1,"published context");
        require(liveServices==hcas&&liveChannels==3*queues&&liveRegistrations==hcas,"retained ownership");
        for(int peer=0;peer<4;++peer)for(int queue=0;queue<queues;++queue){const auto handle=context->channels[peer*queues+queue];if(peer==rank){require(handle.peer==-1,"self slot");continue;}require(handle.peer==peer&&handle.hca==queue%hcas&&handle.queue==queue/hcas,"logical mapping");}
        bool repeated=false;try{service.setup(reinterpret_cast<void*>(0x100000),4096);}catch(const Error&){repeated=true;}require(repeated,"repeat setup");
       }else require(createdServices==0,"invalid configuration created service");
      }
      require(liveServices==0&&liveChannels==0&&liveAllocations==0&&liveRegistrations==0,"leaked owner");
     }
     for(const char* value:{"0","-1","65","4junk",""}){setenv("MSCCLPP_EP_GPUNETIO_QPS_PER_PEER",value,1);auto bootstrap=std::make_shared<Bootstrap>();createdServices=0;bool rejected=false;try{EpGpuNetIoService service(bootstrap,"nic0",0);service.setup(reinterpret_cast<void*>(1),4096);}catch(const Error&){rejected=true;}require(rejected&&createdServices==0,"invalid count accepted");}
     unsetenv("MSCCLPP_EP_GPUNETIO_QPS_PER_PEER");
     for(int mismatch=1;mismatch<=5;++mismatch){auto bootstrap=std::make_shared<Bootstrap>();bootstrap->mismatch=mismatch;createdServices=createdSemaphores=0;bool rejected=false;try{EpGpuNetIoService service(bootstrap,"nic0,nic1",0);service.setup(reinterpret_cast<void*>(1),4096);}catch(const Error&){rejected=true;}require(rejected&&createdSemaphores==0,"collective failure not stopped");if(mismatch<5)require(createdServices==0,"geometry failure posted setup");require(liveServices==0&&liveRegistrations==0,"collective failure leak");}
     for(int mode=0;mode<2;++mode){registrationCalls=createdSemaphores=0;registrationFailure=mode==0?1:-1;semaphoreFailure=mode==1?1:-1;bool rejected=false;try{EpGpuNetIoService service(std::make_shared<Bootstrap>(),"nic0,nic1",0);service.setup(reinterpret_cast<void*>(1),4096);}catch(const Error&){rejected=true;}require(rejected&&liveServices==0&&liveChannels==0&&liveAllocations==0&&liveRegistrations==0,"partial setup cleanup");}
    }
    """
        self.run_native(native)

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
            "queryLocalPort(local)",
            "pathMtu(",
            "doca_verbs_qp_attr_set_path_mtu(attr, mtu)",
        )

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

    def test_actual_adapter_batches_and_signal_bindings(self):
        native = "#define TEST_DEVICE_QPS\n#define main sharedRoutingChecks\n"
        implementation = (
            source(IMPL)
            .replace('#include "doca_gpunetio_device.h"', "")
            .replace('#include "../assert_device.hpp"', "#include <mscclpp/assert_device.hpp>")
        )
        header = (
            source(HEADER)
            .replace('#include "device.hpp"', "")
            .replace('#include "internal/port_channel_gpunetio_device_impl.hpp"', implementation)
        )
        native += (
            source("test/unit/gpunetio_channel_test.cc").replace(
                "#include <mscclpp/port_channel_gpunetio_device.hpp>", header
            )
            + "\n#undef main\n"
        )
        native += r"""
    constexpr int DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT=0;
    constexpr int DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD=0;
    constexpr int DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB=0;
    constexpr uint64_t DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE=1<<20;
    constexpr int DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE=8;
    constexpr int DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE=2;
    struct Wqe{uint64_t ticket,dst,src,bytes;uint32_t remote,local;};
    Wqe entries[3];std::vector<int> events;doca_gpu_dev_verbs_qp* expectedQp;
    template<int Mode>uint64_t doca_gpu_dev_verbs_reserve_wq_slots(doca_gpu_dev_verbs_qp* qp,int count,int){require(qp==expectedQp&&count==3,"batch reserve");events.push_back(1);return 1023;}
    Wqe* doca_gpu_dev_verbs_get_wqe_ptr(doca_gpu_dev_verbs_qp*,uint64_t ticket){return &entries[ticket%3];}
    void doca_gpu_dev_verbs_wqe_prepare_write(doca_gpu_dev_verbs_qp* qp,Wqe* entry,uint64_t ticket,int opcode,int flags,int,uint64_t dst,uint32_t remote,uint64_t src,uint32_t local,uint64_t bytes){require(qp==expectedQp&&opcode==8&&flags==2,"batch WQE");*entry={ticket,dst,src,bytes,remote,local};events.push_back(2);}
    template<int Mode>void doca_gpu_dev_verbs_mark_wqes_ready(doca_gpu_dev_verbs_qp* qp,uint64_t first,uint64_t last){require(qp==expectedQp&&first==1023&&last==1025,"batch ready");events.push_back(3);}
    template<int Mode,int Scope,int Handler>void doca_gpu_dev_verbs_submit(doca_gpu_dev_verbs_qp* qp,uint64_t end,int){require(qp==expectedQp&&end==1026,"batch submit");events.push_back(4);}
    namespace mscclpp::ep {
    """
        native += structure(source("src/ext/ep/include/gpu_net_io.hpp"), "EpGpuNetIoDeviceContext")
        native += r"""
    }
    int main(){
     using namespace mscclpp;using namespace mscclpp::ep;
     for(int hcas:{1,2,4,8,64})for(int queues:{1,4,8,64}){
      if(queues<hcas||queues%hcas)continue;
      std::vector<std::vector<doca_gpu_dev_verbs_qp>> qps(hcas,std::vector<doca_gpu_dev_verbs_qp>(2*queues/hcas));
      std::vector<GpuNetIoDeviceContext> contexts(hcas);
      std::vector<PortChannelDeviceHandle> handles(2*queues);
      std::vector<std::array<GpuNetIoMemoryDeviceHandle,2>> memories(hcas);
      std::vector<uint64_t> scratch(2*queues),inbound(queues),expected(queues);
      for(int hca=0;hca<hcas;++hca){contexts[hca].qps=qps[hca].data();contexts[hca].numPeers=2;contexts[hca].numQpsPerPeer=queues/hcas;contexts[hca].atomicResultBase=reinterpret_cast<uintptr_t>(scratch.data()+hca*2*queues/hcas);contexts[hca].atomicResultLkey=100+hca;
       memories[hca]={GpuNetIoMemoryDeviceHandle{0x100000+uint64_t(hca)*0x100000,1<<20,uint32_t(11+hca),1},GpuNetIoMemoryDeviceHandle{0x200000+uint64_t(hca)*0x100000,1<<20,uint32_t(21+hca),0}};
      }
      for(int queue=0;queue<queues;++queue){const int hca=queue%hcas;BasePortChannelDeviceHandle base(&contexts[hca],1,queue/hcas,0,{0x800000,8,31,1},&inbound[queue],&expected[queue],memories[hca].data(),2);handles[queues+queue]=PortChannelDeviceHandle(base,0,1);}
      EpGpuNetIoDeviceContext adapter{handles.data(),2,queues,hcas};
      for(int queue=0;queue<queues;++queue){const int hca=queue%hcas;expectedQp=&qps[hca][queues/hcas+queue/hcas];events.clear();
       adapter.putBatched3(1,queue,100,200,4096,300,400,32,500,600,32);
       require(events==std::vector<int>{1,2,2,2,3,4},"batch publication order");
       for(int index=0;index<3;++index){const auto entry=entries[index];require(entry.ticket==uint64_t(1023+index)&&entry.dst==memories[hca][0].base+100+index*200&&entry.src==memories[hca][1].base+200+index*200,"batch addresses/wrap");require(entry.remote==__builtin_bswap32(memories[hca][0].key)&&entry.local==__builtin_bswap32(memories[hca][1].key),"batch HCA keys");}
       events.clear();bool rejected=false;try{adapter.putBatched3(1,queue,0,0,8,0,0,8,1<<20,0,8);}catch(const std::runtime_error&){rejected=true;}require(rejected&&events.empty(),"invalid final write reserved WQEs");
       adapter.putWithSignal(1,8,16,32,64,1,queue);require(selectedQp==expectedQp&&signalDestination.addr==memories[hca][0].base+64&&signalDestination.key==__builtin_bswap32(memories[hca][0].key),"symmetric flag binding");
      }
     }
    }
    """
        self.run_native("#include <array>\n" + native)

    def run_native(self, native):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ unavailable")
        with tempfile.TemporaryDirectory(prefix="ep-combine-pipeline-") as directory:
            binary = str(Path(directory) / "check")
            compiled = subprocess.run(
                [compiler, "-std=c++20", "-I" + str(ROOT / "include"), "-x", "c++", "-", "-o", binary],
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
using EpGpuNetIoDeviceContext = Gin;
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


if __name__ == "__main__":
    unittest.main()
