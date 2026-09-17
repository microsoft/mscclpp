# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU validation of the expanded feature/ep port, not GPU/NIC correctness proof."""

from itertools import product
import math
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_gpunetio_benchmark_port as benchmark_tests
from test_gpunetio_benchmark_port import _api_types, _load, _args
from test_gpunetio_feature_port import HOST_PREAMBLE, block, code, function, source, structure
import test_gpunetio_multi_qp as native_tests

KERNEL = "src/ext/ep/topk_expanded.cu"


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


if __name__ == "__main__":
    unittest.main()
