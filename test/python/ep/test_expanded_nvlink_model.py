# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU model/source checks for expanded NVLink optimization, not GPU validation."""

from pathlib import Path
import random
import subprocess
import unittest

from test_rank_major_topk_expanded import Sample, _dispatch_reference, _combine_reference

ROOT = Path(__file__).resolve().parents[3]
KERNEL = "src/ext/ep/low_latency/topk_expanded.cu"
BASE = "6f682e9ccacf5708bdce2b6dd71fa4edc57a47e7"


def function(text, name):
    start = text.index(name + "(")
    begin = text.index("{", start)
    end, depth = begin + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def routing_image_model(samples, capacity, topk, experts, sentinel, rng):
    """Model shuffled token/slot producers, source publication, then receiver metadata.

    This deliberately does not implement any CUDA, TMA, or cross-rank ordering.
    The GPU regression remains mandatory for those properties.
    """
    ranks = len(samples)
    local_experts = experts // ranks
    source_ids = [[sentinel] * (capacity * topk) for _ in samples]
    source_weights = [[0.0] * (capacity * topk) for _ in samples]
    counts = [[0] * ranks for _ in samples]
    payloads = [{} for _ in samples]
    tasks = [(source, token, k) for source in range(ranks) for token in range(capacity) for k in range(topk)]
    rng.shuffle(tasks)
    for source, token, k in tasks:
        sample = samples[source]
        if token >= len(sample.x):
            continue
        expert = sample.ids[token][k]
        if not 0 <= expert < experts:
            continue
        selection = token * topk + k
        source_ids[source][selection] = expert
        source_weights[source][selection] = 1.0 if sample.weights is None else sample.weights[token][k]
        destination = expert // local_experts
        counts[source][destination] += 1
        row = source * capacity * topk + selection
        if row in payloads[destination]:
            raise AssertionError("two producers wrote the same expanded payload row")
        payloads[destination][row] = sample.x[token]
    result = []
    for destination in range(ranks):
        ids, weights = [], []
        for source in range(ranks):
            for row, expert in enumerate(source_ids[source]):
                local = 0 <= expert < experts and expert // local_experts == destination
                ids.append(expert if local else sentinel)
                weights.append(source_weights[source][row] if local else 0.0)
        result.append((ids, weights, [c[destination] for c in counts], payloads[destination]))
    return result


class ExpandedNvlinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / KERNEL).read_text()

    def test_published_routing_matches_fixed_row_oracle(self):
        rng = random.Random(1729)
        for ranks in (1, 2, 8, 16, 64):
            for capacity in (1, 7, 133):
                for topk in (1, 8, 9):
                    experts = ranks * 4
                    samples = []
                    for source in range(ranks):
                        n = capacity if source % 3 == 0 else rng.randrange(capacity + 1)
                        ids = [
                            [rng.choice((-1, experts, 2**40, rng.randrange(experts))) for _ in range(topk)]
                            for _ in range(n)
                        ]
                        weights = [[rng.choice((0.0, -0.0, 0.5, -1.0)) for _ in range(topk)] for _ in range(n)]
                        x = [[float(source), float(token)] for token in range(n)]
                        samples.append(Sample("model", x, ids, None if source % 2 else weights))
                    for sentinel in (-99, experts):
                        result = routing_image_model(samples, capacity, topk, experts, sentinel, rng)
                        for destination, actual in enumerate(result):
                            ref = _dispatch_reference(samples, destination, capacity, topk, experts, sentinel)
                            self.assertEqual(actual, (ref.ids, ref.weights, ref.counts, ref.payloads))

    def test_fallback_and_public_layout_unchanged(self):
        old = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE}:{KERNEL}"], text=True)
        for name in (
            "dispatchTopkExpandedKernel",
            "combineTopkExpandedKernel",
            "postMarkers",
            "drainMarkers",
            "waitSource",
            "finishCollective",
            "validate",
        ):
            self.assertEqual(function(old, name), function(self.source, name), name)
        for name in (
            "src/ext/ep/config.hpp",
            "src/ext/ep/low_latency/config.cuh",
            "python/mscclpp/ep/low_latency.py",
            "python/mscclpp/ep/types.py",
            "test/python/ep/ep_bench_mscclpp.py",
            "src/ext/ep/low_latency/combine.cu",
            "src/ext/ep/low_latency/dispatch.cu",
        ):
            self.assertEqual(
                (ROOT / name).read_bytes(),
                subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE}:{name}"]),
                name,
            )

    def test_dispatch_no_per_peer_payload_loop_or_metadata_race(self):
        send = function(self.source, "dispatchTopkExpandedNvlinkKernel")
        self.assertNotIn("for (int peer", send)
        self.assertNotIn("gpuNetIo_->", send)
        self.assertNotIn("bulkStoreWaitSource", send)
        self.assertIn("mscclpp::bulkStoreWait();", send)
        self.assertIn("atomicAdd(state.dispatchRankPayloadSlots_ + expert / localExperts, 1)", send)
        self.assertIn("outputCount[source] = counts[comm.rank_]", send)
        self.assertNotIn("atomicAdd(outputCount", send)
        self.assertNotIn("mappedBuffer(outputIds", send)
        self.assertNotIn("mappedBuffer(outputWeights", send)
        self.assertEqual(send.count("state.combineSyncer_->sync(gridDim.x)"), 3)
        self.assertLess(send.index("bulkStoreWait()"), send.index("signalLocal("))
        self.assertLess(send.index("outputWeights[destination]"), send.index("finishCollective("))
        runtime = (ROOT / "src/ext/ep/ll_runtime.cc").read_text()
        self.assertIn("cudaMemsetAsync(workspace_, 0, static_cast<size_t>(numRanks_) * sizeof(int), stream)", runtime)

    def test_combine_load_and_wait_hoisted_from_vector_loop(self):
        combine = function(self.source, "combineTopkExpandedNvlinkKernel")
        vector_loop = combine[combine.index("for (int v = threadIdx.x;") :]
        vector_loop = vector_loop[: vector_loop.index("__threadfence_system()")]
        self.assertNotIn("waitFlag", vector_loop)
        self.assertNotIn("bulkLoad", vector_loop)
        self.assertNotIn("mappedBuffer", vector_loop)
        self.assertIn("if (weight != 0.0f)", combine)
        self.assertLess(combine.index("if (weight != 0.0f)"), combine.index("bulkLoad("))
        self.assertIn("for (int slot = 0; slot < topk; ++slot)", combine)
        self.assertIn("fmaf(values.data[0], weight, sum[p].x)", combine)
        self.assertIn("barriers[k].wait(phase)", combine)
        self.assertEqual(combine.count("state.combineSyncer_->sync(gridDim.x)"), 2)
        finish = combine.index("finishCollective(")
        self.assertIn("state.combineSyncer_->sync(gridDim.x)", combine[:finish])
        self.assertIn("state.combineSyncer_->sync(gridDim.x)", combine[finish:])

    def test_conditional_barrier_phases_and_duplicate_row_addresses(self):
        # A skipped lane must not advance its mbarrier parity. Different lanes
        # can therefore have different phases when a block handles many tokens.
        sequences = ((True, False, True, True), (False, True, False, True), (False,) * 4)
        for active in sequences:
            phase, waits = 0, []
            for loaded in active:
                if loaded:
                    waits.append(phase)
                    phase ^= 1
            self.assertEqual(waits, [i % 2 for i in range(sum(active))])
        for hidden in (2048, 4096, 4352, 6656, 7168, 8192, 8704, 9216):
            for topk in (1, 8, 9):
                shared_rows = [k * hidden * 2 for k in range(topk)]
                self.assertEqual(len(set(shared_rows)), topk)
                self.assertTrue(all(offset % 16 == 0 for offset in shared_rows))
                self.assertLess(topk * (hidden * 2 + 8 + 4), 226 * 1024)
        duplicate = Sample("duplicate", [[8.0]], [[0, 0]], [[0.5, -0.25]])
        self.assertEqual(_combine_reference(duplicate, 8), [[0.75]])

    def test_collective_selection_requires_mapped_peers_and_resources(self):
        runtime = (ROOT / "src/ext/ep/ll_runtime.cc").read_text()
        self.assertIn("MSCCLPP_EP_EXPANDED_NVLINK_FASTPATH", runtime)
        self.assertIn("enabled && allMapped && commContext_.gpuNetIo_ == nullptr", runtime)
        self.assertIn("communicator_->bootstrap()->allGather(ready.data(), sizeof(int))", runtime)
        self.assertIn("ipcDomainSize >= numRanks_", runtime)
        self.assertIn("nvlinkFastPathAvailable(hidden_, numTopk_, commContext_)", runtime)
        self.assertIn(">= MaxDispatchBlocks", function(self.source, "nvlinkKernelFits"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
