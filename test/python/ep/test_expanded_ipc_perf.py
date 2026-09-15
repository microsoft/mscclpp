# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU scheduling models/source guards, not CUDA memory-order or performance tests.

Release/acquire transitivity, completed TMA stores and serialized launches are
assumptions of the model. Run the existing expanded GPU regression on both paths
before accepting timing; CPU success cannot establish any GPU/NVLink behavior.
"""

from collections import Counter
from itertools import permutations
from pathlib import Path
import random
import subprocess
import unittest

from test_rank_major_topk_expanded import Sample, _dispatch_reference
from test_topk_expanded_transport_model import body, definitions, locate, statement_end, tokens

ROOT = Path(__file__).resolve().parents[3]
BASE = "44b1d9c70e9588ef69308abd5b4886878b334489"
NATIVE = "src/ext/ep/low_latency/topk_expanded.cu"
IPC = "src/ext/ep/low_latency/topk_expanded_ipc.cuh"


def original(path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE}:{path}"], text=True)


class Retirement:
    """One invocation: workers retire; controller acknowledges; peer ACKs gate exit."""

    def __init__(self, ranks, blocks):
        self.ranks, self.blocks = ranks, blocks
        self.done = [set() for _ in range(ranks)]
        self.control_done = set()
        self.acks = set()
        self.finished = set()

    def retire(self, rank, block):
        if block == 0 or block >= self.blocks or block in self.done[rank]:
            raise ValueError("invalid/duplicate worker retirement")
        self.done[rank].add(block)

    def acknowledge(self, rank):
        if rank in self.acks or rank not in self.control_done or len(self.done[rank]) != self.blocks - 1:
            raise ValueError("ACK before all local work")
        self.acks.add(rank)

    def finish(self, rank):
        if rank not in self.acks or len(self.acks) != self.ranks:
            raise ValueError("return/buffer reuse before all peer ACKs")
        self.finished.add(rank)


class IpcRetirementModelTests(unittest.TestCase):
    def test_randomized_skew_and_dispatch_combine_counter_reuse(self):
        rng = random.Random(42017)
        for ranks in (1, 2, 8, 16, 64):
            for blocks in (ranks + 2, 130):
                counters = [0] * ranks
                epoch = [0] * ranks
                for _ in range(6):  # alternate dispatch/combine, independently of captured host epochs
                    model = Retirement(ranks, blocks)
                    events = [(rank, block) for rank in range(ranks) for block in range(blocks)]
                    rng.shuffle(events)
                    for rank, block in events:
                        if block == 0:
                            model.control_done.add(rank)
                        else:
                            model.retire(rank, block)
                            counters[rank] += 1
                        if rank in model.control_done and counters[rank] == blocks - 1:
                            model.acknowledge(rank)
                            counters[rank] = 0  # every increment arrived before reset
                            epoch[rank] += 1
                        if len(model.acks) != ranks:
                            with self.assertRaises(ValueError):
                                model.finish(rank)
                    for rank in range(ranks):
                        model.finish(rank)
                    self.assertEqual(counters, [0] * ranks)
                    self.assertEqual(len(set(epoch)), 1)
                    self.assertTrue(all(len(done) == blocks - 1 for done in model.done))

    def test_ack_cannot_cover_missing_or_duplicate_block(self):
        for order in permutations((1, 2, 3)):
            model = Retirement(2, 4)
            model.control_done.add(0)
            for block in order[:-1]:
                model.retire(0, block)
                with self.assertRaises(ValueError):
                    model.acknowledge(0)
            with self.assertRaises(ValueError):
                model.retire(0, order[0])
            model.retire(0, order[-1])
            model.acknowledge(0)
            with self.assertRaises(ValueError):
                model.finish(0)  # the other rank still reads this rank's aliased storage

    def test_control_block_is_part_of_lifetime_proof(self):
        model = Retirement(1, 3)
        model.retire(0, 1)
        model.retire(0, 2)
        with self.assertRaises(ValueError):
            model.acknowledge(0)
        model.control_done.add(0)
        model.acknowledge(0)
        model.finish(0)

    def test_device_cache_generation_projection_at_wrap(self):
        # Every source is refreshed every invocation, even if the batch is empty.
        for target in (1, 2, (1 << 32) - 1, 1 << 32, (1 << 32) + 1):
            old = (target - 1) & 0xFFFFFFFF
            new = target & 0xFFFFFFFF
            self.assertNotEqual(old, new)
            for ranks in (1, 8, 16, 64):
                cache = [old] * ranks
                for source in reversed(range(ranks)):
                    self.assertNotEqual(cache[source], new)
                    cache[source] = new
                self.assertEqual(cache, [new] * ranks)

    def test_dense_completion_and_unique_warpgroup_ownership(self):
        for workers in (8, 16, 128):
            for count in (0, 1, 2, 8, 128, 133, 257, 1024):
                per_group = 16 if count <= workers else 8
                groups = 16 // per_group
                assignments = [
                    list(range(b * groups + g, count, workers * groups)) for b in range(workers) for g in range(groups)
                ]
                self.assertEqual(sorted(t for items in assignments for t in items), list(range(count)))
                for k in (1, 8, 9):
                    # All active metadata slots count, including invalid routes/zero weights.
                    self.assertEqual(sum(len(items) * k for items in assignments), count * k)

    def test_mixed_opt_in_or_unmapped_rank_disables_collectively(self):
        # Mirrors the boolean setup gate, not bootstrap communication itself.
        for ranks in (1, 2, 8, 16, 64):
            ready = [True] * ranks
            self.assertTrue(all(ready))
            for rank in range(ranks):
                for requested, mapped, no_gin in ((False, True, True), (True, False, True), (True, True, False)):
                    votes = ready.copy()
                    votes[rank] = requested and mapped and no_gin
                    self.assertFalse(all(votes))

    def test_completed_phase_counters_do_not_carry_into_next_phase(self):
        for ranks in (1, 8, 16, 64):
            counter = [0] * ranks
            for blocks in (130, 129, ranks + 2, ranks + 1, 130, 129):
                model = Retirement(ranks, blocks)
                model.control_done.update(range(ranks))
                for rank in range(ranks):
                    self.assertEqual(counter[rank], 0)
                    for block in range(1, blocks):
                        model.retire(rank, block)
                        counter[rank] += 1
                    self.assertEqual(counter[rank], blocks - 1)
                    model.acknowledge(rank)
                    counter[rank] = 0
                for rank in range(ranks):
                    model.finish(rank)

    def test_ipc_scatter_and_dense_metadata_match_existing_oracle(self):
        rng = random.Random(91723)
        for ranks in (1, 2, 8, 16):
            for capacity in (1, 8, 133, 257):
                for topk in (1, 8, 9):
                    experts = ranks * 4
                    samples = []
                    for rank in range(ranks):
                        n = capacity if rank % 2 == 0 else rng.randrange(capacity + 1)
                        ids = [
                            [rng.choice((-1, experts, 1 << 40, rng.randrange(experts))) for _ in range(topk)]
                            for _ in range(n)
                        ]
                        weights = [[rng.choice((0.0, -0.0, 0.5, -1.0)) for _ in range(topk)] for _ in range(n)]
                        samples.append(
                            Sample(
                                "ipc-model",
                                [[float(rank), float(t)] for t in range(n)],
                                ids,
                                None if rank % 2 else weights,
                            )
                        )
                    for destination in range(ranks):
                        ids, weights, counts, payload = (
                            [experts] * (ranks * capacity * topk),
                            [0.0] * (ranks * capacity * topk),
                            [0] * ranks,
                            {},
                        )
                        completions = Counter()
                        for rank, sample in enumerate(samples):
                            for token, route in enumerate(sample.ids):
                                for slot, expert in enumerate(route):
                                    row = (rank * capacity + token) * topk + slot
                                    completions[rank] += 1  # dense publication also covers invalid selections
                                    if 0 <= expert < experts and expert // 4 == destination:
                                        ids[row] = expert
                                        weights[row] = 1.0 if sample.weights is None else sample.weights[token][slot]
                                        payload[row] = sample.x[token]
                                        counts[rank] += 1
                            self.assertEqual(completions[rank], len(sample.x) * topk)
                        ref = _dispatch_reference(samples, destination, capacity, topk, experts, experts)
                        self.assertEqual(
                            (ids, weights, counts, payload), (ref.ids, ref.weights, ref.counts, ref.payloads)
                        )


class IpcSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ipc = tokens((ROOT / IPC).read_text())
        cls.native = tokens((ROOT / NATIVE).read_text())

    def assert_order(self, code, *snippets):
        start = 0
        for snippet in snippets:
            start = locate(code, snippet, start) + len(tokens(snippet))

    def test_original_gpunetio_kernel_code_unchanged(self):
        # Network launch routing and cached-readiness templates are additive;
        # verify original kernels, public layout and reduction arithmetic exactly.
        from test_expanded_gpunetio_fastpath import FastPathSourceTests

        FastPathSourceTests.test_original_kernels_public_layout_and_math_preserved(self)

    def test_opt_in_collective_full_domain_gate(self):
        runtime = tokens((ROOT / "src/ext/ep/ll_runtime.cc").read_text())
        for snippet in (
            'std::getenv("MSCCLPP_EP_EXPANDED_IPC_FASTPATH")',
            "outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED && ipcDomainSize >= numRanks_",
            "(requested == nullptr || std::atoi(requested) != 0) && allMapped && commContext_.gpuNetIo_ == nullptr",
            "communicator_->bootstrap()->allGather(enabled.data(), sizeof(int))",
            "std::all_of(enabled.begin(), enabled.end(), [](int value) { return value != 0; })",
        ):
            locate(runtime, snippet)
        locate(tokens((ROOT / "src/ext/ep/include/api.cuh").read_text()), "bool expandedIpcFastPath_ = false")

    def test_retirement_storage_is_private_and_unused_by_baseline_expanded(self):
        base = tokens(original(NATIVE))
        self.assertNotIn("dispatchNumRecvTasks_", base)
        config = tokens((ROOT / "src/ext/ep/low_latency/config.cuh").read_text())
        locate(config, "dispatchNumRecvTasks_ = cursor++")
        locate(config, "combineRankReadyEpochs_ = reinterpret_cast<uint32_t*>(cursor)")
        runtime = (ROOT / "src/ext/ep/ll_runtime.cc").read_text()
        self.assertIn("cudaMemset(workspace_, 0, workspaceBytes_)", runtime)
        # Per-dispatch slot reset must not reset retirement or readiness counters.
        self.assertIn("cudaMemsetAsync(workspace_, 0, static_cast<size_t>(numRanks_) * sizeof(int), stream)", runtime)

    def test_full_grid_residency_and_original_launch_geometry_remain(self):
        for name, kernel in (
            ("launchDispatch", "ipc::dispatchKernel<Hidden>"),
            ("launchCombine", "ipc::combineKernel<Hidden, UseTma>"),
        ):
            code = body(self.native, name)
            self.assert_order(
                code,
                "if (comm.expandedIpcFastPath_)",
                "EP_HOST_ASSERT(configureKernel(",
                kernel,
                ">= blocks)",
                "<<<blocks",
                "return;",
            )
        locate(self.native, "const int blocks = numBlocks + 1")

    def test_no_network_operations_or_grid_barriers_in_specialization(self):
        for forbidden in ("gpuNetIo_", "__trap", "combineSyncer_", "flush", "put", "atomicAdd", "finishCollective"):
            self.assertNotIn(forbidden, self.ipc)
        for name in ("dispatchKernel", "combineKernel"):
            code = body(self.ipc, name)
            self.assertEqual(code.count("retireAndAck"), 1)
            self.assert_order(code, "const uint64_t target", "retireAndAck(", "= target;")

    def test_retirement_release_sequence_before_ack(self):
        code = body(self.ipc, "retireAndAck")
        self.assert_order(
            code,
            "__syncthreads()",
            "auto* retired = state.dispatchNumRecvTasks_",
            "if (blockIdx.x != 0)",
            "mscclpp::memoryOrderRelease",
            "return;",
            "mscclpp::memoryOrderAcquire",
            "static_cast<int>(gridDim.x) - 1",
            "*retired = 0",
            "__syncthreads()",
            "release(remote + transport.rank_)",
            "__syncthreads()",
            "wait(flags + peer, target)",
            "__syncthreads()",
        )
        self.assertNotIn("epoch_", self.ipc)  # no captured host generation

    def test_dispatch_payload_and_metadata_publication_retained(self):
        send = body(self.ipc, "send")
        self.assert_order(
            send,
            "barrier->arriveAndExpect(bytes)",
            "mscclpp::bulkLoad(",
            "barrier->wait(phase)",
            "mscclpp::bulkFence()",
            "mscclpp::bulkStore(",
            "mscclpp::bulkStoreCommit()",
            "mscclpp::bulkStoreWait()",
            "atomicAdd_block(completed + peer, 1)",
            "__threadfence_system()",
            "mscclpp::memoryOrderRelease",
        )
        notify = body(self.ipc, "notify")
        self.assert_order(
            notify,
            "__threadfence_system()",
            "__syncthreads()",
            "mscclpp::memoryOrderAcquire",
            "work.numTokens_ * topk",
            "layout.expandedCounts_",
            "__threadfence_system()",
            "release(",
        )

    def test_readiness_system_wait_is_centralized(self):
        publish = body(self.ipc, "publishReady")
        self.assert_order(publish, "release(", "wait(", "state.combineRankReadyEpochs_", "mscclpp::memoryOrderRelease")
        check = body(self.ipc, "ready")
        locate(check, "mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>")
        gather = body(self.ipc, "gather")
        self.assertNotIn("scopeSystem", gather)
        self.assertNotIn("gpuNetIoCombineFlagsBuffer_", gather)
        self.assert_order(
            gather,
            "weight != 0.0f",
            "while (__any_sync(0xffffffff, pending))",
            "pending && ready(state, source, target)",
            "mscclpp::bulkLoad(",
            "if (live) barriers[lane].wait(phase)",
            "__syncthreads()",
            "fmaf(",
        )
        self.assertNotIn("^=", gather)
        locate(gather, "if constexpr (UseTma) __syncthreads()")


if __name__ == "__main__":
    unittest.main(verbosity=2)
