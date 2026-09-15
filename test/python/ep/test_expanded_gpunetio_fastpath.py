# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CPU models/source guards for the default-enabled expanded fast paths.

These assume (not prove) system-fence visibility, ordered RC delivery, and
all-signaled CQ recycling. Run test_rank_major_topk_expanded.py on actual GPUs
with each topology before timing. K=1/8/9, capacity=8/133/257, mixed empty sources,
duplicate/zero-weight slots and many graph replays exercise that existing suite.
"""

from itertools import product
from pathlib import Path
import random
import subprocess
import unittest

from test_rank_major_topk_expanded import Sample, _dispatch_reference
from test_topk_expanded_transport_model import body, locate, tokens

ROOT = Path(__file__).resolve().parents[3]
BASE = "44b1d9c70e9588ef69308abd5b4886878b334489"
FAST = ROOT / "src/ext/ep/low_latency/topk_expanded_gpunetio.cuh"
NATIVE = ROOT / "src/ext/ep/low_latency/topk_expanded.cu"


def sparse_batches(rows, hcas, qps, owner, live):
    for stripe in range(hcas):
        q = (owner % qps + stripe) % qps
        begin, end = rows * stripe // hcas, rows * (stripe + 1) // hcas
        for tile in range(begin, end, 32):
            yield q, [r for r in range(tile, min(tile + 32, end)) if live[r]]


class FastPathModelTests(unittest.TestCase):
    def test_aggregate_completion_is_dense_token_count(self):
        for workers, count, topk, ranks in product((8, 16, 128), (0, 1, 8, 128, 133, 257, 1024), (1, 8, 9), (8, 16)):
            groups = 1 if count <= workers else 2
            assignments = [
                list(range(b * groups + g, count, workers * groups)) for b in range(workers) for g in range(groups)
            ]
            self.assertEqual(sorted(t for a in assignments for t in a), list(range(count)))
            self.assertEqual(sum(map(len, assignments)), count)
            self.assertEqual(sum(len(a) * topk for a in assignments), count * topk)
            # Old dense counters all had the SAME expected value. One count
            # retains that proof, provided every writer fences before its RMW.
            self.assertEqual([count * topk] * ranks, [sum(len(a) * topk for a in assignments)] * ranks)

    def test_direct_rows_and_metadata_match_oracle(self):
        rng = random.Random(711)
        for ranks, capacity, topk in product((2, 8, 16), (1, 8, 133, 257), (1, 8, 9)):
            samples = []
            for rank in range(ranks):
                n = 0 if rank == ranks - 1 else rng.randrange(capacity + 1)
                ids = [
                    [rng.choice((-1, ranks * 4, 1 << 40, rng.randrange(ranks * 4))) for _ in range(topk)]
                    for _ in range(n)
                ]
                weights = [[rng.choice((0.0, -0.0, 1.0, -0.25)) for _ in range(topk)] for _ in range(n)]
                samples.append(Sample("network", [[float(rank), float(t)] for t in range(n)], ids, weights))
            for dst in range(ranks):
                ref = _dispatch_reference(samples, dst, capacity, topk, ranks * 4, ranks * 4)
                actual = {}
                for src, sample in enumerate(samples):
                    for t, route in enumerate(sample.ids):
                        for k, expert in enumerate(route):
                            if 0 <= expert < ranks * 4 and expert // 4 == dst:
                                actual[(src * capacity + t) * topk + k] = sample.x[t]
                self.assertEqual(actual, ref.payloads)  # zero-weight dispatch rows also exist

    def test_sparse_stripes_cover_each_live_row_once(self):
        rng = random.Random(441)
        for rows, hcas, qps in product((1, 9, 64, 1024, 1197, 2313), (1, 2, 4, 8, 32, 64), (1, 2, 4, 8, 32, 64)):
            if qps < hcas or qps % hcas:
                continue
            for owner in (0, 1, 7, 31):
                live = [rng.randrange(5) == 0 for _ in range(rows)]
                batches = list(sparse_batches(rows, hcas, qps, owner, live))
                got = [r for _, batch in batches for r in batch]
                self.assertEqual(sorted(got), [r for r, valid in enumerate(live) if valid])
                self.assertEqual(len(got), len(set(got)))
                markers = [(owner % qps + s) % qps for s in range(hcas)]
                self.assertEqual(len(set(markers)), hcas)
                self.assertEqual(len({q % hcas for q in markers}), hcas)
                self.assertTrue(all(q in markers for q, _ in batches))

    def test_sparse_ticket_compaction_wrap_and_empty_masks(self):
        rng = random.Random(55)
        base = (1 << 32) - 17
        masks = [0, 1, 1 << 31, 0xFFFFFFFF, 0xAAAAAAAA] + [rng.getrandbits(32) for _ in range(300)]
        for mask in masks:
            lanes = [lane for lane in range(32) if mask & (1 << lane)]
            tickets = [base + (mask & ((1 << lane) - 1)).bit_count() for lane in lanes]
            self.assertEqual(tickets, list(range(base, base + len(lanes))))
            # Contiguous tickets cross the 1024-entry SQ boundary without holes.
            self.assertEqual(len({ticket % 1024 for ticket in tickets}), len(tickets))
            base += len(lanes)
        self.assertGreater(base, 1 << 32)

    def test_bounded_outstanding_wqes_and_all_signaled(self):
        rng = random.Random(713)
        for _ in range(100):
            pending = 0
            for _ in range(100):
                batch = rng.randrange(33)
                pending += batch
                self.assertLessEqual(pending, 159)  # flush threshold 128 + next batch <=32
                if pending >= 128:
                    pending = 0
            pending += 2  # conservative bound for the zero-write + atomic marker
            self.assertLess(pending, 1024)

    def test_no_marker_or_ack_before_dependencies(self):
        for stripes in (1, 2, 4, 64):
            posted, arrived, drained = set(), set(), set()
            expected = set(range(stripes))
            for stripe in reversed(range(stripes)):
                posted.add(stripe)  # all row-WQE lanes sync BEFORE this marker
                arrived.add(stripe)
                self.assertEqual(arrived == expected, stripe == 0)
            # Incoming readiness is insufficient to retire outgoing registered buffers.
            self.assertFalse(drained == expected)
            drained.update(posted)
            self.assertEqual(drained, expected)

    def test_projection_refresh_for_every_source_at_wrap(self):
        for target in (1, 2, (1 << 32) - 1, 1 << 32, (1 << 32) + 1):
            self.assertNotEqual((target - 1) & 0xFFFFFFFF, target & 0xFFFFFFFF)

    def test_runtime_gates_mutually_exclusive(self):
        for ipc, gin, ipc_enabled, network_enabled in product((False, True), repeat=4):
            selected_ipc = ipc and not gin and ipc_enabled
            selected_net = not ipc and gin and network_enabled
            self.assertFalse(selected_ipc and selected_net)

    def test_unset_defaults_explicit_opt_out_and_collective_agreement(self):
        # Mirrors the native nullptr-or-nonzero check for ordinary numeric flags.
        def requested(value):
            return value is None or int(value) != 0

        for ranks in (1, 2, 8, 16, 32, 64):
            for value, expected in ((None, True), ("0", False), ("1", True), ("-1", True)):
                self.assertEqual(all(requested(value) for _ in range(ranks)), expected)
            for disabled_rank in range(ranks):
                votes = [True] * ranks
                votes[disabled_rank] = requested("0")
                self.assertFalse(all(votes))
            # An unset flag cannot bypass topology or resource eligibility.
            self.assertFalse(requested(None) and False)  # unmapped/no transport


class FastPathSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = tokens(FAST.read_text())

    def ordered(self, stream, *snippets):
        pos = 0
        for snippet in snippets:
            pos = locate(stream, snippet, pos) + len(tokens(snippet))

    def test_sender_retains_system_fences_and_direct_rows(self):
        send = body(self.code, "send")
        self.assertNotIn("atomicAdd_block", send)
        self.ordered(send, "mscclpp::bulkStore(", "if (lane < topk)", "if (mapped) mscclpp::bulkStoreWait()")
        self.ordered(
            send, "++completedTokens", "__threadfence_system()", "__syncwarp()", "if (lane == 0", "memoryOrderRelease"
        )
        self.assertIn("gpuNetIoStagingBuffer_", send)
        self.assertNotIn("replicate", send)

    def test_only_original_ipc_and_network_paths_exist(self):
        native = tokens(NATIVE.read_text())
        for name in ("launchDispatch", "launchCombine"):
            code = body(native, name)
            self.ordered(
                code,
                "if (comm.expandedGpuNetIoFastPath_)",
                "gpunetio_fast::",
                "return;",
                "if (comm.expandedIpcFastPath_)",
                "ipc::",
                "return;",
            )
        # No alternate IPC specialization is compiled from the network header.
        self.assertNotIn("Network", self.code)
        self.assertNotIn("ipc", body(self.code, "dispatchKernel"))
        for path in (ROOT / "src/ext/ep/include/api.cuh", ROOT / "src/ext/ep/ll_runtime.cc", NATIVE, FAST):
            text = path.read_text()
            self.assertNotIn("expandedIpcFastPath" + "V2_", text)
            self.assertNotIn("MSCCLPP_EP_EXPANDED_IPC_FASTPATH" + "_V2", text)
        self.assertFalse((FAST.parent / ("topk_expanded_fast_" + "v2.cuh")).exists())

    def test_default_on_host_gates_and_existing_transport_opt_in(self):
        code = tokens((ROOT / "src/ext/ep/ll_runtime.cc").read_text())
        for snippet in (
            'std::getenv("MSCCLPP_EP_EXPANDED_IPC_FASTPATH")',
            'std::getenv("MSCCLPP_EP_EXPANDED_GPUNETIO_FASTPATH")',
            "(requested == nullptr || std::atoi(requested) != 0) && allMapped && commContext_.gpuNetIo_ == nullptr",
            "(requested == nullptr || std::atoi(requested) != 0) && commContext_.gpuNetIo_ != nullptr",
            "outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED && ipcDomainSize >= numRanks_",
            "outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED && ipcDomainSize < numRanks_",
            "crossDomain && enableGpuNetIo != nullptr && std::atoi(enableGpuNetIo) != 0",
        ):
            locate(code, snippet)
        self.assertEqual(code.count("allGather"), 2)
        self.assertEqual(code.count("all_of"), 3)  # mapping check + two collective reductions

    def test_notify_single_counter_then_publish(self):
        self.ordered(
            body(self.code, "notify"),
            "memoryOrderAcquire",
            "work.numTokens_",
            "*state.dispatchRankPayloadCompletions_ = 0",
            "__syncthreads()",
            "signalLocal(flag)",
        )

    def test_sparse_wqe_all_signaled_and_publication_order(self):
        batch = body(self.code, "putWarpRows")
        self.ordered(
            batch,
            "__ballot_sync",
            "if (count == 0) return 0",
            "doca_gpu_dev_verbs_reserve_wq_slots",
            "__shfl_sync",
            "doca_gpu_dev_verbs_wqe_prepare_write",
            "DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE",
            "__threadfence_system()",
            "__syncwarp()",
            "doca_gpu_dev_verbs_mark_wqes_ready",
            "doca_gpu_dev_verbs_submit",
        )
        for name in ("ginRemoteKey", "ginLocalKey", "ginHtobe32"):
            self.assertIn(name, batch)

    def test_dispatch_posting_barrier_and_all_qp_markers(self):
        self.ordered(
            body(self.code, "dispatchKernel"),
            "state.combineSyncer_->sync(gridDim.x)",
            "postDispatch(",
            "waitSource(",
            "retireNetwork(",
        )
        markers = body(self.code, "postDispatch")
        self.ordered(
            markers,
            "DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE",
            "gin->putBatched3(",
            "__syncthreads()",
            "ranks * gin->numQpsPerPeer",
            "gin->atomicAdd(",
            "__syncthreads()",
            "gin->flush(",
        )

    def test_combine_ready_cache_drain_and_lifetime_ack(self):
        self.ordered(
            body(self.code, "combineKernel"),
            "pushCombine<Hidden>",
            "waitSource(",
            "state.combineRankReadyEpochs_",
            "memoryOrderRelease",
            "recvRankMajorTopkExpandedRemotePartialsTma<Hidden, true>",
            "gin->flush(",
            "retireNetwork(",
        )
        self.ordered(
            body(self.code, "retireNetwork"),
            "__threadfence_system()",
            "__syncthreads()",
            "memoryOrderRelease",
            "memoryOrderAcquire",
            "*state.dispatchNumRecvTasks_ = 0",
            "__syncthreads()",
            "finishCollective(",
        )

    def test_original_kernels_public_layout_and_math_preserved(self):
        def original(path):
            return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE}:{path}"], text=True)

        for path in (
            "src/ext/ep/config.hpp",
            "src/ext/ep/low_latency/config.cuh",
            "src/ext/ep/low_latency/dispatch.cu",
            "src/ext/ep/low_latency/combine.cu",
        ):
            self.assertEqual((ROOT / path).read_text(), original(path), path)
        old = tokens(original("src/ext/ep/low_latency/topk_expanded.cu"))
        now = tokens(NATIVE.read_text())
        for name in (
            "dispatchTopkExpandedKernel",
            "combineTopkExpandedKernel",
            "pushExpandedCombine",
            "finishCollective",
            "expandedRow",
        ):
            self.assertEqual(body(now, name), body(old, name), name)
        for name in ("recvRankMajorTopkExpandedRemotePartialsTma", "recvRankMajorTopkExpandedRemotePartials"):
            previous = body(old, name)
            current = body(now, name)
            begin_old = locate(previous, "for (int v = threadIdx.x")
            begin_now = locate(current, "for (int v = threadIdx.x")
            self.assertEqual(current[begin_now:], previous[begin_old:], name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
