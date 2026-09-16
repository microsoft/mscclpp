# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Standalone CPU regressions for the native GPUNetIO feature port.

Run directly with Python's standard library; no Torch, MPI, CUDA or EP imports.
Source guards check the actual native function bodies, not comments. Protocol
models exercise counter/slot invariants, not GPU memory ordering. When g++ is
available, compile actual WorkspaceView, layout helpers and rankMajorDispatch
from this checkout with small host-only dependency stubs. C++ goes to compiler
stdin: only the temporary executable is written, never a generated source file.
Neither successful host execution nor source fences prove GPU/NIC correctness.
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
DISPATCH = "src/ext/ep/dispatch/common.cuh"
COMBINE = "src/ext/ep/combine/common.cuh"
WORKSPACE = "src/ext/ep/common/latency.cuh"
CONFIG = "src/ext/ep/include/config.hpp"
HOST_DISPATCH = "src/ext/ep/dispatch/rank_major_dispatch.cu"
HOST_CONTEXT = "src/ext/ep/latency.cc"
VERBS = "include/mscclpp/internal/port_channel_gpunetio_device_impl.hpp"
RANKS = (1, 2, 8, 16, 64)
RAW_STRING = re.compile(r'(?:u8|u|U|L)?R"([^()\s\\]{0,16})\(')


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def mask_cpp(text):
    """Blank comments and quoted/raw literals while preserving source positions.

    Braces inside comments, escaped strings and raw strings must not terminate
    an extracted native body. Reject malformed input rather than testing a
    truncated function. This is a lexical scanner, not a C++ preprocessor.
    """
    masked = list(text)
    i = 0
    while i < len(text):
        start = i
        if text.startswith("//", i):
            end = text.find("\n", i + 2)
            i = len(text) if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise AssertionError("Unterminated C++ comment")
            i = end + 2
        elif raw := RAW_STRING.match(text, i):
            terminator = ")" + raw.group(1) + '"'
            end = text.find(terminator, raw.end())
            if end < 0:
                raise AssertionError("Unterminated C++ raw string")
            i = end + len(terminator)
        elif text[i] in "\"'":
            quote = text[i]
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == quote:
                    i += 1
                    break
                else:
                    i += 1
            else:
                raise AssertionError("Unterminated C++ quoted literal")
        else:
            i += 1
            continue
        for j in range(start, min(i, len(text))):
            if text[j] != "\n":
                masked[j] = " "
    return "".join(masked)


def matching_delimiter(masked, start):
    """Match nested delimiters in already-masked code, never with regex .*."""
    pairs = {"{": "}", "(": ")", "[": "]"}
    if start >= len(masked) or masked[start] not in pairs:
        raise AssertionError("Expected opening delimiter")
    stack = []
    for i in range(start, len(masked)):
        char = masked[i]
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                raise AssertionError("Unbalanced C++ delimiters")
            if not stack:
                return i
    raise AssertionError("Unterminated C++ block")


def unique_match(masked, pattern):
    matches = list(re.finditer(pattern, masked))
    if len(matches) != 1:
        raise AssertionError(f"Expected one match for {pattern!r}, found {len(matches)}")
    return matches[0]


def function(text, name):
    """Extract a complete definition (return type through closing brace).

    Required native functions have a simple return type and no trailing return
    or constructor initializer. Calls do not match the return-type anchor.
    Templates/macros outside the definition are supplied explicitly by the host
    harness; the signature and body themselves are copied unchanged.
    """
    masked = mask_cpp(text)
    match = unique_match(masked, r"\b(?:void|bool|int|size_t)\s+" + re.escape(name) + r"\s*\(")
    params_end = matching_delimiter(masked, match.end() - 1)
    brace = params_end + 1
    while brace < len(masked) and masked[brace].isspace():
        brace += 1
    if brace == len(masked) or masked[brace] != "{":
        raise AssertionError(f"Expected definition of {name}, not declaration")
    end = matching_delimiter(masked, brace)
    return text[match.start() : end + 1]


def block(text, anchor):
    """Extract a braced region after an unambiguous regex anchor."""
    masked = mask_cpp(text)
    match = unique_match(masked, anchor)
    brace = masked.find("{", match.end())
    if brace < 0:
        raise AssertionError(f"Missing block after {anchor}")
    end = matching_delimiter(masked, brace)
    return text[brace : end + 1]


def structure(text, name):
    return f"struct {name} " + block(text, r"\bstruct\s+" + re.escape(name) + r"\b") + ";"


def code(text):
    return re.sub(r"\s+", "", mask_cpp(text))


NESTED_CPP_FIXTURE = r"""
// void wanted() { misleading(); }
void wanted(int n) {
  const char* a = "escaped quote: \" }";
  const char* b = R"tag({ unmatched } } \" /* })tag";
  char c = '}'; /* } */
  if (n) { for (int j = 0; j < n; ++j) { sink(j); } }
  final_call();
}
void other() { wrong(); }
"""


class ExtractionTests(unittest.TestCase):
    def test_nested_braces_comments_and_literals_do_not_truncate(self):
        extracted = function(NESTED_CPP_FIXTURE, "wanted")
        self.assertIn("final_call();", extracted)
        self.assertNotIn("other", extracted)
        self.assertNotIn("misleading", extracted)
        self.assertEqual(code(block(extracted, r"if\s*\(n\)")), "{for(intj=0;j<n;++j){sink(j);}}")

    def test_missing_ambiguous_or_unbalanced_source_fails_closed(self):
        for text in ("void x();", "void x() {", "void x() {} void x() {}", "void y() {}"):
            with self.subTest(text=text), self.assertRaises(AssertionError):
                function(text, "x")
        for text in ("/* }", '"unterminated', 'R"tag(unterminated'):
            with self.subTest(text=text), self.assertRaises(AssertionError):
                mask_cpp(text)


class NativeSourceTests(unittest.TestCase):
    def assert_ordered(self, text, *statements):
        compact = code(text)
        previous = 0
        for statement in statements:
            needle = code(statement)
            position = compact.find(needle, previous)
            self.assertGreaterEqual(position, 0, f"Missing/out-of-order statement: {statement}")
            previous = position + len(needle)

    def test_flush_drains_latest_reserved_ticket_and_get_waits_own_ticket(self):
        native = source(VERBS)
        flush = function(native, "GpuNetIoDeviceContext::flush")
        self.assert_ordered(
            flush,
            "uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, "
            "DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_rsvd_index);",
            "if (ticket == 0) return;",
            "doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);",
            "while (doca_gpu_dev_verbs_poll_one_cq_at<"
            "DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(cq, ticket - 1) == EBUSY)",
        )
        self.assertNotIn("doca_gpu_dev_verbs_wait(", code(flush))
        get = function(native, "GpuNetIoDeviceContext::get")
        self.assert_ordered(
            get,
            "doca_gpu_dev_verbs_ticket_t ticket;",
            "doca_gpu_dev_verbs_get_thread<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>"
            "(qp, raddr, laddr, size, laddr, &ticket);",
            "doca_gpu_dev_verbs_wait(qp, ticket);",
        )
        self.assertNotIn("sq_rsvd_index", code(get))
        self.assertNotIn("flush(", code(get))

    def test_registered_sender_stores_are_system_fenced_before_nic_puts(self):
        send = function(source(DISPATCH), "sendRankMajorGpuNetIo")
        self.assert_ordered(
            send,
            "auto* stagingBase = reinterpret_cast<uint8_t*>(transport.gpuNetIoStagingBuffer_);",
            "for (int i = 0; i < NumVec; ++i) dst[i] = src[i];",
            "slotIds[laneId] = isLocal ? candidateExpert : invalidTokenExpertId;",
            "slotWeights[laneId] = isLocal ? candidateWeight : 0.0f;",
            "__syncwarp();",
            "__threadfence_system();",
            "__syncwarp();",
            "gin->put(destinationRank, transport.symmetricOffset(remoteToken), "
            "transport.symmetricOffset(slot), HiddenBytes);",
            "gin->put(destinationRank, transport.symmetricOffset(remoteIds),",
            "gin->putWithSignal(destinationRank, transport.symmetricOffset(remoteWeights),",
            "transport.symmetricOffset(remoteFlag), /*signalValue=*/1);",
            "gin->flush(destinationRank);",
        )
        initialize = function(source(HOST_CONTEXT), "LatencyContext::initialize")
        self.assertIn("svc->setup(symmetricBuffer_,static_cast<size_t>(symmetricBufferBytes_));", code(initialize))
        self.assertIn("deviceContext_.gpuNetIoStagingBuffer_=layout.gpuNetIoStagingBuffer_;", code(initialize))

    def test_count_scratch_is_registered_disjoint_and_system_fenced(self):
        write = function(source(DISPATCH), "writeRankMajorCounts")
        remote = block(write, r"if\s*\(transport.gpuNetIo_\s*!=\s*nullptr\s*&&\s*!transport.isNvlinkPeer\(dstRank\)\)")
        self.assert_ordered(
            remote,
            "auto* scratch = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + 2 * nRanks + dstRank;",
            "scratch->write(static_cast<uint32_t>(rankTokenCounts[dstRank]), epoch);",
            "__threadfence_system();",
            "auto* remotePacket = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + nRanks + transport.rank_;",
            "gin->put(dstRank, transport.symmetricOffset(remotePacket), "
            "transport.symmetricOffset(scratch), sizeof(mscclpp::LL8Packet));",
            "gin->flush(dstRank);",
            "continue;",
        )
        self.assertNotIn("gpuNetIoStagingBuffer_", code(write))
        self.assertIn("destinationPackets[nRanks+transport.rank_].write(", code(write))
        offset = function(source(CONFIG), "rankMajorTopkIdsOffset")
        self.assert_ordered(
            offset,
            "const size_t metadataPackets = static_cast<size_t>(numRanks) + numExperts;",
            "const size_t countPackets = static_cast<size_t>(3) * numRanks;",
            "const size_t packets = metadataPackets > countPackets ? metadataPackets : countPackets;",
            "return configAlign<size_t>(packets * sizeof(mscclpp::LL8Packet), BufferAlignmentBytes);",
        )

    def test_recv_persists_zero_counts_before_positive_count_gate(self):
        recv = function(source(DISPATCH), "dispatchRecvRankMajor")
        self.assert_ordered(
            recv,
            "auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + nRanks;",
            "outputCount[sourceRank] = nRankTokens;",
            "sharedMem[0] = nRankTokens;",
            "const int nRankTokens = sharedMem[0];",
            "if (threadIdx.x == 0) { workspaceView.dispatchRecvCounts_[sourceRank] = nRankTokens; }",
            "if (threadIdx.x == 0 && nRankTokens > 0)",
        )
        self.assertEqual(code(recv).count("dispatchRecvCounts_[sourceRank]="), 1)

    def test_remote_recv_uses_monotonic_per_source_flags_not_ipc(self):
        recv = function(source(DISPATCH), "dispatchRecvRankMajor")
        remote = block(
            recv, r"if\s*\(transport.gpuNetIo_\s*!=\s*nullptr\s*&&\s*!transport.isNvlinkPeer\(sourceRank\)\)"
        )
        self.assert_ordered(
            remote,
            "auto* flags = reinterpret_cast<volatile uint64_t*>(transport.gpuNetIoFlagsBuffer_);",
            "const uint64_t target = workspaceView.dispatchArrivedBaseline_[sourceRank] + "
            "static_cast<uint64_t>(nRankTokens);",
            "while (flags[sourceRank] < target) { }",
            "workspaceView.dispatchArrivedBaseline_[sourceRank] = target;",
            "return;",
        )
        self.assertNotIn("baseMemoryChannels_", code(remote))
        self.assertNotRegex(code(recv), r"flags\[sourceRank\](?:=(?!=)|\+=|-=|\+\+|--)")
        self.assertNotIn("memset", code(recv))
        self.assert_ordered(recv, remote, "transport.baseMemoryChannels_[sourceRank].wait(-1);")

    def test_expert_scheduler_has_no_network_protocol(self):
        scheduler = code(function(source(DISPATCH), "dispatchRecvScheduler"))
        for forbidden in ("gpuNetIo", "GPUNETIO", "dispatchArrivedBaseline_", "dispatchRecvCounts_", "gin->"):
            self.assertNotIn(forbidden, scheduler)
        self.assertIn("transport.baseMemoryChannels_[sourceRank].wait(-1);", scheduler)

    def test_remote_publish_continues_before_any_concurrent_slot_reset(self):
        publish = function(source(DISPATCH), "publishDispatchPayloads")
        loop = block(publish, r"for\s*\(int dstRank\s*=\s*threadId;")
        remote = block(loop, r"if\s*\(transport.gpuNetIo_\s*!=\s*nullptr\s*&&\s*!transport.isNvlinkPeer\(dstRank\)\)")
        self.assertEqual(code(remote), "{continue;}")
        self.assert_ordered(
            loop,
            remote,
            "if (expectedPayloadCount > 0)",
            "workspaceView.dispatchRankPayloadCompletions_ + dstRank, mscclpp::memoryOrderAcquire) "
            "!= expectedPayloadCount",
            "workspaceView.dispatchRankPayloadSlots_[dstRank] = 0;",
            "workspaceView.dispatchRankPayloadCompletions_[dstRank] = 0;",
        )
        self.assertEqual(code(publish).count("dispatchRankPayloadSlots_[dstRank]=0;"), 1)
        self.assertEqual(code(publish).count("dispatchRankPayloadCompletions_[dstRank]=0;"), 1)

    def test_host_predispatch_reset_uses_workspace_member_not_offset_zero(self):
        host = function(source(HOST_DISPATCH), "rankMajorDispatch")
        self.assert_ordered(
            host,
            "const WorkspaceView workspace(context.workspace_, context.numRanks_, workload.numExperts_);",
            "CUDA_CHECK(cudaMemsetAsync(workspace.dispatchRankPayloadSlots_, 0, "
            "static_cast<size_t>(context.numRanks_) * sizeof(int), stream));",
            "dispatchAlgorithm<DispatchLayout::RANK_MAJOR, RankMajorDispatchKernelSelector>(",
        )
        self.assertEqual(code(host).count("cudaMemsetAsync("), 1)
        self.assertNotIn("cudaMemsetAsync(context.workspace_", code(host))

    def test_combine_network_uses_two_distinct_epoch_phases(self):
        body = function(source(COMBINE), "combineBody")
        remote = block(body, r"if\s*\(transport.gpuNetIo_\s*!=\s*nullptr\)")
        self.assert_ordered(
            remote,
            "const uint32_t epoch = workload.epoch_ * 2;",
            "synchronizeRankMajorCombine(transport, nRanks, epoch, workspaceView);",
            "sendRankMajorCombinePush<Hidden>(",
            "recvRankMajorCombinePush<Hidden>(",
            "synchronizeRankMajorCombine(transport, nRanks, epoch + 1, workspaceView);",
            "return;",
        )
        self.assertEqual(code(remote).count("synchronizeRankMajorCombine("), 2)

    def test_combine_signals_all_then_waits_all_excluding_remote_and_self(self):
        sync = function(source(COMBINE), "synchronizeRankMajorCombine")
        control = block(sync, r"if\s*\(blockIdx.x\s*==\s*0\s*&&\s*threadId\s*==\s*0\)")
        # Exact loop bodies reject the old signal/wait in one peer iteration.
        self.assertEqual(
            code(control),
            code("""{
              for (int peerRank = 0; peerRank < nRanks; ++peerRank) {
                if (transport.isSelf(peerRank) || !transport.isNvlinkPeer(peerRank)) continue;
                transport.baseMemoryChannels_[peerRank].relaxedSignal();
              }
              for (int peerRank = 0; peerRank < nRanks; ++peerRank) {
                if (transport.isSelf(peerRank) || !transport.isNvlinkPeer(peerRank)) continue;
                transport.baseMemoryChannels_[peerRank].relaxedWait(-1);
              }
            }"""),
        )
        self.assert_ordered(sync, control, "__syncthreads();", "workspaceView.combineReadyEpoch_, epoch,")

    def test_combine_push_has_one_final_flush_and_retains_plus_rank_landing(self):
        send = function(source(COMBINE), "sendRankMajorCombinePush")
        rows = block(send, r"for\s*\(int slot\s*=\s*0;")
        self.assertNotIn("flush(", code(rows))
        self.assertEqual(code(send).count("gin->flush("), 1)
        self.assert_ordered(
            send,
            "const int nRowsToOwner = workspaceView.dispatchRecvCounts_[sourceRank];",
            "if (nRowsToOwner <= 0) return;",
            rows,
            "gin->flush(sourceRank);",
        )
        self.assertIn("static_cast<size_t>(nRanks+transport.rank_*maxTokensPerRank+slot)", code(rows))
        last = block(rows, r"if\s*\(slot\s*==\s*nRowsToOwner\s*-\s*1\)")
        self.assertIn("gin->putWithSignal(", code(last))
        self.assertIn("transport.symmetricOffset(remoteFlag),1);", code(last))
        recv = function(source(COMBINE), "recvRankMajorCombinePush")
        self.assertIn("static_cast<size_t>(nRanks+destinationRank*maxTokensPerRank+destinationSlot)", code(recv))
        self.assert_ordered(
            recv,
            "if (!sendsToRank) continue;",
            "const uint64_t target = workspaceView.combineArrivedBaseline_[destinationRank] + 1;",
            "while (flags[destinationRank] < target) { }",
            "workspaceView.combineArrivedBaseline_[destinationRank] = target;",
            "workspaceView.combineSyncer_->sync(gridDim.x);",
        )
        self.assertNotRegex(code(recv), r"flags\[destinationRank\](?:=(?!=)|\+=|-=|\+\+|--)")

    def test_host_disallows_expert_major_and_direct_send_over_network(self):
        context = source(HOST_CONTEXT)
        cross_domain = block(context, r"if\s*\(numRanksPerIpcDomain_\s*<\s*numRanks_\)")
        enabled = block(
            cross_domain, r"if\s*\(enableGpuNetIo\s*!=\s*nullptr\s*&&\s*std::atoi\(enableGpuNetIo\)\s*!=\s*0\)"
        )
        self.assertEqual(
            code(enabled),
            code("""{
              available_ = outputLayout_ == DispatchLayout::RANK_MAJOR &&
                           combineMode_ == CombineMode::RANK_LOCAL_REDUCE;
            }"""),
        )
        initialize = function(context, "LatencyContext::initialize")
        self.assert_ordered(initialize, "EP_HOST_ASSERT(available_);", "svc->setup(")


class ArrivalModel:
    """Nonblocking CPU interpretation of a source's native arrival wait.

    Tests supply NIC observations explicitly. A false result stands for a
    blocked poll; only this source's baseline/count may change at the receiver.
    Future counter observations are arithmetic stress, not a claim that future
    payload-buffer reuse is safe without the protocol's other lifetime rules.
    """

    def __init__(self, ranks):
        self.flags = [0] * ranks
        self.baseline = [0] * ranks
        self.counts = [-1] * ranks

    def receive(self, peer, count):
        self.counts[peer] = count
        if count == 0:
            return True
        target = self.baseline[peer] + count
        if self.flags[peer] < target:
            return False
        self.baseline[peer] = target
        return True


class ProtocolModelTests(unittest.TestCase):
    """Deliberately CPU models: source guards above tie these to native policy."""

    def test_cumulative_dispatch_generations_zero_and_future_overshoot(self):
        for ranks in RANKS:
            with self.subTest(ranks=ranks):
                model = ArrivalModel(ranks)
                totals = [0] * ranks
                for generation in range(256):
                    for peer in range(ranks):
                        count = (generation * 7 + peer * 3) % 11
                        target = model.baseline[peer] + count
                        before = model.baseline[:]
                        if count and model.flags[peer] < target:
                            self.assertFalse(model.receive(peer, count))
                            self.assertEqual(model.baseline, before)
                        # Inject only monotonic NIC observations, including
                        # equality and overshoot of this source's target.
                        model.flags[peer] = max(model.flags[peer], target + (generation + peer) % 5)
                        observed = model.flags[:]
                        self.assertTrue(model.receive(peer, count))
                        totals[peer] += count
                        self.assertEqual(model.baseline, totals)
                        self.assertEqual(model.flags, observed)  # no receiver resets NIC state
                        self.assertEqual(model.counts[peer], count)
                self.assertTrue(all(value > 0 for value in totals))

    def test_overshoot_must_not_be_absorbed_or_reset_at_receiver(self):
        # Two sources advance independently; a zero-count generation consumes
        # no arrival credit and overwrites the previous nonzero combine count.
        model = ArrivalModel(2)
        model.baseline[:] = [5, 20]
        model.flags[:] = [12, 21]
        model.counts[:] = [7, 9]
        self.assertTrue(model.receive(0, 3))
        self.assertTrue(model.receive(1, 0))
        self.assertEqual(model.baseline, [8, 20])
        self.assertEqual(model.counts, [3, 0])
        self.assertEqual(model.flags, [12, 21])
        self.assertTrue(model.receive(0, 4))  # consume the four excess credits
        self.assertEqual(model.baseline, [12, 20])
        self.assertFalse(model.receive(0, 1))  # no more source-zero arrivals
        self.assertTrue(model.receive(1, 1))  # source-one credit is independent
        self.assertEqual(model.baseline, [12, 21])
        self.assertEqual(model.flags, [12, 21])
        # Regression witnesses: either absorbing the observation or clearing
        # the NIC flag would block the otherwise-ready next four-token batch.
        self.assertFalse(12 >= 12 + 4)
        self.assertFalse(0 >= 8 + 4)

    def test_unsafe_mid_send_reset_has_duplicate_slot_witness(self):
        for rows in (2, 3, 17, 128):
            for reset_after in range(1, rows):
                with self.subTest(rows=rows, reset_after=reset_after):
                    slots = list(range(reset_after)) + list(range(rows - reset_after))
                    self.assertLess(len(set(slots)), rows)
                    self.assertEqual(slots[0], slots[reset_after])

    def test_predispatch_reset_keeps_many_graph_pairs_dense_without_mid_send_reset(self):
        for ranks in RANKS:
            slots = [999] * ranks
            dispatch_baseline = [0] * ranks
            combine_baseline = [0] * ranks
            for pair in range(200):
                # Stream-ordered reset executes before ANY producers, every pair.
                slots[:] = [0] * ranks
                for peer in range(ranks):
                    rows = (pair * 13 + peer) % 33
                    allocated = []
                    for _ in range(rows):
                        allocated.append(slots[peer])
                        slots[peer] += 1
                        # A remote notifier may run here, but must not reset.
                    self.assertEqual(allocated, list(range(rows)))
                    dispatch_baseline[peer] += rows
                    combine_baseline[peer] += int(rows > 0)
                saved = (dispatch_baseline[:], combine_baseline[:])
                slots[:] = [0] * ranks
                self.assertEqual((dispatch_baseline, combine_baseline), saved)

    def test_combine_generations_zero_rows_signal_once_not_per_row(self):
        for ranks in RANKS:
            flags = [0] * ranks
            baseline = [0] * ranks
            for pair in range(200):
                for peer in range(ranks):
                    rows = (pair + peer * 5) % 19
                    puts = ["put"] * max(0, rows - 1) + (["signal", "flush"] if rows else [])
                    self.assertEqual(puts.count("signal"), int(rows > 0))
                    self.assertEqual(puts.count("flush"), int(rows > 0))
                    if rows:
                        flags[peer] += 1
                        target = baseline[peer] + 1
                        self.assertGreaterEqual(flags[peer], target)
                        baseline[peer] = target
                    self.assertEqual(flags[peer], baseline[peer])

    def test_phase_epochs_do_not_alias_previous_exit_and_next_entry(self):
        previous_exit = 0
        for epoch in range(1, 10001):
            entry, exit_epoch = epoch * 2, epoch * 2 + 1
            self.assertNotEqual(previous_exit, entry)
            ready_cache = entry
            self.assertNotEqual(ready_cache, exit_epoch)  # worker must wait at exit
            previous_exit = exit_epoch
            # Counterexample: reusing the entry target for exit lets a worker
            # pass before the control block actually publishes exit readiness.
            legacy_ready_cache = epoch
            legacy_exit_target = epoch
            self.assertEqual(legacy_ready_cache, legacy_exit_target)

    def test_count_ranges_and_landing_are_disjoint(self):
        for ranks in RANKS:
            received = set(range(ranks, 2 * ranks))
            scratch = set(range(2 * ranks, 3 * ranks))
            self.assertFalse(received & scratch)
            for experts in (ranks, 2 * ranks, 256):
                self.assertLess(max(scratch), max(ranks + experts, 3 * ranks))
            for capacity in (1, 8, 133, 257):
                if ranks + ranks * capacity > 32768:
                    continue
                landing = [ranks + peer * capacity + slot for peer in range(ranks) for slot in range(capacity)]
                self.assertEqual(len(landing), len(set(landing)))
                self.assertGreaterEqual(min(landing), ranks)
                self.assertLess(max(landing), 32768)


HOST_PREAMBLE = r"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define MSCCLPP_HOST_DEVICE_INLINE inline
using std::size_t;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
namespace mscclpp {
// Layout-only substitutes, NOT semaphore/sync or packet implementations.
struct DeviceSemaphore { int value; };
struct DeviceSyncer { unsigned int counters[3]; unsigned int current; };
struct alignas(8) LL8Packet { uint32_t data; uint32_t flag; };
}
constexpr int MaxWorkerBlocks = 128;
constexpr size_t BufferAlignmentBytes = 128;
template <typename T> constexpr T configAlign(T value, T alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}
"""


HOST_DISPATCH_STUBS = r"""
enum class DispatchLayout { RANK_MAJOR };
struct RankMajorDispatchKernelSelector {};
struct Workload { int numExperts_; };
struct DeviceContext { void* workspace_; int numRanks_; };
using cudaStream_t = void*;
void* expected_slots;
size_t expected_bytes;
cudaStream_t expected_stream;
std::vector<std::string> events;
int cudaMemsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) {
  // Check before writing: an offset-zero/oversized regression fails safely.
  require(ptr == expected_slots, "reset did not use actual WorkspaceView slots");
  require(value == 0 && bytes == expected_bytes, "reset must cover exactly R ints");
  require(stream == expected_stream, "reset used a different stream");
  events.push_back("reset");
  std::memset(ptr, value, bytes);
  return 0;
}
#define CUDA_CHECK(call) require((call) == 0, "fake CUDA call failed")
template <DispatchLayout Layout, typename Selector>
void dispatchAlgorithm(void*, void*, int*, int*, float*, int64_t*, int*, const void*,
                       const int64_t*, const float*, const Workload& workload, void*,
                       const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  static_assert(Layout == DispatchLayout::RANK_MAJOR);
  require(events == std::vector<std::string>{"reset"}, "launch preceded/reset was missing");
  require(numBlocks == 7 && stream == expected_stream, "launch parameters changed");
  WorkspaceView view(context.workspace_, context.numRanks_, workload.numExperts_);
  for (int peer = 0; peer < context.numRanks_; ++peer)
    require(view.dispatchRankPayloadSlots_[peer] == 0, "producer saw a stale slot");
  events.push_back("dispatch");
}
"""


HOST_MAIN = r"""
size_t check_workspace(int ranks, int experts, int capacity, int topk) {
  const size_t bytes = WorkspaceView::numBytes(ranks, experts, capacity, topk);
  // uint64_t storage supplies allocation alignment and leading/trailing canaries.
  std::vector<uint64_t> storage((bytes + 7) / 8 + 16);
  auto* storage_bytes = reinterpret_cast<unsigned char*>(storage.data());
  const size_t storage_size = storage.size() * sizeof(uint64_t);
  std::memset(storage_bytes, 0xa5, storage_size);
  auto* base = storage_bytes + 64;
  WorkspaceView view(base, ranks, experts);
  size_t cursor = 0;
  auto field = [&](const void* ptr, size_t count, size_t element_size, size_t alignment) {
    const auto* address = reinterpret_cast<const unsigned char*>(ptr);
    require(address == base + cursor, "field overlap, gap or incorrect field order");
    require(reinterpret_cast<uintptr_t>(ptr) % alignment == 0, "misaligned field");
    require(cursor + count * element_size <= bytes, "field exceeds numBytes");
    cursor += count * element_size;
  };
  field(view.dispatchArrivedBaseline_, ranks, sizeof(uint64_t), alignof(uint64_t));
  field(view.combineArrivedBaseline_, ranks, sizeof(uint64_t), alignof(uint64_t));
  field(view.dispatchRecvCounts_, ranks, sizeof(int), alignof(int));
  field(view.dispatchRankPayloadSlots_, ranks, sizeof(int), alignof(int));
  field(view.dispatchRankPayloadCompletions_, ranks, sizeof(int), alignof(int));
  field(view.dispatchLocalPayloadReady_, 1, sizeof(mscclpp::DeviceSemaphore), alignof(mscclpp::DeviceSemaphore));
  field(view.dispatchExpertCopiedCounts_, experts, sizeof(int), alignof(int));
  field(view.dispatchRankReadyEpochs_, ranks, sizeof(uint32_t), alignof(uint32_t));
  field(view.dispatchRecvTasks_, MaxWorkerBlocks, sizeof(RecvTask), alignof(RecvTask));
  field(view.dispatchTasksReadyEpoch_, 1, sizeof(uint32_t), alignof(uint32_t));
  field(view.dispatchNumRecvTasks_, 1, sizeof(int), alignof(int));
  field(view.combineRankReadyEpochs_, ranks, sizeof(uint32_t), alignof(uint32_t));
  field(view.combineReadyEpoch_, 1, sizeof(uint32_t), alignof(uint32_t));
  field(view.combineSyncer_, 1, sizeof(mscclpp::DeviceSyncer), alignof(mscclpp::DeviceSyncer));
  field(view.rankMajorSendIndices_, static_cast<size_t>(capacity) * topk, sizeof(int), alignof(int));
  require(cursor == bytes, "numBytes disagrees with constructor's final field");
  require(reinterpret_cast<void*>(view.dispatchRankPayloadSlots_) != base, "slots unexpectedly at offset zero");

  Workload workload{experts};
  DeviceContext context{base, ranks};
  int stream_canary = 42;
  expected_stream = &stream_canary;
  expected_slots = view.dispatchRankPayloadSlots_;
  expected_bytes = static_cast<size_t>(ranks) * sizeof(int);
  const size_t reset_begin = reinterpret_cast<unsigned char*>(expected_slots) - storage_bytes;
  for (int generation = 0; generation < 8; ++generation) {
    for (int peer = 0; peer < ranks; ++peer) {
      view.dispatchArrivedBaseline_[peer] = (uint64_t{1} << 40) + generation * 257 + peer;
      view.combineArrivedBaseline_[peer] = (uint64_t{1} << 48) + generation * 17 + peer;
      view.dispatchRecvCounts_[peer] = (generation + peer) % 7;
      view.dispatchRankPayloadSlots_[peer] = 1000 + peer + generation;
      view.dispatchRankPayloadCompletions_[peer] = 2000 + peer;
    }
    for (size_t index = 0; index < static_cast<size_t>(capacity) * topk; ++index)
      view.rankMajorSendIndices_[index] = static_cast<int>(index) + generation;
    const std::vector<unsigned char> before(storage_bytes, storage_bytes + storage_size);
    events.clear();
    // Actual extracted host wrapper, not a hand-written imitation of its reset.
    rankMajorDispatch(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                      nullptr, nullptr, nullptr, workload, nullptr, context, 7, expected_stream);
    require(events == std::vector<std::string>{"reset", "dispatch"}, "wrong reset/launch order");
    for (size_t i = 0; i < storage_size; ++i) {
      const bool reset_byte = i >= reset_begin && i < reset_begin + expected_bytes;
      require(storage_bytes[i] == (reset_byte ? 0 : before[i]),
              "reset corrupted baselines/counts/indices/another field or canary");
    }
  }
  return 8;
}

void check_metadata(int ranks, int experts, int capacity, int topk) {
  const size_t packet_bytes = sizeof(mscclpp::LL8Packet);
  const size_t packets = std::max(static_cast<size_t>(ranks + experts), size_t{3} * ranks);
  const size_t expected = ((packets * packet_bytes + 127) / 128) * 128;
  const size_t ids = rankMajorTopkIdsOffset(ranks, experts);
  const size_t weights = rankMajorTopkWeightsOffset(ranks, experts, capacity, topk);
  const size_t tokens = rankMajorTokenOffset(ranks, experts, capacity, topk);
  const size_t entries = static_cast<size_t>(ranks) * capacity * topk;
  require(ids == expected, "topk ids must allocate aligned max(R+E,3R) packets");
  require(ids % 128 == 0 && weights % 128 == 0 && tokens % 128 == 0, "metadata alignment");
  require(weights >= ids + entries * sizeof(int), "ids/weights overlap");
  require(tokens >= weights + entries * sizeof(float), "weights/tokens overlap");
  std::vector<unsigned char> counts(ids + 128, 0xa5);
  // Exercise every scratch packet; [R,2R) receive packets and ids remain canaries.
  for (int peer = 0; peer < ranks; ++peer) {
    const size_t scratch = static_cast<size_t>(2 * ranks + peer) * packet_bytes;
    require(scratch >= static_cast<size_t>(2 * ranks) * packet_bytes, "scratch overlaps receive counts");
    require(scratch + packet_bytes <= ids, "scratch overlaps ids");
    std::memset(counts.data() + scratch, 0x5a, packet_bytes);
  }
  for (size_t i = 0; i < counts.size(); ++i) {
    const bool scratch = i >= size_t{2} * ranks * packet_bytes && i < size_t{3} * ranks * packet_bytes;
    require(counts[i] == (scratch ? 0x5a : 0xa5), "scratch corrupted receive counts or ids");
  }
}

int main() {
  try {
    size_t cases = 0, resets = 0;
    for (int ranks : {1, 2, 8, 16, 64})
      for (int experts : {ranks, 2 * ranks, 256})
        for (int capacity : {1, 8, 133, 257})
          for (int topk : {1, 8, 32}) {
            resets += check_workspace(ranks, experts, capacity, topk);
            check_metadata(ranks, experts, capacity, topk);
            ++cases;
          }
    std::cout << "layout_cases=" << cases << " host_dispatch_resets=" << resets << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""


def host_translation_unit():
    workspace = source(WORKSPACE)
    config = source(CONFIG)
    return "\n".join(
        [
            HOST_PREAMBLE,
            structure(workspace, "RecvTask"),
            structure(workspace, "WorkspaceView"),
            function(config, "rankMajorTopkIdsOffset"),
            function(config, "rankMajorTopkWeightsOffset"),
            function(config, "rankMajorTokenOffset"),
            HOST_DISPATCH_STUBS,
            function(source(HOST_DISPATCH), "rankMajorDispatch"),
            HOST_MAIN,
        ]
    )


class ExtractedHostCppTests(unittest.TestCase):
    def test_actual_workspace_layout_metadata_and_host_reset(self):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ unavailable; source guards and CPU models still run")
        translation_unit = host_translation_unit()
        with tempfile.TemporaryDirectory(prefix="ep-feature-port-cpu-") as directory:
            executable = str(Path(directory) / "workspace-regression")
            compiled = subprocess.run(
                [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic", "-x", "c++", "-", "-o", executable],
                input=translation_unit,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, "Host C++ compile failed:\n" + compiled.stdout + compiled.stderr)
            executed = subprocess.run([executable], capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(
                executed.returncode, 0, "Host C++ regression failed:\n" + executed.stdout + executed.stderr
            )
            self.assertEqual(executed.stdout.strip(), "layout_cases=180 host_dispatch_resets=1440")


if __name__ == "__main__":
    unittest.main()
