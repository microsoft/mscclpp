# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Standard-library CPU models and lexical guards for the expanded transport.

These tests do not import the extension, allocate CUDA memory, compile CUDA, or
establish NIC/TMA memory ordering. Ordered per-QP delivery and system-scope
publication are model assumptions, not GPU test results. Scalar arithmetic uses
small exactly representable values, not an emulation of BF16/FP32 instructions.
The separate C++ layout test checks the actual Layout and legacy size formulas.
"""

from collections import Counter, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import permutations, product
import math
from pathlib import Path
import random
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
EP = ROOT / "src/ext/ep"
# Immutable pre-expanded port baseline, not HEAD (which changes after a commit).
LEGACY_BASE = "4cc4276d9c5bf31589b180bedca967ffb0968282"
RANKS = (2, 8, 16, 32, 64)
QPS = (1, 2, 4, 8)
TOPKS = (1, 8, 9)
HIDDEN = (2048, 4096, 4352, 6656, 7168, 8192, 8704, 9216)


# Strings must be consumed before looking for comment delimiters inside them.
# This is a token guard, not a C++ parser or a raw-source snapshot comparison.
_LEX = re.compile(
    r"(?P<comment>//[^\n]*|/\*.*?\*/)|(?P<space>\s+)"
    r'|R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"'
    r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r"|[A-Za-z_]\w*|[0-9][A-Za-z0-9_.']*"
    r"|::|->|\+\+|--|&&|\|\||==|!=|<=|>=|\+=|-=|\^=|&=|\|=|<<|>>|[^\s]",
    re.DOTALL,
)


def tokens(text):
    result = []
    for match in _LEX.finditer(text):
        if match.lastgroup in ("comment", "space"):
            continue
        token = match.group()
        # clang-format may split a long printf string into adjacent literals.
        # C++ concatenates them; preserve content while ignoring that formatting.
        if result and token.startswith('"') and result[-1].startswith('"'):
            result[-1] = result[-1][:-1] + token[1:]
        else:
            result.append(token)
    return tuple(result)


def locate(stream, fragment, start=0):
    needle = tokens(fragment) if isinstance(fragment, str) else tuple(fragment)
    for i in range(start, len(stream) - len(needle) + 1):
        if stream[i : i + len(needle)] == needle:
            return i
    raise AssertionError(f"missing code tokens: {' '.join(needle)}")


def closing(stream, start):
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for i in range(start, len(stream)):
        token = stream[i]
        if token in pairs:
            stack.append(pairs[token])
        elif token in pairs.values():
            if not stack or stack.pop() != token:
                raise AssertionError("unbalanced source delimiters")
            if not stack:
                return i
    raise AssertionError("unterminated source scope")


def definitions(stream, name):
    result = []
    for i, token in enumerate(stream[:-1]):
        if token == name and stream[i + 1] == "(":
            end = closing(stream, i + 1) + 1
            if end < len(stream) and stream[end] == "{":
                result.append((end, closing(stream, end)))
    if not result:
        raise AssertionError(f"no definition of {name}")
    return result


def body(stream, name):
    spans = definitions(stream, name)
    if len(spans) != 1:
        raise AssertionError(f"ambiguous definition of {name}")
    begin, end = spans[0]
    return stream[begin + 1 : end]


def statement_end(stream, start):
    if stream[start] == "{":
        return closing(stream, start) + 1
    if stream[start] in ("if", "for", "while", "switch"):
        head = start + 1 + (stream[start + 1] == "constexpr")
        end = statement_end(stream, closing(stream, head) + 1)
        if stream[start] == "if" and end < len(stream) and stream[end] == "else":
            end = statement_end(stream, end + 1)
        return end
    i = start
    while i < len(stream):
        if stream[i] == ";":
            return i + 1
        i = closing(stream, i) + 1 if stream[i] in ("(", "[", "{") else i + 1
    raise AssertionError("unterminated statement")


def enclosing_controls(stream, position):
    result = []
    for i, token in enumerate(stream[:position]):
        if token not in ("if", "for", "while", "switch"):
            continue
        head = i + 1 + (stream[i + 1] == "constexpr")
        if stream[head] != "(":
            continue
        end = closing(stream, head)
        begin = end + 1
        if begin <= position < statement_end(stream, begin):
            result.append((token, stream[head + 1 : end]))
    return result


def assert_unconditional_dispatch_vote(stream):
    vote = locate(stream, "if (__any_sync(0xffffffff, remote))")
    if stream.count("__any_sync") != 1:
        raise AssertionError("dispatch needs exactly one full-warp remote vote")
    controls = enclosing_controls(stream, vote)
    if [kind for kind, _ in controls] != ["for"] or "token" not in controls[0][1]:
        raise AssertionError("remote vote is not unconditional within the token loop")


def strip_exported_route(stream, kind):
    """Remove only the include and the first statement of the exported wrapper."""
    result = list(stream)
    include = tokens('#include "topk_expanded.cuh"')
    at = locate(tuple(result), include)
    del result[at : at + len(include)]
    stream = tuple(result)
    begin, _ = definitions(stream, kind)[-1]
    start = begin + 1
    guard = tokens("if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED)")
    if stream[start : start + len(guard)] != guard:
        raise AssertionError("expanded routing is not the first exported statement")
    end = statement_end(stream, start)
    route = stream[start:end]
    if kind == "dispatch":
        call = """topk_expanded::dispatch(output, outputTopkIdx, outputTopkWeights, outputCount,
                  input, topkIdx, topkWeights, workload, comm, workspace, numBlocks, stream);"""
        extra = ""
    else:
        call = """topk_expanded::combine(output, input, topkIdx, topkWeights, workload,
                  comm, workspace, numBlocks, stream);"""
        extra = "EP_HOST_ASSERT(mode == CombineMode::RANK_LOCAL_REDUCE);"
    if route != guard + tokens("{" + extra + call + "return;}"):
        raise AssertionError("unexpected code in the exported expanded route")
    return stream[:start] + stream[end:]


def code_changes(before, after):
    matcher = SequenceMatcher(None, before, after, autojunk=False)
    return [
        (op, " ".join(before[a:b]), " ".join(after[c:d])) for op, a, b, c, d in matcher.get_opcodes() if op != "equal"
    ]


def integer_constant(stream, name):
    i = locate(stream, (name, "="))
    if stream[i + 3] != ";":
        raise AssertionError(f"{name} is no longer a literal; update the sizing model")
    return int(stream[i + 2], 0)


def align(size, alignment=128):
    return (size + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Topology:
    ranks: int
    domains: int
    qps: int
    hcas: int

    def __post_init__(self):
        if self.ranks not in RANKS or self.domains not in (1, 2):
            raise ValueError("model requires one IPC domain or two equal domains")
        if self.qps not in QPS or not 0 < self.hcas <= self.qps or self.qps % self.hcas:
            raise ValueError("H must divide Q, with 0 < H <= Q")

    def mapped(self, a, b):
        size = self.ranks // self.domains
        return a // size == b // size

    def combine_qps(self, owner, source):
        if self.mapped(owner, source):
            return (0,)
        return tuple((owner % self.qps + stripe) % self.qps for stripe in range(self.hcas))

    def dispatch_qps(self, owner, source):
        return (0,) if self.mapped(owner, source) else tuple(range(self.qps))


def topologies():
    for ranks, domains, qps in product(RANKS, (1, 2), QPS):
        for hcas in range(1, qps + 1):
            if qps % hcas == 0:
                yield Topology(ranks, domains, qps, hcas)


def valid_expert(expert, experts):
    # Compare the original int64 value, NEVER a narrowed int32 value.
    return 0 <= expert < experts


def expanded_row(source, token, slot, capacity, topk):
    return (source * capacity + token) * topk + slot


def dispatch_image(source, ids, weights, ranks, capacity, topk, invalid=-1):
    experts = ranks * 4
    metadata = [[(invalid, 0.0)] * (capacity * topk) for _ in range(ranks)]
    counts = [0] * ranks
    payload = {}
    for token, selections in enumerate(ids):
        for k, expert in enumerate(selections):
            if not valid_expert(expert, experts):
                continue
            peer = expert // 4
            weight = 1.0 if weights is None else weights[token][k]
            metadata[peer][token * topk + k] = (expert, weight)
            counts[peer] += 1  # Selected routes, including duplicates and zero weights.
            payload[peer, expanded_row(source, token, k, capacity, topk)] = (source, token)
    return metadata, counts, payload


def gather_address(topology, owner, contributor, token, k, capacity, topk):
    if topology.mapped(owner, contributor):
        return ("ipc", contributor, expanded_row(owner, token, k, capacity, topk))
    return ("landing", owner, expanded_row(contributor, token, k, capacity, topk))


def combine_scalar(topology, owner, token, ids, weights, capacity, read):
    result = 0.0
    for k, expert in enumerate(ids):
        if not valid_expert(expert, topology.ranks * 4):
            continue
        weight = 1.0 if weights is None else weights[k]
        if weight == 0.0:
            continue  # Including -0.0: do not even obtain/read a payload address.
        address = gather_address(topology, owner, expert // 4, token, k, capacity, len(ids))
        result += read(address) * weight
    return result


class OrderedMarkers:
    """One source/destination pair; arbitrary inter-QP progress, ordered QP WQEs."""

    def __init__(self, qps):
        self.queues = {q: deque() for q in qps}
        self.flags = Counter()
        self.markers = {q: [] for q in qps}
        self.payload = set()
        self.counts = {}
        self.count_writes = Counter()

    def post(self, generation, payloads, count=None, seed=0):
        rng = random.Random(seed)
        for q, queue in self.queues.items():
            # Different lanes/blocks may submit payloads in any order within a QP.
            entries = list(payloads.get(q, ()))
            rng.shuffle(entries)
            queue.extend(("payload", generation, entry) for entry in entries)
            if q == 0 and count is not None:
                queue.append(("count", generation, count))
            queue.append(("marker", generation, None))  # Even empty QPs/peers.

    def deliver(self, q):
        kind, generation, value = self.queues[q].popleft()
        if kind == "marker":
            self.flags[q] += 1  # Native atomicAdd increments; it does not write host epoch.
            self.markers[q].append(generation)
        elif kind == "count":
            self.counts[generation] = value
            self.count_writes[generation] += 1
        else:
            self.payload.add((generation, value))

    def ready(self, generation):
        return all(self.flags[q] >= generation for q in self.queues)

    def complete(self, seed=0):
        rng = random.Random(seed)
        while active := [q for q, queue in self.queues.items() if queue]:
            self.deliver(rng.choice(active))


class FinishAck:
    """Aliased expert storage cannot be reused until all readers AND drains finish."""

    def __init__(self, ranks):
        self.ranks = ranks
        self.readers = set()
        self.drains = set()
        self.acks = set()

    def publish(self, rank):
        if rank not in self.readers or rank not in self.drains:
            raise AssertionError("ACK before the local grid's readers/drains completed")
        self.acks.update((receiver, rank) for receiver in range(self.ranks))

    def reusable(self, rank):
        return all((rank, peer) in self.acks for peer in range(self.ranks))


def allocation_regions(ranks, capacity, topk, hidden, staging_slots, max_qps):
    """Integer offsets only; actual C++ Layout is tested in the companion unit."""
    rows = ranks * capacity * topk
    experts = ranks * 4
    ids = align((ranks + experts) * 8)  # LL8Packet storage header, NOT count protocol.
    weights = align(ids + rows * 4)
    data = align(weights + rows * 4)
    bf16 = align(hidden * 2 + topk * 8 + 4, 32)
    fp8 = align(hidden + (hidden // 128) * 4 + topk * 8 + 4, 32)
    dispatch = align((ranks + experts) * 8) + ranks * capacity * align(max(bf16, fp8))
    recv = align(max(dispatch, data + rows * hidden * 2, rows * hidden * 2))
    stride = align(hidden * 2 + topk * 8)
    regions = {"ids": (ids, rows * 4), "weights": (weights, rows * 4), "tokens": (data, rows * hidden * 2)}
    cursor = 2 * recv
    for name, size in (
        ("staging", max(staging_slots, capacity) * stride),
        ("dispatch_flags", ranks * max_qps * 8),
        ("combine_flags", ranks * max_qps * 8),
        ("landing", rows * hidden * 2),
        ("send_ids", rows * 4),
        ("send_weights", rows * 4),
        ("sync_flags", ranks * 8),
        ("sync_epoch", 8),
        ("counts", ranks * 4),
        ("count_staging", ranks * 4),
    ):
        regions[name] = (cursor, size)
        cursor += align(size)
    return regions, cursor, stride, recv


class SourceLexerTests(unittest.TestCase):
    def test_comments_cannot_satisfy_guards(self):
        text = "/* __any_sync(0xffffffff, remote) */ // LL8Packet\n int real = 1;"
        self.assertEqual(tokens(text), tokens("int real=1;"))
        with self.assertRaises(AssertionError):
            locate(tokens(text), "__any_sync")

    def test_strings_raw_strings_and_nested_scopes(self):
        text = 'void f() { if (x) { f("// not a comment"); } auto x=R"d(/*raw*/)d"; }'
        code = tokens(text)
        self.assertIn('"// not a comment"', code)
        self.assertIn('R"d(/*raw*/)d"', code)
        self.assertEqual(body(code, "f")[-1], ";")

    def test_vote_guard_rejects_both_braced_and_unbraced_divergence(self):
        good = "for (int token=0; token<n; ++token) { if (__any_sync(0xffffffff, remote)) { send(); } }"
        assert_unconditional_dispatch_vote(tokens(good))
        for bad in (
            good.replace("if (__any_sync", "if (remote) if (__any_sync"),
            good.replace("if (__any_sync", "if (lane < topk) { if (__any_sync") + "}",
            good.replace("0xffffffff", "__activemask()"),
        ):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                assert_unconditional_dispatch_vote(tokens(bad))

    def test_token_diff_ignores_formatting_but_detects_legacy_edits(self):
        before = tokens("void legacy(){ int count = 1; }")
        self.assertFalse(code_changes(before, tokens("void legacy() { /* note */ int count=1; }")))
        self.assertTrue(code_changes(before, tokens("void legacy(){ int count = 2; }")))


class RoutingModelTests(unittest.TestCase):
    def test_full_int64_bounds_before_narrowing(self):
        for ranks in RANKS:
            experts = ranks * 4
            for value in (-(1 << 63), -(1 << 32), -1, experts, (1 << 32) + 1, (1 << 63) - 1):
                with self.subTest(ranks=ranks, expert=value):
                    self.assertFalse(valid_expert(value, experts))
            self.assertTrue(valid_expert(0, experts))
            self.assertTrue(valid_expert(experts - 1, experts))
            # A narrowing-before-validation regression can turn a huge ID into expert 1.
            self.assertEqual(((1 << 32) + 1) & 0xFFFFFFFF, 1)

    def test_selected_routes_dense_metadata_padding_and_counts(self):
        for ranks, topk, weighted in product(RANKS, TOPKS, (False, True)):
            for source in range(ranks):
                with self.subTest(ranks=ranks, source=source, topk=topk, weighted=weighted):
                    choices = (source * 4, source * 4, (ranks - 1) * 4 + 3, -1, ranks * 4, (1 << 32) + 1)
                    ids = [tuple(choices[(t + k) % len(choices)] for k in range(topk)) for t in range(3)]
                    weights = [tuple((0.0, 0.5, -1.0)[k % 3] for k in range(topk)) for _ in ids] if weighted else None
                    metadata, counts, payload = dispatch_image(source, ids, weights, ranks, 5, topk)
                    expected_counts = Counter(e // 4 for row in ids for e in row if 0 <= e < ranks * 4)
                    self.assertEqual(counts, [expected_counts[p] for p in range(ranks)])
                    self.assertEqual(len(payload), sum(counts))
                    for peer in range(ranks):
                        self.assertEqual(len(metadata[peer]), 5 * topk)
                        for token, k in product(range(5), range(topk)):
                            selected = token < 3 and 0 <= ids[token][k] < ranks * 4 and ids[token][k] // 4 == peer
                            expected = (
                                (ids[token][k], 1.0 if weights is None else weights[token][k])
                                if selected
                                else (-1, 0.0)
                            )
                            self.assertEqual(metadata[peer][token * topk + k], expected)
                            row = (source * 5 + token) * topk + k
                            self.assertEqual((peer, row) in payload, selected)

    def test_empty_sources_clear_every_destination_and_padding(self):
        for ranks, topk, invalid in product(RANKS, TOPKS, (-1, 256)):
            metadata, counts, payload = dispatch_image(ranks - 1, [], None, ranks, 133, topk, invalid)
            self.assertEqual(counts, [0] * ranks)
            self.assertFalse(payload)
            self.assertTrue(all(row == [(invalid, 0.0)] * (133 * topk) for row in metadata))

    def test_duplicates_are_distinct_routes_even_with_zero_weight(self):
        metadata, counts, payload = dispatch_image(1, [(4, 4, 4)], [(0.0, -0.0, 2.0)], 2, 1, 3)
        self.assertEqual(counts, [0, 3])
        self.assertEqual(metadata[1], [(4, 0.0), (4, -0.0), (4, 2.0)])
        self.assertEqual(set(payload), {(1, 3), (1, 4), (1, 5)})

    def test_world_indices_for_ipc_and_remote_landing(self):
        for topology in topologies():
            for owner, source in product(range(topology.ranks), repeat=2):
                with self.subTest(topology=topology, owner=owner, source=source):
                    mapped = topology.domains == 1 or (owner < topology.ranks // 2) == (source < topology.ranks // 2)
                    self.assertEqual(topology.mapped(owner, source), mapped)
                    address = gather_address(topology, owner, source, 132, 8, 133, 9)
                    if mapped:
                        self.assertEqual(address, ("ipc", source, (owner * 133 + 132) * 9 + 8))
                    else:
                        # Sender reads owner-major expert output; receiver lands by world contributor rank.
                        pushed = ("landing", owner, (source * 133 + 132) * 9 + 8)
                        self.assertEqual(address, pushed)
                        self.assertNotEqual(address[2], (owner * 133 + 132) * 9 + 8)

    def test_zero_weight_nan_and_invalid_rows_are_never_read(self):
        for domains, topk in product((1, 2), TOPKS):
            topology = Topology(8, domains, 4, 2)
            ids = ([4, 4, -1, 1 << 32, 32, 0, 4, 4, 4])[:topk]
            weights = ([0.0, -0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0])[:topk]
            allowed = {gather_address(topology, 7, 0, 0, 5, 1, topk)} if topk > 5 else set()
            reads = []

            def read(address):
                self.assertIn(address, allowed, "payload was read before the invalid/zero-weight guard")
                reads.append(address)
                return 3.0

            result = combine_scalar(topology, 7, 0, ids, weights, 1, read)
            self.assertEqual(result, 6.0 if allowed else 0.0)
            self.assertEqual(set(reads), allowed)
            self.assertFalse(math.isnan(result))
            self.assertTrue(math.isnan(float("nan") * 0.0))  # Multiplying instead of skipping is NOT safe.

    def test_duplicate_experts_may_produce_different_slot_payloads(self):
        for topology in (Topology(8, 1, 8, 4), Topology(8, 2, 8, 4)):
            ids, weights = (4, 4, 4), (0.5, -2.0, 0.0)
            addresses = [gather_address(topology, 7, 1, 0, k, 1, 3) for k in range(3)]
            values = dict(zip(addresses, (4.0, 7.0, float("nan"))))
            self.assertEqual(len(values), 3)
            reads = []

            def read(address):
                reads.append(address)
                return values[address]

            self.assertEqual(combine_scalar(topology, 7, 0, ids, weights, 1, read), -12.0)
            self.assertEqual(reads, addresses[:2])

    def test_missing_weights_default_to_one_in_original_slot_order(self):
        topology = Topology(2, 2, 2, 1)
        reads = []

        def read(address):
            reads.append(address[2])
            return float(address[2] + 1)

        self.assertEqual(combine_scalar(topology, 0, 0, (4, 4, 4), None, 1, read), 15.0)
        self.assertEqual(reads, [3, 4, 5])

    def test_tma_barrier_phase_advances_only_for_loaded_slots(self):
        for topk in (1, 8):
            phases = [0] * topk
            loads = [0] * topk
            for token in range(133):
                for k in range(topk):
                    before = phases[k]
                    valid = (token + k) % 3 != 0
                    if valid:
                        loads[k] += 1
                        phases[k] ^= 1  # BulkBarrier.wait(uint32_t&) performs this update.
                    self.assertEqual(phases[k], loads[k] % 2)
                    if not valid:
                        self.assertEqual(phases[k], before)


class ReadinessModelTests(unittest.TestCase):
    def test_selected_routes_and_dense_metadata_are_ready_before_unique_count_use(self):
        for topology in topologies():
            for source, topk in product((0, topology.ranks - 1), TOPKS):
                choices = (0, 0, topology.ranks * 4 - 1, -1, (1 << 32) + 1)
                ids = [tuple(choices[k % len(choices)] for k in range(topk))]
                metadata, counts, payload = dispatch_image(source, ids, None, topology.ranks, 3, topk)
                received_counts = {}
                for peer in range(topology.ranks):
                    qps = topology.dispatch_qps(peer, source)
                    model = OrderedMarkers(qps)
                    writes = {q: [] for q in qps}
                    for k, expert in enumerate(ids[0]):
                        if valid_expert(expert, topology.ranks * 4) and expert // 4 == peer:
                            q = 0 if topology.mapped(peer, source) else (expert % 4) % topology.qps
                            row = expanded_row(source, 0, k, 3, topk)
                            self.assertIn((peer, row), payload)
                            writes[q].append(("row", row))
                    # Both metadata images are dense, including peers with no selected payloads.
                    writes[0].extend(
                        (
                            ("ids", tuple(e for e, _ in metadata[peer])),
                            ("weights", tuple(w for _, w in metadata[peer])),
                        )
                    )
                    required = {(1, entry) for entries in writes.values() for entry in entries}
                    model.post(1, writes, counts[peer], seed=source + peer)
                    while any(model.queues.values()):
                        # Deliberately finish nonzero queues before the metadata/count queue.
                        q = next(q for q in reversed(qps) if model.queues[q])
                        model.deliver(q)
                        if model.ready(1):
                            self.assertEqual(model.payload, required)
                            self.assertEqual(model.count_writes, {1: 1})
                            received_counts[peer, source] = model.counts[1]
                    self.assertEqual(received_counts[peer, source], counts[peer])
                self.assertEqual(sum(received_counts.values()), len(payload))

    def test_dispatch_fixed_all_qps_including_empty_peers(self):
        for topology in topologies():
            for owner in range(topology.ranks):
                for source in (owner, (owner + topology.ranks // 2) % topology.ranks):
                    qps = topology.dispatch_qps(owner, source)
                    expected = 1 if topology.mapped(owner, source) else topology.qps
                    self.assertEqual(len(qps), expected)
                    model = OrderedMarkers(qps)
                    model.post(1, {}, count=0)
                    self.assertFalse(model.ready(1))
                    model.complete()
                    self.assertTrue(model.ready(1))
                    self.assertEqual(model.count_writes, {1: 1})
                    self.assertEqual(model.counts, {1: 0})

    def test_combine_per_hca_markers_unique_and_physically_striped(self):
        for topology in topologies():
            for owner in range(topology.ranks):
                source = (owner + topology.ranks // 2) % topology.ranks
                qps = topology.combine_qps(owner, source)
                self.assertEqual(len(set(qps)), len(qps))
                if topology.mapped(owner, source):
                    self.assertEqual(qps, (0,))
                else:
                    self.assertEqual(
                        qps, tuple((owner % topology.qps + h) % topology.qps for h in range(topology.hcas))
                    )
                    self.assertEqual({q % topology.hcas for q in qps}, set(range(topology.hcas)))
                model = OrderedMarkers(qps)
                model.post(1, {})
                for q in qps[:-1]:
                    model.deliver(q)
                    self.assertFalse(model.ready(1))
                model.deliver(qps[-1])
                self.assertTrue(model.ready(1))

    def test_stripe_ranges_partition_rows_including_empty_stripes(self):
        for capacity, topk, hcas in product((1, 128, 133, 4096), TOPKS, QPS):
            rows = capacity * topk
            ranges = [range(rows * h // hcas, rows * (h + 1) // hcas) for h in range(hcas)]
            self.assertEqual([r for stripe in ranges for r in stripe], list(range(rows)))
            self.assertEqual(sum(len(stripe) for stripe in ranges), rows)

    def test_every_same_qp_payload_permutation_precedes_marker(self):
        for order in permutations(("row0", "row1", "row2")):
            for qps in QPS:
                model = OrderedMarkers(range(qps))
                model.post(1, {qps - 1: order}, count=3)
                # Explicitly exercise all six submission orders, not only RNG samples.
                tail = model.queues[qps - 1]
                remainder = [event for event in tail if event[0] != "payload"]
                model.queues[qps - 1] = deque([("payload", 1, row) for row in order] + remainder)
                while any(model.queues.values()):
                    q = next(q for q in reversed(range(qps)) if model.queues[q])
                    model.deliver(q)
                    if model.ready(1):
                        self.assertEqual(model.payload, {(1, row) for row in order})
                        self.assertEqual(model.counts, {1: 3})
                self.assertEqual(model.count_writes, {1: 1})

    def test_replay_generations_change_active_qps_without_skipping_empty_ones(self):
        for qps in QPS:
            model = OrderedMarkers(range(qps))
            for generation in range(1, 10):
                count = 0 if generation % 3 == 0 else generation
                active = generation % qps
                rows = [(generation, k) for k in range(count)]
                model.post(generation, {active: rows}, count=count, seed=generation)
                self.assertFalse(model.ready(generation))
                model.complete(seed=generation)
                self.assertTrue(model.ready(generation))
                self.assertEqual(model.counts[generation], count)
                self.assertEqual(model.count_writes[generation], 1)
                self.assertTrue(all(model.flags[q] == generation for q in range(qps)))
                self.assertTrue(all(history == list(range(1, generation + 1)) for history in model.markers.values()))

    def test_combine_replays_mark_empty_physical_stripes(self):
        for topology in topologies():
            owner, source = topology.ranks - 1, 0
            model = OrderedMarkers(topology.combine_qps(owner, source))
            for generation in range(1, 10):
                q = tuple(model.queues)[generation % len(model.queues)]
                rows = {} if generation % 2 else {q: [(owner, source, generation)]}
                model.post(generation, rows, seed=generation)
                model.complete(generation)
                self.assertTrue(model.ready(generation))
                self.assertTrue(all(v == generation for v in model.flags.values()))
            self.assertFalse(model.count_writes)  # Combine markers do not republish dispatch counts.

    def test_negative_control_missing_zero_qp_marker_stalls_next_generation(self):
        model = OrderedMarkers((0, 1))
        model.post(1, {0: ["row"]}, count=1)
        model.queues[1].clear()  # Wrong: only mark QPs which carried payload.
        model.complete()
        self.assertFalse(model.ready(1))
        model.post(2, {1: ["new-row"]}, count=1)
        model.complete()
        self.assertFalse(model.ready(2))
        self.assertEqual(model.markers[1], [2])
        self.assertEqual(model.flags[1], 1)

    def test_negative_control_duplicate_marker_can_fake_readiness(self):
        model = OrderedMarkers((0, 1))
        model.post(1, {})
        model.queues[0].append(("marker", 1, None))
        model.complete()
        model.post(2, {0: ["pending"]})
        model.deliver(1)
        self.assertTrue(model.ready(2))  # Counts alone cannot detect duplicate publication.
        self.assertNotIn((2, "pending"), model.payload)
        self.assertEqual(model.markers[0], [1, 1])

    def test_negative_control_marker_before_payload_is_not_a_delivery_proof(self):
        model = OrderedMarkers((0,))
        model.post(1, {0: ["pending"]}, count=1)
        model.queues[0].rotate(1)  # Wrong: move marker ahead of its preceding writes.
        model.deliver(0)
        self.assertTrue(model.ready(1))
        self.assertFalse(model.payload)
        self.assertNotIn(1, model.counts)

    def test_finish_ack_blocks_alias_reuse_until_all_readers_and_drains(self):
        for ranks in RANKS:
            ack = FinishAck(ranks)
            for rank in range(ranks - 1):
                ack.readers.add(rank)
                ack.drains.add(rank)
                ack.publish(rank)
            self.assertFalse(any(ack.reusable(rank) for rank in range(ranks)))
            with self.assertRaises(AssertionError):
                ack.publish(ranks - 1)
            ack.readers.add(ranks - 1)
            with self.assertRaises(AssertionError):
                ack.publish(ranks - 1)
            ack.drains.add(ranks - 1)
            ack.publish(ranks - 1)
            self.assertTrue(all(ack.reusable(rank) for rank in range(ranks)))

    def test_reject_unsupported_domain_or_hca_geometry(self):
        for args in ((8, 4, 4, 2), (8, 2, 4, 8), (8, 2, 8, 3), (8, 2, 4, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Topology(*args)


class AllocationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = tokens((EP / "config.hpp").read_text())
        cls.slots = integer_constant(cls.config, "GpuNetIoStagingSlots")
        cls.max_qps = integer_constant(cls.config, "GpuNetIoMaxQpsPerPeer")
        cls.alignment = integer_constant(cls.config, "BufferAlignmentBytes")

    def test_actual_staging_constant_not_assumed_to_be_128_or_4096(self):
        self.assertGreater(self.slots, 4096)
        self.assertEqual(self.alignment, 128)
        self.assertGreaterEqual(self.max_qps, max(QPS))

    def test_alignment_and_nonoverlap_at_requested_capacities(self):
        for ranks, capacity, topk, hidden in product(RANKS, (1, 128, 133, 4096), TOPKS, HIDDEN):
            with self.subTest(ranks=ranks, capacity=capacity, topk=topk, hidden=hidden):
                regions, total, stride, recv = allocation_regions(
                    ranks, capacity, topk, hidden, self.slots, self.max_qps
                )
                self.assertEqual(stride % 128, 0)
                self.assertEqual(total % 128, 0)
                previous_end = 0
                for offset, size in regions.values():
                    self.assertEqual(offset % 128, 0)
                    self.assertGreaterEqual(offset, previous_end)
                    self.assertLessEqual(offset + size, total)
                    previous_end = offset + size
                self.assertLessEqual(sum(regions["tokens"]), recv)
                self.assertLessEqual(capacity * stride, regions["staging"][1])
                self.assertEqual(regions["landing"][1], ranks * capacity * topk * hidden * 2)
                for name in ("ids", "weights", "send_ids", "send_weights"):
                    self.assertEqual(regions[name][1], ranks * capacity * topk * 4)

    def test_actual_staging_crossover_and_last_private_token_slot(self):
        for capacity, topk in product((self.slots - 1, self.slots, self.slots + 1), TOPKS):
            regions, _, stride, _ = allocation_regions(8, capacity, topk, 4096, self.slots, self.max_qps)
            self.assertEqual(regions["staging"][1], max(self.slots, capacity) * stride)
            last_byte = regions["staging"][0] + (capacity - 1) * stride + 4096 * 2
            self.assertLessEqual(last_byte, regions["dispatch_flags"][0])

    def test_private_token_staging_has_no_ring_reuse(self):
        for capacity in (1, 128, 133, 4096, self.slots + 1):
            stride = align(4096 * 2 + 9 * 8)
            offsets = [token * stride for token in range(capacity)]
            self.assertEqual(len(set(offsets)), capacity)
            self.assertTrue(all(b - a >= 4096 * 2 for a, b in zip(offsets, offsets[1:])))
        self.assertEqual((self.slots % self.slots) * stride, 0)  # Ring-address negative control.
        self.assertNotEqual(self.slots * stride, 0)

    def test_warpgroup_token_ownership_and_private_completion_stage_sizing(self):
        config = tokens((EP / "low_latency/config.cuh").read_text())
        warps = integer_constant(config, "DispatchNWarps")
        minimum = integer_constant(config, "DispatchMinNWarpsPerGroup")
        max_groups = warps // minimum
        for ranks, capacity, topk in product(RANKS, (1, 128, 133, 4096), TOPKS):
            payload_blocks = ranks
            for n_tokens in (0, capacity):
                per_group = (
                    warps if n_tokens <= payload_blocks else (warps // 2 if n_tokens <= 2 * payload_blocks else minimum)
                )
                groups = warps // per_group
                ownership = [
                    range((block - 1) * groups + group, n_tokens, payload_blocks * groups)
                    for block in range(1, payload_blocks + 1)
                    for group in range(groups)
                ]
                assigned = [token for stage in ownership for token in stage]
                self.assertEqual(sorted(assigned), list(range(n_tokens)))
                self.assertEqual(sum(len(stage) * topk for stage in ownership), n_tokens * topk)
                send_slots = max(ranks, max_groups * 32)
                control = align((send_slots + max_groups * ranks) * 4)
                counters = [(send_slots + group * ranks) * 4 for group in range(max_groups)]
                self.assertEqual(len(set(counters)), max_groups)
                self.assertLessEqual(counters[-1] + ranks * 4, control)
                for hidden in HIDDEN:
                    stride = align(align(hidden * 2 + topk * 8 + 4, 32))
                    barrier_base = control + max_groups * stride
                    shared_bytes = control + max_groups * (stride + 8)  # BulkBarrier has one aligned uint64.
                    self.assertEqual(barrier_base % 8, 0)
                    self.assertLessEqual(shared_bytes, 226 * 1024)
                    for group in range(max_groups):
                        self.assertEqual((control + group * stride) % 128, 0)
                        self.assertLessEqual(control + group * stride + hidden * 2, barrier_base)

    def test_counts_and_metadata_never_share_a_staging_address(self):
        for ranks in RANKS:
            regions, _, _, _ = allocation_regions(ranks, 133, 9, 4096, self.slots, self.max_qps)
            count_start, count_size = regions["counts"]
            staged_start, staged_size = regions["count_staging"]
            self.assertEqual((count_size, staged_size), (ranks * 4, ranks * 4))
            self.assertGreaterEqual(staged_start, count_start + count_size)
            for peer in range(ranks):
                self.assertGreaterEqual(staged_start + peer * 4, sum(regions["send_weights"]))
                self.assertLess(staged_start + peer * 4, staged_start + staged_size)

    def test_world_source_qp_flag_indices_do_not_alias(self):
        for ranks, qps in product(RANKS, QPS):
            indices = [source * self.max_qps + q for source, q in product(range(ranks), range(qps))]
            self.assertEqual(len(set(indices)), ranks * qps)
            self.assertLess(max(indices), ranks * self.max_qps)
        # The native interface must bound Q: Q=max+1 aliases the next source.
        self.assertEqual(0 * self.max_qps + self.max_qps, 1 * self.max_qps + 0)


class NativeSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = tokens((EP / "low_latency/topk_expanded.cu").read_text())
        cls.config = tokens((EP / "config.hpp").read_text())
        cls.runtime = tokens((EP / "ll_runtime.cc").read_text())
        cls.bulk = tokens((ROOT / "include/mscclpp/bulk_device.hpp").read_text())

    def assertCode(self, stream, *fragments):
        for fragment in fragments:
            locate(stream, fragment)

    def assertOrder(self, stream, *fragments):
        at = 0
        for fragment in fragments:
            at = locate(stream, fragment, at) + len(tokens(fragment))

    def test_release_build_qp_bounds_before_any_flag_access(self):
        check = body(self.native, "validateGpuNetIo")
        self.assertCode(
            check,
            "qps <= 0 || qps > GpuNetIoMaxQpsPerPeer || hcas <= 0 || hcas > qps || qps % hcas != 0",
            "__trap()",
        )
        for name in ("dispatchTopkExpandedKernel", "combineTopkExpandedKernel"):
            self.assertOrder(body(self.native, name), "validateGpuNetIo(transport)", "const Layout layout")

    def test_legacy_dispatch_and_combine_tokens_outside_only_exported_route(self):
        for kind in ("dispatch", "combine"):
            path = f"src/ext/ep/low_latency/{kind}.cu"
            baseline = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{LEGACY_BASE}:{path}"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout
            current = tokens((ROOT / path).read_text())
            with self.subTest(kernel=kind):
                remaining = strip_exported_route(current, kind)
                self.assertEqual(code_changes(tokens(baseline), remaining), [], "legacy CUDA code changed")
                # A legacy arithmetic edit must not be hidden by the allowed route removal.
                changed = remaining + tokens("legacy_unexpected_change();")
                self.assertTrue(code_changes(tokens(baseline), changed))

    def test_real_cpp_layout_unit_keeps_legacy_formula_coverage(self):
        cpp = tokens((ROOT / "test/python/ep/test_topk_expanded_layout.cpp").read_text())
        check = body(cpp, "checkLayout")
        self.assertCode(
            check,
            "for (bool rankMajor : {false, true})",
            "rankMajor ? tokenBytes : static_cast<size_t>(experts) * capacity * hidden * sizeof(Bf16)",
            "2 * expectedRecv + stagingBytes + 2 * flagsBytes + alignBytes(tokenBytes)",
            "legacy(reinterpret_cast<void*>(base), capacity, hidden, ranks, experts, topk, rankMajor)",
            "explicitLegacy(reinterpret_cast<void*>(base), capacity, hidden, ranks, experts, topk, rankMajor, false)",
            "requireNoExpandedPointers(*layout)",
            "layout->totalBytes_ == expectedTotal",
            "layout->rankMajorExpertOutputBuffer_ != layout->rankMajorTokenBuffer_",
        )
        self.assertCode(body(cpp, "main"), "checkOriginalShapes()", "checkBoundaryShapes()")

    def test_dispatch_upstream_warpgroup_initialization_and_tma(self):
        init = body(self.native, "initRankMajorSendState")
        self.assertCode(
            init,
            "dispatchNWarpsPerGroup(nTokens, nPayloadBlocks)",
            "const int nWarpGroups = DispatchNWarps / nWarpsPerGroup",
            "if (warpId % nWarpsPerGroup != 0) return false",
            "state.tokenStride_ = nPayloadBlocks * nWarpGroups",
            "state.firstTokenIdx_ = (static_cast<int>(blockIdx.x) - 1) * nWarpGroups + state.warpGroupId_",
            "state.stagedToken_ = sharedTokenBase + state.warpGroupId_ * stride",
            "state.bulkBarrier_ = barriers + state.warpGroupId_",
            "state.bulkPhase_ = 0",
        )
        send = body(self.native, "dispatchSendRankMajorTopkExpandedBf16")
        self.assertOrder(
            send,
            "initRankMajorSendState(",
            "send.bulkBarrier_->arriveAndExpect(bytes)",
            "mscclpp::bulkLoad(",
            "send.bulkBarrier_->wait(send.bulkPhase_)",
            "mscclpp::bulkFence()",
            "mscclpp::bulkStore(",
            "mscclpp::bulkStoreCommit()",
            "mscclpp::bulkStoreWait()",
        )

    def test_dispatch_full_warp_vote_and_private_registered_token_slots(self):
        send = body(self.native, "dispatchSendRankMajorTopkExpandedBf16")
        assert_unconditional_dispatch_vote(send)
        self.assertCode(send, "static_cast<size_t>(token) * layout.gpuNetIoSlotStride_", "staged[v] = shared[v]")
        self.assertOrder(
            send,
            "staged[v] = shared[v]",
            "__syncwarp()",
            "__threadfence_system()",
            "__syncwarp()",
            "if (remote)",
            "transport.gpuNetIo_->put(",
        )
        self.assertNotIn("GpuNetIoStagingSlots", send)  # No modulo-ring addressing.

    def test_dispatch_per_stage_dense_metadata_completion_counters(self):
        send = body(self.native, "dispatchSendRankMajorTopkExpandedBf16")
        self.assertCode(
            send,
            "sharedMem + sendSlots + send.warpGroupId_ * ranks",
            "for (int peer = lane; peer < ranks; peer += WARP_SIZE) completions[peer] = 0",
            "ids[offset] = local ? static_cast<int>(expert) : work.invalidTokenExpertId_",
            "wgts[offset] = local ? weight : 0.0f",
            "atomicAdd_block(completions + peer, 1)",
            "workspace.dispatchRankPayloadCompletions_ + peer, completions[peer], mscclpp::memoryOrderRelease",
        )
        notify = body(self.native, "dispatchRankMajorTopkExpandedNotify")
        self.assertCode(
            notify,
            "const int expected = work.numTokens_ * topk",
            "workspace.dispatchRankPayloadCompletions_[peer] = 0",
            "static_cast<size_t>(work.numTokens_) * topk + threadIdx.x",
            "ids[base + i] = work.invalidTokenExpertId_",
            "wgts[base + i] = 0.0f",
        )
        shared = body(self.native, "dispatchControlBytes")
        self.assertCode(shared, "sendSlots + DispatchMaxNWarpGroups * ranks")

    def test_int64_validation_precedes_narrowing_on_every_route(self):
        self.assertCode(self.native, "validExpert(int64_t expert, int nExperts)")
        self.assertCode(body(self.native, "validExpert"), "return expert >= 0 && expert < nExperts")
        for name in (
            "dispatchSendRankMajorTopkExpandedBf16",
            "dispatchRankMajorTopkExpandedNotify",
            "recvRankMajorTopkExpandedRemotePartialsTma",
            "recvRankMajorTopkExpandedRemotePartials",
        ):
            code = body(self.native, name)
            self.assertOrder(code, "const int64_t expert", "validExpert(expert, work.numExperts_)")
        self.assertCode(
            body(self.native, "validate"), "static_cast<int64_t>(work.maxTokensPerRank_) * work.numTopk_ <= INT32_MAX"
        )

    def test_counts_are_int_buffers_not_ll8_or_captured_host_epochs(self):
        self.assertNotIn("LL8Packet", self.native)
        self.assertNotIn("epoch_", self.native)
        notify = body(self.native, "dispatchRankMajorTopkExpandedNotify")
        self.assertCode(
            notify,
            "atomicAdd_block(counts + expert / localExperts, 1)",
            "remoteCount[transport.rank_] = counts[peer]",
            "static_cast<int*>(layout.expandedCountStaging_) + peer",
            "static_cast<int*>(layout.expandedCounts_) + transport.rank_",
            "transport.symmetricOffset(staged), sizeof(int), 0",
        )
        self.assertNotIn("weight", notify)
        self.assertOrder(
            body(self.native, "dispatchTopkExpandedKernel"),
            "waitSource(",
            "outputCount[blockIdx.x] = static_cast<int*>(layout.expandedCounts_)[blockIdx.x]",
        )

    def test_dispatch_posts_dense_metadata_after_payload_barrier_then_every_qp_marker_and_drain(self):
        kernel = body(self.native, "dispatchTopkExpandedKernel")
        self.assertOrder(
            kernel,
            "dispatchRankMajorTopkExpandedNotify(",
            "__threadfence_system()",
            "state.combineSyncer_->sync(gridDim.x)",
            "postRemoteDispatchMetadataAndMarkers(",
        )
        post = body(self.native, "postRemoteDispatchMetadataAndMarkers")
        self.assertCode(
            post,
            "const size_t source = static_cast<size_t>(peer) * rows",
            "const size_t dest = static_cast<size_t>(transport.rank_) * rows",
            "rows * sizeof(int), 0",
            "rows * sizeof(float), 0",
        )
        self.assertOrder(post, "gin->put(", "__syncthreads()", "gin->atomicAdd(", "__syncthreads()", "gin->flush(")
        for call in ("gin->atomicAdd", "gin->flush"):
            controls = enclosing_controls(post, locate(post, call))
            self.assertEqual([kind for kind, _ in controls], ["for", "if"])
            self.assertIn("numQpsPerPeer", controls[0][1])
            self.assertEqual(controls[1][1], tokens("!transport.isNvlinkPeer(peer)"))

    def test_combine_stripe_mapping_and_world_landing_indices(self):
        self.assertCode(
            body(self.native, "markerQp"),
            "(owner % transport.gpuNetIo_->numQpsPerPeer + stripe) % transport.gpuNetIo_->numQpsPerPeer",
        )
        ready = body(self.native, "sourceReady")
        self.assertCode(
            ready,
            "markerQp(transport, transport.rank_, stripe)",
            "flags + static_cast<size_t>(source) * GpuNetIoMaxQpsPerPeer + q",
        )
        self.assertCode(
            body(self.native, "markerCount"),
            "combine ? transport.gpuNetIo_->numHcas : transport.gpuNetIo_->numQpsPerPeer",
        )
        push = body(self.native, "pushExpandedCombine")
        self.assertCode(
            push,
            "markerQp(transport, owner, stripe)",
            "rows * stripe / gin->numHcas",
            "rows * (stripe + 1) / gin->numHcas",
            "(owner * rows + row) * bytes",
            "(static_cast<size_t>(transport.rank_) * rows + row) * bytes",
        )
        row = body(self.native, "expandedRow")
        self.assertCode(
            row,
            "const int rowRank = mapped ? transport.rank_ : source",
            "((static_cast<size_t>(rowRank) * capacity + token) * topk + slot) * bytes",
        )

    def test_combine_empty_stripes_still_mark_and_counts_are_not_added(self):
        push = body(self.native, "pushExpandedCombine")
        controls = enclosing_controls(push, locate(push, "gin->atomicAdd"))
        self.assertEqual(controls, [("if", tokens("threadIdx.x < gin->numHcas"))])
        self.assertNotIn("expandedCounts_", push)
        self.assertNotIn("dispatchRecvCounts_", push)
        self.assertCode(push, "gin->atomicAdd(owner, transport.symmetricOffset(flag), 1, q)")

    def test_combine_skips_zero_weight_before_payload_in_both_gathers_and_push(self):
        for name in ("recvRankMajorTopkExpandedRemotePartialsTma", "recvRankMajorTopkExpandedRemotePartials"):
            code = body(self.native, name)
            self.assertOrder(code, "weight != 0.0f", "expandedRow(")
            self.assertCode(code, "fmaf(")
        tma = body(self.native, "recvRankMajorTopkExpandedRemotePartialsTma")
        self.assertOrder(tma, "if (!validRows[k]) continue", "sharedRows[k * vectors + v]")
        fallback = body(self.native, "recvRankMajorTopkExpandedRemotePartials")
        self.assertOrder(fallback, "for (int k = 0; k < work.numTopk_; ++k)", "if (owner < 0) continue", "expandedRow(")
        self.assertOrder(
            body(self.native, "pushExpandedCombine"),
            "if (!validExpert(ids[row], work.numExperts_) || wgts[row] == 0.0f) continue",
            "gin->put(",
        )

    def test_tma_contributor_readiness_and_reference_phase_update(self):
        tma = body(self.native, "recvRankMajorTopkExpandedRemotePartialsTma")
        self.assertOrder(
            tma,
            "uint32_t phase = 0",
            "while (__any_sync(0xffffffff, pending))",
            "sourceReady(",
            "if (ready)",
            "expandedRow(",
            "barriers[lane].arriveAndExpect(bytes)",
            "mscclpp::bulkLoad(",
            "if (valid) barriers[lane].wait(phase)",
        )
        self.assertCode(self.bulk, "void wait(uint32_t& phase,")
        self.assertCode(body(self.bulk, "wait"), "phase ^= 1u")
        self.assertNotIn("^=", tma)  # Avoid a second unconditional phase toggle on skipped rows.
        self.assertCode(
            self.native,
            "RankMajorTmaMaxNTopk = 8",
            "CombineMaxNTopk = 9",
            "launchCombine<H, true>",
            "launchCombine<H, false>",
        )

    def test_final_drains_grid_barriers_and_finish_ack_are_retained(self):
        combine = body(self.native, "combineTopkExpandedKernel")
        self.assertOrder(
            combine,
            "recvRankMajorTopkExpandedRemotePartials<Hidden>",
            "gin->flush(blockIdx.x, markerQp(transport, blockIdx.x, threadIdx.x))",
            "__threadfence_system()",
            "state.combineSyncer_->sync(gridDim.x)",
            "finishCollective(",
            "state.combineArrivedBaseline_[comm.rank_] = target",
            "state.combineSyncer_->sync(gridDim.x)",
        )
        dispatch = body(self.native, "dispatchTopkExpandedKernel")
        self.assertOrder(
            dispatch,
            "postRemoteDispatchMetadataAndMarkers(",
            "waitSource(",
            "state.combineSyncer_->sync(gridDim.x)",
            "finishCollective(",
            "state.dispatchArrivedBaseline_[comm.rank_] = target",
            "state.combineSyncer_->sync(gridDim.x)",
        )
        finish = body(self.native, "finishCollective")
        self.assertOrder(
            finish,
            "++*epoch",
            "__syncthreads()",
            "const uint64_t target = *epoch",
            "signalLocal(",
            "transport.gpuNetIo_->atomicAdd(",
            "__syncthreads()",
            "flagReady(",
            "transport.gpuNetIo_->flush(peer, 0)",
            "__syncthreads()",
        )
        self.assertCode(finish, "remote + transport.rank_", "flags + peer")

    def test_replay_uses_separate_device_baselines_and_monotonic_flags(self):
        for phase in ("dispatch", "combine"):
            code = body(self.native, f"{phase}TopkExpandedKernel")
            self.assertCode(code, f"const uint64_t target = state.{phase}ArrivedBaseline_[comm.rank_] + 1")
        self.assertCode(body(self.native, "flagReady"), "mscclpp::memoryOrderAcquire) >= target")
        self.assertCode(body(self.native, "signalLocal"), "flag, 1, mscclpp::memoryOrderRelease")
        self.assertNotIn("epoch_", self.native)

    def test_expanded_layout_has_separate_metadata_counts_and_capacity_sizing(self):
        layout = body(self.config, "Layout")
        self.assertCode(
            layout,
            "topkExpanded && maxTokensPerRank > GpuNetIoStagingSlots ? maxTokensPerRank : GpuNetIoStagingSlots",
            "static_cast<size_t>(gpuNetIoStagingRows) * gpuNetIoSlotStride_",
            "rankMajorExpertOutputBuffer_ = topkExpanded ? rankMajorTokenBuffer_ : combineRecvBuffer_",
            "expandedSendIds_ = base + expandedIdsOffset",
            "expandedSendWeights_ = base + expandedWeightsOffset",
            "expandedCounts_ = base + expandedCountsOffset",
            "expandedCountStaging_ = base + expandedCountStagingOffset",
            "expandedCountStagingOffset = expandedCountsOffset + expandedCountBytes",
            "if (topkExpanded) totalBytes_ = expandedCountStagingOffset + expandedCountBytes",
        )
        self.assertOrder(
            layout,
            "expandedIdsOffset = totalBytes_",
            "expandedWeightsOffset =",
            "expandedSyncOffset =",
            "expandedEpochOffset =",
            "expandedCountsOffset =",
            "expandedCountStagingOffset =",
        )

    def test_host_keeps_fixed_shape_and_registered_alias_contract(self):
        for kind in ("dispatch", "combine"):
            code = body(self.runtime, kind)
            self.assertCode(
                code,
                "maxTokensPerRank == maxTokensPerRank_ && numTopk == numTopk_ && hidden == hidden_",
                "dispatchDataType == low_latency::DispatchDataType::BF16",
            )
        self.assertCode(
            body(self.runtime, "dispatch"),
            "output == allocationLayout.rankMajorTokenBuffer_",
            "outputTopkIdx == allocationLayout.rankMajorTopkIdsBuffer_",
            "outputTopkWeights == allocationLayout.rankMajorTopkWeightsBuffer_",
        )
        self.assertCode(
            body(self.runtime, "combine"),
            "input == allocationLayout.rankMajorExpertOutputBuffer_",
            "mode == low_latency::CombineMode::RANK_LOCAL_REDUCE",
        )


if __name__ == "__main__":
    unittest.main()
