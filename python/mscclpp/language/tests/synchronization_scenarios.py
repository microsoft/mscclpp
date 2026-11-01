# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU scenarios for token diagnostics, run against an installed MSCCL++ package.

Run with ``python3 python/mscclpp/language/tests/synchronization_scenarios.py``.
The oracle executes authored streams using token counters, independently of the
diagnostic's graph and IR traversal. No GPU execution is performed.
"""

from collections import Counter
from itertools import permutations, product
import json

from mscclpp.language.channel import MemoryChannel, PortChannel, SwitchChannel
from mscclpp.language.collectives import AllGather, TestCollective
from mscclpp.language.internal.types import BufferType, SyncType
from mscclpp.language.loop import LoopIterationContext
from mscclpp.language.program import CollectiveProgram
from mscclpp.language.rank import Buffer, Rank, Semaphore
from mscclpp.language.thread_block_group import ThreadBlockGroup


def token_status(streams):
    """Run one event per ready thread block until completion or a fixed point."""
    produced = Counter()
    consumed = Counter()
    expected = Counter()
    for (rank, _), events in streams.items():
        for kind, peer, slot in events:
            key = (rank, peer, slot) if kind == "signal" else (peer, rank, slot)
            expected[key] += 1 if kind == "signal" else -1
    if any(expected.values()):
        return "mismatch"
    cursor = dict.fromkeys(streams, 0)
    while True:
        progress = False
        for owner, events in streams.items():
            rank, _ = owner
            if cursor[owner] == len(events):
                continue
            kind, peer, slot = events[cursor[owner]]
            key = (rank, peer, slot) if kind == "signal" else (peer, rank, slot)
            if kind == "wait" and produced[key] == consumed[key]:
                continue
            (produced if kind == "signal" else consumed)[key] += 1
            cursor[owner] += 1
            progress = True
        if not progress:
            return "acyclic" if all(cursor[owner] == len(events) for owner, events in streams.items()) else "cycle"


def author_streams(streams, peers, relaxed=False, data_sync=SyncType.both):
    channels = {}
    for rank, destinations in enumerate(peers):
        occurrences = Counter()
        for peer in destinations:
            channels[rank, peer, occurrences[peer]] = MemoryChannel(peer, rank)
            occurrences[peer] += 1
    for (rank, tb), events in streams.items():
        for kind, peer, slot in events:
            getattr(channels[rank, peer, slot], kind)(tb=tb, relaxed=relaxed, data_sync=data_sync)


def check_streams(streams, peers, relaxed=False, instances=1, instr_fusion=True):
    with CollectiveProgram(
        "token_scenario", TestCollective(len(peers), 1, 1), len(peers), instances=instances, instr_fusion=instr_fusion
    ) as program:
        author_streams(streams, peers, relaxed)
        before = [gpu.to_dict() for gpu in program.gpus]
        diagnostic = program.diagnose_synchronization()
        expected = token_status(streams)
        if diagnostic.status != expected:
            raise RuntimeError(f"Token oracle {expected}, diagnostic {diagnostic}, streams {streams}")
        if program.diagnose_synchronization() != diagnostic or before != [gpu.to_dict() for gpu in program.gpus]:
            raise RuntimeError("Repeated diagnosis changed the report or authored operations")
        if diagnostic.status == "cycle":
            cycle = diagnostic.locations
            if len(cycle) < 2 or cycle[0] != cycle[-1]:
                raise RuntimeError(f"Cycle witness is not closed: {diagnostic}")
            # Check each witness edge directly against the scenario's program order
            # or the corresponding ordinal in the peer's signal stream.
            for source, target in zip(cycle, cycle[1:]):
                if source[:2] == target[:2] and target[2] == source[2] + 1:
                    continue
                src_kind, src_peer, src_slot = streams[source[:2]][source[2]]
                dst_kind, dst_peer, dst_slot = streams[target[:2]][target[2]]
                source_ordinal = streams[source[:2]][: source[2] + 1].count(("signal", src_peer, src_slot))
                target_ordinal = streams[target[:2]][: target[2] + 1].count(("wait", dst_peer, dst_slot))
                if (src_kind, dst_kind, source[0], target[0], src_slot, source_ordinal) != (
                    "signal",
                    "wait",
                    dst_peer,
                    src_peer,
                    dst_slot,
                    target_ordinal,
                ):
                    raise RuntimeError(f"Invalid cycle edge {source} -> {target}: {diagnostic}")
        return diagnostic


def check_serialization(name, author, expected, **options):
    plans = []
    for diagnose in (False, True):
        with CollectiveProgram(name, TestCollective(2, 2, 2), 2, **options) as program:
            author(program)
            if diagnose:
                diagnostic = program.diagnose_synchronization()
                if diagnostic.status != expected:
                    raise RuntimeError(f"{name}: expected {expected}, got {diagnostic}")
            plans.append(program.to_json())
            if diagnose and program.diagnose_synchronization().status != "unsupported":
                raise RuntimeError(f"{name}: post-processed operations were accepted")
    if plans[0] != plans[1]:
        raise RuntimeError(f"{name}: diagnosis changed serialized output")


def allgather():
    plans = []
    for diagnose in (False, True):
        with CollectiveProgram("allgather_tokens", AllGather(2, 1, True), 2, instances=3) as program:
            for rank in range(2):
                peer = 1 - rank
                channel = MemoryChannel(peer, rank)
                channel.signal(tb=0, relaxed=True)
                channel.wait(tb=0, relaxed=True)
                channel.put(
                    Rank(peer).get_output_buffer()[rank : rank + 1],
                    Rank(rank).get_output_buffer()[rank : rank + 1],
                    tb=0,
                )
                channel.signal(tb=0)
                channel.wait(tb=0)
            if diagnose and program.diagnose_synchronization().status != "acyclic":
                raise RuntimeError("The allgather token graph was not accepted")
            plans.append(program.to_json())
    if plans[0] != plans[1]:
        raise RuntimeError("Diagnosis changed the replicated allgather plan")


def boundary_scenarios():
    balanced = {(0, 0): [("signal", 1, 0), ("wait", 1, 0)], (1, 0): [("signal", 0, 0), ("wait", 0, 0)]}
    peers = [[1], [0]]
    for fusion in (False, True):
        for instances in (1, 3):
            check_serialization(
                "balanced",
                lambda _: author_streams(balanced, peers),
                "acyclic",
                instances=instances,
                instr_fusion=fusion,
            )
    cycle = {owner: list(reversed(events)) for owner, events in balanced.items()}
    check_serialization("cycle", lambda _: author_streams(cycle, peers), "cycle")
    mismatch = {(0, 0): [("signal", 1, 0)], (1, 0): [("wait", 0, 1)]}
    check_serialization("mismatch", lambda _: author_streams(mismatch, [[1, 1], [0, 0]]), "mismatch")
    check_serialization("empty", lambda _: None, "acyclic")
    check_serialization("unpaired", lambda _: MemoryChannel(1, 0), "unsupported")
    check_serialization("reuse", lambda _: author_streams(balanced, peers), "unsupported", reuse_resources=True)
    check_serialization("ll", lambda _: author_streams(balanced, peers), "unsupported", protocol="LL")
    check_serialization("port", lambda _: PortChannel(1, 0), "unsupported")
    check_serialization("switch", lambda _: SwitchChannel([0, 1], BufferType.input), "unsupported")
    check_serialization("barrier", lambda _: Rank(0).barrier([0, 1]), "unsupported")
    check_serialization("semaphore", lambda _: Semaphore(0, 1).acquire(tb=0), "unsupported")
    check_serialization(
        "packets",
        lambda _: Rank(0).copy_packets(Buffer(0, 1)[0:1], Rank(0).get_input_buffer()[0:1], tb=0),
        "unsupported",
    )
    check_serialization(
        "tb_group",
        lambda _: Rank(0).copy(
            Rank(0).get_output_buffer()[0:2], Rank(0).get_input_buffer()[0:2], tb_group=ThreadBlockGroup([0, 1])
        ),
        "unsupported",
    )
    for sync in (SyncType.none, SyncType.before, SyncType.after):
        check_serialization("nondefault_sync", lambda _: author_streams(balanced, peers, data_sync=sync), "unsupported")
    for kind in ("signal", "wait"):
        shared = {
            (0, 0): [(kind, 1, 0)],
            (0, 1): [(kind, 1, 0)],
            (1, 0): [("wait" if kind == "signal" else "signal", 0, 0)] * 2,
        }
        check_serialization("shared_channel", lambda _: author_streams(shared, peers), "unsupported")

    def pipeline(program):
        with LoopIterationContext(1, 2):
            author_streams(balanced, peers)
            if program.diagnose_synchronization().status != "unsupported":
                raise RuntimeError("An active pipeline was accepted")

    check_serialization("pipeline", pipeline, "unsupported")
    with CollectiveProgram("post_process", TestCollective(2, 1, 1), 2, instances=2) as program:
        author_streams(balanced, peers)
        program.post_process_operations()
        if program.diagnose_synchronization().status != "unsupported":
            raise RuntimeError("Explicit post-processing was accepted")
    with CollectiveProgram("amended", TestCollective(2, 1, 1), 2) as program:
        forward, reverse = MemoryChannel(1, 0), MemoryChannel(0, 1)
        forward.signal(tb=0)
        if program.diagnose_synchronization().status != "mismatch":
            raise RuntimeError("Missing wait was accepted")
        reverse.wait(tb=0)
        if program.diagnose_synchronization().status != "acyclic":
            raise RuntimeError("Diagnosis retained stale state after completing the plan")
    allgather()


def main():
    cases = 0
    for count in range(4):
        orders = sorted(set(permutations(["signal"] * count + ["wait"] * count)))
        for left, right in product(orders, repeat=2):
            streams = {(rank, 0): [(kind, 1 - rank, 0) for kind in order] for rank, order in enumerate((left, right))}
            for relaxed in (False, True):
                check_streams(streams, [[1], [0]], relaxed=relaxed)
                cases += 1
    orders = list(permutations([("signal", 0), ("signal", 1), ("wait", 0), ("wait", 1)]))
    for left, right in product(orders, repeat=2):
        streams = {
            (rank, 0): [(kind, 1 - rank, slot) for kind, slot in order] for rank, order in enumerate((left, right))
        }
        check_streams(streams, [[1, 1], [0, 0]])
        cases += 1
    for choices in product((False, True), repeat=3):
        streams = {}
        for rank, wait_first in enumerate(choices):
            events = [("signal", (rank + 1) % 3, 0), ("wait", (rank - 1) % 3, 0)]
            streams[rank, rank + 1] = list(reversed(events)) if wait_first else events
        check_streams(streams, [[2, 1], [2, 0], [0, 1]], instances=3)
        cases += 1
    independent = {
        (0, 1): [("signal", 1, 0)],
        (0, 3): [("wait", 1, 0)],
        (1, 2): [("signal", 0, 0)],
        (1, 4): [("wait", 0, 0)],
    }
    check_streams(independent, [[1], [0]])
    boundary_scenarios()
    print(
        json.dumps(
            {"token_oracle_comparisons": cases + 1, "boundary_scenarios": "passed", "serialization_preserved": True}
        )
    )


if __name__ == "__main__":
    main()
