# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Literal

from mscclpp.language.internal.operations import SignalOperation, WaitOperation
from mscclpp.language.internal.types import ChannelType, Instruction, SyncType


@dataclass(frozen=True)
class SynchronizationDiagnostic:
    """A diagnostic for the authored memory-channel token graph.

    Attributes:
        status: One of ``acyclic``, ``mismatch``, ``cycle``, or ``unsupported``.
        message: The result and its scope, or the reason analysis stopped.
        locations: Zero-based (rank, thread block, operation index) positions.
            A cycle repeats its first location at the end, in dependency order.
    """

    status: Literal["acyclic", "mismatch", "cycle", "unsupported"]
    message: str
    locations: tuple[tuple[int, int, int], ...] = ()


def diagnose_synchronization(program):
    if program._operations_processed:
        return SynchronizationDiagnostic("unsupported", "Call diagnose_synchronization() before post-processing.")
    if program.protocol != "Simple":
        return SynchronizationDiagnostic("unsupported", "Only Simple protocol plans are supported.")
    if program.loop_context is not None:
        return SynchronizationDiagnostic("unsupported", "An active pipeline context is not supported.")
    if program.reuse_resources:
        return SynchronizationDiagnostic("unsupported", "Resource reuse can carry semaphore state across plans.")
    if not isinstance(program.instances, int) or program.instances < 1:
        return SynchronizationDiagnostic("unsupported", "Instance replication requires a positive integer.")

    endpoints = set()
    channel_keys = {}
    for gpu in program.gpus:
        if gpu._nvls_channels or any(kind != ChannelType.memory for kind in gpu._channels):
            return SynchronizationDiagnostic("unsupported", f"Rank {gpu.id} has non-memory channels.")
        if gpu.semaphores:
            return SynchronizationDiagnostic("unsupported", f"Rank {gpu.id} has intra-rank semaphores.")
        peer_counts = defaultdict(int)
        channels = gpu._channels.get(ChannelType.memory)
        for channel_id, peer in enumerate(channels.connected_to if channels else ()):
            if peer == gpu.id or not 0 <= peer < program.num_ranks:
                return SynchronizationDiagnostic("unsupported", f"Rank {gpu.id} has unsupported peer {peer}.")
            # The executor pairs channel endpoints by their occurrence for each peer.
            key = (gpu.id, peer, peer_counts[peer])
            peer_counts[peer] += 1
            endpoints.add(key)
            channel_keys[gpu.id, channel_id] = key
    for src, dst, occurrence in channel_keys.values():
        if (dst, src, occurrence) not in endpoints:
            return SynchronizationDiagnostic(
                "unsupported",
                f"Memory channel {occurrence} from rank {src} to rank {dst} has no authored peer endpoint.",
            )

    nonblocking = {
        Instruction.nop,
        Instruction.copy,
        Instruction.reduce,
        Instruction.get,
        Instruction.put,
        Instruction.read_reduce,
        Instruction.reduce_send,
        Instruction.read_reduce_send,
    }
    predecessors = {}
    signals = defaultdict(list)
    waits = defaultdict(list)
    descriptions = {}
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            previous = None
            for index, op in enumerate(tb.ops):
                location = (gpu.id, tb.id, index)
                if getattr(op, "tbg_info", None) is not None:
                    return SynchronizationDiagnostic(
                        "unsupported", "Thread-block groups are not supported.", (location,)
                    )
                if not isinstance(op, (SignalOperation, WaitOperation)):
                    if op.name not in nonblocking:
                        return SynchronizationDiagnostic(
                            "unsupported", f"Operation {op.name} is not supported.", (location,)
                        )
                    continue
                if op.channel_type != ChannelType.memory or op.data_sync != SyncType.both or len(op.channel_ids) != 1:
                    return SynchronizationDiagnostic(
                        "unsupported",
                        "Signal/wait operations require one memory channel and SyncType.both.",
                        (location,),
                    )
                local_id = next(iter(op.channel_ids))
                channel_id = tb._channels[ChannelType.memory].channel_ids[local_id]
                src, dst, occurrence = channel_keys[gpu.id, channel_id]
                if isinstance(op, WaitOperation):
                    src, dst = dst, src
                    events = waits
                else:
                    events = signals
                key = (src, dst, occurrence)
                if events[key] and events[key][0][:2] != location[:2]:
                    return SynchronizationDiagnostic(
                        "unsupported",
                        f"Memory channel {occurrence} from rank {src} to rank {dst} has multiple {op.name} thread blocks.",
                        (events[key][0], location),
                    )
                events[key].append(location)
                descriptions[location] = (
                    f"rank {gpu.id}, thread block {tb.id}, operation {index} ({op.name}, "
                    f"channel {occurrence} from rank {src} to rank {dst}, token {len(events[key])})"
                )
                predecessors[location] = [previous] if previous is not None else []
                previous = location

    for key in dict.fromkeys((*signals, *waits)):
        sent = signals[key]
        received = waits[key]
        if len(sent) != len(received):
            src, dst, occurrence = key
            unmatched = sent[len(received)] if len(sent) > len(received) else received[len(sent)]
            return SynchronizationDiagnostic(
                "mismatch",
                f"Memory channel {occurrence} from rank {src} to rank {dst} has {len(sent)} signal(s) "
                f"and {len(received)} wait(s). First unmatched token at {descriptions[unmatched]}.",
                (unmatched,),
            )
        for signal, wait in zip(sent, received):
            predecessors[wait].append(signal)

    try:
        TopologicalSorter(predecessors).prepare()
    except CycleError as error:
        cycle = tuple(error.args[1])
        return SynchronizationDiagnostic(
            "cycle",
            "Memory-channel token dependency cycle: " + " -> ".join(descriptions[node] for node in cycle),
            cycle,
        )
    return SynchronizationDiagnostic(
        "acyclic",
        "The balanced memory-channel token graph is acyclic; device progress and data visibility are not checked.",
    )
