import argparse
from mscclpp.language import *
from mscclpp.language.buffer import Buffer
from mscclpp.language.collectives import AllGather
from mscclpp.language.types import ChannelType, ReplicationPolicy


def allgather_test(gpus, instances):
    size = gpus
    collective = AllGather(size, 1, False)
    with MSCCLPPProgram(
        "allgather_with_barrier",
        collective,
        size,
        instances,
        protocol="Simple",
        replication_policy=ReplicationPolicy.interleaved,
    ):
        for n in range(gpus):
            c = chunk(n, Buffer.input, 0, 1)
            for peer in range(gpus):
                if n != peer:
                    c.put(peer, Buffer.output, n, sendtb=peer, chan_type=ChannelType.sm)
                else:
                    c.copy(n, Buffer.output, n, sendtb=peer)
            # explicit barrier
            r = rank(n)
            r.barrier(tb_list=list(range(gpus)))
            for peer in range(gpus):
                if n != peer:
                    c.signal(peer, Buffer.output, n, sendtb=peer, chan_type=ChannelType.sm)

        for n in range(gpus):
            for peer in range(gpus):
                c = chunk(n, Buffer.output, peer, 1)
                if n != peer:
                    c.wait(peer, Buffer.input, peer, recvtb=peer, chan_type=ChannelType.sm)

        Json()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument("num_gpus", type=int, help="number of gpus")
parser.add_argument("instances", type=int, help="number of instances")
args = parser.parse_args()
allgather_test(args.num_gpus, args.instances)
