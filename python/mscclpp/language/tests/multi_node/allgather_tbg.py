# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *
import mscclpp


def allgather_example(
    name, num_gpus, tbg_size, gpus_per_node, num_threads_per_block, min_message_size, max_message_size
):
    nodes = num_gpus // gpus_per_node
    chunksperloop = 1
    collective = AllGather(num_gpus, chunksperloop, False)
    with CollectiveProgram(
        name,
        collective,
        num_gpus,
        protocol="Simple",
        num_threads_per_block=num_threads_per_block,
        reuse_resources=False,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Creating Channels
        memory_channels = {}
        port_channels = {}
        scratch_buffer = []
        tbgs = []
        tb_offset = 1
        total_tb = gpus_per_node * tbg_size + tb_offset
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                next_rank_id = (gpu + gpus_per_node * (node + 1)) % num_gpus
                previous_rank_id = (gpu + gpus_per_node * (node - 1)) % num_gpus
                scratch_buffer.append(Buffer(src_rank_id, num_gpus))
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        for tb in range(tbg_size):
                            dst_rank_id = peer + gpus_per_node * node
                            memory_channels[(dst_rank_id, src_rank_id, tb)] = MemoryChannel(dst_rank_id, src_rank_id)
                port_channels[src_rank_id, 0] = PortChannel(next_rank_id, src_rank_id)
                if nodes != 2:
                    port_channels[src_rank_id, 1] = PortChannel(previous_rank_id, src_rank_id)
                else:
                    port_channels[src_rank_id, 1] = port_channels[src_rank_id, 0]

        for gpu in range(gpus_per_node):
            tbgs.append(
                ThreadBlockGroup(
                    tb_list=[i for i in range(gpu * tbg_size + tb_offset, gpu * tbg_size + tb_offset + tbg_size)]
                )
            )

        # Ensuring the intra node ranks are ready to receive data
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + gpus_per_node * node
                        tbg_id = peer if peer < gpu else peer - 1
                        for i in range(tbgs[tbg_id].numtb()):
                            tb = tbgs[tbg_id].tb_list[i]
                            memory_channels[(dst_rank_id, src_rank_id, i)].signal(
                                tb=tb, data_sync=SyncType.none, relaxed=True
                            )

        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                for peer in range(gpus_per_node):
                    if peer != gpu:
                        dst_rank_id = peer + gpus_per_node * node
                        tbg_id = peer if peer < gpu else peer - 1
                        for i in range(tbgs[tbg_id].numtb()):
                            tb = tbgs[tbg_id].tb_list[i]
                            memory_channels[(dst_rank_id, src_rank_id, i)].wait(
                                tb=tb, data_sync=SyncType.after, relaxed=True
                            )

        # Ensuring inter node ranks are ready to receive data
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                src_rank_id = gpu + gpus_per_node * node
                port_channels[src_rank_id, 1].signal(tb=0, data_sync=SyncType.none)
                port_channels[src_rank_id, 0].wait(tb=0, data_sync=SyncType.after)

        for step in range(nodes):

            # Internode AllGather
            for node in range(nodes):
                for gpu in range(gpus_per_node):
                    src_rank_id = gpu + gpus_per_node * node
                    src_rank = Rank(src_rank_id)
                    src_input_buffer = src_rank.get_input_buffer()
                    next_rank_id = (gpu + gpus_per_node * (node + 1)) % num_gpus
                    index = (gpu + gpus_per_node * (node - step)) % num_gpus
                    if step != 0:
                        port_channels[src_rank_id, 1].wait(tb=0)
                        src_rank.barrier(tb_list=[tb for tb in range(total_tb)])
                    if step != nodes - 1:
                        if step == 0:
                            port_channels[src_rank_id, 0].put_with_signal_and_flush(
                                scratch_buffer[next_rank_id][index : index + 1], src_input_buffer[0:1], tb=0
                            )
                        else:
                            port_channels[src_rank_id, 0].put_with_signal_and_flush(
                                scratch_buffer[next_rank_id][index : index + 1],
                                scratch_buffer[src_rank_id][index : index + 1],
                                tb=0,
                            )

            # Intranode AllGather
            for node in range(nodes):
                for gpu in range(gpus_per_node):
                    src_rank_id = gpu + gpus_per_node * node
                    src_rank = Rank(src_rank_id)
                    input_buffer = src_rank.get_input_buffer()
                    output_buffer = src_rank.get_output_buffer()
                    local_index = (gpu + gpus_per_node * (node - step)) % num_gpus
                    for peer in range(gpus_per_node):
                        if peer != gpu:
                            dst_rank_id = peer + gpus_per_node * node
                            tbg_id = peer if peer < gpu else peer - 1
                            remote_index = (peer + gpus_per_node * (node - step)) % num_gpus
                            if step == 0:
                                memory_channels[(dst_rank_id, src_rank_id, 0)].put(
                                    scratch_buffer[dst_rank_id][local_index : local_index + 1],
                                    input_buffer[0:1],
                                    tb_group=tbgs[tbg_id],
                                )
                            else:
                                memory_channels[(dst_rank_id, src_rank_id, 0)].put(
                                    scratch_buffer[dst_rank_id][local_index : local_index + 1],
                                    scratch_buffer[src_rank_id][local_index : local_index + 1],
                                    tb_group=tbgs[tbg_id],
                                )
                            for i in range(tbgs[tbg_id].numtb()):
                                tb = tbgs[tbg_id].tb_list[i]
                                memory_channels[(dst_rank_id, src_rank_id, i)].signal(tb=tb, data_sync=SyncType.before)
                                memory_channels[(dst_rank_id, src_rank_id, i)].wait(tb=tb, data_sync=SyncType.after)
                            src_rank.copy(
                                output_buffer[remote_index : remote_index + 1],
                                scratch_buffer[src_rank_id][remote_index : remote_index + 1],
                                tb_group=tbgs[tbg_id],
                            )
                    if step == 0:
                        src_rank.copy(
                            output_buffer[local_index : local_index + 1],
                            input_buffer[0:1],
                            tb_group=tbgs[gpus_per_node - 1],
                        )
                    else:
                        src_rank.copy(
                            output_buffer[local_index : local_index + 1],
                            scratch_buffer[src_rank_id][local_index : local_index + 1],
                            tb_group=tbgs[gpus_per_node - 1],
                        )

        print(JSON())


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="name of the program")
parser.add_argument("--num_gpus", type=int, help="number of gpus")
parser.add_argument("--gpus_per_node", type=int, help="number of gpus per node")
parser.add_argument("--tbg_size", type=int, help="number of thread blocks in the thread block group")
parser.add_argument("--num_threads_per_block", type=int, default=1024, help="number of threads per block")
parser.add_argument("--min_message_size", type=int, default=0, help="minimum message size")
parser.add_argument("--max_message_size", type=int, default=2**64 - 1, help="maximum message size")

args = parser.parse_args()

allgather_example(
    args.name,
    args.num_gpus,
    args.gpus_per_node,
    args.tbg_size,
    args.num_threads_per_block,
    args.min_message_size,
    args.max_message_size,
)
