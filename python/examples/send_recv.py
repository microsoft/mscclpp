# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import time

import mscclpp


def main(args):
    if args.root:
        rank = 0
    else:
        rank = 1

    boot = mscclpp.TcpBootstrap.create(rank, 2)
    boot.initialize(args.if_ip_port_trio)

    comm = mscclpp.Communicator(boot)

    if args.gpu:
        import torch

        print("Allocating GPU memory")
        memory = torch.zeros(args.num_elements, dtype=torch.int32)
        memory = memory.to("cuda")
        ptr = memory.data_ptr()
        size = memory.numel() * memory.element_size()
    else:
        from array import array

        print("Allocating host memory")
        memory = array("i", [0] * args.num_elements)
        ptr, elements = memory.buffer_info()
        size = elements * memory.itemsize
    my_reg_mem = comm.register_memory(ptr, size, mscclpp.Transport.IB0)

    conn = comm.connect_on_setup((rank + 1) % 2, 0, mscclpp.Transport.IB0)

    other_reg_mem = None
    if rank == 0:
        other_reg_mem = comm.recv_memory_on_setup((rank + 1) % 2, 0)
    else:
        comm.send_memory_on_setup(my_reg_mem, (rank + 1) % 2, 0)

    comm.setup()

    if rank == 0:
        other_reg_mem = other_reg_mem.get()

    if rank == 0:
        for i in range(args.num_elements):
            memory[i] = i + 1
        conn.write(other_reg_mem, 0, my_reg_mem, 0, size)
        print("Done sending")
    else:
        print("Checking for correctness")
        # polling
        for _ in range(args.polling_num):
            all_correct = True
            for i in range(args.num_elements):
                if memory[i] != i + 1:
                    all_correct = False
                    print(f"Error: Mismatch at index {i}: expected {i + 1}, got {memory[i]}")
                    break
            if all_correct:
                print("All data matched expected values")
                break
            else:
                time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("if_ip_port_trio", type=str)
    parser.add_argument("-r", "--root", action="store_true")
    parser.add_argument("-n", "--num-elements", type=int, default=10)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--polling_num", type=int, default=100)
    args = parser.parse_args()

    main(args)
