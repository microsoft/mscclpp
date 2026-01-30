# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# LD_PRELOAD=<MSCCLPP_REPO>/build/lib/libmscclpp_nccl.so MSCCLPP_DISABLE_CHANNEL_CACHE=true  torchrun --nnodes=1 --nproc_per_node=8 memory_report.py
import os, sys
import torch
import torch.distributed as dist


def memory_report(d) -> str:
    """
    One-line CUDA memory report for the current device.
    """
    if not torch.cuda.is_available():
        return "MEMORY REPORT: CUDA not available"
    torch.cuda.synchronize(d)

    allocated = torch.cuda.memory_allocated(d)
    reserved = torch.cuda.memory_reserved(d)
    max_alloc = torch.cuda.max_memory_allocated(d)
    max_resv = torch.cuda.max_memory_reserved(d)

    free_b, total_b = torch.cuda.mem_get_info(d)  # (free, total) in bytes
    used_b = total_b - free_b

    to_gib = lambda b: f"{b / (1024**3):.2f} GiB"
    return (
        "MEMORY REPORT: "
        f"torch allocated: {to_gib(allocated)} | "
        f"torch reserved: {to_gib(reserved)} | "
        f"max torch allocated: {to_gib(max_alloc)} | "
        f"max torch reserved: {to_gib(max_resv)} | "
        f"total memory used: {to_gib(used_b)} | "
        f"total memory: {to_gib(total_b)}"
    )


def main():
    # torchrun provides these envs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    nelems = 1024 * 1024 * 32  # 32M elements
    torch.cuda.set_device(local_rank)
    backend = "nccl"

    # init default PG
    dist.init_process_group(backend=backend, init_method="env://")
    if rank == 0:
        print(
            f"[world_size={world_size}] torch={torch.__version__}, cuda={torch.version.cuda}, backend={backend}",
            flush=True,
        )
    dist.barrier()

    # make a subgroup over all ranks (you can change to a subset to test)
    group_ranks = list(range(world_size))
    if rank == 0:
        print(f"Creating new_group with ranks={group_ranks}", flush=True)
    grp0 = dist.new_group(ranks=group_ranks, backend=backend)
    x = torch.ones(nelems, device=local_rank, dtype=torch.float32) * (rank + 1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=grp0)

    grp1 = dist.new_group(ranks=list(range(world_size)), backend=backend)
    x = torch.ones(nelems, device=local_rank, dtype=torch.float32) * (rank + 1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=grp1)

    dist.barrier()

    print(memory_report(local_rank))
    dist.destroy_process_group(grp0)
    dist.destroy_process_group(grp1)
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[rank {os.getenv('RANK','?')}] EXCEPTION: {e}", file=sys.stderr, flush=True)
        raise
