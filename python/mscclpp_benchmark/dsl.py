# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from mscclpp import Algorithm

logger = logging.getLogger(__name__)
DEFAULT_DSL_TBG = (1, 2, 4, 8)
DEFAULT_DSL_TPB = (256, 512, 768, 1024)


def compile_dsl_algorithms(
    collective: str,
    tbg_values: Iterable[int],
    tpb_values: Iterable[int],
    *,
    rank: int,
    world_size: int,
    nranks_per_node: int,
    in_place: bool = True,
) -> list[Algorithm]:
    """Compile and return the multi-node DSL variants for ``collective``.

    Each (thread_block_group_size, num_threads_per_block) pair is a separate compiled plan, since
    DSL algorithms bake their launch geometry into the plan and ignore the nblocks/nthreads passed
    to execute(). The variant name must encode those values and the buffer mode: the plan cache
    key includes none of them, so variants sharing a name would silently resolve to the same
    cached plan.

    Only allgather_multi_nodes has an out-of-place form; the allreduce and reducescatter builders
    reduce into the input buffer, so nothing is compiled for them when ``in_place`` is False.
    """
    import mscclpp
    from mscclpp.language.utils import AlgoSpec

    if nranks_per_node <= 0 or world_size % nranks_per_node != 0:
        return []
    n_nodes = world_size // nranks_per_node
    if n_nodes < 2:
        return []

    if collective == "allreduce":
        if not in_place:
            return []
        from mscclpp.default_algos import allreduce_multi_nodes
        from mscclpp.language.collectives import AllReduce

        builder = allreduce_multi_nodes
        collective_op = AllReduce(world_size, 1, True)
        name_prefix = "dsl_allreduce"
    elif collective == "allgather":
        from mscclpp.default_algos import allgather_multi_nodes
        from mscclpp.language.collectives import AllGather

        builder = allgather_multi_nodes
        collective_op = AllGather(world_size, 1, in_place)
        name_prefix = "dsl_allgather"
    elif collective == "reducescatter":
        if not in_place:
            return []
        from mscclpp.default_algos import reducescatter_multi_nodes
        from mscclpp.language.collectives import ReduceScatter

        builder = reducescatter_multi_nodes
        collective_op = ReduceScatter(world_size, 1, True)
        name_prefix = "dsl_reducescatter"
    else:
        message = f"Unsupported collective for DSL algorithms: {collective}"
        logger.error(message)
        raise ValueError(message)

    algorithms: list[Algorithm] = []
    for tbg in tbg_values:
        for tpb in tpb_values:
            spec = AlgoSpec(
                name=f"{name_prefix}_{n_nodes}node_{tbg}TBG_{tpb}TPB_{'ip' if in_place else 'oop'}",
                collective=collective_op,
                nranks_per_node=nranks_per_node,
                world_size=world_size,
                in_place=in_place,
                instances=1,
                protocol="LL",
                auto_sync=False,
                num_threads_per_block=tpb,
                reuse_resources=True,
                use_double_scratch_buffer=True,
                min_message_size=1 << 10,
                max_message_size=8 << 20,
            )
            algorithms.append(mscclpp.compile(builder, spec, rank, thread_block_group_size=tbg))
    return algorithms
