# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

import pytest

from mscclpp.default_algos import allgather_multi_nodes, allreduce_multi_nodes
from mscclpp.language.collectives import AllGather, AllReduce
from mscclpp.language.utils import AlgoSpec


def _allgather_spec(*, in_place: bool = False) -> AlgoSpec:
    return AlgoSpec(
        name="test_allgather_multi_nodes",
        collective=AllGather(16, 1, in_place),
        nranks_per_node=8,
        world_size=16,
        in_place=in_place,
        instances=1,
        protocol="LL",
        auto_sync=False,
        num_threads_per_block=1024,
        reuse_resources=True,
        use_double_scratch_buffer=True,
        min_message_size=1 << 10,
        max_message_size=8 << 20,
        tags={"default": 1},
    )


def _allreduce_spec() -> AlgoSpec:
    return AlgoSpec(
        name="test_allreduce_multi_nodes",
        collective=AllReduce(16, 1, True),
        nranks_per_node=8,
        world_size=16,
        in_place=True,
        instances=1,
        protocol="LL",
        auto_sync=False,
        num_threads_per_block=1024,
        reuse_resources=True,
        use_double_scratch_buffer=True,
        min_message_size=1 << 10,
        max_message_size=8 << 20,
        tags={"default": 1},
    )


def test_allreduce_multi_nodes_builds_serializable_program() -> None:
    program = allreduce_multi_nodes(_allreduce_spec(), 1)
    program.post_process_operations()
    payload = json.loads(program.to_json())

    assert payload["name"] == "test_allreduce_multi_nodes"
    assert payload["collective"] == "allreduce"
    assert payload["inplace"] is True
    assert len(payload["gpus"]) == 16
    assert all(gpu["input_chunks"] == 16 for gpu in payload["gpus"])
    assert all(gpu["output_chunks"] == 16 for gpu in payload["gpus"])
    assert all(gpu["scratch_chunks"] == 36 for gpu in payload["gpus"])
    assert all(gpu["threadblocks"] for gpu in payload["gpus"])


@pytest.mark.parametrize("in_place", [False, True])
def test_allgather_multi_nodes_builds_serializable_program(
    in_place: bool,
) -> None:
    program = allgather_multi_nodes(_allgather_spec(in_place=in_place))
    program.post_process_operations()
    payload = json.loads(program.to_json())

    assert payload["name"] == "test_allgather_multi_nodes"
    assert payload["collective"] == "allgather"
    assert len(payload["gpus"]) == 16
    assert all(gpu["input_chunks"] == 1 for gpu in payload["gpus"])
    assert all(gpu["output_chunks"] == 16 for gpu in payload["gpus"])
    assert all(gpu["scratch_chunks"] == 16 for gpu in payload["gpus"])

    if not in_place:
        for rank, gpu in enumerate(payload["gpus"]):
            local_copy = gpu["threadblocks"][0]["ops"][0]
            assert local_copy["name"] == "copy"
            assert local_copy["src_buff"] == [{"type": "i", "index": 0, "size": 1}]
            assert local_copy["dst_buff"] == [{"type": "o", "index": rank, "size": 1}]


def test_allgather_multi_nodes_rejects_invalid_spec() -> None:
    spec = _allgather_spec()
    invalid_collective = AlgoSpec(
        **{
            **spec.__dict__,
            "collective": AllReduce(16, 1, False),
        }
    )

    with pytest.raises(ValueError, match="AllGather"):
        allgather_multi_nodes(invalid_collective)
    invalid_protocol = AlgoSpec(
        **{
            **spec.__dict__,
            "protocol": "Simple",
        }
    )
    with pytest.raises(ValueError, match="protocol='LL'"):
        allgather_multi_nodes(invalid_protocol)
    invalid_chunk_factor = AlgoSpec(
        **{
            **spec.__dict__,
            "collective": AllGather(16, 2, False),
        }
    )
    with pytest.raises(ValueError, match="chunk_factor=1"):
        allgather_multi_nodes(invalid_chunk_factor)
    inconsistent_in_place = AlgoSpec(
        **{
            **spec.__dict__,
            "in_place": True,
        }
    )
    with pytest.raises(ValueError, match="must match"):
        allgather_multi_nodes(inconsistent_in_place)
