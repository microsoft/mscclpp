# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any
from unittest.mock import patch

import pytest

from mscclpp.ep import MoECommunicatorConfig, MoEMode
from mscclpp.ep.latency import LatencyContext
from mscclpp.ep.throughput import ThroughputContext


class _FakeCommGroup:
    my_rank = 0
    nranks = 4


def _config(**kwargs: Any) -> MoECommunicatorConfig:
    return MoECommunicatorConfig(
        comm=_FakeCommGroup(),
        num_experts=16,
        num_local_experts=4,
        hidden_size=4096,
        topk=2,
        max_tokens_per_rank=8,
        **kwargs,
    )


@patch("torch.cuda.current_device", return_value=0)
def test_latency_default_block_counts(_current_device: Any) -> None:
    context = LatencyContext(_config())

    assert (context.dispatch_blocks, context.combine_blocks) == (130, 128)


@patch("torch.cuda.current_device", return_value=0)
def test_latency_expands_single_block_count(_current_device: Any) -> None:
    context = LatencyContext(_config(num_blocks=66))

    assert (context.dispatch_blocks, context.combine_blocks) == (66, 64)


@patch("torch.cuda.current_device", return_value=0)
def test_latency_accepts_independent_block_counts(_current_device: Any) -> None:
    context = LatencyContext(_config(num_blocks=(130, 32)))

    assert (context.dispatch_blocks, context.combine_blocks) == (130, 32)


@pytest.mark.parametrize("num_blocks", [(130, 0), (130, 129)])
@patch("torch.cuda.current_device", return_value=0)
def test_latency_rejects_invalid_combine_count(_current_device: Any, num_blocks: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="combine block count must be between 1 and 128"):
        LatencyContext(_config(num_blocks=num_blocks))


@pytest.mark.parametrize("num_blocks", [(130,), (130, 128, 64)])
@patch("torch.cuda.current_device", return_value=0)
def test_latency_rejects_malformed_block_pair(_current_device: Any, num_blocks: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="must contain exactly"):
        LatencyContext(_config(num_blocks=num_blocks))


@patch("torch.cuda.current_device", return_value=0)
def test_throughput_defaults_to_equal_block_counts(_current_device: Any) -> None:
    context = ThroughputContext(_config(mode=MoEMode.THROUGHPUT))

    assert (context.dispatch_blocks, context.combine_blocks) == (20, 20)


@patch("torch.cuda.current_device", return_value=0)
def test_throughput_expands_single_block_count(_current_device: Any) -> None:
    context = ThroughputContext(_config(mode=MoEMode.THROUGHPUT, num_blocks=12))

    assert (context.dispatch_blocks, context.combine_blocks) == (12, 12)


@patch("torch.cuda.current_device", return_value=0)
def test_throughput_accepts_independent_block_counts(_current_device: Any) -> None:
    context = ThroughputContext(_config(mode=MoEMode.THROUGHPUT, num_blocks=(16, 8)))

    assert (context.dispatch_blocks, context.combine_blocks) == (16, 8)
