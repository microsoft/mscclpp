# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native layout support and layout-conversion coverage, without CUDA."""

import itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import pytest
import torch
import yaml

from ..communication_backends import create_backend
from ..memory import estimate_tensor_bytes
from ..config import SUPPORTED_LAYOUTS, load_config, validate_config
from ..layout import LayoutMLP, assignment_counts, local_route_ids
from .test_moe_bench import tiny_config


@pytest.mark.parametrize(
    "backend,mode,layout",
    list(
        itertools.product(("mscclpp", "nccl"), ("latency", "throughput"), ("expert_major", "rank_major", "token_major"))
    ),
)
def test_support_matrix(backend, mode, layout, monkeypatch):
    config = tiny_config()
    config["input"]["tokens_per_rank"] = 4
    config["environment"].update(communication_backend=backend, mode=mode, layout=layout)
    if layout in SUPPORTED_LAYOUTS[backend][mode]:
        validate_config(config)
    else:
        with pytest.raises(ValueError, match=f"{backend} mode={mode}.*layout={layout}"):
            validate_config(config)
        with pytest.raises(ValueError, match=f"{backend} mode={mode}.*layout={layout}"):
            create_backend(config, 0, None, None, None, None, None, None)


def test_old_yaml_gets_defaults_without_mutation():
    path = Path(__file__).resolve().parents[1] / "configs.yaml"
    document = yaml.safe_load(path.read_text())
    del document["environment"]["mode"], document["environment"]["layout"]
    with patch("builtins.open", mock_open(read_data=yaml.safe_dump(document))):
        config = load_config(path, num_gpus=4, model="small", input_name="decode", routing="balanced")
    assert config["environment"]["mode"] == "latency"
    assert config["environment"]["layout"] == "expert_major"
    assert "mode" not in document["environment"]
    config["environment"].pop("mode")
    config["environment"].pop("layout")
    validate_config(config)


def test_rank_zone_counts_mask_undefined_padding_metadata():
    backend = SimpleNamespace(
        route_ids=torch.tensor([[0, 1], [0, 0], [2, -1], [0, 0]]),
        route_id_base=0,
        route_valid_rows=torch.tensor([True, False, True, False]),
    )
    assert assignment_counts(local_route_ids(backend), 4).tolist() == [1, 1, 1, 0]


def test_nccl_throughput_transfer_alignment():
    config = tiny_config()
    config["environment"].update(communication_backend="nccl", mode="throughput", layout="token_major")
    with pytest.raises(ValueError, match="even tokens_per_rank"):
        validate_config(config)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("num_experts", 4096, "1024 local experts"),
        ("tokens_per_rank", 32769, "65536 rows"),
    ],
)
def test_mscclpp_throughput_capacity_errors(field, value, match):
    config = tiny_config()
    config["environment"].update(mode="throughput", layout="rank_major")
    config["input" if field == "tokens_per_rank" else "model"][field] = value
    with pytest.raises(ValueError, match=match):
        validate_config(config)


def test_native_payload_capacity_and_adapter_memory():
    config = tiny_config()
    baseline = estimate_tensor_bytes(config, [2, 0, 1, 0])
    config["environment"].update(mode="throughput", layout="rank_major")
    assert estimate_tensor_bytes(config, [2, 0, 1, 0]) > baseline + 65536 * (16384 + 768)
    config["input"]["tokens_per_rank"] = 32768
    config["model"]["d_model"] = 9216
    with pytest.raises(ValueError, match="1 GiB pool"):
        validate_config(config)


def test_initial_layout_failure_drains_native_collective(monkeypatch):
    from ..worker import Pipeline
    from .. import worker

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.counts = [1]
    pipeline.recv = torch.empty(1, 2, 4, dtype=torch.bfloat16)
    pipeline.expert_output = torch.ones_like(pipeline.recv)
    pipeline.mlp = Mock()
    pipeline.backend = SimpleNamespace(
        dispatch=Mock(return_value=(pipeline.recv, torch.tensor([2]))),
        combine=Mock(),
        combine_buffer=pipeline.expert_output,
    )
    monkeypatch.setattr(worker.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(worker.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        worker.dist, "all_gather_object", lambda values, value: values.__setitem__(slice(None), [value, None])
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with pytest.raises(RuntimeError, match="differ from fixed routing"):
        pipeline._initialize_compute(False)
    pipeline.backend.combine.assert_called_once()
    pipeline.mlp.assert_not_called()
    assert not pipeline.expert_output.count_nonzero()


def test_throughput_empty_native_pointer_contract(monkeypatch):
    # Regression for a genuinely empty receive rank: empty Torch tensors have
    # data_ptr()==0 even when their storage aliases the native registered pool.
    ep = pytest.importorskip("mscclpp.ep")
    from mscclpp.ep import throughput

    empty = torch.empty(0, 16, dtype=torch.bfloat16)
    cpp = SimpleNamespace(dispatch=Mock(), dispatch_output_buffer_ptr=Mock(return_value=1234), combine=Mock())
    context = SimpleNamespace(
        dispatch_blocks=20,
        output_layout=ep.DispatchLayout.TOKEN_MAJOR,
        group_size=2,
        rank=0,
        combine_blocks=20,
    )
    runtime = SimpleNamespace(
        context=context, cpp_runtime=cpp, _allocate_recv=Mock(return_value=empty), _validate_combine=Mock()
    )
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(throughput, "_stream_ptr", lambda: 0)
    send_head = torch.empty(3, 2, dtype=torch.int32)
    throughput.ThroughputRuntime._dispatch_throughput(
        runtime,
        torch.empty(3, 16),
        None,
        None,
        None,
        None,
        torch.empty(3, 2, dtype=torch.bool),
        None,
        0,
        torch.empty(2, 2, dtype=torch.int32),
        torch.empty(2, 20, dtype=torch.int32),
        1,
    )
    assert empty.data_ptr() == 0
    assert cpp.dispatch.call_args.args[0] == 1234
    handle = SimpleNamespace(_context=SimpleNamespace(recv_topk_weights=torch.empty(0, 2), send_head=send_head))
    throughput.ThroughputRuntime.combine.__wrapped__(runtime, empty, handle, out=None, stream=None)
    args = cpp.combine.call_args.args
    assert args[1] == args[3] == 0  # optional received/combined weights are both null


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("base", [0, 4])
def test_pack_reduce_multiple_sources_padding_empty_experts(empty, base):
    # Two source-rank zones. Row zero selects two local experts; padding payload
    # is NaN and must never enter compute. Expert 2 has no assignments.
    ids = torch.tensor([[0, 1], [1, -1], [-1, -1], [0, 3], [-1, -1], [3, -1]])
    if empty:
        ids.fill_(-1)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0], [0, 0], [0.2, 0.8], [0, 0], [0.6, 0]])
    tokens = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).bfloat16()
    tokens[0, 2] = tokens[1, 1] = float("nan")
    counts = assignment_counts(ids, 4).tolist()
    recv = torch.full((4, 6, 4), float("nan"), dtype=torch.bfloat16)
    out = torch.empty_like(recv)
    backend = SimpleNamespace(
        route_ids=ids + base,
        route_id_base=base,
        route_weights=weights,
        combine_buffer=torch.empty_like(tokens),
    )

    class MLP:
        metadata = {"mlp_timed_operations": "test expert scaling"}

        def __call__(self, packed, result):
            for expert, count in enumerate(counts):
                result[expert, :count] = packed[expert, :count] * (expert + 1)

    adapter = LayoutMLP(MLP(), counts, recv, out, backend)
    for _ in range(2):
        adapter(tokens, out)
        expected = torch.zeros(6, 4)
        for row in range(6):
            for slot in range(2):
                expert = ids[row, slot].item()
                if expert >= 0:
                    value = (tokens.reshape(6, 4)[row] * (expert + 1)).float()
                    expected[row] += value * weights[row, slot]
        torch.testing.assert_close(backend.combine_buffer.reshape(6, 4), expected.bfloat16(), rtol=0, atol=0)
        # Native row ordering may change on every dispatch, even for fixed routes.
        tokens = tokens.flip(0)
        ids = ids.reshape(2, 3, 2).flip(0).reshape(6, 2)
        weights = weights.reshape(2, 3, 2).flip(0).reshape(6, 2)
        backend.route_ids.copy_(ids + base)
        backend.route_weights.copy_(weights)
