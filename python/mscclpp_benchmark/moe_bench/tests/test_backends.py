# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only adapter contract tests; no native communication runtime is needed."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ..communication_backends import create_backend
from ..communication_backends import nccl as nccl_adapter
from .test_moe_bench import tiny_config


def buffers():
    x = torch.empty(3, 16, dtype=torch.bfloat16)
    ids = torch.zeros(3, 2, dtype=torch.int64)
    weights = torch.full((3, 2), 0.5, dtype=torch.float32)
    recv = torch.empty(4, 6, 16, dtype=torch.bfloat16)
    return x, ids, weights, recv, torch.empty_like(recv), torch.empty_like(x)


@pytest.fixture
def fake_nccl(monkeypatch):
    monkeypatch.setitem(sys.modules, "mscclpp", None)
    destroyed = []
    comm = SimpleNamespace(destroy=Mock(side_effect=lambda: destroyed.append("comm")))
    handle = SimpleNamespace(
        destroy=Mock(side_effect=lambda: destroyed.append("handle")), dispatch=Mock(), combine=Mock()
    )
    group = SimpleNamespace(
        destroy=Mock(side_effect=lambda: destroyed.append("group")), create_handle=Mock(return_value=handle)
    )
    core = ModuleType("nccl.core")
    core.get_unique_id = Mock(return_value="unique-id")
    core.Communicator = SimpleNamespace(init=Mock(return_value=comm))
    core.get_version = lambda: SimpleNamespace(libnccl=SimpleNamespace(version="2.30.7"))
    ep = ModuleType("nccl.ep")
    ep.Algorithm = SimpleNamespace(LOW_LATENCY="low-latency", HIGH_THROUGHPUT="high-throughput")
    ep.Layout = SimpleNamespace(EXPERT_MAJOR="expert-major", RANK_MAJOR="rank-major", FLAT="flat")
    ep.ExpertIdKind = SimpleNamespace(LOCAL="local")
    ep.Group = SimpleNamespace(create=Mock(return_value=group))
    ep.Tensor = Mock(side_effect=lambda buffer: SimpleNamespace(buffer=buffer))
    for name in (
        "GroupConfig",
        "AllocConfig",
        "HandleConfig",
        "DispatchConfig",
        "CombineConfig",
        "DispatchInputs",
        "DispatchOutputs",
        "CombineInputs",
        "CombineOutputs",
        "LayoutInfo",
    ):
        setattr(ep, name, Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)))
    namespace = ModuleType("nccl")
    namespace.core, namespace.ep = core, ep
    for name, module in (("nccl", namespace), ("nccl.core", core), ("nccl.ep", ep)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(nccl_adapter, "version", lambda name: {"nccl4py": "0.6.0", "nccl-extensions": "0.1.0"}[name])
    stream = SimpleNamespace(cuda_stream=123)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: destroyed.append("synchronize"))

    def broadcast(objects, src):
        assert src == 0
        assert objects[0] in (None, "unique-id")
        objects[0] = "unique-id"

    monkeypatch.setattr(nccl_adapter.dist, "broadcast_object_list", broadcast)
    return SimpleNamespace(core=core, ep=ep, group=group, handle=handle, destroyed=destroyed, stream=stream)


@pytest.mark.parametrize(
    "mode,layout", [("latency", "rank_major"), ("throughput", "expert_major"), ("throughput", "token_major")]
)
def test_nccl_native_layout_wiring(fake_nccl, mode, layout):
    config = tiny_config()
    config["environment"].update(communication_backend="nccl", mode=mode, layout=layout)
    config["input"]["tokens_per_rank"] = 4
    x, ids, weights, recv, expert_output, output = buffers()

    def create_handle(*args, layout_info, **kwargs):
        if layout_info is not None:
            layout_info.expert_counters.buffer.copy_(torch.tensor([2, 0, 1, 0]))
            layout_info.recv_total_counter.buffer.fill_(3)
            if layout_info.expert_offsets is not None:
                layout_info.expert_offsets.buffer.copy_(torch.tensor([0, 2, 2, 3]))
        return fake_nccl.handle

    fake_nccl.group.create_handle.side_effect = create_handle
    adapter = create_backend(config, 0, x, ids, weights, recv, expert_output, output)
    assert adapter.adapt_layout
    assert adapter.combine_inputs.tokens.buffer is adapter.combine_buffer
    assert adapter.combine_outputs.topk_weights is None
    assert adapter.dispatch_inputs.topk_weights.buffer is weights
    assert (
        fake_nccl.group.create_handle.call_args.args[0]
        == {"rank_major": "rank-major", "expert_major": "expert-major", "token_major": "flat"}[layout]
    )
    if mode == "throughput":
        capacity = 16 if layout == "expert_major" else 8
        assert adapter.recv.shape == (capacity, 4096)
        assert adapter.layout is None
        assert fake_nccl.ep.Group.create.call_args.args[1].max_recv_tokens_per_rank == capacity
        if layout == "expert_major":
            assert adapter.route_ids[:4, 0].tolist() == [0, 0, 2, -1]
            assert adapter.dispatch_outputs.topk_weights.buffer.shape == (capacity,)
        else:
            assert adapter.route_valid_rows.tolist() == [True] * 3 + [False] * 5
    else:
        assert adapter.recv.shape == (2, 4, 4096)
        assert adapter.layout.recv_topk_idx_kind == "local"
    adapter.close()


def test_nccl_missing_throughput_api_fails_before_collectives(fake_nccl):
    config = tiny_config()
    config["environment"].update(communication_backend="nccl", mode="throughput", layout="token_major")
    del fake_nccl.ep.Algorithm.HIGH_THROUGHPUT
    with pytest.raises(RuntimeError, match="does not expose a throughput algorithm"):
        nccl_adapter.preflight(config)
    fake_nccl.core.get_unique_id.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_nccl_fixed_buffer_wiring(fake_nccl, rank):
    config = tiny_config()
    config["environment"]["communication_backend"] = "nccl"
    x, ids, weights, recv, expert_output, output = buffers()
    adapter = create_backend(config, rank, x, ids, weights, recv, expert_output, output)
    fake_nccl.core.Communicator.init.assert_called_once_with(nranks=2, rank=rank, unique_id="unique-id")
    assert fake_nccl.core.get_unique_id.call_count == (1 if rank == 0 else 0)
    group_config = fake_nccl.ep.Group.create.call_args.args[1]
    assert group_config.algorithm == "low-latency"
    assert group_config.num_experts == 8
    assert group_config.max_dispatch_tokens_per_rank == 3
    assert group_config.max_recv_tokens_per_rank == 6
    assert group_config.max_token_bytes == 8192
    args, kwargs = fake_nccl.group.create_handle.call_args
    assert args[0] == "expert-major"
    assert args[1].buffer is ids and ids.dtype == torch.int64
    assert kwargs["stream"] == 123 and kwargs["layout_info"] is None
    for stream_ptr in (234, 345):
        fake_nccl.stream.cuda_stream = stream_ptr
        tokens, counts = adapter.dispatch()
        adapter.combine()
        assert tokens is recv and counts.shape == (4,) and counts.dtype == torch.int32
        args, kwargs = fake_nccl.handle.dispatch.call_args
        assert args[0].tokens.buffer is x
        assert args[1].tokens.buffer is recv
        assert kwargs["layout_info"].expert_counters.buffer is counts
        assert kwargs["stream"] == stream_ptr
        args, kwargs = fake_nccl.handle.combine.call_args
        assert args[0].tokens.buffer is expert_output
        assert args[0].tokens.buffer is not recv
        assert args[1].tokens.buffer is output
        assert args[1].topk_weights.buffer is weights
        assert kwargs["stream"] == stream_ptr
    assert adapter.metadata["combine_mode"] == "weighted_topk"
    assert adapter.metadata["nccl_ep_version"] == "0.1.0"
    assert "mscclpp_version" not in adapter.metadata
    adapter.close()
    adapter.close()
    assert fake_nccl.destroyed == ["synchronize", "handle", "group", "comm"]


@pytest.mark.parametrize("failure", ["group", "handle", "descriptor", "teardown"])
def test_nccl_resource_cleanup_on_failure(fake_nccl, failure):
    failing = {
        "group": fake_nccl.ep.Group.create,
        "handle": fake_nccl.group.create_handle,
        "descriptor": fake_nccl.ep.DispatchInputs,
        "teardown": fake_nccl.handle.destroy,
    }[failure]
    failing.side_effect = RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        adapter = nccl_adapter.NcclBackend(tiny_config(), 0, *buffers())
        adapter.close()
    expected = {
        "group": ["comm"],
        "handle": ["group", "comm"],
        "descriptor": ["synchronize", "handle", "group", "comm"],
        "teardown": ["synchronize", "group", "comm"],
    }
    assert fake_nccl.destroyed == expected[failure]


def test_nccl_missing_dependency_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "nccl", None)
    monkeypatch.delitem(sys.modules, "nccl.core", raising=False)
    monkeypatch.delitem(sys.modules, "nccl.ep", raising=False)
    with pytest.raises(RuntimeError, match=r"nccl-extensions\[cu13\]") as error:
        nccl_adapter.NcclBackend(tiny_config(), 0, *buffers())
    assert isinstance(error.value.__cause__, ImportError)


def test_imports_are_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.path.insert(0, sys.argv[1])
class BlockOptionalRuntimes:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('mscclpp', 'nccl'):
            raise AssertionError('Eager optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptionalRuntimes())
from moe_bench import worker
from moe_bench.communication_backends import nccl, mscclpp
""",
            str(Path(__file__).resolve().parents[2]),
        ],
        check=True,
        timeout=30,
    )


def test_unknown_backend_is_not_substituted():
    config = tiny_config()
    config["environment"]["communication_backend"] = "absent"
    with pytest.raises(ValueError, match="Unsupported communication backend"):
        create_backend(config, 0, *buffers())


def test_invalid_multinode_launch_rejected_before_cuda(monkeypatch):
    from ..worker import initialize

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("LOCAL_RANK", "0")
    set_device = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    with pytest.raises(ValueError, match="Invalid torchrun"):
        initialize()
    set_device.assert_not_called()


@pytest.fixture
def fake_mscclpp(monkeypatch):
    from ..communication_backends import mscclpp as adapter_module

    native = ModuleType("mscclpp")
    native.__version__ = "test-version"
    native.CommGroup = Mock(return_value=SimpleNamespace(nranks_per_ipc_domain=2))
    ep = ModuleType("mscclpp.ep")
    native.ep = ep
    ep.MoEMode = SimpleNamespace(LATENCY="latency")
    ep.DispatchLayout = SimpleNamespace(EXPERT_MAJOR="expert-major")
    ep.CombineMode = SimpleNamespace(RANK_LOCAL_REDUCE="rank-local-reduce")
    tensors = buffers()
    x, ids, weights, recv, expert_output, output = tensors
    counts = torch.tensor([1, 2, 3, 0])
    dispatched = SimpleNamespace(tokens=recv, layout=SimpleNamespace(num_tokens_per_expert=counts))
    comm = SimpleNamespace(
        is_available=lambda: True,
        is_internode=lambda: False,
        dispatch=Mock(return_value=(dispatched, "handle")),
        combine=Mock(),
    )
    ep.MoECommunicator = Mock(return_value=comm)
    monkeypatch.setitem(sys.modules, "mscclpp", native)
    monkeypatch.setitem(sys.modules, "mscclpp.ep", ep)
    monkeypatch.setitem(sys.modules, "nccl", None)
    monkeypatch.setattr(adapter_module.dist.distributed_c10d, "_get_default_group", lambda: "gloo")
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    return SimpleNamespace(native=native, ep=ep, tensors=tensors, counts=counts, comm=comm)


@pytest.mark.parametrize("world,rank,internode", [(2, 0, False), (8, 5, True)])
def test_mscclpp_adapter_preserves_contract(fake_mscclpp, world, rank, internode):
    native, ep, tensors, counts, comm = (
        fake_mscclpp.native,
        fake_mscclpp.ep,
        fake_mscclpp.tensors,
        fake_mscclpp.counts,
        fake_mscclpp.comm,
    )
    x, ids, weights, recv, expert_output, output = tensors
    native.CommGroup.return_value.nranks_per_ipc_domain = world
    comm.is_internode = lambda: internode
    config = tiny_config()
    config["environment"]["num_gpus"] = world
    adapter = create_backend(config, rank, *tensors)
    tokens, actual_counts = adapter.dispatch()
    adapter.combine()
    assert tokens is recv and actual_counts is counts
    comm.dispatch.assert_called_once_with(x, ids, weights, output_buffer=recv)
    comm.combine.assert_called_once_with(expert_output, "handle", out=output)
    assert ep.MoECommunicator.call_args.kwargs["combine_mode"] == "rank-local-reduce"
    assert adapter.metadata["mscclpp_version"] == "test-version"
    assert adapter.metadata["ipc_domain_ranks"] == world
    assert adapter.metadata["communication_transport"] == ("cuda_ipc_mnnvl" if internode else "cuda_ipc")
    assert adapter.route_id_base == rank * (8 // world)
    assert ep.MoECommunicator.call_args.kwargs["num_local_experts"] == 8 // world
    assert "nccl_ep_version" not in adapter.metadata
    adapter.close()
    adapter.close()
    torch.cuda.synchronize.assert_called_once_with(x.device)


def test_split_ipc_domain_rejected_before_native_runtime(fake_mscclpp):
    config = tiny_config()
    config["environment"]["num_gpus"] = 8
    with pytest.raises(RuntimeError, match="domain has 2 of 8 ranks"):
        create_backend(config, 0, *fake_mscclpp.tensors)
    fake_mscclpp.ep.MoECommunicator.assert_not_called()


def test_forced_partial_ipc_domain_rejected(fake_mscclpp, monkeypatch):
    monkeypatch.setenv("MSCCLPP_EP_IPC_DOMAIN_SIZE", "1")
    with pytest.raises(RuntimeError, match="Unset MSCCLPP_EP_IPC_DOMAIN_SIZE"):
        create_backend(tiny_config(), 0, *fake_mscclpp.tensors)
    fake_mscclpp.ep.MoECommunicator.assert_not_called()


def test_multinode_worker_uses_local_device_and_global_rank(monkeypatch):
    from .. import worker

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    set_device = Mock()
    initialize = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(worker.dist, "init_process_group", initialize)
    monkeypatch.setattr(worker.dist, "get_rank", lambda: 5)
    monkeypatch.setattr(worker.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch, "set_num_threads", Mock())
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)
    assert worker.initialize() == (5, 8, torch.device("cuda", 1))
    set_device.assert_called_once_with(1)
    assert initialize.call_args.args == ("gloo",)


@pytest.mark.parametrize(
    "local_rank,local_world,world,visible",
    [
        (-1, 4, 8, 4),
        (4, 4, 8, 4),
        (0, 0, 8, 4),
        (0, 4, 0, 4),
        (0, 4, -8, 4),
        (0, 4, 2, 4),
        (0, 4, 6, 4),
        (0, 4, 8, 2),
    ],
)
def test_invalid_worker_rank_layout_fails_before_cuda(monkeypatch, local_rank, local_world, world, visible):
    from .. import worker

    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world))
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible)
    set_device = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    with pytest.raises(ValueError):
        worker.initialize()
    set_device.assert_not_called()
