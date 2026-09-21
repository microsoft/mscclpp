# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import builtins
import importlib.util
import logging
import sys
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from mscclpp_benchmark import dsl


@pytest.fixture
def dsl_runtime(monkeypatch):
    modules = {
        name: ModuleType(name)
        for name in (
            "mscclpp",
            "mscclpp.default_algos",
            "mscclpp.language",
            "mscclpp.language.collectives",
            "mscclpp.language.utils",
        )
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    runtime = modules["mscclpp"]
    runtime.compile = Mock(
        side_effect=lambda builder, spec, rank, **kwargs: SimpleNamespace(
            name=spec.name, collective=spec.collective.name, reset=Mock()
        )
    )
    for name, class_name in (
        ("allreduce", "AllReduce"),
        ("allgather", "AllGather"),
        ("reducescatter", "ReduceScatter"),
    ):
        setattr(modules["mscclpp.default_algos"], f"{name}_multi_nodes", Mock())
        setattr(
            modules["mscclpp.language.collectives"],
            class_name,
            Mock(return_value=SimpleNamespace(name=name)),
        )
    modules["mscclpp.language.utils"].AlgoSpec = SimpleNamespace
    return runtime


@pytest.mark.parametrize(
    "collective,in_place",
    [("allreduce", True), ("allgather", True), ("allgather", False), ("reducescatter", True)],
)
def test_compiles_dsl_variants(dsl_runtime, collective, in_place):
    algorithms = dsl.compile_dsl_algorithms(
        collective, (1, 4), (256, 512), rank=3, world_size=16, nranks_per_node=8, in_place=in_place
    )

    assert len(algorithms) == dsl_runtime.compile.call_count == 4
    for algorithm, call, (tbg, tpb) in zip(algorithms, dsl_runtime.compile.call_args_list, product((1, 4), (256, 512))):
        builder, spec, rank = call.args
        assert builder is getattr(sys.modules["mscclpp.default_algos"], f"{collective}_multi_nodes")
        assert rank == 3
        assert call.kwargs == ({} if collective == "allgather" else {"thread_block_group_size": tbg})
        assert algorithm.name == f"dsl_{collective}_2node_{tbg}TBG_{tpb}TPB_{'ip' if in_place else 'oop'}"
        assert algorithm.collective == collective
        assert vars(spec) == {
            "name": algorithm.name,
            "collective": spec.collective,
            "nranks_per_node": 8,
            "world_size": 16,
            "in_place": in_place,
            "instances": 1,
            "protocol": "LL",
            "auto_sync": False,
            "num_threads_per_block": tpb,
            "reuse_resources": True,
            "use_double_scratch_buffer": True,
            "min_message_size": 1024,
            "max_message_size": 8 * 1024 * 1024,
        }
    class_name = {"allreduce": "AllReduce", "allgather": "AllGather", "reducescatter": "ReduceScatter"}[collective]
    getattr(sys.modules["mscclpp.language.collectives"], class_name).assert_called_once_with(16, 1, in_place)


@pytest.mark.parametrize("world_size,nranks_per_node", [(16, 0), (16, -1), (15, 8), (8, 8)])
def test_skips_unsupported_topologies(dsl_runtime, world_size, nranks_per_node):
    assert (
        dsl.compile_dsl_algorithms(
            "allreduce", (1,), (256,), rank=0, world_size=world_size, nranks_per_node=nranks_per_node
        )
        == []
    )
    dsl_runtime.compile.assert_not_called()


@pytest.mark.parametrize("collective", ["allreduce", "reducescatter"])
def test_skips_unsupported_buffer_modes(dsl_runtime, collective):
    assert (
        dsl.compile_dsl_algorithms(collective, (1,), (256,), rank=0, world_size=16, nranks_per_node=8, in_place=False)
        == []
    )
    dsl_runtime.compile.assert_not_called()


def test_unsupported_collective_logs_and_raises(dsl_runtime, caplog):
    message = "Unsupported collective for DSL algorithms: unsupported"
    with pytest.raises(ValueError, match=message):
        dsl.compile_dsl_algorithms("unsupported", (1,), (256,), rank=0, world_size=16, nranks_per_node=8)
    assert (dsl.__name__, logging.ERROR, message) in caplog.record_tuples
    dsl_runtime.compile.assert_not_called()


def test_compilation_errors_propagate(dsl_runtime):
    dsl_runtime.compile.side_effect = RuntimeError("Compilation failed")
    with pytest.raises(RuntimeError, match="Compilation failed"):
        dsl.compile_dsl_algorithms("allreduce", (1,), (256,), rank=0, world_size=16, nranks_per_node=8)


def test_dsl_module_defers_mscclpp_imports(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "mscclpp" or name.startswith("mscclpp."):
            raise AssertionError(f"Unexpected eager import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("_test_dsl", dsl.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_DSL_TBG == (1, 2, 4, 8)
    assert module.DEFAULT_DSL_TPB == (256, 512, 768, 1024)


@pytest.mark.parametrize("enable_dsl", [False, True])
def test_comm_owns_registration_and_cleanup(monkeypatch, dsl_runtime, enable_dsl):
    gpu = ModuleType("mscclpp_benchmark.gpu")
    gpu.current_device = Mock()
    gpu.device_name = Mock()
    gpu.set_device = Mock()
    monkeypatch.setitem(sys.modules, gpu.__name__, gpu)
    spec = importlib.util.spec_from_file_location("_test_comm", Path(dsl.__file__).with_name("comm.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_ensure_device", Mock())
    monkeypatch.setattr(module, "_mscclpp", lambda: dsl_runtime)
    dsl_runtime.RawGpuBuffer = Mock()
    dsl_runtime.Executor = Mock()
    native = SimpleNamespace(collective="allgather", name="native", reset=Mock())
    builder = Mock()
    builder.return_value.build_default_algorithms.return_value = [native]
    dsl_runtime.ext = SimpleNamespace(AlgorithmCollectionBuilder=builder)
    group = SimpleNamespace(my_rank=3, nranks=16, nranks_per_node=8, communicator=object())

    comm = module.Comm(
        group,
        hardware_profile=module.HardwareProfile("H100", 16),
        collective="allgather",
        enable_dsl=enable_dsl,
        buffer_mode="out-of-place",
        dsl_tbg=(4,),
        dsl_tpb=(512,),
    )

    assert comm.algorithms["allgather"]["native"] is native
    expected_names = {"dsl_allgather_2node_4TBG_512TPB_oop"} if enable_dsl else set()
    assert comm.dsl_algorithms == expected_names
    assert set(comm.algorithms["allgather"]) == {"native"} | expected_names
    assert dsl_runtime.compile.call_count == int(enable_dsl)
    dsl_runtime.Executor.assert_called_once_with(group.communicator)
    algorithms = list(comm.algorithms["allgather"].values())
    comm.close()
    for algorithm in algorithms:
        algorithm.reset.assert_called_once_with()
    assert comm.algorithms == {}
    assert comm._executor is None
    assert comm._scratch_buffer is None
    builder.reset.assert_called_once_with()
