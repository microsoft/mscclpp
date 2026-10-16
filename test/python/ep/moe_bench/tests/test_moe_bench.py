# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests and opt-in two-GPU mathematical reference coverage."""

import copy
import csv
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

if __package__:
    from ..compute_backends.torch_reference import ExpertMLP
    from ..config import SUPPORTED_LAYOUTS, load_config
    from ..memory import estimate_tensor_bytes
    from ..results import latency_stats, save_results, summarize, write_json
    from ..workload import make_expert_weights, make_input
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compute_backends.torch_reference import ExpertMLP
    from config import SUPPORTED_LAYOUTS, load_config
    from memory import estimate_tensor_bytes
    from results import latency_stats, save_results, summarize, write_json
    from workload import make_expert_weights, make_input


def tiny_config():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs.yaml",
        num_gpus=2,
        model="small",
        input_name="decode",
        routing="balanced",
        overrides={"warmup": 0, "iterations": 3},
    )
    config["model"]["intermediate_size"] = 32
    config["input"]["tokens_per_rank"] = 3
    return config


def timing(total, dispatch=10, mlp=60):
    return {"dispatch_us": dispatch, "mlp_us": mlp, "combine_us": total - dispatch - mlp, "total_us": total}


def test_summary_uses_maximum_per_iteration():
    rows = summarize([[timing(100), timing(300)], [timing(200), timing(100)]], tokens_per_rank=10)
    assert rows[0]["mean_latency_us"] == 250
    assert rows[0]["p50_latency_us"] == 250
    assert rows[0]["p95_latency_us"] == 295
    assert rows[0]["throughput_tokens_per_s"] == 80000
    assert len(rows) == 9
    assert rows[4]["mean_latency_us"] == 200
    assert rows[8]["mean_latency_us"] == 150
    assert rows[1]["phase"] == "dispatch"
    assert rows[1]["mean_latency_us"] == 10
    assert rows[2]["phase"] == "mlp"
    assert rows[2]["mean_latency_us"] == 60
    assert rows[3]["phase"] == "combine"
    assert rows[3]["mean_latency_us"] == 130
    assert rows[1]["throughput_tokens_per_s"] == ""


@pytest.mark.parametrize("values", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_timings(values):
    with pytest.raises(ValueError):
        latency_stats(values)


def test_mismatched_iterations():
    with pytest.raises(ValueError):
        summarize([[1, 2], [3]], 10)


@pytest.mark.parametrize("field", ["dispatch_us", "mlp_us", "combine_us", "total_us"])
def test_invalid_phase_sample(field):
    sample = timing(100)
    sample[field] = float("nan")
    with pytest.raises(ValueError):
        summarize([[sample]], 10)
    del sample[field]
    with pytest.raises(ValueError, match="Each iteration"):
        summarize([[sample]], 10)


def test_zero_work_phase():
    rows = summarize([[timing(100, mlp=0)]], 10)
    assert rows[2]["mean_latency_us"] == 0


def test_timing_events_materialized_before_measurement(monkeypatch):
    from ..worker import make_timing_events

    recorded = []

    class Stream:
        def synchronize(self):
            recorded.append("synchronize")

    stream = Stream()

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self, target):
            assert target is stream
            recorded.append(self)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    events = make_timing_events(3)
    assert len(events) == 3
    assert all(len(boundaries) == 4 for boundaries in events)
    assert recorded == [event for boundaries in events for event in boundaries] + ["synchronize"]


def test_result_files(tmp_path):
    config = tiny_config()
    write_json(tmp_path / "run.json", {"metadata": {}, "presets": {"model": "tiny"}, "status": "running"})
    save_results(
        tmp_path,
        [[timing(100), timing(110), timing(120)], [timing(120), timing(100), timing(130)]],
        config,
        {"compute": "reference"},
        {"assignments_per_expert": [3, 3], "rank_max_over_mean": 1.0},
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["iterations.csv", "run.json", "summary.csv"]
    with (tmp_path / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert rows[0]["scope"] == "global"
    with (tmp_path / "iterations.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == 6
    assert samples[-1] == {
        "iteration": "2",
        "rank": "1",
        "dispatch_us": "10",
        "mlp_us": "60",
        "combine_us": "60",
        "total_us": "130",
    }
    metadata = json.loads((tmp_path / "run.json").read_text())
    assert metadata["metadata"]["compute"] == "reference"
    assert metadata["status"] == "running"  # Only the launcher declares success.
    assert metadata["schema_version"] == 2
    assert "no synchronization" in metadata["measurement"]["phase_timing"]


def test_compute_valid_rows_only_and_distinct_weights():
    config = tiny_config()
    config["model"]["d_model"] = 16
    config["model"]["num_experts"] = 6
    counts = [2, 0, 1]
    compute = ExpertMLP(config, 0, counts, "cpu")
    tokens = torch.randn(3, 4, 16, dtype=torch.bfloat16)
    out = torch.full_like(tokens, float("nan"))
    compute(tokens, out)
    for expert, count in enumerate(counts):
        gate, up, down = compute.weights[expert]
        x = tokens[expert, :count]
        expected = (F.silu(x @ gate) * (x @ up)) @ down
        torch.testing.assert_close(out[expert, :count], expected, rtol=0, atol=0)
        assert torch.isnan(out[expert, count:]).all()
    assert not torch.equal(compute.weights[0][0], compute.weights[1][0])
    again = make_expert_weights(config["model"], 0, config["benchmark"]["seed"], "cpu")
    assert torch.equal(compute.weights[0][0], again[0])
    assert estimate_tensor_bytes(config, counts) >= sum(
        matrix.numel() * matrix.element_size() for weights in compute.weights for matrix in weights
    )


def test_input_reproducibility_without_global_rng_mutation():
    config = tiny_config()
    state = torch.random.get_rng_state()
    first = make_input(config, 0, "cpu")
    assert torch.equal(first, make_input(config, 0, "cpu"))
    assert not torch.equal(first, make_input(config, 1, "cpu"))
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "options,environment",
    [
        ([], {"num_gpus": 2, "communication_backend": "mscclpp", "compute_backend": "torch", "cuda_graph": False}),
        (
            ["--backend", "nccl", "--graph", "mlp"],
            {"num_gpus": 4, "communication_backend": "nccl", "compute_backend": "torch", "cuda_graph": "mlp"},
        ),
    ],
)
def test_launcher_records_failure_without_summary(tmp_path, monkeypatch, options, environment):
    import yaml
    from .. import run as launcher

    environment = {"mode": "latency", "layout": "expert_major", **environment}
    config = tiny_config()
    document = {
        "models": {"tiny": config["model"]},
        "inputs": {"tiny": config["input"]},
        "environment": {key: value for key, value in config["environment"].items() if key != "num_gpus"},
        "routing": {"balanced": config["routing"]},
        "benchmark": config["benchmark"],
    }
    config_path = tmp_path / "configs.yaml"
    config_path.write_text(yaml.safe_dump(document))
    output_dir = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(launcher.__file__).resolve()),
            "--config",
            str(config_path),
            "--model",
            "tiny",
            "--input",
            "tiny",
            "--routing",
            "balanced",
            "--output-dir",
            str(output_dir),
            *options,
        ],
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: environment["num_gpus"])

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(stdout="revision\n" if command[-1] == "HEAD" else "")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 1
    files = list(output_dir.glob("*/*"))
    assert len(files) == 1 and files[0].name == "run.json"
    run = json.loads(files[0].read_text())
    assert run["status"] == "failed"
    assert run["failure"]["type"] == "SystemExit"
    assert run["config"]["benchmark"]["output_dir"] == str(output_dir)
    assert run["config"]["environment"] == environment
    assert run["presets"] == {"model": "tiny", "input": "tiny", "routing": "balanced"}
    assert "--standalone" in run["invocation"]["worker_command"]
    assert f"--nproc-per-node={environment['num_gpus']}" in run["invocation"]["worker_command"]


def test_cli_environment_defaults():
    from ..run import parse_args

    args = parse_args(["--model", "small", "--input", "decode", "--routing", "balanced"])
    assert args.backend is args.graph is args.mode is args.layout is None
    assert not hasattr(args, "gpus")


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--backend", "unknown"),
        ("--mode", "auto"),
        ("--layout", "rank_major_topk_expanded"),
        ("--gpus", "8"),
        ("--gpus", "2"),
        ("--gpus", "4"),
        ("--gpus", "true"),
        ("--gpus", "2.0"),
        ("--graph", "full"),
        ("--graph", "false"),
        ("--graph", "eager"),
        ("--environment", "local"),
    ],
)
def test_cli_rejects_invalid_environment_options(flag, value):
    from ..run import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(["--model", "small", "--input", "decode", "--routing", "balanced", flag, value])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], {}),
        (["--backend", "nccl"], {"communication_backend": "nccl"}),
        (["--layout", "rank_major"], {"layout": "rank_major"}),
        (["--mode", "throughput", "--layout", "token_major"], {"mode": "throughput", "layout": "token_major"}),
        (["--graph", "none"], {"cuda_graph": False}),
        (["--graph", "mlp"], {"cuda_graph": "mlp"}),
        (
            ["--backend", "nccl", "--graph", "mlp"],
            {"communication_backend": "nccl", "cuda_graph": "mlp"},
        ),
    ],
)
@pytest.mark.parametrize("num_gpus", [2, 4])
@pytest.mark.parametrize("gloo_interface", [None, "eth0"])
def test_launcher_resolves_environment_and_worker_command(
    tmp_path, monkeypatch, options, expected, num_gpus, gloo_interface
):
    import yaml
    from .. import run as launcher

    document = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs.yaml").read_text())
    document["environment"]["cuda_graph"] = "mlp"
    config_path = tmp_path / "configs.yaml"
    config_path.write_text(yaml.safe_dump(document))
    output_dir = tmp_path / "runs"
    argv = [
        str(Path(launcher.__file__).resolve()),
        "--config",
        str(config_path),
        "--model",
        "scaled_small",
        "--input",
        "decode",
        "--routing",
        "balanced",
        "--warmup",
        "200",
        "--iterations",
        "1000",
        "--output-dir",
        str(output_dir),
        *options,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: num_gpus)
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("PET_LOCAL_ADDR", raising=False)
    if gloo_interface is not None:
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", gloo_interface)
    expected = {**document["environment"], **expected, "num_gpus": num_gpus}
    commands = []

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(stdout="revision\n" if command[-1] == "HEAD" else "")
        commands.append(command)
        assert kwargs["env"]["GLOO_SOCKET_IFNAME"] == (gloo_interface or "lo")
        assert kwargs["env"]["PATH"] == os.environ["PATH"]
        run_dir = Path(command[-1])
        metadata = json.loads((run_dir / "run.json").read_text())
        assert metadata["status"] == "running"
        assert metadata["config"]["environment"] == expected
        (run_dir / "summary.csv").write_text("")
        (run_dir / "iterations.csv").write_text("")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher.main()
    (run_path,) = output_dir.glob("*/run.json")
    metadata = json.loads(run_path.read_text())
    assert metadata["status"] == "completed"
    assert metadata["metadata"]["gloo_socket_ifname"] == (gloo_interface or "lo")
    assert os.environ.get("GLOO_SOCKET_IFNAME") == gloo_interface
    assert metadata["config"]["environment"] == expected
    assert metadata["config"]["model"] == document["models"]["scaled_small"]
    assert metadata["config"]["input"] == document["inputs"]["decode"]
    assert metadata["config"]["routing"] == document["routing"]["balanced"]
    assert metadata["config"]["benchmark"] == {
        "warmup": 200,
        "iterations": 1000,
        "seed": 42,
        "output_dir": str(output_dir),
    }
    assert metadata["presets"] == {"model": "scaled_small", "input": "decode", "routing": "balanced"}
    assert metadata["invocation"]["argv"] == [sys.executable, *argv]
    assert shlex.split(metadata["invocation"]["command"]) == [sys.executable, *argv]
    world = num_gpus
    assert commands == [
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--local-addr=127.0.0.1",
            "--nnodes=1",
            f"--nproc-per-node={world}",
            str(Path(launcher.__file__).with_name("worker.py").resolve()),
            "--run-dir",
            str(run_path.parent),
        ]
    ]
    assert shlex.split(metadata["invocation"]["worker_command"]) == commands[0]


@pytest.mark.parametrize("num_gpus", [0, 1, 3, 8])
def test_launcher_rejects_unsupported_visible_gpu_count(tmp_path, monkeypatch, num_gpus):
    from .. import run as launcher

    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--model", "small", "--input", "decode", "--routing", "balanced", "--output-dir", str(tmp_path)],
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: num_gpus)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not launch workers"))
    with pytest.raises(RuntimeError, match=f"uses all visible GPUs.*found {num_gpus}"):
        launcher.main()
    assert not list(tmp_path.iterdir())


def assert_idle_gpus():
    """Opt-in tests must not compete with other users' GPU workloads."""
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    print(f"GPU availability before test:\n{usage}", flush=True)
    assert not processes.strip(), "Shared GPUs occupied; refusing to run"
    for row in csv.reader(usage.splitlines()):
        _, used, free, utilization = map(int, row)
        assert used <= 64 and free > 1024 and utilization == 0, "Shared GPUs not idle; refusing to run"


@pytest.mark.parametrize("backend", ["mscclpp", "nccl"])
def test_distributed_reference(backend):
    variable = "MOE_BENCH_GPU_TESTS" if backend == "mscclpp" else "MOE_BENCH_NCCL_GPU_TESTS"
    if os.environ.get(variable) != "1":
        pytest.skip(f"Set {variable}=1 for the {backend} two-GPU test")
    assert_idle_gpus()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            backend,
        ],
        check=True,
        timeout=600,
    )


@torch.inference_mode()
def distributed_reference(backend):
    import torch.distributed as dist
    from worker import Pipeline, initialize

    rank, world, device = initialize()
    pipeline = None
    try:
        combinations = [(mode, layout) for mode, layouts in SUPPORTED_LAYOUTS[backend].items() for layout in layouts]
        for (mode, layout), graph_mode, routing in itertools.product(
            combinations,
            (False, "mlp"),
            (
                {"kind": "balanced"},
                {"kind": "uniform"},
                {"kind": "hot_experts", "expert_ids": [0, 1], "probability_mass": 0.999},
            ),
        ):
            config = copy.deepcopy(tiny_config())
            config["routing"] = routing
            config["environment"]["cuda_graph"] = graph_mode
            config["environment"]["communication_backend"] = backend
            config["environment"].update(mode=mode, layout=layout)
            if backend == "nccl" and mode == "throughput":
                config["input"]["tokens_per_rank"] = 4
            if rank == 0:
                print(f"Testing {backend}/{mode}/{layout}, graph={graph_mode}, routing={routing['kind']}", flush=True)
            pipeline = Pipeline(config, rank, device)
            assert (pipeline.mlp_graph is not None) == (graph_mode == "mlp")
            local_experts = config["model"]["num_experts"] // world
            for _ in range(2):
                events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                pipeline.step(events)
                events[-1].synchronize()
                phases = [events[i].elapsed_time(events[i + 1]) for i in range(3)]
                assert all(value >= 0 for value in phases)
                assert sum(phases) == pytest.approx(events[0].elapsed_time(events[-1]), abs=0.002, rel=1e-4)
                # Compute the selected experts locally without dispatch.
                routes = torch.zeros(
                    config["input"]["tokens_per_rank"],
                    config["model"]["top_k"],
                    config["model"]["d_model"],
                    device=device,
                    dtype=torch.bfloat16,
                )
                for expert in range(config["model"]["num_experts"]):
                    gate, up, down = make_expert_weights(config["model"], expert, config["benchmark"]["seed"], device)
                    selected = pipeline.ids == expert
                    token, slot = selected.nonzero(as_tuple=True)
                    x = pipeline.x[token]
                    routes[token, slot] = (F.silu(x @ gate) * (x @ up)) @ down
                expected = torch.zeros_like(pipeline.x, dtype=torch.float32)
                if backend == "nccl":
                    # NCCL-EP weights and sums BF16 expert outputs in FP32,
                    # without MSCCL++'s intermediate BF16 rank partials.
                    for slot in range(config["model"]["top_k"]):
                        expected.addcmul_(routes[:, slot].float(), pipeline.weights[:, slot].unsqueeze(-1))
                else:
                    for destination in range(world):
                        partial = torch.zeros_like(expected)
                        for slot in range(config["model"]["top_k"]):
                            selected = pipeline.ids[:, slot] // local_experts == destination
                            weight = pipeline.weights[:, slot].masked_fill(~selected, 0).unsqueeze(-1)
                            partial.addcmul_(routes[:, slot].float(), weight)
                        expected.add_(partial.to(torch.bfloat16).float())
                expected = expected.to(torch.bfloat16)
                valid = torch.tensor(int(torch.allclose(pipeline.output, expected, atol=0.025, rtol=0.025)))
                dist.all_reduce(valid, op=dist.ReduceOp.MIN)
                assert valid.item(), f"Distributed expert computation mismatch for {routing}"
                pipeline.x.mul_(0.5)
            dist.barrier()
            pipeline.close()
            pipeline = None
        if rank == 0:
            print("Distributed MoE reference comparisons passed", flush=True)
    finally:
        try:
            if pipeline is not None:
                pipeline.close()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    distributed_reference(sys.argv[1])
