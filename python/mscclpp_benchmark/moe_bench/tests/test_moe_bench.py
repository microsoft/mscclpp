# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for configuration, results, and launching."""

import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

if __package__:
    from ..config import load_config
    from ..results import latency_stats, save_results, summarize, write_json
    from ..workload import make_input
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config
    from results import latency_stats, save_results, summarize, write_json
    from workload import make_input


def tiny_config():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs.yaml",
        num_gpus=2,
        model="small",
        input_name="decode",
        routing="balanced",
        overrides={"warmup": 0, "iterations": 3},
    )
    config["model"]["intermediate_size"] = 128
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


class FakeEvent:
    def __init__(self, time):
        self.time = time

    def elapsed_time(self, end):
        return end.time - self.time


def layer_pipeline(layer_graph):
    from ..worker import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.backend = SimpleNamespace(dispatch=Mock(), combine=Mock())
    pipeline.mlp = Mock()
    pipeline.mlp_graph = None
    pipeline.layer_graph = layer_graph
    pipeline.phase_events = [FakeEvent(time) for time in (1, 2, 4, 7)]
    pipeline.output, pipeline.expert_output = "output", "expert_output"
    return pipeline


def test_layer_graph_step_replays_without_eager_collectives():
    graph = SimpleNamespace(replay=Mock())
    pipeline = layer_pipeline(graph)
    outer = [FakeEvent(time) for time in (0, 10, 20, 30)]
    for _ in range(3):
        assert pipeline.step(outer) == "output"
    assert graph.replay.call_count == 3
    pipeline.backend.dispatch.assert_not_called()
    pipeline.backend.combine.assert_not_called()
    pipeline.mlp.assert_not_called()
    # Phases come from the events captured inside the graph, not the eager events.
    assert pipeline.read_latencies(outer) == {
        "dispatch_us": 1000,
        "mlp_us": 2000,
        "combine_us": 3000,
        "total_us": 6000,
    }


def test_eager_step_uses_iteration_events(monkeypatch):
    pipeline = layer_pipeline(None)
    pipeline.backend.dispatch.return_value = ("tokens", None)
    stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    recorded = []
    events = [
        SimpleNamespace(record=lambda target, index=index: recorded.append((index, target))) for index in range(4)
    ]
    assert pipeline.step(events) == "output"
    assert recorded == [(index, stream) for index in range(4)]
    pipeline.backend.dispatch.assert_called_once_with()
    pipeline.mlp.assert_called_once_with("tokens", "expert_output")
    pipeline.backend.combine.assert_called_once_with()
    timed = [FakeEvent(time) for time in (0, 5, 6, 9)]
    assert pipeline.read_latencies(timed)["total_us"] == 9000


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
        (
            [],
            {
                "num_gpus": 2,
                "communication_backend": "mscclpp",
                "compute_backend": "cutlass_flashinfer",
                "cuda_graph": False,
            },
        ),
        (
            ["--backend", "nccl", "--graph", "mlp"],
            {
                "num_gpus": 4,
                "communication_backend": "nccl",
                "compute_backend": "cutlass_flashinfer",
                "cuda_graph": "mlp",
            },
        ),
    ],
)
def test_launcher_records_failure_without_summary(tmp_path, monkeypatch, options, environment):
    import yaml
    from .. import run as launcher
    from ..compute_backends import flashinfer as compute_flashinfer

    monkeypatch.setattr(compute_flashinfer, "preflight", lambda config, device: None)

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
        (["--graph", "layer"], {"cuda_graph": "layer"}),
        (
            ["--mode", "throughput", "--layout", "rank_major", "--graph", "layer"],
            {"mode": "throughput", "layout": "rank_major", "cuda_graph": "layer"},
        ),
        (
            ["--backend", "nccl", "--graph", "layer"],
            {"communication_backend": "nccl", "cuda_graph": "layer"},
        ),
        (
            ["--backend", "nccl", "--graph", "mlp"],
            {"communication_backend": "nccl", "cuda_graph": "mlp"},
        ),
    ],
)
@pytest.mark.parametrize("num_gpus", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("gloo_interface", [None, "eth0"])
def test_launcher_resolves_environment_and_worker_command(
    tmp_path, monkeypatch, options, expected, num_gpus, gloo_interface
):
    import yaml
    from .. import run as launcher
    from ..compute_backends import flashinfer as compute_flashinfer

    monkeypatch.setattr(compute_flashinfer, "preflight", lambda config, device: None)

    document = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs.yaml").read_text())
    document["environment"]["cuda_graph"] = "mlp"
    document["models"]["scaled_small"]["num_experts"] = 2 * num_gpus
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


def test_launcher_rejects_no_visible_gpus(tmp_path, monkeypatch):
    from .. import run as launcher

    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--model", "small", "--input", "decode", "--routing", "balanced", "--output-dir", str(tmp_path)],
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not launch workers"))
    with pytest.raises(RuntimeError, match="at least one visible GPU.*found none"):
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


def test_elementwise_check_catches_one_bad_token_at_prefill_scale():
    from ..reference import compare_outputs

    generator = torch.Generator().manual_seed(0)
    expected = torch.randn(4096, 1024, generator=generator)
    good = expected * (1 + 0.002 * torch.randn(expected.shape, generator=generator))
    stats, first = compare_outputs(good, expected)
    assert stats["mismatched_elements"] == stats["non_finite_elements"] == 0 and first is None
    assert 0 < stats["max_tolerance_fraction"] < 1
    bad = good.clone()
    bad[7] = torch.randn(1024, generator=generator)
    stats, first = compare_outputs(bad, expected)
    # One wrong token barely moves the aggregate error, but every element is checked.
    assert stats["relative_l2"] < 0.03
    assert stats["mismatched_elements"] > 900 and first["token"] == 7
    bad[0, 0] = float("nan")
    stats, _ = compare_outputs(bad, expected)
    assert stats["non_finite_elements"] == 1 and stats["relative_l2"] is None


@pytest.mark.parametrize("failing_rank", [None, 1])
def test_check_output_agrees_across_ranks(monkeypatch, failing_rank):
    from .. import reference, worker

    expected = torch.ones(4, 8)

    def gather(values, value):
        for rank in range(2):
            stats, first = reference.compare_outputs(
                expected + (1 if rank == failing_rank else 0) * (torch.arange(32).reshape(4, 8) == 9), expected
            )
            values[rank] = {"rank": rank, **stats, "first_mismatch": first}

    monkeypatch.setattr(worker.dist, "all_gather_object", gather)
    if failing_rank is None:
        summary = worker.check_output(expected, expected, 0, 2, 0.05, 0.01, "initial input")
        assert summary["total_elements"] == 64 and summary["mismatched_elements"] == 0
    else:
        with pytest.raises(
            RuntimeError, match=r"\(changed input\).* 1/64 elements .* rank 1: \{'token': 1, 'column': 1"
        ):
            worker.check_output(expected, expected, 0, 2, 0.05, 0.01, "changed input")


def test_check_tolerance_options():
    from ..run import parse_args

    base = ["--model", "small", "--input", "decode", "--routing", "balanced"]
    args = parse_args(base + ["--check", "--rtol", "0.1", "--atol", "0"])
    assert (args.rtol, args.atol) == (0.1, 0.0)
    for extra in (["--rtol", "0.1"], ["--check", "--atol", "-1"], ["--check", "--rtol", "nan"]):
        with pytest.raises(SystemExit):
            parse_args(base + extra)


def test_check_before_timing_uses_changed_input_and_restores_it(monkeypatch):
    from .. import reference, worker

    x = torch.arange(6.0).reshape(2, 3)
    original = x.clone()
    seen = []

    def step():
        pipeline.output = x * 2

    pipeline = SimpleNamespace(x=x, ids="ids", weights="weights", step=step, output=None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(reference, "reference_output", lambda config, x, ids, weights, magnitude: (x * 2, x.abs()))

    def check(actual, expected, rank, world, rtol, atol, label, magnitude):
        seen.append((label, torch.equal(actual, expected), x.clone()))
        return {"label": label}

    monkeypatch.setattr(worker, "check_output", check)
    expected, checks = worker.check_before_timing(pipeline, {}, 0, 1, 0.05, 0.01)
    assert [label for label, _, _ in seen] == ["initial input", "changed input"]
    assert all(matched for _, matched, _ in seen)
    assert torch.equal(seen[0][2], original) and not torch.equal(seen[1][2], original)
    assert torch.equal(x, original) and torch.equal(expected[0], original * 2)
    assert torch.equal(expected[1], original.abs())
    assert set(checks) == {"initial", "changed_input"}


def test_worker_failure_is_recorded_for_launcher(tmp_path, monkeypatch):
    from .. import run as launcher
    from .. import worker

    write_json(tmp_path / "run.json", {"config": {}, "check_tolerance": {"rtol": 0.1, "atol": None}})
    received = {}

    def fail(config, run_dir, check, rtol, atol):
        received.update(check=check, rtol=rtol, atol=atol)
        raise RuntimeError("element mismatch on rank 1")

    monkeypatch.setattr(worker, "run", fail)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(sys, "argv", ["worker.py", "--run-dir", str(tmp_path), "--check"])
    with pytest.raises(RuntimeError, match="element mismatch"):
        worker.main()
    assert received == {"check": True, "rtol": 0.1, "atol": None}
    write_json(tmp_path / "worker-failure-rank3.json", {"rank": 3, "type": "ValueError", "message": "later"})
    assert launcher.read_worker_failure(tmp_path) == {
        "rank": 1,
        "type": "RuntimeError",
        "message": "element mismatch on rank 1",
        "failed_ranks": [1, 3],
    }
    assert launcher.read_worker_failure(tmp_path / "missing") is None


def test_identity_encoding_is_exact_in_bf16():
    from .test_exact_communication import IdentityExperts, encoded_rows

    ids = torch.arange(32)
    rows = encoded_rows(ids, 64, "cpu")
    assert torch.equal(rows.float(), (8 * ids[:, None] + torch.arange(64) % 8).float())
    # Half-weighted BF16 partials and their sum stay exact.
    assert torch.equal((rows.float() * 0.5).bfloat16().float() * 2, rows.float())
    out = torch.empty_like(rows)
    assert IdentityExperts()(rows, out) is out and torch.equal(out, rows)


def test_rounding_slack_covers_cancelling_bf16_partials():
    from ..reference import ROUNDING_SLACK, compare_outputs

    # Two destination partials of about 3 nearly cancel; a one-ULP (2**-6) difference
    # in one BF16 partial is large relative to the result but small relative to the terms.
    partials = torch.tensor([[3.0, -2.90625]])
    expected = partials.sum(dim=1, keepdim=True)
    actual = expected + 2.0**-6
    magnitude = partials.abs().sum(dim=1, keepdim=True)
    assert compare_outputs(actual, expected)[0]["mismatched_elements"] == 1
    stats, _ = compare_outputs(actual, expected, magnitude=magnitude)
    assert stats["mismatched_elements"] == 0 and stats["max_tolerance_fraction"] < 1
    assert ROUNDING_SLACK * magnitude.item() >= 2.0**-6
    # A wrong result of the terms' own size still fails with the slack.
    stats, _ = compare_outputs(expected + 1.0, expected, magnitude=magnitude)
    assert stats["mismatched_elements"] == 1


def test_node_locality_topology_is_filled_and_verified(tmp_path, monkeypatch):
    from .. import run as launcher
    from .. import worker

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    argv = ["run.py", "--model", "small", "--input", "decode", "--routing", "cross_node_heavy"]
    monkeypatch.setattr(sys, "argv", argv + ["--output-dir", str(tmp_path)])
    # One node: the launcher fills gpus_per_node=4, so there is no remote node.
    with pytest.raises(ValueError, match="more than one node"):
        launcher.main()
    assert not list(tmp_path.iterdir())
    config = tiny_config()
    config["environment"]["num_gpus"] = 8
    config["routing"] = {"kind": "locality", "scope": "node", "local_fraction": 0.9, "gpus_per_node": 4}
    monkeypatch.setattr(worker.dist, "get_world_size", lambda: 8)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    with pytest.raises(ValueError, match="gpus_per_node=4 does not match the 8 GPUs launched per node"):
        worker.Pipeline(config, 0, "cpu")
