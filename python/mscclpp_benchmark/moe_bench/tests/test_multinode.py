# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only node discovery, launch, cancellation, and result collection tests."""

import copy
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .. import multinode
from .. import run as launcher


def nodes(count=2):
    return [
        dict(
            ip=f"10.0.0.{rank + 1}",
            target=f"tester@10.0.0.{rank + 1}",
            hostname=f"node-{rank}",
            root=f"/deploy/node-{rank}/bundle",
            python=f"/deploy/node-{rank}/bundle/.venv/bin/python",
            interface=f"eth{rank}",
            num_gpus=4,
            torch="test-torch",
            cuda="13.0",
            mscclpp="test-mscclpp",
            bundle_id=f"bundle-{rank}",
            native_wheel_sha256="same-native-wheel",
            cuda_visible_devices=None,
        )
        for rank in range(count)
    ]


@pytest.mark.parametrize("value", ["", "host", "10.0.0.1,", "10.0.0.1,10.0.0.1", "::1"])
def test_invalid_nodes(value):
    with pytest.raises(ValueError):
        multinode.parse_nodes(value)


def test_parse_nodes_preserves_rank_order():
    assert multinode.parse_nodes(" 10.0.0.2,10.0.0.1 ") == ["10.0.0.2", "10.0.0.1"]


def mock_discovery(monkeypatch, discovered):
    replies = []
    for node in discovered:
        replies.extend(
            [
                SimpleNamespace(stdout=json.dumps({key: node[key] for key in ("root", "python")})),
                SimpleNamespace(
                    stdout=json.dumps(
                        {key: value for key, value in node.items() if key not in ("root", "python", "ip", "target")}
                    )
                ),
            ]
        )
    ssh = Mock(side_effect=replies)
    monkeypatch.setattr(multinode, "ssh", ssh)
    return ssh


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_discovery_accepts_n_nodes(monkeypatch, count):
    expected = nodes(count)
    ssh = mock_discovery(monkeypatch, expected)
    actual = multinode.discover_nodes([node["ip"] for node in expected], "tester", "/deploy")
    assert actual == expected
    assert ssh.call_count == 2 * count
    for call in ssh.call_args_list:
        assert call.kwargs == dict(capture_output=True, text=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_gpus", 2),
        ("torch", "other"),
        ("cuda", "12.0"),
        ("mscclpp", "other"),
        ("native_wheel_sha256", "other"),
    ],
)
def test_incompatible_nodes_rejected(monkeypatch, field, value):
    discovered = nodes()
    discovered[1][field] = value
    mock_discovery(monkeypatch, discovered)
    with pytest.raises(ValueError, match=f"matching {field}"):
        multinode.discover_nodes([node["ip"] for node in discovered], "tester", "/deploy")


def test_discovery_rejects_no_gpus(monkeypatch):
    discovered = nodes(1)
    discovered[0]["num_gpus"] = 0
    mock_discovery(monkeypatch, discovered)
    with pytest.raises(ValueError, match="No GPUs visible"):
        multinode.discover_nodes(["10.0.0.1"], "tester", "/deploy")


@pytest.mark.parametrize("user", ["-option", "a@host", "a b", "a;echo bad"])
def test_invalid_ssh_user_rejected_before_contact(monkeypatch, user):
    ssh = Mock()
    monkeypatch.setattr(multinode, "ssh", ssh)
    with pytest.raises(ValueError, match="Invalid SSH username"):
        multinode.discover_nodes(["10.0.0.1"], user, "/deploy")
    ssh.assert_not_called()


@pytest.mark.parametrize(
    "error,message",
    [
        (subprocess.CalledProcessError(1, ["ssh"], stderr="incomplete deployment"), "incomplete deployment"),
        (subprocess.TimeoutExpired(["ssh"], 120), "timed out on 10.0.0.1"),
    ],
)
def test_discovery_errors_include_node(monkeypatch, error, message):
    monkeypatch.setattr(multinode, "ssh", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match=message):
        multinode.discover_nodes(["10.0.0.1"], "tester", "/deploy")


def deployment(parent, name, created_at, complete=True):
    root = parent / name
    root.mkdir()
    (root / "moe_deployment.json").write_text(json.dumps({"created_at": created_at}))
    if complete:
        python = root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
    return root


def discover_directory(directory):
    return subprocess.run(
        [sys.executable, "-c", multinode.DISCOVER, str(directory)], text=True, capture_output=True, timeout=10
    )


def test_discover_newest_or_exact_deployment(tmp_path):
    older = deployment(tmp_path, "moe_old", "2026-01-01T00:00:00+00:00")
    newer = deployment(tmp_path, "moe_new", "2026-02-01T00:00:00+00:00")
    for requested, expected in [(tmp_path, newer), (older, older)]:
        result = discover_directory(requested)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"root": str(expected), "python": str(expected / ".venv/bin/python")}


def test_incomplete_newest_deployment_does_not_silently_fall_back(tmp_path):
    deployment(tmp_path, "moe_old", "2026-01-01T00:00:00+00:00")
    deployment(tmp_path, "moe_new", "2026-02-01T00:00:00+00:00", complete=False)
    result = discover_directory(tmp_path)
    assert result.returncode != 0
    assert "Deployment is incomplete" in result.stderr


def test_missing_deployment_is_actionable(tmp_path):
    result = discover_directory(tmp_path)
    assert result.returncode != 0
    assert "finish deploy_moe_bench.py first" in result.stderr


@pytest.mark.parametrize("count", [1, 2, 3, 5])
@pytest.mark.parametrize("check", [False, True])
def test_worker_commands_map_global_and_local_ranks(count, check):
    discovered = nodes(count)
    for rank, node in enumerate(discovered):
        command = multinode.worker_command(node, rank, discovered, "test_run", 29501, check)
        assert command[:4] == [node["python"], "-m", "torch.distributed.run", f"--nnodes={count}"]
        assert f"--node-rank={rank}" in command
        assert "--nproc-per-node=4" in command
        assert "--master-addr=10.0.0.1" in command
        assert "--master-port=29501" in command
        assert f"--local-addr={node['ip']}" in command
        assert f"{node['root']}/runs/test_run/benchmark/worker.py" in command
        assert ("--check" in command) == check
        assert command[command.index("--run-dir") + 1] == f"{node['root']}/runs/test_run"


class Process:
    def __init__(self, code=0, wait_errors=()):
        self.code = code
        self.stdin = Mock()
        self.wait = Mock(side_effect=[*wait_errors, code])
        self.terminate = Mock()
        self.kill = Mock()

    def poll(self):
        return self.code


def mock_launch(monkeypatch, tmp_path, processes):
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text('{"status":"running"}')
    ssh = Mock()
    run = Mock()
    popen = Mock(side_effect=processes)
    monkeypatch.setattr(multinode, "ssh", ssh)
    monkeypatch.setattr(multinode.subprocess, "run", run)
    monkeypatch.setattr(multinode.subprocess, "Popen", popen)
    return run_dir, ssh, run, popen


def test_launch_snapshots_workers_and_collects_only_rank_zero_results(monkeypatch, tmp_path):
    discovered = nodes(3)
    discovered[1]["cuda_visible_devices"] = "2,3,0,1"
    processes = [Process() for _ in discovered]
    run_dir, ssh, run, popen = mock_launch(monkeypatch, tmp_path, processes)
    source = tmp_path / "source"
    source.mkdir()
    for relative in (
        "worker.py",
        "multinode.py",
        "communication_backends/mscclpp.py",
        "tests/test_skip.py",
        ".venv/secret.py",
        "venv/library.py",
        "__pycache__/skip.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n")
    monkeypatch.setattr(multinode, "__file__", str(source / "multinode.py"))
    commands = [
        multinode.worker_command(node, rank, discovered, run_dir.name, 29500, True)
        for rank, node in enumerate(discovered)
    ]
    downloaded = {
        "run.json": '{"status":"running","downloaded":true}',
        "summary.csv": "summary\n",
        "iterations.csv": "iterations\n",
    }

    def transfer(command, **kwargs):
        if ":" in command[-2]:
            filename = command[-2].rsplit("/", 1)[-1]
            destination = run_dir / filename
            before = destination.read_bytes() if destination.exists() else None
            Path(command[-1]).write_text(downloaded[filename])
            assert (destination.read_bytes() if destination.exists() else None) == before
        return SimpleNamespace(returncode=0)

    run.side_effect = transfer
    assert multinode.launch(discovered, run_dir, commands, 60) == 0
    with tarfile.open(run_dir / "workers.tar.gz") as archive:
        assert set(archive.getnames()) == {
            "run.json",
            "benchmark/worker.py",
            "benchmark/multinode.py",
            "benchmark/communication_backends/mscclpp.py",
        }
    assert ssh.call_count == 6
    for rank, call in enumerate(popen.call_args_list):
        invocation = shlex.split(call.args[0][-1])
        assert f"GLOO_SOCKET_IFNAME=eth{rank}" in invocation
        assert f"MSCCLPP_SOCKET_IFNAME=eth{rank}" in invocation
        assert "MSCCLPP_EP_ENABLE_GPUNETIO=0" in invocation
        assert json.loads(invocation[-1]) == commands[rank]
        assert ("CUDA_VISIBLE_DEVICES=2,3,0,1" in invocation) == (rank == 1)
        assert "StrictHostKeyChecking=yes" in call.args[0]
    transfers = [call.args[0] for call in run.call_args_list]
    assert len(transfers) == 6
    for node, transfer in zip(discovered, transfers[:3]):
        assert transfer[-1] == f"{node['target']}:{node['root']}/runs/test_run/workers.tar.gz"
    assert [command[-2] for command in transfers[3:]] == [
        f"{discovered[0]['target']}:{discovered[0]['root']}/runs/test_run/{filename}"
        for filename in ("run.json", "summary.csv", "iterations.csv")
    ]
    for command in transfers[3:]:
        destination = Path(command[-1])
        filename = command[-2].rsplit("/", 1)[-1]
        assert destination.parent == run_dir
        assert destination != run_dir / filename
        assert not destination.exists()
        assert (run_dir / filename).read_text() == downloaded[filename]
    for process in processes:
        process.stdin.close.assert_called_once()
        process.wait.assert_called_once_with(timeout=30)


@pytest.mark.parametrize("failure", ["worker", "timeout", "spawn", "collect"])
def test_failed_launch_cancels_workers_and_never_collects_success(monkeypatch, tmp_path, failure):
    discovered = nodes()
    processes = [Process(code=1 if failure == "worker" else None), Process()]
    run_dir, ssh, run, popen = mock_launch(monkeypatch, tmp_path, processes)
    if failure == "spawn":
        popen.side_effect = [processes[0], OSError("ssh spawn failed")]
    if failure == "timeout":
        monkeypatch.setattr(multinode.time, "monotonic", Mock(side_effect=[0, 2]))
    if failure == "collect":
        processes[0].code = 0
        run.side_effect = [SimpleNamespace(returncode=0)] * 2 + [subprocess.CalledProcessError(1, "scp")]
    expected = {
        "worker": RuntimeError,
        "timeout": TimeoutError,
        "spawn": OSError,
        "collect": subprocess.CalledProcessError,
    }[failure]
    with pytest.raises(expected):
        multinode.launch(discovered, run_dir, [["worker-0"], ["worker-1"]], 1)
    for process in processes[: 1 if failure == "spawn" else 2]:
        process.stdin.close.assert_called_once()
    assert run.call_count == (3 if failure == "collect" else 2)


def test_unresponsive_ssh_is_terminated_then_killed(monkeypatch, tmp_path):
    expired = subprocess.TimeoutExpired(["ssh"], 20)
    process = Process(code=None, wait_errors=[expired, expired])
    run_dir, *_ = mock_launch(monkeypatch, tmp_path, [process])
    monkeypatch.setattr(multinode.time, "monotonic", Mock(side_effect=[0, 2]))
    with pytest.raises(TimeoutError):
        multinode.launch(nodes(1), run_dir, [["worker"]], 1)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 3


def test_transfer_failure_starts_no_workers(monkeypatch, tmp_path):
    run_dir, ssh, run, popen = mock_launch(monkeypatch, tmp_path, [])
    run.side_effect = subprocess.CalledProcessError(1, "scp")
    with pytest.raises(subprocess.CalledProcessError):
        multinode.launch(nodes(1), run_dir, [["worker"]], 1)
    popen.assert_not_called()


@pytest.mark.parametrize("filename", ["run.json", "summary.csv", "iterations.csv"])
@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_partial_download_preserves_existing_result(monkeypatch, tmp_path, filename, failure):
    run_dir, _, transfer, _ = mock_launch(monkeypatch, tmp_path, [Process()])
    original = '{"status":"running"}'
    destination = run_dir / filename
    destination.write_text(original)
    downloads = []
    error = subprocess.CalledProcessError(1, ["scp"]) if failure == "exit" else subprocess.TimeoutExpired(["scp"], 120)

    def scp(command, **kwargs):
        if ":" in command[-2]:
            path = Path(command[-1])
            downloads.append(path)
            if command[-2].endswith("/" + filename):
                path.write_text('{"status":')
                raise error
            path.write_text('{"status":"running"}')
        return SimpleNamespace(returncode=0)

    transfer.side_effect = scp
    with pytest.raises(type(error)) as raised:
        multinode.launch(nodes(1), run_dir, [["worker"]], 60)
    assert raised.value is error
    assert destination.read_text() == original
    assert all(not path.exists() for path in downloads)


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_controller_records_partial_download_failure(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--model",
            "small",
            "--input",
            "decode",
            "--routing",
            "balanced",
            "--nodelist",
            "192.0.2.1",
            "--output-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(multinode, "discover_nodes", lambda *args: nodes(1))
    monkeypatch.setattr(multinode, "ssh", Mock())
    monkeypatch.setattr(multinode.subprocess, "Popen", Mock(return_value=Process()))
    error = subprocess.CalledProcessError(1, ["scp"]) if failure == "exit" else subprocess.TimeoutExpired(["scp"], 120)
    downloads = []

    def transfer(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(stdout="revision\n" if command[-1] == "HEAD" else "")
        if ":" in command[-2]:
            path = Path(command[-1])
            downloads.append(path)
            path.write_text('{"status":')
            raise error
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(multinode.subprocess, "run", transfer)
    with pytest.raises(type(error)) as raised:
        launcher.main()
    assert raised.value is error
    (record_path,) = tmp_path.glob("*/run.json")
    record = json.loads(record_path.read_text())
    assert record["status"] == "failed"
    assert record["failure"] == {"type": type(error).__name__, "message": str(error)}
    assert record["config"]["environment"]["num_gpus"] == 4
    assert record["metadata"]["nodes"] == nodes(1)
    assert record["finished_at"]
    assert len(downloads) == 1 and not downloads[0].exists()


def test_supervisor_eof_terminates_its_process_group(tmp_path):
    pid_file = tmp_path / "worker.pid"
    command = [
        sys.executable,
        "-c",
        "import os,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)",
        str(pid_file),
    ]
    process = subprocess.Popen(
        [sys.executable, "-c", multinode.SUPERVISOR, json.dumps(command)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    worker_pid = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and process.poll() is None:
            if pid_file.exists() and (contents := pid_file.read_text()).strip():
                worker_pid = int(contents)
                break
            time.sleep(0.02)
        assert worker_pid is not None
        process.stdin.close()
        assert process.wait(timeout=15) != 0
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            if worker_pid is not None:
                os.killpg(worker_pid, signal.SIGKILL)
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


@pytest.mark.parametrize(
    "exit_mode,runner",
    [("eof", "synthetic"), ("signal", "synthetic"), ("launcher_exit", "synthetic"), ("eof", "torchrun")],
)
def test_supervisor_reaps_isolated_descendants(tmp_path, exit_mode, runner):
    pid_file = tmp_path / "isolated.pids"
    worker = """
import os,pathlib,signal,subprocess,sys,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
pathlib.Path(sys.argv[1]).write_text(f'{os.getpid()} {child.pid}')
while True:
    time.sleep(1)
"""
    launcher = """
import pathlib,signal,subprocess,sys,time
worker=subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1]],start_new_session=True)
def stop(signum,frame):
    worker.terminate()
    try:
        worker.wait(timeout=30)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
while not pathlib.Path(sys.argv[1]).exists():
    time.sleep(0.02)
if sys.argv[3]=='launcher_exit':
    raise SystemExit(7)
while True:
    time.sleep(1)
"""
    command = [sys.executable, "-c", launcher, str(pid_file), worker, exit_mode]
    if runner == "torchrun":
        worker_file = tmp_path / "isolated_worker.py"
        worker_file.write_text(worker)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--local-addr=127.0.0.1",
            "--nnodes=1",
            "--nproc-per-node=1",
            str(worker_file),
            str(pid_file),
        ]
    process = subprocess.Popen(
        [sys.executable, "-c", multinode.SUPERVISOR, json.dumps(command)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pids = []
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.poll() is None:
            if pid_file.exists() and len(contents := pid_file.read_text().split()) == 2:
                pids = list(map(int, contents))
                break
            time.sleep(0.02)
        if not pids:
            process.stdin.close()
            process.wait(timeout=25)
            pytest.fail(f"Worker did not become ready:\n{process.stderr.read().decode()}")
        if exit_mode == "eof":
            process.stdin.close()
        elif exit_mode == "signal":
            process.terminate()
        assert process.wait(timeout=25) != 0
        errors = process.stderr.read()
        assert b"Worker cleanup timed out" not in errors
        if runner == "synthetic":
            assert errors == b""
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
        process.stderr.close()


@pytest.mark.parametrize("returncode", [0, 7])
def test_supervisor_preserves_normal_returncode(returncode):
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            multinode.SUPERVISOR,
            json.dumps([sys.executable, "-c", f"raise SystemExit({returncode})"]),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.wait(timeout=15) == returncode
    finally:
        process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def test_distributed_controller_does_not_import_torch(tmp_path):
    script = """
import sys
sys.path.insert(0,sys.argv[1])
class NoTorch:
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0]=='torch':
            raise AssertionError('Distributed controller imported Torch')
sys.meta_path.insert(0,NoTorch())
from moe_bench import run,multinode
run.sys.argv=['run.py','--model','small','--input','decode','--routing','balanced','--nodelist','10.0.0.1','--output-dir',sys.argv[2]]
multinode.discover_nodes=lambda *args:[dict(ip='10.0.0.1',num_gpus=4,root='/deploy',python='/deploy/.venv/bin/python',interface='eth0')]
def stop(*args):
    raise RuntimeError('reached distributed launch without Torch')
multinode.launch=stop
try:
    run.main()
except RuntimeError as error:
    assert str(error)=='reached distributed launch without Torch'
else:
    raise AssertionError('Expected distributed launch')
"""
    subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2]), str(tmp_path)], check=True, timeout=15
    )


@pytest.mark.parametrize(
    "option,value", [("--master-port", "0"), ("--master-port", "65536"), ("--timeout", "0"), ("--timeout", "-1")]
)
def test_invalid_distributed_cli_options(option, value):
    with pytest.raises(SystemExit):
        launcher.parse_args(["--model", "small", "--input", "decode", "--routing", "balanced", option, value])


@pytest.mark.parametrize(
    "backend,check,compute,message",
    [
        ("nccl", False, "torch", "requires --backend mscclpp"),
        ("mscclpp", True, "cutlass_flashinfer", "requires the Torch BF16"),
    ],
)
def test_incompatible_distributed_options_rejected(monkeypatch, backend, check, compute, message):
    from .test_moe_bench import tiny_config

    config = copy.deepcopy(tiny_config())
    config["environment"].update(communication_backend=backend, compute_backend=compute)
    args = ["run.py", "--model", "small", "--input", "decode", "--routing", "balanced", "--nodelist", "10.0.0.1"]
    if check:
        args.append("--check")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(multinode, "discover_nodes", lambda *args: nodes(1))
    monkeypatch.setattr(launcher, "load_config", lambda *args, **kwargs: config)
    with pytest.raises(ValueError, match=message):
        launcher.main()


@pytest.mark.parametrize("outcome", ["success", "worker_failure", "missing_results"])
def test_controller_persists_global_configuration_and_final_status(monkeypatch, tmp_path, outcome):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--model",
            "small",
            "--input",
            "decode",
            "--routing",
            "balanced",
            "--nodelist",
            "10.0.0.1,10.0.0.2",
            "--iterations",
            "10",
            "--output-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(multinode, "discover_nodes", lambda *args: nodes())

    def launch(discovered, run_dir, commands, timeout):
        assert len(discovered) == len(commands) == 2
        running = json.loads((run_dir / "run.json").read_text())
        assert running["status"] == "running"
        if outcome == "worker_failure":
            raise RuntimeError("test worker failure")
        if outcome == "success":
            for name in ("summary.csv", "iterations.csv"):
                (run_dir / name).write_text("test results\n")
        return 0

    monkeypatch.setattr(multinode, "launch", launch)
    if outcome == "success":
        launcher.main()
    else:
        with pytest.raises(RuntimeError, match="test worker failure|without required result"):
            launcher.main()
    records = list(tmp_path.glob("*/run.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["config"]["environment"]["num_gpus"] == 8
    assert record["config"]["benchmark"]["iterations"] == 10
    assert record["metadata"]["num_nodes"] == 2
    assert record["metadata"]["nodes"] == nodes()
    assert len(record["invocation"]["worker_commands"]) == 2
    assert record["status"] == ("completed" if outcome == "success" else "failed")
    if outcome != "success":
        assert record["failure"]["type"] == "RuntimeError"
