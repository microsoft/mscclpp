# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline bundle construction, validation, and SSH orchestration tests."""

import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import deploy_moe_bench as deploy


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--nodelist", "10.0.4.159,10.0.4.159"],
        ["--nodelist", "10.0.4.159,"],
        ["--nodelist", "-oProxyCommand=bad"],
        ["--nodelist", "10.0.4.159", "--user", "user;bad"],
        ["--nodelist", "10.0.4.159", "--remote-dir", "../existing"],
        ["--nodelist", "10.0.4.159", "--remote-dir", "~/some directory"],
        ["--build-only", "--bundle", "bundle.tar"],
    ],
)
def test_invalid_deployment_arguments(options):
    with pytest.raises(SystemExit):
        deploy.parse_args(options)


def test_node_list():
    args = deploy.parse_args(["--nodelist", "10.0.4.159, 10.0.4.69", "--user", "azhpcuser"])
    assert args.nodes == ["10.0.4.159", "10.0.4.69"]


def test_platform_probe_matches_local(monkeypatch):
    monkeypatch.setattr(deploy.subprocess, "check_output", lambda *args, **kwargs: "10.0\n10.0\n")
    expected = deploy.platform_info()
    code = (
        "import subprocess\n"
        "subprocess.check_output=lambda *args,**kwargs: '10.0\\n10.0\\n'\n" + deploy.PLATFORM_PROBE
    )
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == expected


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "platform_info", lambda: {"python": [3, 12], "machine": "aarch64"})
    root = tmp_path / "source"
    benchmark = root / "test/python/ep/moe_bench"
    benchmark.mkdir(parents=True)
    (benchmark / "run.py").write_text("benchmark")
    (root / "LICENSE").write_text("MIT")
    (root / "secret.txt").write_text("not part of the bundle")
    commands = []
    source = {"git_commit": "a" * 40, "git_dirty": False, "git_status": []}
    monkeypatch.setattr(deploy, "source_info", lambda root: source)

    def fake_run(command, **kwargs):
        commands.append(command)
        if "wheel" in command:
            wheels = Path(command[command.index("--wheel-dir") + 1])
            (wheels / "mscclpp-1.0-cp312-linux_aarch64.whl").write_bytes(b"native wheel")
        elif "download" in command:
            wheels = Path(command[command.index("--dest") + 1])
            (wheels / "torch-1.0-py3-none-any.whl").write_bytes(b"dependency wheel")
        else:
            return SimpleNamespace(stdout="test/python/ep/moe_bench/run.py\0")

    monkeypatch.setattr(deploy, "run", fake_run)
    path = deploy.build_bundle(root, tmp_path / "output", "13")
    assert sum("wheel" in command for command in commands) == 1
    assert "--only-binary=:all:" in commands[1]
    assert "[cuda13,moe-benchmark]" in commands[1][-1]
    return path


def test_bundle_contents_and_offline_install(bundle, tmp_path, monkeypatch):
    manifest = deploy.read_manifest(bundle)
    assert manifest["source"]["git_commit"] == "a" * 40
    assert "secret.txt" not in manifest["files"]
    assert "deploy_moe_bench.py" in manifest["files"]
    destination = tmp_path / "installed"
    with tarfile.open(bundle) as archive:
        archive.extractall(destination, filter="data")
    monkeypatch.setattr(deploy.venv, "EnvBuilder", Mock())
    commands = []
    monkeypatch.setattr(deploy, "run", lambda command, **kwargs: commands.append(command))
    deploy.install_bundle(destination)
    assert "--no-index" in commands[0]
    assert "--find-links" in commands[0]
    assert "[cuda13,moe-benchmark]" in commands[0][-1]
    assert "torch.cuda.is_available()" in commands[1][-1]
    (destination / "test/python/ep/moe_bench/run.py").write_text("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        deploy.install_bundle(destination)


def test_install_refuses_existing_environment(bundle, tmp_path, monkeypatch):
    directory = tmp_path / "existing"
    with tarfile.open(bundle) as archive:
        archive.extractall(directory, filter="data")
    (directory / ".venv").mkdir()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        deploy.install_bundle(directory)


def test_deployment_checks_every_node_before_copying(bundle, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(deploy.platform_info()))

    monkeypatch.setattr(deploy, "run", fake_run)
    deploy.deploy_bundle(bundle, ["10.0.4.159", "10.0.4.69"], "azhpcuser", ".local/moe", "python3")
    assert [command[0] for command in calls] == ["ssh", "ssh", "ssh", "scp", "ssh", "ssh", "scp", "ssh"]
    assert all("StrictHostKeyChecking=yes" in command for command in calls)
    assert calls[2][-1].startswith("mkdir -p")
    assert " && mkdir -- " in calls[2][-1]
    assert calls[3][-1].startswith("azhpcuser@10.0.4.159:")
    assert "tar -xf" in calls[4][-1]
    assert "--install" in calls[4][-1]
    assert deploy.sha256(bundle) in calls[4][-1]


def test_incompatible_node_stops_before_copying(bundle, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        actual = {**deploy.platform_info(), "machine": "incompatible"}
        return SimpleNamespace(stdout=json.dumps(actual))

    monkeypatch.setattr(deploy, "run", fake_run)
    with pytest.raises(ValueError, match="matching Python ABI"):
        deploy.deploy_bundle(bundle, ["10.0.4.159"], "azhpcuser", ".local/moe", "python3")
    assert len(calls) == 1


def test_transfer_failure_is_not_reported_as_success(bundle, monkeypatch):
    def fake_run(command, **kwargs):
        if command[0] == "scp":
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=json.dumps(deploy.platform_info()))

    monkeypatch.setattr(deploy, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        deploy.deploy_bundle(bundle, ["10.0.4.159"], "azhpcuser", ".local/moe", "python3")


def test_benchmark_uses_bundled_source_metadata(tmp_path, monkeypatch):
    import torch
    from .. import run as launcher

    config_path = Path(launcher.__file__).with_name("configs.yaml")
    source = {"git_commit": "a" * 40, "git_dirty": False, "git_status": []}
    (tmp_path / deploy.MANIFEST).write_text(json.dumps({"source": source}))
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "test/python/ep/moe_bench/run.py"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--config",
            str(config_path),
            "--model",
            "small",
            "--input",
            "decode",
            "--routing",
            "balanced",
            "--output-dir",
            str(tmp_path / "results"),
        ],
    )

    def fake_run(command, **kwargs):
        assert command[0] != "git"
        directory = Path(command[-1])
        (directory / "summary.csv").touch()
        (directory / "iterations.csv").touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher.main()
    (result,) = (tmp_path / "results").glob("*/run.json")
    saved = json.loads(result.read_text())
    assert saved["status"] == "completed"
    assert saved["metadata"]["git_commit"] == source["git_commit"]
