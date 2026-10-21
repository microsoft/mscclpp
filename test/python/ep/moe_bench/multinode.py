# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SSH orchestration for one MoE workload across a shared MNNVL domain."""

import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile
import time

SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=2",
]

DISCOVER = """
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]).expanduser()
if not root.is_absolute():
    root=pathlib.Path.home()/root
if (root/'moe_deployment.json').is_file():
    candidates=[root]
else:
    candidates=[p for p in root.glob('moe_*') if (p/'moe_deployment.json').is_file()]
    candidates.sort(key=lambda p:json.loads((p/'moe_deployment.json').read_text())['created_at'],reverse=True)
if not candidates:
    raise RuntimeError('No deployment found under '+str(root)+'; finish deploy_moe_bench.py first')
root=candidates[0]
python=root/'.venv/bin/python'
if not python.is_file():
    raise RuntimeError('Deployment is incomplete: '+str(root))
print(json.dumps(dict(root=str(root),python=str(python))))
"""

PROBE = """
import json,os,pathlib,socket,subprocess,sys
import torch,mscclpp,mscclpp.ep
root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/'moe_deployment.json').read_text())
interfaces=json.loads(subprocess.check_output(['ip','-j','addr'],text=True))
interface=next((entry['ifname'] for entry in interfaces
    if any(addr['local']==sys.argv[2] for addr in entry.get('addr_info',[]))),None)
if interface is None:
    raise RuntimeError('Node IP is not assigned to a local network interface: '+sys.argv[2])
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable on '+socket.gethostname())
print(json.dumps(dict(hostname=socket.gethostname(),interface=interface,
    num_gpus=torch.cuda.device_count(),torch=torch.__version__,cuda=torch.version.cuda,
    mscclpp=mscclpp.__version__,bundle_id=manifest['bundle_id'],
    native_wheel_sha256=manifest['files']['wheels/'+manifest['wheel']],
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))))
"""

SUPERVISOR = """
import ctypes,json,os,pathlib,select,signal,subprocess,sys,time
# torchrun workers start separate sessions. Adopt them if their launcher exits
# before it can clean up, rather than leaving them owned by init.
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(36,1,0,0,0)!=0:
    raise OSError(ctypes.get_errno(),'Cannot enable child subreaper')
stopping=False
def stop(signum,frame):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
process=subprocess.Popen(json.loads(sys.argv[1]),stdin=subprocess.DEVNULL,start_new_session=True)
try:
    while not stopping and process.poll() is None:
        if select.select([sys.stdin],[],[],0.2)[0] and not sys.stdin.buffer.read(1):
            break
finally:
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    # Killing an adopted parent may expose more descendants on the next pass.
    # Only inspect our own children, never unrelated processes on shared nodes.
    children_file=pathlib.Path(f'/proc/self/task/{os.getpid()}/children')
    deadline=time.monotonic()+5
    while True:
        process.poll()
        children=[int(pid) for pid in children_file.read_text().split()]
        if not children:
            break
        for pid in children:
            try:
                os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            if pid!=process.pid:
                os.waitpid(pid,os.WNOHANG)
        if time.monotonic()>=deadline:
            raise RuntimeError(f'Worker cleanup timed out; surviving child PIDs: {children}')
        time.sleep(0.02)
    process.wait(timeout=0)
sys.exit(process.returncode)
"""


def parse_nodes(value):
    nodes = [str(ipaddress.IPv4Address(node.strip())) for node in value.split(",")]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("Node list must contain distinct IPv4 addresses")
    return nodes


def ssh(node, command, **kwargs):
    return subprocess.run(["ssh", *SSH_OPTIONS, node, command], check=True, timeout=120, **kwargs)


def discover_nodes(nodes, user, remote_dir):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", user):
        raise ValueError("Invalid SSH username")
    resolved = []
    for node in nodes:
        target = f"{user}@{node}"
        try:
            location = json.loads(
                ssh(target, shlex.join(["python3", "-c", DISCOVER, remote_dir]), capture_output=True, text=True).stdout
            )
            details = json.loads(
                ssh(
                    target,
                    shlex.join([location["python"], "-c", PROBE, location["root"], node]),
                    capture_output=True,
                    text=True,
                ).stdout
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Node preflight failed on {node}:\n{error.stderr}") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Node preflight timed out on {node}") from error
        if details["num_gpus"] < 1:
            raise ValueError(f"No GPUs visible on {node}")
        resolved.append({"ip": node, "target": target, **location, **details})
    for node in resolved[1:]:
        for key in ("num_gpus", "torch", "cuda", "mscclpp", "native_wheel_sha256"):
            if node[key] != resolved[0][key]:
                raise ValueError(f"Nodes must have matching {key}: {resolved[0]['ip']} vs {node['ip']}")
    return resolved


def worker_command(node, node_rank, nodes, run_id, port, check):
    directory = str(Path(node["root"]) / "runs" / run_id)
    command = [
        node["python"],
        "-m",
        "torch.distributed.run",
        f"--nnodes={len(nodes)}",
        f"--node-rank={node_rank}",
        f"--nproc-per-node={node['num_gpus']}",
        f"--master-addr={nodes[0]['ip']}",
        f"--master-port={port}",
        f"--local-addr={node['ip']}",
        "--rdzv-conf=timeout=120",
        f"{directory}/benchmark/worker.py",
        "--run-dir",
        directory,
    ]
    if check:
        command.append("--check")
    return command


def launch(nodes, run_dir, commands, timeout):
    if not nodes or len(nodes) != len(commands):
        raise ValueError("Each node requires exactly one worker command")
    archive_path = run_dir / "workers.tar.gz"
    with tarfile.open(archive_path, "x:gz") as archive:
        archive.add(run_dir / "run.json", arcname="run.json")
        source = Path(__file__).parent
        for directory, subdirectories, filenames in os.walk(source):
            subdirectories[:] = sorted(
                name
                for name in subdirectories
                if not name.startswith(".") and name not in ("tests", "__pycache__", "venv")
            )
            for filename in sorted(filenames):
                if filename.endswith(".py"):
                    path = Path(directory) / filename
                    archive.add(path, arcname=str(Path("benchmark") / path.relative_to(source)))
    for node in nodes:
        directory = str(Path(node["root"]) / "runs" / run_dir.name)
        ssh(
            node["target"],
            f"mkdir -p -- {shlex.quote(str(Path(directory).parent))} && mkdir -- {shlex.quote(directory)}",
        )
        subprocess.run(
            ["scp", *SSH_OPTIONS, str(archive_path), f"{node['target']}:{directory}/workers.tar.gz"],
            check=True,
            timeout=120,
        )
        ssh(node["target"], f"cd -- {shlex.quote(directory)} && tar -xzf workers.tar.gz")
    processes = []
    logs = []
    started = time.monotonic()
    try:
        for rank, (node, command) in enumerate(zip(nodes, commands)):
            environment = {
                "GLOO_SOCKET_IFNAME": node["interface"],
                "MSCCLPP_SOCKET_IFNAME": node["interface"],
                "MSCCLPP_EP_ENABLE_GPUNETIO": "0",
                "PYTHONUNBUFFERED": "1",
            }
            if node["cuda_visible_devices"] is not None:
                environment["CUDA_VISIBLE_DEVICES"] = node["cuda_visible_devices"]
            invocation = [
                "env",
                *(f"{key}={value}" for key, value in environment.items()),
                node["python"],
                "-c",
                SUPERVISOR,
                json.dumps(command),
            ]
            log = (run_dir / f"node-{rank}.log").open("wb")
            logs.append(log)
            process = subprocess.Popen(
                ["ssh", *SSH_OPTIONS, node["target"], shlex.join(invocation)],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
        while True:
            codes = [process.poll() for process in processes]
            for rank, code in enumerate(codes):
                if code is not None and code != 0:
                    path = run_dir / f"node-{rank}.log"
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 8000))
                        tail = stream.read().decode(errors="replace")
                    raise RuntimeError(
                        f"Node {nodes[rank]['ip']} failed (exit {code}); {path}\n{tail}\n"
                        "Cross-node CUDA IPC requires working MNNVL Fabric imports and a configured "
                        "NVIDIA IMEX domain on every node; a shared fabric ID alone is not sufficient."
                    )
            if all(code == 0 for code in codes):
                break
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Distributed run exceeded {timeout}s; inspect node logs in {run_dir}")
            time.sleep(0.2)
    finally:
        for process in processes:
            process.stdin.close()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()
    first = nodes[0]
    directory = str(Path(first["root"]) / "runs" / run_dir.name)
    for filename in ("run.json", "summary.csv", "iterations.csv"):
        with tempfile.NamedTemporaryFile(dir=run_dir, prefix=f".{filename}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            subprocess.run(
                ["scp", *SSH_OPTIONS, f"{first['target']}:{directory}/{filename}", str(temporary)],
                check=True,
                timeout=120,
            )
            temporary.replace(run_dir / filename)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Collected distributed results: {run_dir}", flush=True)
    return 0
