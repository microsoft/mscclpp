# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Launch a reference MoE benchmark locally or across MNNVL-connected nodes."""

import argparse
from datetime import datetime, timezone
import json
import os
import getpass
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

if __package__:
    from .config import load_config
    from .results import write_json
else:
    from config import load_config
    from results import write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs.yaml"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--backend", choices=("mscclpp", "nccl"), help="Override environment.communication_backend")
    parser.add_argument(
        "--mode", choices=("latency", "throughput"), help="Native communication algorithm; default latency"
    )
    parser.add_argument(
        "--layout",
        choices=("expert_major", "rank_major", "token_major"),
        help="Native dispatch layout; default expert_major",
    )
    parser.add_argument("--compute-backend", choices=("torch", "cutlass_flashinfer"))
    parser.add_argument("--activation-dtype", choices=("bf16", "mxfp8"), help="Internal compute activation format")
    parser.add_argument("--weight-dtype", choices=("bf16", "mxfp4"), help="Internal compute weight format")
    parser.add_argument(
        "--graph", choices=("none", "mlp"), help="Override environment.cuda_graph (none disables graphs)"
    )
    parser.add_argument("--routing", required=True)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--nodelist", help="Comma-separated IPv4 addresses in one shared MNNVL domain")
    parser.add_argument("--user", default=getpass.getuser(), help="SSH user for --nodelist")
    parser.add_argument("--remote-dir", default=".local/share/mscclpp/moe-bench", help="Deployment directory or parent")
    parser.add_argument("--master-port", type=int, default=29500, help="Cross-node coordination port")
    parser.add_argument("--timeout", type=int, default=600, help="Distributed job timeout in seconds")
    parser.add_argument(
        "--check", action="store_true", help="Check Torch BF16 output against a reference before timing"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.master_port <= 65535 or args.timeout <= 0:
        parser.error("--master-port must be in [1, 65535] and --timeout must be positive")
    return args


def main():
    args = parse_args()

    nodes = None
    if args.nodelist:
        if __package__:
            from . import multinode
        else:
            import multinode
        nodes = multinode.discover_nodes(multinode.parse_nodes(args.nodelist), args.user, args.remote_dir)
        num_gpus = len(nodes) * nodes[0]["num_gpus"]
    else:
        import torch

        num_gpus = torch.cuda.device_count()
    if nodes is None and num_gpus not in (2, 4):
        raise RuntimeError(f"MoE benchmark uses all visible GPUs on this node and supports 2 or 4; found {num_gpus}")
    config = load_config(
        args.config,
        num_gpus=num_gpus,
        model=args.model,
        input_name=args.input,
        routing=args.routing,
        environment_overrides={
            "communication_backend": args.backend,
            "mode": args.mode,
            "layout": args.layout,
            "compute_backend": args.compute_backend,
            "cuda_graph": False if args.graph == "none" else args.graph,
        },
        compute_overrides={"activation_dtype": args.activation_dtype, "weight_dtype": args.weight_dtype},
        overrides={
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "output_dir": args.output_dir,
        },
    )
    if nodes and config["environment"]["communication_backend"] != "mscclpp":
        raise ValueError("--nodelist currently requires --backend mscclpp over shared MNNVL")
    if args.check and config["environment"]["compute_backend"] != "torch":
        raise ValueError("--check requires the Torch BF16 compute backend")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(config["benchmark"]["output_dir"]).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--local-addr=127.0.0.1",
        "--nnodes=1",
        f"--nproc-per-node={config['environment']['num_gpus']}",
        str(Path(__file__).with_name("worker.py").resolve()),
        "--run-dir",
        str(run_dir),
    ]
    if nodes:
        commands = [
            multinode.worker_command(node, rank, nodes, run_dir.name, args.master_port, args.check)
            for rank, node in enumerate(nodes)
        ]
        command = commands[0]
    elif args.check:
        command.append("--check")
    root = Path(__file__).resolve().parents[4]
    deployment = root / "moe_deployment.json"
    if deployment.is_file():
        with deployment.open() as stream:
            source = json.load(stream)["source"]
    else:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout
        source = {"git_commit": revision, "git_dirty": bool(status), "git_status": status.splitlines()}
    worker_env = os.environ.copy()
    worker_env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    run = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "invocation": {
            "argv": [sys.executable, *sys.argv],
            "command": shlex.join([sys.executable, *sys.argv]),
            "worker_command": shlex.join(command),
            "working_directory": str(Path.cwd()),
            "config_path": str(args.config.resolve()),
        },
        "presets": {
            "model": args.model,
            "input": args.input,
            "routing": args.routing,
        },
        "config": config,
        "metadata": {
            **source,
            "gloo_socket_ifname": worker_env["GLOO_SOCKET_IFNAME"],
        },
    }
    if nodes:
        run["invocation"]["worker_commands"] = [shlex.join(item) for item in commands]
        run["metadata"]["nodes"] = nodes
        run["metadata"]["num_nodes"] = len(nodes)
        run["metadata"]["gloo_socket_ifname"] = [node["interface"] for node in nodes]
    write_json(run_dir / "run.json", run)
    print(
        f"Launching reference MoE benchmark on all {num_gpus} visible GPUs"
        f" across {len(nodes) if nodes else 1} node(s); output: {run_dir}",
        flush=True,
    )
    returncode = None
    try:
        if not nodes and config["environment"]["compute_backend"] == "cutlass_flashinfer":
            if __package__:
                from .compute_backends.flashinfer import preflight
            else:
                from compute_backends.flashinfer import preflight
            for device in range(num_gpus):
                preflight(config, device)
        if nodes:
            returncode = multinode.launch(nodes, run_dir, commands, args.timeout)
        else:
            returncode = subprocess.run(command, check=False, env=worker_env).returncode
        if returncode != 0:
            raise SystemExit(returncode if returncode > 0 else 128 - returncode)
        for filename in ("summary.csv", "iterations.csv"):
            if not (run_dir / filename).is_file():
                returncode = 1
                raise RuntimeError(f"Workers exited without required result {filename}")
    finally:
        with (run_dir / "run.json").open() as stream:
            run = json.load(stream)
        run["status"] = "completed" if returncode == 0 else "failed"
        run["worker_returncode"] = returncode
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        if returncode != 0:
            error = sys.exc_info()[1]
            run["failure"] = {
                "type": type(error).__name__ if error is not None else "WorkerFailure",
                "message": str(error) if error is not None else f"Worker exit code {returncode}",
            }
        write_json(run_dir / "run.json", run)
        if returncode != 0:
            print(f"Run failed; metadata: {run_dir / 'run.json'} (see error above)", file=sys.stderr)


if __name__ == "__main__":
    main()
