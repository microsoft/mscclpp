# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Launch a single-node reference MoE benchmark from named YAML presets."""

import argparse
from datetime import datetime, timezone
import json
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
    parser.add_argument("--gpus", type=int, choices=(2, 4), help="Override environment.num_gpus")
    parser.add_argument(
        "--graph", choices=("none", "mlp"), help="Override environment.cuda_graph (none disables graphs)"
    )
    parser.add_argument("--routing", required=True)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    config = load_config(
        args.config,
        model=args.model,
        input_name=args.input,
        routing=args.routing,
        environment_overrides={
            "communication_backend": args.backend,
            "mode": args.mode,
            "layout": args.layout,
            "compute_backend": args.compute_backend,
            "num_gpus": args.gpus,
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
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(config["benchmark"]["output_dir"]).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={config['environment']['num_gpus']}",
        str(Path(__file__).with_name("worker.py").resolve()),
        "--run-dir",
        str(run_dir),
    ]
    root = Path(__file__).resolve().parents[4]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
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
        "metadata": {"git_commit": revision, "git_dirty": bool(status), "git_status": status.splitlines()},
    }
    write_json(run_dir / "run.json", run)
    print(f"Launching reference MoE benchmark; output: {run_dir}", flush=True)
    returncode = None
    try:
        import torch

        available = torch.cuda.device_count()
        required = config["environment"]["num_gpus"]
        if available < required:
            raise RuntimeError(f"Requested {required} GPUs, but only {available} CUDA GPUs are visible")
        if config["environment"]["compute_backend"] == "cutlass_flashinfer":
            if __package__:
                from .compute_flashinfer import preflight
            else:
                from compute_flashinfer import preflight
            for device in range(required):
                preflight(config, device)
        returncode = subprocess.run(command, check=False).returncode
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
