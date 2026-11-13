# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Sweep MegaMoE benchmark candidates and build a token-range policy profile.

``sweep`` runs ``mscclpp.ext.megamoe.benchmark`` once per candidate through a user-provided
multi-rank launcher and writes one JSON result per run. ``build`` reads such results and, for
each workload and token count, selects the candidate (kernel, ``sm_margin``, fused FC1 delay)
with the lowest mean latency across routing patterns, emitting ``megamoe_policies.json``
entries whose token ranges end at each measured token count.

Example::

    python -m mscclpp.ext.megamoe.tune sweep --grid grid.json --out results/ \\
        --launcher "mpirun -np 32 --hostfile hosts python -m mscclpp.ext.megamoe.benchmark"
    python -m mscclpp.ext.megamoe.tune build results/*.json --output megamoe_policies.json
"""

import argparse
from dataclasses import asdict, fields
import datetime
import glob
import itertools
import json
from pathlib import Path
import shlex
import subprocess

from .jit import W4A8KernelConfig
from .policy import PROFILE_KIND, PROFILE_VERSION, gpu_key, mean_latency, validate_megamoe_policies

_CONFIG_FIELDS = tuple(field.name for field in fields(W4A8KernelConfig))
_LICENSE = "Copyright (c) Microsoft Corporation. Licensed under the MIT License."


def _kernel_flags(kernel):
    flags = []
    for name, value in asdict(W4A8KernelConfig(**kernel)).items():
        if name == "tile_n":
            continue
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            flags.append(option if value else "--no-" + option[2:])
        else:
            flags.extend((option, str(value)))
    return flags


def _sweep(args):
    grid = json.loads(Path(args.grid).read_text())
    base = grid.get("base_args", [])
    kernels = grid["kernels"]
    margins = grid.get("sm_margins", [0])
    patterns = grid.get("patterns", ["balanced"])
    tokens = grid["tokens"]
    fused = grid.get("fused", [False])
    delays = grid.get("fc1_delays_ns", [None])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    launcher = shlex.split(args.launcher)
    for (name, kernel), margin, pattern, count, mode in itertools.product(
        kernels.items(), margins, patterns, tokens, fused
    ):
        for delay in delays if mode else [None]:
            tag = f"{'fused' if mode else 'routed'}-{name}-{pattern}-t{count}-m{margin}" + (
                "" if delay is None else f"-d{delay}"
            )
            path = out / f"{tag}.json"
            if path.exists() and not args.force:
                continue
            command = [
                *launcher,
                *base,
                "--tokens",
                str(count),
                "--sm-margin",
                str(margin),
                "--routing-pattern",
                pattern,
                *_kernel_flags(kernel),
                "--json-output",
                str(path),
            ]
            if mode:
                command.append("--fused-shared-expert")
                if delay is not None:
                    command.extend(("--shared-fc1-delay-ns", str(delay)))
            print(" ".join(shlex.quote(part) for part in command), flush=True)
            if not args.dry_run:
                status = subprocess.run(command, timeout=args.timeout, check=False).returncode
                if status:
                    print(f"FAILED ({status}): {tag}", flush=True)


def _record(path):
    data = json.loads(Path(path).read_text())
    c = data["configuration"]
    fused = bool(c.get("fused_shared_expert"))
    metric = "fused_latency_us_max_across_ranks" if fused else "latency_us_max_across_ranks"
    kernel_config = {k: v for k, v in data["ranks"][0]["kernel_config"].items() if k in _CONFIG_FIELDS}
    clamp = c.get("gate_up_clamp", -1.0)
    match = {
        "world_size": c["world_size"],
        "hidden": c["hidden"],
        "intermediate": c["intermediate"],
        "local_experts": c["experts"] // c["world_size"],
        "top_k": c["top_k"],
        "weight_dtype": c["weight_dtype"],
        "activation_dtype": c["activation_dtype"],
        "gate_up_clamp": "negative" if clamp < 0 else "nonnegative",
        "shared": {"hidden": c["shared_hidden"], "intermediate": c["shared_intermediate"]} if fused else None,
    }
    runtime = {"sm_margin": c["sm_margin"], "fc1_delay_ns": data.get("fused_fc1_delay_ns") if fused else None}
    return {
        "gpu": gpu_key(data["hardware"]),
        "match": match,
        "tokens": c["tokens"],
        "pattern": c["routing_pattern"],
        "kernel": asdict(W4A8KernelConfig(**kernel_config)),
        "runtime": runtime,
        "latency_us": data[metric]["median"],
    }


def build_policies(records, *, provenance=None):
    """Return a validated policy profile from benchmark records produced by ``_record``."""
    groups = {}
    for r in records:
        key = (r["gpu"], json.dumps(r["match"], sort_keys=True), r["tokens"])
        candidate = json.dumps({"kernel": r["kernel"], "runtime": r["runtime"]}, sort_keys=True)
        group = groups.setdefault(key, {"patterns": set(), "candidates": {}})
        group["patterns"].add(r["pattern"])
        per_pattern = group["candidates"].setdefault(candidate, {})
        previous = per_pattern.get(r["pattern"])
        per_pattern[r["pattern"]] = r["latency_us"] if previous is None else min(previous, r["latency_us"])
    selected = {}
    for (gpu, match, count), group in groups.items():
        patterns = sorted(group["patterns"])
        scored = [
            (mean_latency(per_pattern.get(p) for p in patterns), candidate, per_pattern)
            for candidate, per_pattern in group["candidates"].items()
        ]
        best = min(scored, key=lambda item: item[0])
        if best[0] == float("inf"):
            continue
        selected.setdefault((gpu, match), []).append((count, best, len(scored), patterns))
    profile = {"_license": _LICENSE, "version": PROFILE_VERSION, "kind": PROFILE_KIND}
    for (gpu, match), items in sorted(selected.items()):
        section = profile.setdefault(gpu, {"entries": [], "provenance": dict(provenance or {})})
        if not section["provenance"]:
            section["provenance"] = {"source": "mscclpp.ext.megamoe.tune build"}
        low = 1
        for count, (mean, candidate, per_pattern), total, patterns in sorted(items):
            chosen = json.loads(candidate)
            section["entries"].append(
                {
                    "match": {**json.loads(match), "tokens": [low, count]},
                    "kernel": chosen["kernel"],
                    "runtime": {k: v for k, v in chosen["runtime"].items() if v is not None},
                    "measurement": {
                        "metric": "median latency (us), slowest rank",
                        "tokens": count,
                        "mean_us": round(mean, 3),
                        "per_pattern_us": {p: round(per_pattern[p], 3) for p in patterns},
                        "candidates": total,
                    },
                }
            )
            low = count + 1
    return validate_megamoe_policies(profile)


def _build(args):
    paths = sorted({path for pattern in args.results for path in glob.glob(pattern)})
    records = []
    for path in paths:
        try:
            records.append(_record(path))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            print(f"skipping {path}: {error}")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    profile = build_policies(
        records,
        provenance={"created_at": now, "source": f"mscclpp.ext.megamoe.tune build ({len(records)} results)"},
    )
    Path(args.output).write_text(json.dumps(profile, indent=2) + "\n")
    print(f"wrote {args.output} from {len(records)} results")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep", help="run benchmark candidates from a grid JSON")
    sweep.add_argument(
        "--grid",
        required=True,
        help='JSON: {"kernels": {name: W4A8KernelConfig fields}, "tokens": [...], "patterns": [...], '
        '"sm_margins": [...], "fused": [false, true], "fc1_delays_ns": [...], "base_args": [...]}',
    )
    sweep.add_argument("--out", required=True, help="directory for per-run benchmark JSON results")
    sweep.add_argument("--launcher", required=True, help="multi-rank command that runs the benchmark module")
    sweep.add_argument("--timeout", type=float, default=600)
    sweep.add_argument("--force", action="store_true", help="rerun cases whose result already exists")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.set_defaults(func=_sweep)
    build = sub.add_parser("build", help="select per-token-range policies from benchmark JSON results")
    build.add_argument("results", nargs="+", help="result JSON files or glob patterns")
    build.add_argument("--output", required=True)
    build.set_defaults(func=_build)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
