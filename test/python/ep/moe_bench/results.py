# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Persistence and metrics for fixed-workload reference MoE runs."""

import csv
import json
import math
import statistics
from pathlib import Path

PHASES = ("dispatch", "mlp", "combine", "total")


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def latency_stats(values, *, allow_zero=False):
    if not values or any(not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero) for value in values):
        raise ValueError("Latency samples must be finite, positive (nonnegative for phases), and nonempty")
    ordered = sorted(values)

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "mean_latency_us": statistics.mean(values),
        "p50_latency_us": percentile(0.5),
        "p95_latency_us": percentile(0.95),
    }


def summarize(latencies, tokens_per_rank):
    if not latencies or not latencies[0] or any(len(row) != len(latencies[0]) for row in latencies):
        raise ValueError("Each rank must supply the same nonzero number of iterations")
    per_rank = []
    for rank, samples in enumerate(latencies):
        for sample in samples:
            if set(sample) != {f"{phase}_us" for phase in PHASES}:
                raise ValueError("Each iteration must contain dispatch_us, mlp_us, combine_us, and total_us")
        for phase in PHASES:
            stats = latency_stats([sample[f"{phase}_us"] for sample in samples], allow_zero=phase != "total")
            per_rank.append({"scope": "rank", "rank": rank, "phase": phase, **stats, "throughput_tokens_per_s": ""})
    global_stats = latency_stats([max(sample["total_us"] for sample in values) for values in zip(*latencies)])
    throughput = len(latencies) * tokens_per_rank * 1e6 / global_stats["mean_latency_us"]
    rows = [{"scope": "global", "rank": "", "phase": "total", **global_stats, "throughput_tokens_per_s": throughput}]
    rows.extend(per_rank)
    return rows


def save_results(run_dir, latencies, config, metadata, expert_load):
    run_dir = Path(run_dir)
    rows = summarize(latencies, config["input"]["tokens_per_rank"])
    with (run_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "iterations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["iteration", "rank", *(f"{phase}_us" for phase in PHASES)])
        writer.writeheader()
        for iteration, durations in enumerate(zip(*latencies)):
            writer.writerows({"iteration": iteration, "rank": rank, **value} for rank, value in enumerate(durations))
    with (run_dir / "run.json").open() as stream:
        run = json.load(stream)
    run["metadata"].update(metadata)
    run["expert_load"] = expert_load
    run["schema_version"] = 2
    run["measurement"] = {
        "latency": "CUDA event interval from dispatch through combine, including host launch gaps",
        "distributed_latency": "maximum rank duration per iteration, not a globally synchronized wall clock",
        "throughput": "original global input tokens / mean distributed iteration duration",
        "routing": "fixed; generation and expert counts excluded from timing",
        "compute": "BF16 PyTorch reference SwiGLU, not optimized grouped GEMM",
        "warmup_included": False,
        "percentiles": "linear interpolation",
        "phase_timing": "adjacent CUDA events on the same stream; no synchronization between phases",
        "event_initialization": "all native CUDA events materialized before warmup; warmup includes phase recording",
        "mlp": "all expert GEMMs, SwiGLU, and per-expert launch gaps; not isolated GEMM kernel time",
        "timing_overhead": "phase event recording is included; communication intervals include peer waits",
        "summary_phases": "per-rank only; global total is not the sum of independently slowest phases",
    }
    write_json(run_dir / "run.json", run)
    print("\nMoE benchmark (PyTorch reference compute)", flush=True)
    print(" | ".join(f"{name}: {value}" for name, value in run["presets"].items()))
    settings = config["benchmark"]
    print(
        f"GPUs: {len(latencies)} | Tokens/rank: {config['input']['tokens_per_rank']} | "
        f"Warmup: {settings['warmup']} | Iterations: {settings['iterations']}"
    )
    print(f"\n{'GPU elapsed latency':<24} {'Mean (us)':>12} {'P50 (us)':>12} {'P95 (us)':>12}")
    for row in rows:
        label = (
            "End-to-end (max rank)"
            if row["scope"] == "global"
            else f"Rank {row['rank']} {row['phase'].upper() if row['phase'] == 'mlp' else row['phase']}"
        )
        print(
            f"{label:<24} {row['mean_latency_us']:12.2f} "
            f"{row['p50_latency_us']:12.2f} {row['p95_latency_us']:12.2f}"
        )
    print(f"\nGlobal throughput: {rows[0]['throughput_tokens_per_s']:.2f} tokens/s")
    counts = expert_load["assignments_per_expert"]
    print(f"Expert load (min/max): {min(counts)} / {max(counts)} assignments")
    print(f"GPU load imbalance: {expert_load['rank_max_over_mean']:.2f}x max/mean")
    print(f"Saved: {run_dir}", flush=True)
