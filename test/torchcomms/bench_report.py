#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate performance report and figures from TorchComms benchmark results.

Reads TorchComms allreduce results and produces:
  - report.txt:    formatted performance table
  - latency.png:   latency vs message size (log-log)
  - bandwidth.png: bus bandwidth vs message size

Not meant to be run directly — called by run_benchmarks.sh.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def format_size(nbytes):
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f}MB"
    elif nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.0f}KB"
    return f"{nbytes}B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchcomms-json", required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    with open(args.torchcomms_json) as f:
        tc_data = json.load(f)

    tc_results = {entry["size"]: entry for entry in tc_data}
    tc_sizes = sorted(tc_results.keys())

    if not tc_sizes:
        print("ERROR: No results found.", file=sys.stderr)
        sys.exit(1)

    # --- Algorithm region spans ---
    algo_regions = []
    prev_algo = tc_results[tc_sizes[0]].get("algorithm", "")
    region_start = tc_sizes[0]
    for i in range(1, len(tc_sizes)):
        cur_algo = tc_results[tc_sizes[i]].get("algorithm", "")
        if cur_algo != prev_algo:
            algo_regions.append((region_start, tc_sizes[i - 1], prev_algo))
            region_start = tc_sizes[i]
            prev_algo = cur_algo
    algo_regions.append((region_start, tc_sizes[-1], prev_algo))

    algo_colors = {
        "allpair_packet": "#E3F2FD",
        "nvls_packet": "#FFF3E0",
        "packet": "#E8F5E9",
        "nvls_warp_pipeline": "#F3E5F5",
        "nvls_block_pipeline": "#FFF9C4",
    }

    def add_algo_regions(ax, ymax):
        for xmin, xmax, algo in algo_regions:
            color = algo_colors.get(algo, "#F5F5F5")
            ax.axvspan(xmin * 0.7, xmax * 1.4, alpha=0.3, color=color, zorder=0)
            label_x = (xmin * xmax) ** 0.5
            ax.text(
                label_x,
                ymax * 0.85,
                algo.replace("_", "\n"),
                fontsize=7,
                ha="center",
                va="top",
                style="italic",
                color="#555555",
            )

    # --- Report ---
    lines = []
    lines.append(f"MSCCL++ AllReduce via TorchComms — {args.nproc}x NVIDIA H100 80GB (NVSwitch)")
    lines.append("")
    lines.append(f"{'Size':<10} {'Time(us)':<12} {'AlgBW(GB/s)':<14} {'BusBW(GB/s)':<14} {'Algorithm':<30}")
    lines.append("-" * 84)

    for size in tc_sizes:
        r = tc_results[size]
        lines.append(
            f"{format_size(size):<10} "
            f"{r['time_us']:<12.1f} "
            f"{r.get('algbw_gbps', 0):<14.1f} "
            f"{r['busbw_gbps']:<14.1f} "
            f"{r.get('algorithm', ''):<30}"
        )

    report_path = os.path.join(args.outdir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- Latency figure ---
    tc_times = [tc_results[s]["time_us"] for s in tc_sizes]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(tc_sizes, tc_times, "o-", linewidth=2.5, markersize=7, label="MSCCL++ via TorchComms", color="#2196F3")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Message Size", fontsize=12)
    ax.set_ylabel("Latency (μs)", fontsize=12)
    ax.set_title(
        f"MSCCL++ AllReduce Latency — {args.nproc}x NVIDIA H100 80GB (single-node, NVSwitch)",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    add_algo_regions(ax, max(tc_times))

    tick_sizes = [
        s
        for s in tc_sizes
        if s
        in [
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            4 * 1024 * 1024,
            16 * 1024 * 1024,
            64 * 1024 * 1024,
            128 * 1024 * 1024,
        ]
    ]
    if tick_sizes:
        ax.set_xticks(tick_sizes)
        ax.set_xticklabels([format_size(s) for s in tick_sizes], rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "latency.png"), dpi=150)
    plt.close()

    # --- Bandwidth figure ---
    tc_algbws = [tc_results[s].get("algbw_gbps", 0) for s in tc_sizes]
    tc_busbws = [tc_results[s]["busbw_gbps"] for s in tc_sizes]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(tc_sizes, tc_busbws, "o-", linewidth=2.5, markersize=7, label="Bus Bandwidth", color="#2196F3")
    ax.plot(tc_sizes, tc_algbws, "s--", linewidth=2, markersize=6, label="Algorithm Bandwidth", color="#FF9800")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Message Size", fontsize=12)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=12)
    ax.set_title(
        f"MSCCL++ AllReduce Bandwidth — {args.nproc}x NVIDIA H100 80GB (single-node, NVSwitch)",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    add_algo_regions(ax, max(max(tc_algbws), max(tc_busbws)))

    if tick_sizes:
        ax.set_xticks(tick_sizes)
        ax.set_xticklabels([format_size(s) for s in tick_sizes], rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "bandwidth.png"), dpi=150)
    plt.close()

    print(f"Report: {report_path}")
    print(f"Figures: {args.outdir}/latency.png, {args.outdir}/bandwidth.png")


if __name__ == "__main__":
    main()
