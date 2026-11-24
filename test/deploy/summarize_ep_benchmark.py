#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Summarize unified EP benchmark logs written by run-ep-benchmark.sh.

Reads ep-<mode>-<backend>.log (benchmark output) and ep-<mode>-<backend>.status (exit code)
from a results directory and writes there:
  results.json  every parsed metric, plus versions and workload
  results.csv   one row per (mode, backend)
  summary.md    comparison tables for the Azure Pipelines run summary page
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
from pathlib import Path

BACKEND_NAMES = {"mscclpp": "MSCCL++ EP", "nccl": "NCCL-EP", "deepep": "DeepEP"}
REFERENCE_BACKEND = "mscclpp"

_NUM = r"([0-9.]+)"
_HOST_LINE = re.compile(rf"^(Dispatch|Combine)(?: \(.+?\))?: avg={_NUM} us, min={_NUM} us, max={_NUM} us")
_HOST_TOTAL = re.compile(rf"^Total \(D\+C\): avg={_NUM} us, min={_NUM} us, max={_NUM} us")
_KERNEL_DISPATCH = re.compile(rf"^Dispatch:\s+avg={_NUM} us, min={_NUM} us .*max={_NUM} us")
_KERNEL_COMBINE = re.compile(rf"^Combine:\s+avg={_NUM} us, min={_NUM} us, max={_NUM} us")
_KERNEL_TOTAL = re.compile(rf"^Total \(D\+C\): avg={_NUM} us \(dispatch avg \+ combine avg\), floor={_NUM} us")
_SUMMARY = re.compile(r"^=== Summary \[(\w+)\] \((\w+), across (\d+) ranks\) ===")
_CONFIG = re.compile(r"^Resolved config: (.+)$")
_BYTES = re.compile(rf"^Logical routed bytes: dispatch={_NUM} MB .*combine={_NUM} MB")


def parse_log(text: str) -> dict:
    """Parse one backend's benchmark summary. Returns {} when no summary was printed."""
    result: dict = {}
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if m := _SUMMARY.match(line):
            result["ranks"] = int(m.group(3))
            continue
        if not result:
            continue
        if m := _CONFIG.match(line):
            result["config"] = dict(item.split("=", 1) for item in m.group(1).split(", ") if "=" in item)
        elif line.startswith("--- Host-observed"):
            section = "host"
        elif line.startswith("--- Kernel-only"):
            section = "kernel"
        elif section == "host" and (m := _HOST_LINE.match(line)):
            phase = m.group(1).lower()
            result[f"host_{phase}_avg_us"], result[f"host_{phase}_min_us"], result[f"host_{phase}_max_us"] = (
                float(v) for v in m.groups()[1:]
            )
        elif section == "host" and (m := _HOST_TOTAL.match(line)):
            result["host_total_avg_us"], result["host_total_min_us"], result["host_total_max_us"] = (
                float(v) for v in m.groups()
            )
        elif section == "kernel" and (m := _KERNEL_DISPATCH.match(line)):
            result["kernel_dispatch_avg_us"], result["kernel_dispatch_min_us"], result["kernel_dispatch_max_us"] = (
                float(v) for v in m.groups()
            )
        elif section == "kernel" and (m := _KERNEL_COMBINE.match(line)):
            result["kernel_combine_avg_us"], result["kernel_combine_min_us"], result["kernel_combine_max_us"] = (
                float(v) for v in m.groups()
            )
        elif section == "kernel" and (m := _KERNEL_TOTAL.match(line)):
            result["kernel_total_avg_us"], result["kernel_total_floor_us"] = (float(v) for v in m.groups())
        elif m := _BYTES.match(line):
            result["dispatch_mb"], result["combine_mb"] = (float(v) for v in m.groups())
    return result


def collect(results_dir: Path, modes: list[str], backends: list[str]) -> list[dict]:
    rows = []
    for mode in modes:
        for backend in backends:
            log = results_dir / f"ep-{mode}-{backend}.log"
            status_file = results_dir / f"ep-{mode}-{backend}.status"
            exit_code = int(status_file.read_text().strip()) if status_file.exists() else None
            metrics = parse_log(log.read_text(errors="replace")) if log.exists() else {}
            if exit_code == 0 and "kernel_total_avg_us" in metrics:
                status = "ok"
            elif exit_code == 0 and metrics:
                status = "no kernel timing"
            elif exit_code is None and not log.exists():
                status = "not run"
            else:
                status = f"failed (exit {exit_code})"
            rows.append({"mode": mode, "backend": backend, "status": status, "exit_code": exit_code, **metrics})
    return rows


def _fmt(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "–"


def _speedup(rows: list[dict], row: dict) -> str:
    """How many times faster MSCCL++ EP is than `row` (kernel-only total). >1 means MSCCL++ is faster."""
    if row["backend"] == REFERENCE_BACKEND:
        return "baseline"
    ref = next((r for r in rows if r["mode"] == row["mode"] and r["backend"] == REFERENCE_BACKEND), None)
    ours = ref.get("kernel_total_avg_us") if ref else None
    theirs = row.get("kernel_total_avg_us")
    if not ours or not theirs:
        return "–"
    return f"{theirs / ours:.2f}×"


def write_markdown(path: Path, rows: list[dict], versions: dict, workload: str, ranks: str) -> None:
    lines = ["# Unified EP benchmark: MSCCL++ EP vs NCCL-EP vs DeepEP", ""]
    lines.append(
        f"Workload: `{workload}` on {ranks} GPUs. Times are microseconds per dispatch + combine, lower is better."
    )
    lines.append("")
    for mode in dict.fromkeys(r["mode"] for r in rows):
        mode_rows = [r for r in rows if r["mode"] == mode]
        lines += [
            f"## {mode.title()} mode",
            "",
            "| Backend | Layout | Dispatch | Combine | **Total** | MSCCL++ speedup | Host-observed total | Status |",
            "|---|---|--:|--:|--:|--:|--:|---|",
        ]
        for r in mode_rows:
            lines.append(
                f"| {BACKEND_NAMES.get(r['backend'], r['backend'])} "
                f"| {r.get('config', {}).get('layout', '–')} "
                f"| {_fmt(r.get('kernel_dispatch_avg_us'))} "
                f"| {_fmt(r.get('kernel_combine_avg_us'))} "
                f"| **{_fmt(r.get('kernel_total_avg_us'))}** "
                f"| {_speedup(mode_rows, r)} "
                f"| {_fmt(r.get('host_total_avg_us'))} "
                f"| {r['status']} |"
            )
        lines.append("")
    lines += [
        "Dispatch, Combine and Total are GPU kernel times (PyTorch profiler, averaged across ranks). "
        "MSCCL++ speedup is the other backend's Total divided by MSCCL++'s; above 1 means MSCCL++ is faster. "
        "Host-observed total is wall-clock time measured from the CPU and includes launch overhead.",
        "",
    ]
    if versions:
        lines += ["## Versions", "", "| Component | Version |", "|---|---|"]
        lines += [f"| {key} | {value} |" for key, value in versions.items()]
        lines.append("")
    path.write_text("\n".join(lines))


CSV_FIELDS = [
    "mode", "backend", "status", "exit_code", "ranks", "layout",
    "kernel_dispatch_avg_us", "kernel_dispatch_min_us", "kernel_combine_avg_us",
    "kernel_total_avg_us", "kernel_total_floor_us",
    "host_dispatch_avg_us", "host_combine_avg_us", "host_total_avg_us", "host_total_min_us", "host_total_max_us",
    "dispatch_mb", "combine_mb",
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--modes", default="latency throughput")
    parser.add_argument("--backends", default="nccl mscclpp deepep")
    parser.add_argument("--workload", default="")
    parser.add_argument("--versions-file", type=Path)
    args = parser.parse_args()

    rows = collect(args.results_dir, args.modes.split(), args.backends.split())
    versions = {}
    if args.versions_file and args.versions_file.exists():
        versions = json.loads(args.versions_file.read_text())
    ranks = ", ".join(sorted({str(r["ranks"]) for r in rows if "ranks" in r})) or "?"

    (args.results_dir / "results.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "workload": args.workload,
                "versions": versions,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    with open(args.results_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "layout": r.get("config", {}).get("layout", "")})
    write_markdown(args.results_dir / "summary.md", rows, versions, args.workload, ranks)
    print((args.results_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
