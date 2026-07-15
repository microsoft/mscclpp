#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unified EP low-latency benchmark driver.

Runs a common low-latency dispatch/combine workload -- identical tokens,
experts, hidden size, top-k, warmup and iteration counts -- against a
selectable expert-parallel library, then prints one normalized summary.

Backends (``--ep-lib``):

* ``mscclpp-cpp`` -- the pure-C++ ``MoERuntime`` benchmark launched with MPI.
* ``nccl-ep``  -- NVIDIA NCCL-EP's ``contrib/nccl_ep/ep_bench`` binary launched
  with ``mpirun`` (HPCX).
* ``both`` / ``all`` -- run mscclpp-cpp then nccl-ep, side by side.

For the mscclpp-vs-nccl-ep *Python* API comparison, use ``run_ep_bench_python.py``.

Both backends emit the identical ``=== Summary (Low Latency, across N ranks) ===``
block, so a single parser reads either one.

NCCL-EP dynamically links its shared libraries (``libnccl.so``, ``libnccl_ep.so``).
Point the driver at the correct build with ``--nccl-lib-path`` (falls back to the
``NCCL_LIB_PATH`` environment variable, else the ``lib`` directory beside the
``--nccl-ep-bench`` build tree); that directory is prepended to ``LD_LIBRARY_PATH``
for the ``ep_bench`` process so the intended NCCL is loaded.

Scope: single or multi-node; pass ``--nodes`` to launch the MPI backends across
multiple hosts.

Examples
--------
Compare both libraries, 4 GPUs, e128::

    python run_ep_bench.py --ep-lib both -e 128 -t 128 -d 7168 -k 8 -w 10 -i 50 \
        --nccl-lib-path /opt/microsoft/mrc/ep/nccl/build/lib

Just mscclpp-cpp with kernel-only timing::

    python run_ep_bench.py --ep-lib mscclpp-cpp -e 128 --kernel-only

Print the commands without running them::

    python run_ep_bench.py --ep-lib both -e 128 --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_cupti_paths() -> tuple[str, str]:
    target_dirs = sorted(glob.glob(os.path.join(CUDA_HOME, "targets", "*")))
    include_candidates = [os.path.join(path, "include") for path in target_dirs]
    include_candidates.append(os.path.join(CUDA_HOME, "extras", "CUPTI", "include"))
    library_candidates = [os.path.join(path, "lib") for path in target_dirs]
    library_candidates.append(os.path.join(CUDA_HOME, "extras", "CUPTI", "lib64"))

    include_dir = next(
        (path for path in include_candidates if os.path.isfile(os.path.join(path, "cupti.h"))),
        "",
    )
    library_dir = next(
        (path for path in library_candidates if glob.glob(os.path.join(path, "libcupti.so*"))),
        "",
    )
    if not include_dir or not library_dir:
        raise SystemExit(f"CUPTI was not found under CUDA_HOME={CUDA_HOME}")
    return include_dir, library_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified EP low-latency benchmark driver (mscclpp EP vs NCCL-EP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ep-lib",
        required=True,
        choices=["mscclpp-cpp", "nccl-ep", "both", "all"],
        help="which expert-parallel library to benchmark. "
        "mscclpp-cpp=MoERuntime (pure C++), nccl-ep=ep_bench. both/all=mscclpp-cpp+nccl-ep.",
    )
    p.add_argument(
        "-a",
        "--algorithm",
        default="ll",
        choices=["ll", "low-latency"],
        help="algorithm mode (only low-latency is wired up here)",
    )

    # Shared problem shape -- passed to whichever backend is selected.
    p.add_argument("-t", "--num-tokens", type=int, default=128, help="tokens per rank")
    p.add_argument(
        "-d",
        "--hidden",
        type=int,
        default=7168,
        choices=(4096, 6656, 7168, 8192, 9216),
        help="hidden dimension",
    )
    p.add_argument("-k", "--num-topk", type=int, default=8, choices=range(1, 10), help="top-k experts per token")
    p.add_argument("-e", "--num-experts", type=int, default=256, help="global number of experts")
    p.add_argument("-w", "--num-warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("-i", "--num-iters", type=int, default=50, help="timed iterations")
    p.add_argument(
        "--dispatch-dtype",
        choices=("bf16", "fp8_e4m3"),
        default="bf16",
        help="MSCCL++ dispatch format; NCCL-EP runs keep their own configured format",
    )
    p.add_argument(
        "--combine-mode",
        choices=("rank_local_reduce", "direct_send"),
        default="rank_local_reduce",
        help="MSCCL++ low-latency combine mode",
    )
    p.add_argument("--num-blocks", type=int, default=130, help="MSCCL++ low-latency dispatch blocks")

    # Launch / fabric.
    p.add_argument("--nproc-per-node", type=int, default=4, help="GPUs (ranks) on this node")
    p.add_argument(
        "--nodes",
        default="",
        help="space-separated node IPs for a multi-node run (first = master). Empty = single " "local node.",
    )
    p.add_argument("--iface", default="", help="optional socket interface name (NCCL/GLOO/UCX)")
    p.add_argument("--hca", default="", help="optional comma-separated mscclpp HCA devices")

    # Kernel-timing options.
    p.add_argument("--cupti-inproc", action="store_true", help="alias for --kernel-only (kept for compatibility)")
    p.add_argument(
        "--kernel-only",
        action="store_true",
        help="compare KERNEL execution time only, stripping host launch overhead "
        "(what ep_bench's CUPTI reports). mscclpp-cpp uses --kernel-timing; nccl-ep uses "
        "ep_bench's built-in CUPTI KernelTimer. The unified table then leads with the "
        "kernel dispatch/combine times and a kernel D+C ratio.",
    )

    # nccl-ep backend.
    p.add_argument(
        "--nccl-lib-path",
        default=os.environ.get("NCCL_LIB_PATH", ""),
        help="directory with libnccl.so / libnccl_ep.so; prepended to LD_LIBRARY_PATH "
        "for ep_bench (falls back to $NCCL_LIB_PATH, else derived from --nccl-ep-bench)",
    )
    p.add_argument(
        "--nccl-ep-bench",
        default="/opt/microsoft/mrc/ep/nccl/build/test/nccl_ep/ep_bench",
        help="path to the NCCL-EP ep_bench binary",
    )
    p.add_argument("--hpcx", default="", help="HPCX install dir (for mpirun); autodetected under /opt if empty")
    p.add_argument(
        "--layout",
        default="em",
        choices=["em", "rm", "fl"],
        help="nccl-ep dispatch layout (em=expert-major, matches mscclpp LL)",
    )

    # mscclpp-cpp backend (pure C++ MoERuntime binary).
    p.add_argument(
        "--mscclpp-cpp-bench",
        default=os.path.join(_HERE, "build", "mscclpp_ep_bench"),
        help="path to the mscclpp_ep_bench C++ binary (built via test/python/ep/CMakeLists.txt)",
    )

    p.add_argument("--dry-run", action="store_true", help="print the backend command(s) and exit")
    args = p.parse_args()

    # These free-form values are interpolated into shell command strings that are
    # executed via bash; constrain them to safe characters to prevent injection
    # and to fail fast on values that would break the launch (spaces, quotes, ...).
    if args.nodes and not re.fullmatch(r"[0-9A-Za-z._:-]+( [0-9A-Za-z._:-]+)*", args.nodes):
        raise SystemExit("--nodes must be space-separated hostnames/IPs")
    if args.iface and not re.fullmatch(r"[0-9A-Za-z._:-]+", args.iface):
        raise SystemExit("--iface must be a valid network interface name")
    if args.hca and not re.fullmatch(r"[0-9A-Za-z._,-]+", args.hca):
        raise SystemExit("--hca must be comma-separated HCA device names")
    if args.dispatch_dtype != "bf16" and args.ep_lib in ("nccl-ep", "both", "all"):
        raise SystemExit("FP8 unified comparison is unsupported because the NCCL-EP command is configured for BF16")
    if args.num_tokens <= 0 or args.num_experts <= 0 or args.nproc_per_node <= 0:
        raise SystemExit("tokens, experts, and nproc-per-node must be positive")
    if args.num_topk > args.num_experts:
        raise SystemExit("num-topk must not exceed num-experts")
    if args.num_warmup < 0 or args.num_iters <= 0:
        raise SystemExit("num-warmup must be non-negative and num-iters must be positive")
    num_nodes = max(1, len(args.nodes.split()))
    num_ranks = num_nodes * args.nproc_per_node
    if args.num_experts % num_ranks != 0:
        raise SystemExit("num-experts must be divisible by the total number of ranks")
    if args.ep_lib in ("mscclpp-cpp", "both", "all"):
        if not num_ranks + 2 <= args.num_blocks <= 130:
            raise SystemExit("num-blocks must be in [total ranks + 2, 130]")

    return args


# ----------------------------------------------------------------------------
# Parsing the common "=== Summary (Low Latency ...) ===" block.
# ----------------------------------------------------------------------------
@dataclass
class Phase:
    avg: float = float("nan")
    min: float = float("nan")
    max: float = float("nan")


@dataclass
class LLResult:
    ep_lib: str
    num_ranks: int = 0
    dispatch: Phase = field(default_factory=Phase)
    combine: Phase = field(default_factory=Phase)
    total: Phase = field(default_factory=Phase)
    # Kernel-only dispatch/combine (avg/min/max) from mscclpp --cupti-inproc or
    # ep_bench's CUPTI KernelTimer, if present.
    kdispatch: Optional[Phase] = None
    kcombine: Optional[Phase] = None
    ok: bool = False


_HOST_RE = {
    "dispatch": re.compile(r"^Dispatch \([^)]+\):\s+avg=([\d.]+)\s*us,\s*min=([\d.]+)\s*us,\s*max=([\d.]+)\s*us"),
    "combine": re.compile(r"^Combine \([^)]+\):\s+avg=([\d.]+)\s*us,\s*min=([\d.]+)\s*us,\s*max=([\d.]+)\s*us"),
    "total": re.compile(r"^Total \(D\+C\):\s+avg=([\d.]+)\s*us,\s*min=([\d.]+)\s*us,\s*max=([\d.]+)\s*us"),
}
_RANKS_RE = re.compile(r"=== Summary \(Low Latency, across (\d+) ranks\) ===")
# Kernel-only Dispatch line, two formats (both carry avg/min/max):
#   mscclpp in-process CUPTI: ``Dispatch: min=M us (representative)  [avg=A, max=X us -- ...]``
#   ep_bench CUPTI:           ``Dispatch: avg=A us, min=M us, max=X us``
_KDISP_REP_RE = re.compile(r"^Dispatch:\s+min=([\d.]+)\s*us \(representative\)\s*\[avg=([\d.]+),\s*max=([\d.]+)")
_KDISP_AMM_RE = re.compile(r"^Dispatch:\s+avg=([\d.]+)\s*us,\s*min=([\d.]+)\s*us,\s*max=([\d.]+)\s*us")
_KCOMB_REP_RE = re.compile(r"^Combine:\s+min=([\d.]+)\s*us \(representative\)\s*\[avg=([\d.]+),\s*max=([\d.]+)")
# Kernel-only Combine line (both backends): ``Combine: avg=A us, min=M us, max=X us`` (no ``(BF16)``).
_KCOMB_RE = re.compile(r"^Combine:\s+avg=([\d.]+)\s*us,\s*min=([\d.]+)\s*us,\s*max=([\d.]+)\s*us")


def parse_ll_summary(text: str, ep_lib: str) -> LLResult:
    res = LLResult(ep_lib=ep_lib)
    for raw in text.splitlines():
        line = raw.strip()
        m = _RANKS_RE.search(line)
        if m:
            res.num_ranks = int(m.group(1))
            continue
        for name, rx in _HOST_RE.items():
            m = rx.match(line)
            if m:
                ph = Phase(float(m.group(1)), float(m.group(2)), float(m.group(3)))
                setattr(res, name, ph)
        # Kernel-only dispatch, first occurrence only. The host lines carry
        # ``(BF16)`` so they never match these bare ``Dispatch:``/``Combine:`` forms.
        if res.kdispatch is None:
            m = _KDISP_REP_RE.match(line)
            if m:  # mscclpp: printed order is min, avg, max
                res.kdispatch = Phase(avg=float(m.group(2)), min=float(m.group(1)), max=float(m.group(3)))
                continue
            m = _KDISP_AMM_RE.match(line)
            if m:  # ep_bench: printed order is avg, min, max
                res.kdispatch = Phase(avg=float(m.group(1)), min=float(m.group(2)), max=float(m.group(3)))
                continue
        if res.kcombine is None and res.kdispatch is not None:
            m = _KCOMB_REP_RE.match(line)
            if m:
                res.kcombine = Phase(avg=float(m.group(2)), min=float(m.group(1)), max=float(m.group(3)))
                continue
            m = _KCOMB_RE.match(line)
            if m:
                res.kcombine = Phase(avg=float(m.group(1)), min=float(m.group(2)), max=float(m.group(3)))
    res.ok = res.dispatch.avg == res.dispatch.avg  # not NaN
    return res


# ----------------------------------------------------------------------------
# Backend command construction.
# ----------------------------------------------------------------------------
def _autodetect_hpcx() -> str:
    import glob

    cands = sorted(glob.glob("/opt/hpcx-*"))
    return cands[0] if cands else ""


def _mpi_launch(args, np_total):
    """Common mpirun prefix. Multi-node when --nodes lists >1 IP (writes a
    hostfile, adds an SSH launcher); otherwise a plain single-node launch."""
    nodes = args.nodes.split()
    setup = ""
    hostfile = ""
    if len(nodes) > 1:
        slots = args.nproc_per_node
        lines = "\\n".join(f"{ip} slots={slots}" for ip in nodes)
        hf = "/tmp/ep_unified_hostfile"
        setup = f"printf '{lines}\\n' > {hf} && "
        hostfile = (
            f"--hostfile {hf} " f'-mca plm_rsh_args "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" '
        )
    iface_arg = f"-mca btl_tcp_if_include {shlex.quote(args.iface)} " if args.iface else ""
    root_arg = "--allow-run-as-root " if os.geteuid() == 0 else ""
    return setup, (
        f"mpirun {root_arg}-np {np_total} {hostfile}--map-by ppr:{args.nproc_per_node}:node --bind-to none "
        f"-mca pml ob1 -mca btl self,vader,tcp {iface_arg}"
        f"-mca coll_hcoll_enable 0 -mca coll_ucc_enable 0 "
    )


def build_nccl_ep_cmd(args: argparse.Namespace) -> str:
    nccl_lib = args.nccl_lib_path
    if not nccl_lib:
        # Derive the libnccl / libnccl_ep directory from the ep_bench binary
        # instead of hard-coding it: <nccl>/build/test/nccl_ep/ep_bench ->
        # <nccl>/build/lib.
        bench_dir = os.path.dirname(os.path.abspath(args.nccl_ep_bench))
        nccl_lib = os.path.join(os.path.dirname(os.path.dirname(bench_dir)), "lib")
    hpcx = args.hpcx or _autodetect_hpcx()
    nodes = args.nodes.split()
    nnodes = max(1, len(nodes))
    np_total = nnodes * args.nproc_per_node
    mnnvl = 1 if nnodes > 1 else 0
    bench_flags = (
        f"-a ll -L {args.layout} -t {args.num_tokens} -d {args.hidden} -k {args.num_topk} "
        f"-e {args.num_experts} -w {args.num_warmup} -i {args.num_iters}"
    )
    setup, mpi_prefix = _mpi_launch(args, np_total)
    opal = f"-x OPAL_PREFIX={shlex.quote(hpcx)}/ompi " if hpcx else ""
    iface_env = (
        f"-x UCX_NET_DEVICES={shlex.quote(args.iface)} " f"-x NCCL_SOCKET_IFNAME={shlex.quote(args.iface)} "
        if args.iface
        else ""
    )
    mpi = (
        f"{mpi_prefix}"
        f"-x LD_LIBRARY_PATH -x PATH -x CUDA_HOME={shlex.quote(CUDA_HOME)} {opal}"
        f"{iface_env}-x UCX_TLS=tcp,sm,self,cuda_copy -x UCX_HANDLE_ERRORS=none "
        f"-x NCCL_NET_PLUGIN=none "
        f"-x NCCL_IB_DISABLE=1 -x NCCL_MNNVL_ENABLE={mnnvl} "
        f"{shlex.quote(args.nccl_ep_bench)} {bench_flags}"
    )
    activation = f"source {shlex.quote(hpcx)}/hpcx-init.sh && hpcx_load && " if hpcx else ""
    return f"{activation}" f"export LD_LIBRARY_PATH={shlex.quote(nccl_lib)}:$LD_LIBRARY_PATH && " f"{setup}{mpi}"


def build_mscclpp_cpp_cmd(args: argparse.Namespace) -> str:
    """Pure-C++ mscclpp_ep_bench (MoERuntime), launched with mpirun -- no Python."""
    hpcx = args.hpcx or _autodetect_hpcx()
    nodes = args.nodes.split()
    nnodes = max(1, len(nodes))
    np_total = nnodes * args.nproc_per_node
    bench_flags = (
        f"-a ll -t {args.num_tokens} -d {args.hidden} -k {args.num_topk} "
        f"-e {args.num_experts} -w {args.num_warmup} -i {args.num_iters} "
        f"--dispatch-dtype {args.dispatch_dtype} --combine-mode {args.combine_mode} --num-blocks {args.num_blocks}"
    )
    if args.kernel_only or args.cupti_inproc:
        bench_flags += " --kernel-timing"
    setup, mpi_prefix = _mpi_launch(args, np_total)
    env_exports = ""
    if args.hca:
        env_exports += f"-x MSCCLPP_HCA_DEVICES={shlex.quote(args.hca)} "
    if args.iface:
        env_exports += (
            f"-x NCCL_SOCKET_IFNAME={shlex.quote(args.iface)} " f"-x MSCCLPP_SOCKET_IFNAME={shlex.quote(args.iface)} "
        )
    mpi = (
        f"{mpi_prefix}"
        f"-x LD_LIBRARY_PATH -x PATH "
        f"{env_exports}"
        f"{shlex.quote(args.mscclpp_cpp_bench)} {bench_flags}"
    )
    _, cupti_lib = _find_cupti_paths()
    activation = f"source {shlex.quote(hpcx)}/hpcx-init.sh && hpcx_load && " if hpcx else ""
    return f"{activation}" f"export LD_LIBRARY_PATH={shlex.quote(cupti_lib)}:$LD_LIBRARY_PATH && " f"{setup}{mpi}"


# ----------------------------------------------------------------------------
# Run + report.
# ----------------------------------------------------------------------------
def run_backend(ep_lib: str, cmd: str, dry_run: bool) -> Optional[LLResult]:
    print(f"\n########## ep-lib={ep_lib} ##########", flush=True)
    print(f"$ {cmd}\n", flush=True)
    if dry_run:
        return None
    proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        print(f"[warn] {ep_lib} exited rc={proc.returncode}", flush=True)
    res = parse_ll_summary(proc.stdout, ep_lib)
    if not res.ok:
        print(f"[warn] could not parse a Low-Latency summary from {ep_lib} output", flush=True)
        return None
    return res


def print_unified(results: list, kernel_only: bool = False) -> None:
    results = [r for r in results if r is not None]
    if not results:
        return
    has_kernel = all(r.kdispatch is not None and r.kcombine is not None for r in results)
    title = "kernel-only" if (kernel_only and has_kernel) else "host-observed"
    print(f"\n=== Unified EP Low-Latency Summary ({title}, us) ===")
    hdr = f"{'metric':<24}" + "".join(f"{r.ep_lib:>14}" for r in results)
    print(hdr)
    print("-" * len(hdr))

    def row(label, fn):
        print(f"{label:<24}" + "".join(f"{fn(r):>14.2f}" for r in results))

    if not (kernel_only and has_kernel):
        # Host-observed dispatch/combine/total, full avg/min/max.
        row("Host Dispatch avg", lambda r: r.dispatch.avg)
        row("Host Dispatch min", lambda r: r.dispatch.min)
        row("Host Dispatch max", lambda r: r.dispatch.max)
        row("Host Combine avg", lambda r: r.combine.avg)
        row("Host Combine min", lambda r: r.combine.min)
        row("Host Combine max", lambda r: r.combine.max)
        row("Host D+C avg", lambda r: r.total.avg)
    if has_kernel:
        if kernel_only:
            row("Kernel Dispatch repr", lambda r: r.kdispatch.min)
            row("Kernel Combine repr", lambda r: r.kcombine.min)
        else:
            row("Kernel Dispatch avg", lambda r: r.kdispatch.avg)
            row("Kernel Dispatch min", lambda r: r.kdispatch.min)
            row("Kernel Dispatch max", lambda r: r.kdispatch.max)
            row("Kernel Combine avg", lambda r: r.kcombine.avg)
            row("Kernel Combine min", lambda r: r.kcombine.min)
            row("Kernel Combine max", lambda r: r.kcombine.max)
    elif kernel_only:
        print(
            "  NOTE: kernel-only requested but kernel timing missing for a backend "
            "(mscclpp-cpp needs --kernel-only; nccl-ep needs CUPTI-enabled ep_bench)."
        )
    if len(results) == 2 and not kernel_only:
        a, b = results
        if a.total.avg == a.total.avg and b.total.avg == b.total.avg and b.total.avg:
            print(f"\nHost D+C ratio {a.ep_lib}/{b.ep_lib} = {a.total.avg / b.total.avg:.2f}x")


def main() -> None:
    args = parse_args()
    if args.ep_lib in ("both", "all"):
        libs = ["mscclpp-cpp", "nccl-ep"]
    else:
        libs = [args.ep_lib]

    builders = {
        "mscclpp-cpp": build_mscclpp_cpp_cmd,
        "nccl-ep": build_nccl_ep_cmd,
    }
    results = []
    for lib in libs:
        cmd = builders[lib](args)
        results.append(run_backend(lib, cmd, args.dry_run))
    if not args.dry_run:
        print_unified(results, kernel_only=args.kernel_only)


if __name__ == "__main__":
    main()
