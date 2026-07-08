#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unified EP low-latency benchmark driver.

Runs the *same* low-latency dispatch/combine benchmark -- identical tokens,
experts, hidden size, top-k, warmup and iteration counts -- against a
selectable expert-parallel library, then prints one normalized summary so the
libraries can be compared apples-to-apples.

Backends (``--ep-lib``):

* ``mscclpp``  -- this repo's :mod:`ep_bench_ll` (MoECommunicator LL) launched
  with ``torchrun``.
* ``nccl-ep``  -- NVIDIA NCCL-EP's ``contrib/nccl_ep/ep_bench`` binary launched
  with ``mpirun`` (HPCX).
* ``both``     -- run mscclpp then nccl-ep and print them side by side.

Both backends emit the identical ``=== Summary (Low Latency, across N ranks) ===``
block (``ep_bench_ll.py`` was written to mirror ``ep_bench``), so a single parser
reads either one.

NCCL-EP dynamically links its shared libraries (``libnccl.so``, ``libnccl_ep.so``).
Point the driver at the correct build with ``--nccl-lib-path`` (falls back to the
``NCCL_LIB_PATH`` environment variable); that directory is prepended to
``LD_LIBRARY_PATH`` for the ``ep_bench`` process so the intended NCCL is loaded.

Scope: single node (``--nproc-per-node`` GPUs). Multi-node runs use the existing
per-backend launchers (mscclpp: run_ep_bench_ll_multinode.sh; nccl-ep: mpirun with
a hostfile); this driver focuses on the common single-node comparison.

Examples
--------
Compare both libraries, 4 GPUs, e128::

    python run_ep_bench.py --ep-lib both -e 128 -t 128 -d 7168 -k 8 -w 10 -i 50 \
        --nccl-lib-path /opt/microsoft/mrc/ep/nccl/build/lib

Just mscclpp with in-process CUPTI kernel timing::

    python run_ep_bench.py --ep-lib mscclpp -e 128 --cupti-inproc

Print the commands without running them::

    python run_ep_bench.py --ep-lib both -e 128 --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

CUDA_INC = "/usr/local/cuda/targets/sbsa-linux/include"
CUDA_LIB = "/usr/local/cuda/targets/sbsa-linux/lib"
_HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified EP low-latency benchmark driver (mscclpp EP vs NCCL-EP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ep-lib", required=True, choices=["mscclpp", "nccl-ep", "both"],
                   help="which expert-parallel library to benchmark")
    p.add_argument("-a", "--algorithm", default="ll", choices=["ll", "low-latency"],
                   help="algorithm mode (only low-latency is wired up here)")

    # Shared problem shape -- passed to whichever backend is selected.
    p.add_argument("-t", "--num-tokens", type=int, default=128, help="tokens per rank")
    p.add_argument("-d", "--hidden", type=int, default=7168, help="hidden dimension")
    p.add_argument("-k", "--num-topk", type=int, default=8, help="top-k experts per token")
    p.add_argument("-e", "--num-experts", type=int, default=256, help="global number of experts")
    p.add_argument("-w", "--num-warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("-i", "--num-iters", type=int, default=50, help="timed iterations")

    # Launch / fabric.
    p.add_argument("--nproc-per-node", type=int, default=4, help="GPUs (ranks) on this node")
    p.add_argument("--iface", default="enP22p1s0f1", help="socket interface name (NCCL/GLOO/UCX)")
    p.add_argument("--hca", default="mlx5_0,mlx5_1,mlx5_2,mlx5_3", help="mscclpp HCA devices")

    # mscclpp backend.
    p.add_argument("--mscclpp-bench", default=os.path.join(_HERE, "ep_bench_ll.py"),
                   help="path to ep_bench_ll.py")
    p.add_argument("--conda-prefix", default=os.path.join(os.path.expanduser("~"), "miniconda3"),
                   help="conda installation prefix for the mscclpp torch env")
    p.add_argument("--conda-env", default="torch", help="conda env name with torch + mscclpp")
    p.add_argument("--cupti-inproc", action="store_true",
                   help="mscclpp: also collect in-process CUPTI kernel-only timing")
    p.add_argument("--torch-profiler", action="store_true",
                   help="mscclpp: run the torch.profiler kernel pass (default: host-observed only)")

    # nccl-ep backend.
    p.add_argument("--nccl-lib-path", default=os.environ.get("NCCL_LIB_PATH", ""),
                   help="directory with libnccl.so / libnccl_ep.so; prepended to LD_LIBRARY_PATH "
                        "for ep_bench (falls back to $NCCL_LIB_PATH)")
    p.add_argument("--nccl-ep-bench", default="/opt/microsoft/mrc/ep/nccl/build/test/nccl_ep/ep_bench",
                   help="path to the NCCL-EP ep_bench binary")
    p.add_argument("--hpcx", default="", help="HPCX install dir (for mpirun); autodetected under /opt if empty")
    p.add_argument("--layout", default="em", choices=["em", "rm", "fl"],
                   help="nccl-ep dispatch layout (em=expert-major, matches mscclpp LL)")

    p.add_argument("--dry-run", action="store_true", help="print the backend command(s) and exit")
    return p.parse_args()


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
    # Kernel-only representative dispatch/combine (mscclpp --cupti-inproc / ep_bench CUPTI), if present.
    kdispatch: Optional[float] = None
    kcombine: Optional[float] = None
    ok: bool = False


_HOST_RE = {
    "dispatch": re.compile(r"^Dispatch \(BF16\):\s+avg=([\d.]+)\s*us,\s*min=([\d.]+)\s*us,\s*max=([\d.]+)\s*us"),
    "combine": re.compile(r"^Combine \(BF16\):\s+avg=([\d.]+)\s*us,\s*min=([\d.]+)\s*us,\s*max=([\d.]+)\s*us"),
    "total": re.compile(r"^Total \(D\+C\):\s+avg=([\d.]+)\s*us,\s*min=([\d.]+)\s*us,\s*max=([\d.]+)\s*us"),
}
_RANKS_RE = re.compile(r"=== Summary \(Low Latency, across (\d+) ranks\) ===")
# Kernel-only representative dispatch line (mscclpp in-process CUPTI / torch.profiler block).
_KDISP_RE = re.compile(r"^Dispatch:\s+min=([\d.]+)\s*us \(representative\)")
_KCOMB_RE = re.compile(r"^Combine:\s+avg=([\d.]+)\s*us")


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
        m = _KDISP_RE.match(line)
        if m:
            res.kdispatch = float(m.group(1))
        elif _KCOMB_RE.match(line) and res.kdispatch is not None and res.kcombine is None:
            # Only the kernel-only Combine line (immediately follows the representative Dispatch line).
            res.kcombine = float(_KCOMB_RE.match(line).group(1))
    res.ok = res.dispatch.avg == res.dispatch.avg  # not NaN
    return res


# ----------------------------------------------------------------------------
# Backend command construction.
# ----------------------------------------------------------------------------
def build_mscclpp_cmd(args: argparse.Namespace) -> str:
    env = (
        f"MSCCLPP_EP_LOCAL_WORLD_SIZE={args.nproc_per_node} "
        f"NCCL_SOCKET_IFNAME={args.iface} GLOO_SOCKET_IFNAME={args.iface} MSCCLPP_SOCKET_IFNAME={args.iface} "
        f"MSCCLPP_HCA_DEVICES={args.hca} NCCL_IB_DISABLE=1 NCCL_MNNVL_ENABLE=0 MSCCLPP_EP_FABRIC_IPC=1"
    )
    bench = args.mscclpp_bench
    bench_flags = (
        f"-a ll -t {args.num_tokens} -d {args.hidden} -k {args.num_topk} "
        f"-e {args.num_experts} -w {args.num_warmup} -i {args.num_iters}"
    )
    cupti_build = ""
    if args.cupti_inproc:
        # In-process CUPTI kernel-only timing (near-zero perturbation, matches
        # ep_bench's KernelTimer). Builds the collector next to the bench if missing.
        bench_flags += " --cupti-inproc"
        env += f" LD_LIBRARY_PATH={CUDA_LIB}:$LD_LIBRARY_PATH"
        so = os.path.join(os.path.dirname(bench), "libcupti_kernel_timer.so")
        src = os.path.join(os.path.dirname(bench), "cupti_kernel_timer.cpp")
        cupti_build = (
            f'if [ ! -f {shlex.quote(so)} ]; then '
            f'g++ -O2 -fPIC -shared {shlex.quote(src)} -o {shlex.quote(so)} '
            f'-I{CUDA_INC} -L{CUDA_LIB} -lcupti; fi && '
        )
    elif args.torch_profiler:
        # Opt-in torch.profiler kernel pass (perturbs the LL recv-spin; the
        # in-process CUPTI path is preferred for kernel numbers).
        pass
    else:
        # Default: clean host-observed only (skip the torch.profiler pass, which
        # is slow and inflates the LL dispatch recv-spin).
        bench_flags += " --no-kernel-timing"
    return (
        f"source {shlex.quote(args.conda_prefix)}/etc/profile.d/conda.sh && "
        f"conda activate {shlex.quote(args.conda_env)} && unset PYTHONPATH && "
        f"{cupti_build}"
        f"export {env} && "
        f"torchrun --standalone --nnodes=1 --nproc_per_node={args.nproc_per_node} "
        f"{shlex.quote(bench)} {bench_flags}"
    )


def _autodetect_hpcx() -> str:
    import glob
    cands = sorted(glob.glob("/opt/hpcx-*"))
    return cands[0] if cands else ""


def build_nccl_ep_cmd(args: argparse.Namespace) -> str:
    nccl_lib = args.nccl_lib_path or "/opt/microsoft/mrc/ep/nccl/build/lib"
    hpcx = args.hpcx or _autodetect_hpcx()
    if not hpcx:
        raise SystemExit("nccl-ep: no HPCX found under /opt; pass --hpcx")
    np = args.nproc_per_node
    bench_flags = (
        f"-a ll -L {args.layout} -t {args.num_tokens} -d {args.hidden} -k {args.num_topk} "
        f"-e {args.num_experts} -w {args.num_warmup} -i {args.num_iters}"
    )
    mpi = (
        f"mpirun -np {np} --map-by ppr:{np}:node --bind-to none "
        f"-mca pml ob1 -mca btl self,vader,tcp -mca btl_tcp_if_include {args.iface} "
        f"-mca coll_hcoll_enable 0 -mca coll_ucc_enable 0 "
        f"-x LD_LIBRARY_PATH -x PATH -x CUDA_HOME=/usr/local/cuda "
        f"-x UCX_NET_DEVICES={args.iface} -x UCX_TLS=tcp,sm,self,cuda_copy -x UCX_HANDLE_ERRORS=none "
        f"-x NCCL_SOCKET_IFNAME={args.iface} -x NCCL_NET_PLUGIN=none "
        f"-x NCCL_IB_DISABLE=1 -x NCCL_MNNVL_ENABLE=0 "
        f"{shlex.quote(args.nccl_ep_bench)} {bench_flags}"
    )
    return (
        f"source {shlex.quote(hpcx)}/hpcx-init.sh && hpcx_load && "
        f"export LD_LIBRARY_PATH={shlex.quote(nccl_lib)}:$LD_LIBRARY_PATH && "
        f"{mpi}"
    )


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


def print_unified(results: list) -> None:
    results = [r for r in results if r is not None]
    if not results:
        return
    print("\n=== Unified EP Low-Latency Summary (host-observed, us) ===")
    hdr = f"{'metric':<18}" + "".join(f"{r.ep_lib:>14}" for r in results)
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("Dispatch avg", lambda r: r.dispatch.avg),
        ("Combine avg", lambda r: r.combine.avg),
        ("Total D+C avg", lambda r: r.total.avg),
        ("Dispatch min", lambda r: r.dispatch.min),
        ("Combine min", lambda r: r.combine.min),
    ]
    for label, fn in rows:
        print(f"{label:<18}" + "".join(f"{fn(r):>14.2f}" for r in results))
    if any(r.kdispatch is not None for r in results):
        print(f"{'Kernel disp(min)':<18}" +
              "".join(f"{(r.kdispatch if r.kdispatch is not None else float('nan')):>14.2f}" for r in results))
        print(f"{'Kernel comb(avg)':<18}" +
              "".join(f"{(r.kcombine if r.kcombine is not None else float('nan')):>14.2f}" for r in results))
    if len(results) == 2:
        a, b = results
        if a.total.avg == a.total.avg and b.total.avg == b.total.avg and b.total.avg:
            ratio = a.total.avg / b.total.avg
            print(f"\nD+C ratio {a.ep_lib}/{b.ep_lib} = {ratio:.2f}x")


def main() -> None:
    args = parse_args()
    libs = ["mscclpp", "nccl-ep"] if args.ep_lib == "both" else [args.ep_lib]
    results = []
    for lib in libs:
        cmd = build_mscclpp_cmd(args) if lib == "mscclpp" else build_nccl_ep_cmd(args)
        results.append(run_backend(lib, cmd, args.dry_run))
    if not args.dry_run:
        print_unified(results)


if __name__ == "__main__":
    main()
