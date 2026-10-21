# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build once and deploy the MSCCL++/Torch BF16 MoE benchmark over SSH."""

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
from importlib import metadata
import ipaddress
import json
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import uuid
import venv

MANIFEST = "moe_deployment.json"
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15"]


def platform_info():
    return {
        "python": list(sys.version_info[:2]),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "soabi": sysconfig.get_config_var("SOABI"),
        "libc": list(platform.libc_ver()),
        "gpu_archs": sorted(
            set(
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
                ).split()
            )
        ),
    }


PLATFORM_PROBE = (
    "import json,platform,subprocess,sys,sysconfig;"
    "print(json.dumps(dict(python=list(sys.version_info[:2]),"
    "implementation=platform.python_implementation(),machine=platform.machine(),"
    "soabi=sysconfig.get_config_var('SOABI'),libc=list(platform.libc_ver()),"
    "gpu_archs=sorted(set(subprocess.check_output("
    "['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],text=True).split())))))"
)


def run(command, **kwargs):
    print(f"+ {shlex.join(map(str, command))}", flush=True)
    return subprocess.run(command, check=True, **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_info(root):
    revision = run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    status = run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout
    return {"git_commit": revision, "git_dirty": bool(status), "git_status": status.splitlines()}


def build_bundle(root, output_dir, cuda):
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source_info(root)
    name = f"moe_{source['git_commit'][:12]}_{uuid.uuid4().hex[:8]}"
    bundle = output_dir / f"{name}.tar"
    with tempfile.TemporaryDirectory(prefix="moe-build-") as temporary:
        stage = Path(temporary)
        wheels = stage / "wheels"
        wheels.mkdir()
        constraints = stage / "constraints.txt"
        constraints.write_text(
            "".join(
                sorted(
                    f"{dist.metadata['Name']}=={dist.version}\n"
                    for dist in metadata.distributions()
                    if dist.metadata["Name"].lower() != "mscclpp"
                )
            )
        )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-cache-dir",
                "--wheel-dir",
                str(wheels),
                str(root),
            ]
        )
        (wheel,) = wheels.glob("mscclpp-*.whl")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--no-cache-dir",
                "--only-binary=:all:",
                "--dest",
                str(wheels),
                "--constraint",
                str(constraints),
                f"{wheel}[cuda{cuda},moe-benchmark]",
            ]
        )
        tracked = run(
            ["git", "-C", str(root), "ls-files", "-z", "--", "test/python/ep/moe_bench"],
            capture_output=True,
            text=True,
        ).stdout.split("\0")
        for relative in filter(None, tracked):
            path = Path(relative)
            if "tests" in path.parts:
                continue
            target = stage / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / path, target)
        shutil.copyfile(root / "LICENSE", stage / "LICENSE")
        shutil.copyfile(__file__, stage / "deploy_moe_bench.py")
        manifest = {
            "schema_version": 1,
            "bundle_id": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "platform": platform_info(),
            "cuda_extra": f"cuda{cuda}",
            "wheel": wheel.name,
            "files": {
                str(path.relative_to(stage)): sha256(path) for path in sorted(stage.rglob("*")) if path.is_file()
            },
        }
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            with tarfile.open(bundle, "x") as archive:
                for path in sorted(stage.iterdir()):
                    archive.add(path, arcname=path.name)
        except (OSError, tarfile.TarError):
            bundle.unlink(missing_ok=True)
            raise
    print(f"Built {bundle}", flush=True)
    return bundle


def read_manifest(bundle):
    with tarfile.open(bundle) as archive:
        stream = archive.extractfile(MANIFEST)
        if stream is None:
            raise ValueError(f"Bundle is missing {MANIFEST}")
        with stream:
            manifest = json.load(stream)
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported deployment manifest version")
    if not re.fullmatch(r"moe_[0-9a-f]{12}_[0-9a-f]{8}", manifest["bundle_id"]):
        raise ValueError("Invalid deployment bundle ID")
    return manifest


def check_platform(expected, actual):
    if expected != actual:
        raise ValueError(
            f"Bundle requires matching Python ABI, CPU/GPU architectures and libc.\n"
            f"Build platform: {expected}\nTarget platform: {actual}"
        )


def install_bundle(directory):
    manifest = json.loads((directory / MANIFEST).read_text())
    check_platform(manifest["platform"], platform_info())
    for relative, checksum in manifest["files"].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != checksum:
            raise ValueError(f"Bundle integrity check failed: {relative}")
    environment = directory / ".venv"
    if environment.exists():
        raise ValueError(f"Refusing to overwrite existing environment: {environment}")
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(directory / "wheels"),
            "--constraint",
            str(directory / "constraints.txt"),
            f"{directory / 'wheels' / manifest['wheel']}[{manifest['cuda_extra']},moe-benchmark]",
        ]
    )
    run(
        [
            str(python),
            "-c",
            "import torch,mscclpp,mscclpp.ep,cupy,yaml;"
            "assert torch.cuda.is_available(), 'CUDA-enabled PyTorch and a working GPU driver are required';"
            "print('Verified MSCCL++',mscclpp.__version__,'Torch',torch.__version__,"
            "'visible GPUs',torch.cuda.device_count())",
        ]
    )
    print(f"Installed benchmark in {directory}", flush=True)


def deploy_bundle(bundle, nodes, user, remote_dir, python):
    manifest = read_manifest(bundle)
    targets = [f"{user}@{node}" for node in nodes]
    for target in targets:
        result = run(
            ["ssh", *SSH_OPTIONS, target, shlex.join([python, "-c", PLATFORM_PROBE])],
            capture_output=True,
            text=True,
        )
        check_platform(manifest["platform"], json.loads(result.stdout))
    checksum = sha256(bundle)
    destination = f"{remote_dir.rstrip('/')}/{manifest['bundle_id']}"
    for node, target in zip(nodes, targets):
        run(
            [
                "ssh",
                *SSH_OPTIONS,
                target,
                f"mkdir -p -- {shlex.quote(remote_dir)} && mkdir -- {shlex.quote(destination)}",
            ]
        )
        scp_target = f"{user}@[{node}]" if ":" in node else target
        run(["scp", *SSH_OPTIONS, str(bundle), f"{scp_target}:{destination}/bundle.tar"])
        verify = (
            "import hashlib,sys\n"
            "digest=hashlib.sha256()\n"
            "with open('bundle.tar','rb') as stream:\n"
            "    for block in iter(lambda: stream.read(1048576),b''):\n"
            "        digest.update(block)\n"
            f"sys.exit(0 if digest.hexdigest()=={checksum!r} else 'Bundle checksum mismatch')"
        )
        command = (
            f"cd -- {shlex.quote(destination)} && "
            f"{shlex.join([python, '-c', verify])} && "
            "tar -xf bundle.tar && "
            f"{shlex.join([python, 'deploy_moe_bench.py', '--install'])} && "
            "rm -- bundle.tar"
        )
        run(["ssh", *SSH_OPTIONS, target, command])
        print(f"Ready on {target}: {destination}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodelist", help="Comma-separated node IP addresses")
    parser.add_argument("--user", default=getpass.getuser(), help="SSH username")
    parser.add_argument("--remote-dir", default=".local/share/mscclpp/moe-bench", help="Parent directory on each node")
    parser.add_argument("--remote-python", default="python3", help="Python interpreter on each node")
    parser.add_argument("--cuda", choices=("12", "13"), default="13", help="CUDA extra to bundle")
    parser.add_argument("--output-dir", type=Path, default=Path("moe_deploy"), help="Local bundle output directory")
    parser.add_argument("--bundle", type=Path, help="Deploy an existing bundle instead of rebuilding")
    parser.add_argument("--build-only", action="store_true", help="Build a bundle without contacting nodes")
    parser.add_argument("--install", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.install:
        return args
    if args.build_only and args.bundle:
        parser.error("--build-only cannot be combined with --bundle")
    if not args.build_only and not args.nodelist:
        parser.error("--nodelist is required unless --build-only is used")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", args.user):
        parser.error("Invalid SSH username")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", args.remote_dir) or ".." in Path(args.remote_dir).parts:
        parser.error("--remote-dir must be a simple path without '..', spaces, or shell expansions")
    if args.nodelist:
        try:
            args.nodes = [str(ipaddress.ip_address(node.strip())) for node in args.nodelist.split(",")]
        except ValueError as error:
            parser.error(str(error))
        if len(args.nodes) != len(set(args.nodes)):
            parser.error("--nodelist contains duplicate nodes")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.install:
        install_bundle(Path(__file__).resolve().parent)
        return
    root = Path(__file__).resolve().parents[1]
    bundle = args.bundle.resolve() if args.bundle else build_bundle(root, args.output_dir.resolve(), args.cuda)
    if not args.build_only:
        deploy_bundle(bundle, args.nodes, args.user, args.remote_dir, args.remote_python)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as error:
        print(f"Deployment failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        sys.exit(1)
