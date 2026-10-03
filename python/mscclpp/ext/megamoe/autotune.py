# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline, sample-set tuning of the native routed-first synthetic MegaMoE layer.

Run with torchrun. Profiles contain local JIT cache keys, not module paths.
``resolve_profile`` only validates and loads a cached winner; it never compiles,
constructs collective contexts, or runs tuning. Applications must build the
same selection key and agree on the returned configuration before constructing
their contexts, whose capacity must be the returned bucket upper bound.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
from types import SimpleNamespace
import uuid

VERSION = 1
PROFILE_KIND = "mscclpp-native-megamoe-profiles"
DEFAULT_RESOURCE_SPLIT = {"route_sm_margin": 32, "shared_sms": 32}
DEFAULT_KERNELS = [
    {"tile_m": 256, "tile_n": 32, "load_stages": 8, "transform_stages": 7, "tile_k": 128},
    {"tile_m": 256, "tile_n": 32, "load_stages": 6, "transform_stages": 7, "tile_k": 128},
    {"tile_m": 256, "tile_n": 64, "load_stages": 6, "transform_stages": 6, "tile_k": 128},
    {"tile_m": 256, "tile_n": 128, "load_stages": 4, "transform_stages": 4, "tile_k": 128},
]
KERNEL_FIELDS = ("tile_m", "tile_n", "load_stages", "transform_stages", "tile_k")
REQUIRED_KERNEL_FIELDS = ("tile_n", "load_stages", "transform_stages")
SHAPE_FIELDS = ("original_hidden", "hidden", "intermediate", "shared_intermediate", "top_k")
FRONTEND_FIELDS = (
    "router_allow_tf32",
    "router_weight_layout",
    "router_probability_order",
    "post_norm",
    "residual",
    "rms_eps",
    "residual_dtype",
    "seed",
)
WORKLOAD_FIELDS = set(SHAPE_FIELDS + FRONTEND_FIELDS + ("local_experts",))


class ProfileMismatchError(ValueError):
    """A saved measurement does not cover the requested exact scope or bucket."""


class CollectiveTuningError(RuntimeError):
    """A preparation/check step failed on one or more ranks."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 2**31 - 1:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**31-1]")
    return value


def _fields(value, allowed, required, label):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"{label}: expected fields {sorted(required)}; allowed fields {sorted(allowed)}")


def _kernel_config(value):
    from .jit import KernelConfig

    _fields(value, KERNEL_FIELDS, REQUIRED_KERNEL_FIELDS, "kernel config")
    for field, setting in value.items():
        _integer(setting, field, 1)
    return KernelConfig(**value)


def _split(value):
    _fields(value, ("route_sm_margin", "shared_sms"), ("route_sm_margin", "shared_sms"), "resource split")
    margin = _integer(value["route_sm_margin"], "route_sm_margin")
    shared = _integer(value["shared_sms"], "shared_sms", 2)
    if margin % 2 or shared % 2 or margin < shared:
        raise ValueError("resource splits require even CTA caps and route_sm_margin >= shared_sms >= 2")
    return dict(value)


def _bucket(value):
    _fields(value, ("min", "max", "samples"), ("min", "max", "samples"), "token bucket")
    lower, upper = _integer(value["min"], "bucket min"), _integer(value["max"], "bucket max", 1)
    samples = value["samples"]
    if lower > upper or not isinstance(samples, list) or not samples:
        raise ValueError("bucket requires min <= max and nonempty positive representative samples")
    for sample in samples:
        _integer(sample, "representative token sample", 1)
        if not lower <= sample <= upper:
            raise ValueError("representative sample is outside its inclusive bucket")
    if len(set(samples)) != len(samples):
        raise ValueError("representative samples must be unique")
    return {"min": lower, "max": upper, "samples": sorted(samples)}


def validate_tuning_input(value):
    """Validate the supported concrete search schema; reject descriptive drafts."""
    fields = ("_license", "version", "kernel_candidates", "resource_splits", "token_buckets", "workloads")
    _fields(value, fields, ("version", "token_buckets"), "tuning input")
    if type(value["version"]) is not int or value["version"] != VERSION:
        raise ValueError("unsupported tuning input version")
    candidates = value.get("kernel_candidates", DEFAULT_KERNELS)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("kernel_candidates must be a nonempty list")
    candidates = [asdict(_kernel_config(candidate)) for candidate in candidates]
    if DEFAULT_KERNELS[0] not in candidates:
        candidates.insert(0, DEFAULT_KERNELS[0].copy())
    if len({_canonical(candidate) for candidate in candidates}) != len(candidates):
        raise ValueError("duplicate kernel candidates")
    candidates.sort(key=lambda candidate: candidate != DEFAULT_KERNELS[0])
    splits = value.get("resource_splits")
    if splits is not None:
        if not isinstance(splits, list) or not splits:
            raise ValueError("resource_splits must be nonempty")
        splits = [_split(split) for split in splits]
        if len({_canonical(split) for split in splits}) != len(splits):
            raise ValueError("duplicate resource splits")
    if not isinstance(value["token_buckets"], list) or not value["token_buckets"]:
        raise ValueError("token_buckets must be nonempty")
    buckets = sorted((_bucket(bucket) for bucket in value["token_buckets"]), key=lambda bucket: bucket["min"])
    if any(left["max"] >= right["min"] for left, right in zip(buckets, buckets[1:])):
        raise ValueError("token buckets must not overlap")
    workloads = value.get("workloads", [{}])
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("workloads must be a nonempty list")
    for workload in workloads:
        _fields(workload, WORKLOAD_FIELDS, (), "workload")
        for field in SHAPE_FIELDS + ("local_experts",):
            if field in workload:
                _integer(workload[field], field, 1)
        if workload.get("top_k", 1) > 32:
            raise ValueError("top_k must not exceed 32")
        for field in ("original_hidden", "hidden", "intermediate", "shared_intermediate"):
            if field in workload and workload[field] % 128:
                raise ValueError(f"{field} must be a multiple of 128")
        for field in ("router_allow_tf32", "post_norm", "residual"):
            if field in workload and not isinstance(workload[field], bool):
                raise ValueError(f"{field} must be bool")
        for field, choices in (
            ("router_weight_layout", ("expert-major", "hidden-major")),
            ("router_probability_order", ("softmax-first", "selected-logits")),
            ("residual_dtype", ("fp32", "bf16")),
        ):
            if field in workload and workload[field] not in choices:
                raise ValueError(f"unsupported {field}")
        if "rms_eps" in workload:
            epsilon = workload["rms_eps"]
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon <= 0
            ):
                raise ValueError("rms_eps must be finite and positive")
        if "seed" in workload:
            _integer(workload["seed"], "seed")
    return {
        "version": VERSION,
        "kernel_candidates": candidates,
        "resource_splits": splits,
        "token_buckets": buckets,
        "workloads": workloads,
    }


def _all_gather(value):
    import torch.distributed as dist

    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


@contextmanager
def collective_error_guard(gather=_all_gather, phase="preparation"):
    """Propagate local host errors before peers enter the next collective phase."""
    error = None
    try:
        yield
    except (
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        AssertionError,
        ImportError,
        MemoryError,
        subprocess.SubprocessError,
    ) as exception:
        error = f"{type(exception).__name__}: {exception}"
    errors = gather(error)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None]
    if failures:
        raise CollectiveTuningError(f"{phase} failed collectively: {'; '.join(failures)}")


def collective_call(function, gather=_all_gather, phase="preparation"):
    """Execute a local step on every rank, aggregating errors but not its result."""
    with collective_error_guard(gather, phase):
        result = function()
    return result


def collective_agree(value, gather=_all_gather, phase="configuration"):
    """Reject mismatched variants/capacities/settings before context creation."""
    values = gather(value)
    if any(_canonical(other) != _canonical(values[0]) for other in values[1:]):
        raise CollectiveTuningError(f"{phase}: ranks disagree")
    return value


def make_selection_key(args, fingerprints, topology):
    """Build an exact synthetic-layer key from all ranks' stable runtime data.

    ``topology`` must include the operator-specified fabric/partition identifier,
    rank-to-host mapping and local device indices. It must describe the same
    active NVLink fabric at tune and replay time, not merely the GPU model.
    """
    if (
        not isinstance(fingerprints, list)
        or not fingerprints
        or not all(isinstance(item, dict) for item in fingerprints)
    ):
        raise ValueError("fingerprints must contain one runtime_fingerprint object per rank")
    _fields(topology, ("fabric", "rank_hosts", "local_devices"), ("fabric", "rank_hosts", "local_devices"), "topology")
    world = len(fingerprints)
    if not isinstance(topology["fabric"], str) or not topology["fabric"].strip():
        raise ValueError("an explicit --topology-id identifying the fabric/partition is required")
    for field in ("rank_hosts", "local_devices"):
        if not isinstance(topology[field], list) or len(topology[field]) != world:
            raise ValueError(f"topology {field} must contain one entry per rank")
        for index in topology[field]:
            _integer(index, f"topology {field}")
    experts = args.experts if args.experts is not None else 16 * world
    if experts < 1 or experts % world:
        raise ValueError("global experts must be divisible by world size")
    key = {
        "world_size": world,
        "shape": {**{field: getattr(args, field) for field in SHAPE_FIELDS}, "local_experts": experts // world},
        "frontend": {field: getattr(args, field) for field in FRONTEND_FIELDS},
        "execution": {
            "schedule": "strict-routed-first-overlap",
            "input_mode": "staged",
            "cuda_graph": True,
            "graph_batch": args.graph_batch,
            "expert_weights": "mxfp8-e4m3fn-e8m0-k32",
            "activation": "bf16-swiglu",
            "gate_up_clamp": -1.0,
            "shared_kernel": "unchanged-builtin",
            "environment_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        },
        "topology": topology,
        "rank_fingerprints": fingerprints,
    }
    return json.loads(_canonical(key))


def collect_selection_key(args, device=None, gather=_all_gather):
    """Collect runtime/topology data once at initialization, never in a forward."""
    from .jit import runtime_fingerprint

    def local_info():
        fingerprint = dict(runtime_fingerprint(device))
        local_device = device if type(device) is int else getattr(device, "index", None)
        if isinstance(device, str) and device.startswith("cuda:"):
            local_device = int(device.split(":", 1)[1])
        if type(local_device) is not int:
            local_device = int(os.environ.get("LOCAL_RANK", 0))
        directory = Path(__file__).parent
        fingerprint["offline_frontend_sha256"] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("autotune.py", "benchmark_shared.py", "benchmark.py", "api.py", "quantization.py")
        }
        return {
            "fingerprint": fingerprint,
            "host": socket.gethostname(),
            "device": local_device,
        }

    infos = gather(collective_call(local_info, gather, "runtime fingerprint"))
    hosts = []
    for info in infos:
        if info["host"] not in hosts:
            hosts.append(info["host"])
    topology = {
        "fabric": args.topology_id,
        "rank_hosts": [hosts.index(info["host"]) for info in infos],
        "local_devices": [info["device"] for info in infos],
    }
    key = collective_call(
        lambda: make_selection_key(args, [info["fingerprint"] for info in infos], topology),
        gather,
        "selection key",
    )
    return collective_agree(key, gather, "selection key")


def _validate_entry(entry):
    _fields(
        entry,
        ("id", "selection_key", "token_bucket", "winner", "measurement", "provenance"),
        ("id", "selection_key", "token_bucket", "winner", "measurement", "provenance"),
        "profile entry",
    )
    if not isinstance(entry["id"], str) or not entry["id"] or not isinstance(entry["selection_key"], dict):
        raise ValueError("profile entry needs a nonempty id and selection_key object")
    _bucket(entry["token_bucket"])
    winner = entry["winner"]
    _fields(
        winner,
        ("kernel_config", "kernel_key", "route_sm_margin", "shared_sms"),
        ("kernel_config", "kernel_key", "route_sm_margin", "shared_sms"),
        "profile winner",
    )
    normalized_kernel_config = asdict(_kernel_config(winner["kernel_config"]))
    _split({field: winner[field] for field in ("route_sm_margin", "shared_sms")})
    if not isinstance(winner["kernel_key"], str) or not winner["kernel_key"]:
        raise ValueError("winner requires a local cache key, not a module path")
    if (winner["kernel_key"] == "builtin") != (normalized_kernel_config == DEFAULT_KERNELS[0]):
        raise ValueError("builtin key and kernel configuration disagree")
    if "/" in winner["kernel_key"] or "\\" in winner["kernel_key"]:
        raise ValueError("winner kernel_key must not be a path")
    if not isinstance(entry["measurement"], dict) or not isinstance(entry["provenance"], dict):
        raise ValueError("profile measurement/provenance must be objects")
    _canonical(entry)
    return entry


def _read_profiles(path):
    with Path(path).open() as source:
        value = json.load(source)
    _fields(value, ("version", "kind", "entries", "history"), ("version", "kind", "entries", "history"), "profiles")
    if type(value["version"]) is not int or value["version"] != VERSION or value["kind"] != PROFILE_KIND:
        raise ValueError("unsupported saved-profile format/version")
    if not isinstance(value["entries"], list) or not isinstance(value["history"], list):
        raise ValueError("profile entries/history must be lists")
    for entry in value["entries"] + value["history"]:
        _validate_entry(entry)
    return value


def save_profile(path, entry):
    """Atomically merge a completed entry under a host filesystem lock.

    Every rank may call this with identical data, so node-local files are
    populated too. Replaced runs are retained in ``history``; disjoint buckets
    and other exact workload keys are preserved. Concurrent overlapping bucket
    definitions are rejected rather than silently selecting one.
    """
    _validate_entry(entry)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        profiles = (
            _read_profiles(path)
            if path.exists()
            else {"version": VERSION, "kind": PROFILE_KIND, "entries": [], "history": []}
        )
        for previous in profiles["history"]:
            if previous["id"] == entry["id"]:
                if _canonical(previous) != _canonical(entry):
                    raise ValueError("conflicting profile content for the same run id")
                return
        replacement = None
        for index, previous in enumerate(profiles["entries"]):
            if previous["id"] == entry["id"]:
                if _canonical(previous) != _canonical(entry):
                    raise ValueError("conflicting profile content for the same run id")
                return
            if previous["selection_key"] != entry["selection_key"]:
                continue
            left, right = previous["token_bucket"], entry["token_bucket"]
            if max(left["min"], right["min"]) <= min(left["max"], right["max"]):
                if (left["min"], left["max"]) != (right["min"], right["max"]):
                    raise ValueError("overlapping saved-profile buckets for the same exact selection key")
                replacement = index
        if replacement is None:
            profiles["entries"].append(entry)
        else:
            profiles["history"].append(profiles["entries"][replacement])
            profiles["entries"][replacement] = entry
        staging = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.pending")
        try:
            with staging.open("x") as destination:
                destination.write(json.dumps(profiles, indent=2, sort_keys=True, allow_nan=False) + "\n")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(staging, path)
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            staging.unlink(missing_ok=True)


@dataclass(frozen=True)
class ResolvedProfile:
    """Prepared rank-local winner and the capacity required before graph capture."""

    kernel: object
    route_sm_margin: int
    shared_sms: int
    max_tokens: int
    entry: dict


def resolve_profile(path, selection_key, tokens, *, cache_dir=None):
    """Load an exact offline winner without any JIT compilation or measurements.

    ``tokens`` is the already agreed maximum local token count across ranks.
    Inclusive bucket bounds are enforced; no interpolation/extrapolation or
    silent baseline fallback is performed. Missing/stale local kernel caches
    are explicit errors and must be prepared separately, before construction.
    """
    from .jit import load_cached_kernel

    _integer(tokens, "tokens")
    entries = _read_profiles(path)["entries"]
    matching = [entry for entry in entries if entry["selection_key"] == selection_key]
    if not matching:
        differing = sorted(
            {
                field
                for entry in entries
                for field in set(entry["selection_key"]) | set(selection_key)
                if entry["selection_key"].get(field) != selection_key.get(field)
            }
        )
        raise ProfileMismatchError(
            f"no exact hardware/topology/workload/environment profile; differing fields: {differing}"
        )
    matches = [entry for entry in matching if entry["token_bucket"]["min"] <= tokens <= entry["token_bucket"]["max"]]
    if len(matches) != 1:
        raise ProfileMismatchError(
            f"tokens={tokens} has {len(matches)} matching buckets; no out-of-range extrapolation"
        )
    entry = matches[0]
    winner = entry["winner"]
    kernel = load_cached_kernel(winner["kernel_key"], cache_dir=cache_dir)
    if asdict(kernel.config) != asdict(_kernel_config(winner["kernel_config"])) or kernel.key != winner["kernel_key"]:
        raise ProfileMismatchError("local cached kernel does not match the saved selection")
    return ResolvedProfile(kernel, winner["route_sm_margin"], winner["shared_sms"], entry["token_bucket"]["max"], entry)


def resolve_for_benchmark(args, device=None, gather=_all_gather):
    """Resolve once collectively and apply the winner before benchmark contexts."""
    key = collect_selection_key(args, device, gather)
    tokens = max(gather(args.tokens))
    resolved = collective_call(
        lambda: resolve_profile(args.tuned_profile, key, tokens, cache_dir=args.cache_dir),
        gather,
        "saved profile loading",
    )
    collective_agree(
        {"winner": resolved.entry["winner"], "max_tokens": resolved.max_tokens},
        gather,
        "saved kernel/capacity/CTA settings",
    )
    args.route_sm_margin, args.shared_sms = resolved.route_sm_margin, resolved.shared_sms
    args.max_tokens = resolved.max_tokens
    return resolved


def select_winner(measurements, representative_samples):
    """Minimize worst sample latency ratio to one common feasible builtin vector.

    Each raw trial records CUDA-event samples for every rank. At each iteration
    the maximum rank time is taken first, then the median across trials and
    iterations. Empty/ragged correctness-only inputs never enter the objective.
    Prefer the default 32/32 builtin reference; otherwise use the first feasible
    builtin in search order. All splits share this denominator, so a slower
    resource split cannot win merely by improving its own slower builtin.
    """
    if not representative_samples:
        raise ValueError("selection needs positive representative token samples")
    for sample in representative_samples:
        _integer(sample, "representative sample", 1)
    valid = []
    for measurement in measurements:
        if measurement.get("status") != "ok":
            continue
        medians = {}
        for tokens in representative_samples:
            samples = []
            for trial in measurement["trials"]:
                ranks = trial["samples"][str(tokens)]["rank_samples_us"]
                if not ranks or not ranks[0] or any(len(rank) != len(ranks[0]) for rank in ranks):
                    raise ValueError("timing requires equally sized nonempty per-rank sample lists")
                if any(not math.isfinite(value) or value <= 0 for rank in ranks for value in rank):
                    raise ValueError("timed latencies must be finite and positive")
                samples.extend(max(values) for values in zip(*ranks))
            if not samples:
                raise ValueError("successful candidates require measured trials")
            medians[str(tokens)] = statistics.median(samples)
        valid.append((measurement, medians))
    baselines = {}
    for measurement, medians in valid:
        if measurement["kernel_config"] == DEFAULT_KERNELS[0]:
            split = _canonical(measurement["resource_split"])
            if split in baselines:
                raise ValueError("duplicate builtin baseline for one resource split")
            baselines[split] = (measurement, medians)
    if not baselines:
        raise ValueError("no correct feasible builtin baseline was measured")
    reference, baseline = baselines.get(_canonical(DEFAULT_RESOURCE_SPLIT), next(iter(baselines.values())))
    scores = []
    for measurement, medians in valid:
        ratios = {str(tokens): medians[str(tokens)] / baseline[str(tokens)] for tokens in representative_samples}
        scores.append(
            {
                "candidate": measurement["candidate"],
                "median_us": medians,
                "reference_candidate": reference["candidate"],
                "reference_split": dict(reference["resource_split"]),
                "reference_median_us": dict(baseline),
                "ratio_to_reference_builtin": ratios,
                "worst_ratio": max(ratios.values()),
            }
        )
    # Stable search order resolves ties; empty inputs never supply zero timings.
    best = min(scores, key=lambda score: score["worst_ratio"])
    return next(item for item in measurements if item["candidate"] == best["candidate"]), scores


def _ragged_reference(config, inputs, ids, scores, weights):
    """Use the benchmark's unchanged oracle, padding only its host-side gather."""
    import torch.nn.functional as functional
    from .benchmark import _reference

    count = inputs.shape[0]
    padding = config.max_tokens - count
    return _reference(
        config,
        functional.pad(inputs, (0, 0, 0, padding)),
        functional.pad(ids, (0, 0, 0, padding)),
        functional.pad(scores, (0, 0, 0, padding)),
        weights,
    )[:count]


def _make_layer(args, count, routed, shared, device, rank):
    import torch
    from .benchmark_shared import _Frontend, _Layer, _Postprocess

    torch.manual_seed(args.seed + 200000 + rank)
    inputs = torch.randn((count, args.original_hidden), device=device, dtype=torch.bfloat16)
    torch.manual_seed(args.seed + 300000)
    postprocess = _Postprocess(
        inputs,
        epsilon=args.rms_eps,
        post_norm=args.post_norm,
        add_residual=args.residual,
        residual_dtype=args.residual_dtype,
    )
    frontend = _Frontend(
        inputs,
        routed.config,
        postprocess=postprocess,
        router_allow_tf32=args.router_allow_tf32,
        router_weight_layout=args.router_weight_layout,
        router_probability_order=args.router_probability_order,
    )
    return _Layer(frontend, routed, shared)


def _run_trial(
    args,
    bucket,
    kernel,
    routed_communicator,
    shared_communicator,
    weights,
    device,
    rank,
    world,
    check,
    independent_reference,
):
    import torch
    import torch.distributed as dist
    from .api import MegaMoE
    from .benchmark_shared import _assert_ctas, _check, _configs, _time_mode

    physical_sms = torch.cuda.get_device_properties(device).multi_processor_count
    route_config, shared_config = collective_call(
        lambda: _configs(args, rank, world, physical_sms), phase="native configuration"
    )
    collective_agree(
        {
            "kernel": kernel.key,
            "max_tokens": args.max_tokens,
            "routed_ctas": physical_sms - args.route_sm_margin,
            "shared_ctas": args.shared_sms,
        },
        phase="trial variant/capacity/CTAs",
    )
    routed = shared = layer = None
    failure = None
    try:
        routed = collective_call(
            lambda: MegaMoE(route_config, routed_communicator, *weights[0], kernel=kernel),
            phase="routed native context preflight",
        )
        shared = collective_call(
            lambda: MegaMoE(shared_config, shared_communicator, *weights[1]),
            phase="shared native context preflight",
        )
        collective_call(lambda: _assert_ctas(routed, shared, physical_sms, args), phase="actual CTA feasibility")
        result = {
            "ranks": _all_gather(
                {
                    "routed_ctas": routed.cta_count,
                    "shared_ctas": shared.cta_count,
                    "workspace_bytes": {"routed": routed.workspace_bytes, "shared": shared.workspace_bytes},
                    "shared_bytes": {"routed": routed.shared_bytes, "shared": shared.shared_bytes},
                }
            ),
            "correctness": {},
            "samples": {},
        }
        if check:
            checks = [(str(tokens), tokens) for tokens in bucket["samples"]]
            checks.append(("empty", 0))
            if world > 1:
                checks.append(("ragged", 0 if rank == 0 else max(1, bucket["max"] // (rank + 1))))
            for label, count in checks:
                layer = collective_call(
                    lambda: _make_layer(args, count, routed, shared, device, rank), phase="correctness frontend"
                )
                correctness = _check(
                    layer,
                    *weights,
                    args,
                    dist.barrier,
                    routed_reference=_ragged_reference,
                    error_guard=lambda: collective_error_guard(phase=f"correctness {label}"),
                    independent_reference=independent_reference
                    and (label not in ("empty", "ragged") or args.full_edge_references),
                )
                result["correctness"][label] = _all_gather(correctness)
                del layer
                layer = None
                gc.collect()
        for tokens in bucket["samples"]:
            layer = collective_call(
                lambda: _make_layer(args, tokens, routed, shared, device, rank), phase="timing frontend"
            )
            samples = _time_mode(layer, "overlap", args, dist.barrier, rank)
            result["samples"][str(tokens)] = {"rank_samples_us": _all_gather(samples)}
            del layer
            layer = None
            gc.collect()
    except CollectiveTuningError as error:
        # Drop traceback-owned graph/layer references before another variant
        # constructs its collective workspace.
        failure = str(error)
    finally:
        torch.cuda.synchronize(device)
        dist.barrier()
        del layer, routed, shared
        gc.collect()
        dist.barrier()
    if failure is not None:
        raise CollectiveTuningError(failure)
    return result


def _parse_args(argv=None):
    from . import benchmark_shared

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("megamoe_tuning.json")))
    parser.add_argument("--profile-output", required=True, help="merged local JSON profiles; written on every node")
    parser.add_argument("--trials", type=int, default=3, help="rounds with rotated candidate order")
    parser.add_argument("--nvcc", default=None)
    parser.add_argument("--cutlass-root", default=None)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument(
        "--full-edge-references",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also run independent CPU oracles for empty and unequal-rank checks",
    )
    argv = sys.argv[1:] if argv is None else list(argv)
    args = benchmark_shared._parse_args(argv, parser=parser)
    options = {argument.split("=", 1)[0] for argument in argv}
    args._cli_resource_split = bool(options & {"--route-sm-margin", "--shared-sms"})
    if "--tokens" in options:
        parser.error("offline token samples/capacity come from token_buckets in --config; --tokens is benchmark-only")
    if min(args.trials, args.compile_timeout) < 1:
        parser.error("--trials and --compile-timeout must be positive")
    if not args.topology_id or not args.topology_id.strip():
        parser.error("--topology-id must identify the active fabric/partition")
    if args.tuned_profile or args.trace_path or args.json_output:
        parser.error("use benchmark_shared for --tuned-profile/--trace-path; tuning writes --profile-output")
    return args


def main(argv=None):
    """Tune once offline; no per-forward search or dispatch is installed."""
    args = _parse_args(argv)
    import torch
    import torch.distributed as dist
    from mscclpp import Communicator, TcpBootstrap
    from .api import is_available
    from .benchmark import _weights
    from .benchmark_shared import _configs, _phase, _scope_report
    from .jit import compile_kernel

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not all(os.environ.get(name) for name in ("MASTER_ADDR", "MASTER_PORT")):
        raise RuntimeError("set explicit MASTER_ADDR and MASTER_PORT, normally using torchrun")
    if not torch.cuda.is_available() or torch.version.hip or not is_available():
        raise RuntimeError("offline native MegaMoE tuning requires an SM100 CUDA native build")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("native MegaMoE tuning requires SM100")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("gloo", rank=rank, world_size=world)
    bootstrap = shared_bootstrap = communicator = shared_communicator = None
    try:

        def read_input():
            with Path(args.config).open() as source:
                return validate_tuning_input(json.load(source))

        tuning = collective_call(read_input, phase="tuning input")
        local_paths = {"config", "profile_output", "cache_dir", "nvcc", "cutlass_root"}
        collective_agree(
            {
                "input": tuning,
                "arguments": {field: value for field, value in vars(args).items() if field not in local_paths},
            },
            phase="tuning plan",
        )
        compiled, compile_errors = [], []
        # No native bootstrap/context exists until all ranks finish all compiles.
        for candidate in tuning["kernel_candidates"]:
            try:
                kernel = collective_call(
                    lambda: compile_kernel(
                        _kernel_config(candidate),
                        cache_dir=args.cache_dir,
                        nvcc=args.nvcc,
                        cutlass_root=args.cutlass_root,
                        timeout=args.compile_timeout,
                    ),
                    phase=f"compile {candidate}",
                )
                collective_agree({"key": kernel.key, "config": asdict(kernel.config)}, phase="compiled module")
                compiled.append(kernel)
            except CollectiveTuningError as error:
                compile_errors.append({"kernel_config": candidate, "error": str(error)})
                _phase(rank, "candidate_compile_rejected", kernel_config=candidate, error=str(error))
                if candidate == DEFAULT_KERNELS[0]:
                    raise
        port = args.bootstrap_port if args.bootstrap_port is not None else int(os.environ["MASTER_PORT"]) + 1
        if not 1 <= port <= 65535:
            raise ValueError("native bootstrap port must be in [1, 65535]")
        bootstrap = TcpBootstrap.create(rank, world)
        bootstrap.initialize(f"{os.environ['MASTER_ADDR']}:{port}")
        communicator = Communicator(bootstrap)
        shared_bootstrap = TcpBootstrap.create(0, 1)
        shared_bootstrap.initialize(TcpBootstrap.create_unique_id())
        shared_communicator = Communicator(shared_bootstrap)
        physical_sms = torch.cuda.get_device_properties(device).multi_processor_count
        for workload in tuning["workloads"]:
            settings = {**vars(args), **{field: value for field, value in workload.items() if field != "local_experts"}}
            if "local_experts" in workload:
                settings["experts"] = workload["local_experts"] * world
            workload_args = SimpleNamespace(**settings)
            key = collect_selection_key(workload_args, device)
            splits = tuning["resource_splits"]
            if args._cli_resource_split or splits is None:
                splits = [_split({"route_sm_margin": args.route_sm_margin, "shared_sms": args.shared_sms})]
            for bucket in tuning["token_buckets"]:
                workload_args.max_tokens, workload_args.tokens = bucket["max"], bucket["max"]
                candidates = []
                for split in splits:
                    for kernel in compiled:
                        trial_args = SimpleNamespace(**{**vars(workload_args), **split})
                        candidate = {
                            "candidate": len(candidates),
                            "kernel_config": asdict(kernel.config),
                            "kernel_key": kernel.key,
                            "resource_split": split,
                            "status": "pending",
                            "trials": [],
                        }
                        try:
                            collective_call(
                                lambda: _configs(trial_args, rank, world, physical_sms), phase="resource feasibility"
                            )
                        except CollectiveTuningError as error:
                            candidate.update(status="rejected", error=str(error))
                        candidates.append(candidate)
                weights = None
                schedule = []
                referenced_kernels = set()
                try:
                    weight_args = SimpleNamespace(**{**vars(workload_args), "route_sm_margin": 0, "shared_sms": 2})
                    configs = collective_call(
                        lambda: _configs(weight_args, rank, world, physical_sms), phase="workload shape"
                    )

                    def make_weights():
                        torch.manual_seed(workload_args.seed + rank)
                        routed_weights = _weights(configs[0], device)
                        torch.manual_seed(workload_args.seed + 100000)
                        return routed_weights, _weights(configs[1], device)

                    weights = collective_call(make_weights, phase="synthetic weights")
                    for trial in range(args.trials):
                        offset = trial % len(candidates)
                        order = candidates[offset:] + candidates[:offset]
                        schedule.append([item["candidate"] for item in order if item["status"] != "rejected"])
                        for candidate in order:
                            if candidate["status"] == "rejected":
                                continue
                            kernel = next(item for item in compiled if item.key == candidate["kernel_key"])
                            trial_args = SimpleNamespace(**{**vars(workload_args), **candidate["resource_split"]})
                            _phase(rank, "tuning_trial", candidate=candidate["candidate"], trial=trial, bucket=bucket)
                            try:
                                check = not candidate["trials"]
                                independent_reference = check and kernel.key not in referenced_kernels
                                result = _run_trial(
                                    trial_args,
                                    bucket,
                                    kernel,
                                    communicator,
                                    shared_communicator,
                                    weights,
                                    device,
                                    rank,
                                    world,
                                    check=check,
                                    independent_reference=independent_reference,
                                )
                                candidate["trials"].append(result)
                                candidate["status"] = "ok"
                                if independent_reference:
                                    referenced_kernels.add(kernel.key)
                            except CollectiveTuningError as error:
                                candidate.update(status="rejected", error=str(error))
                                if candidate["kernel_config"] == DEFAULT_KERNELS[0] and "correctness" in str(error):
                                    raise
                    winner, scores = select_winner(candidates, bucket["samples"])
                    entry = {
                        "id": _all_gather(uuid.uuid4().hex if rank == 0 else None)[0],
                        "selection_key": key,
                        "token_bucket": bucket,
                        "winner": {
                            "kernel_config": winner["kernel_config"],
                            "kernel_key": winner["kernel_key"],
                            **winner["resource_split"],
                        },
                        "measurement": {
                            "objective": "minimum worst representative-sample latency ratio to one common feasible builtin",
                            "metric": "median of samplewise worst-rank full-layer CUDA-event times",
                            "scores": scores,
                            "candidates": candidates,
                            "compile_errors": compile_errors,
                        },
                        "provenance": {
                            "created_at": _all_gather(datetime.now(timezone.utc).isoformat() if rank == 0 else None)[0],
                            "input": tuning,
                            "scope": _scope_report(workload_args),
                            "methodology": "rotating candidate order; warmed CUDA graphs; repeated same input per timing graph",
                            "reference_split": scores[0]["reference_split"],
                            "reference_candidate": scores[0]["reference_candidate"],
                            "reference_selection": "prefer feasible builtin 32/32, otherwise first feasible builtin in search order",
                            "timed_schedule": schedule,
                            "warmup": args.warmup,
                            "iterations": args.iterations,
                            "trials": args.trials,
                            "reference_relative_l2": args.reference_relative_l2,
                            "correctness": (
                                "every candidate checked before timing; one independent representative-sample "
                                "CPU oracle per kernel specialization; changed input/residual graph replay; "
                                f"empty/ragged schedule checks; full_edge_references={args.full_edge_references}"
                            ),
                            "limits": "synthetic seeded weights/routes; only measured samples in this bucket, not a global optimum",
                        },
                    }
                    collective_agree(entry, phase="completed profile")
                    collective_call(lambda: save_profile(args.profile_output, entry), phase="profile persistence")
                    _phase(rank, "profile_saved", path=args.profile_output, winner=entry["winner"], bucket=bucket)
                finally:
                    del weights
                    gc.collect()
    finally:
        torch.cuda.synchronize(device)
        del communicator, shared_communicator
        gc.collect()
        del bootstrap, shared_bootstrap
        gc.collect()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
