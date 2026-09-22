# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only contracts for offline search, atomic profiles and cache-only reuse."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from mscclpp.ext.megamoe import autotune
from mscclpp.ext.megamoe import benchmark_shared
from mscclpp.ext.megamoe.jit import CompiledKernel, KernelConfig


def _input():
    return {"version": 1, "token_buckets": [{"min": 0, "max": 32, "samples": [1, 16, 32]}]}


def _key():
    return autotune.make_selection_key(
        benchmark_shared._parse_args([]),
        [{"gpu": "SM100", "sms": 152, "source": "abc"}] * 2,
        {"fabric": "test-fabric", "rank_hosts": [0, 0], "local_devices": [0, 1]},
    )


def _entry(identifier="first", lower=0, upper=32):
    return {
        "id": identifier,
        "selection_key": _key(),
        "token_bucket": {"min": lower, "max": upper, "samples": [upper]},
        "winner": {
            "kernel_config": asdict(KernelConfig()),
            "kernel_key": "builtin",
            "route_sm_margin": 32,
            "shared_sms": 32,
        },
        "measurement": {"test": "CPU fixture; never a real measurement"},
        "provenance": {"test": True},
    }


def _measurement(index, config, values, split=None):
    return {
        "candidate": index,
        "kernel_config": config,
        "resource_split": split or {"route_sm_margin": 32, "shared_sms": 32},
        "status": "ok",
        "trials": [{"samples": {str(tokens): {"rank_samples_us": ranks} for tokens, ranks in values.items()}}],
    }


class InputTests(unittest.TestCase):
    def test_defaults_are_concrete_and_bounded(self):
        value = autotune.validate_tuning_input(_input())
        self.assertEqual(value["kernel_candidates"], autotune.DEFAULT_KERNELS)
        self.assertIsNone(value["resource_splits"])
        self.assertEqual(value["workloads"], [{}])
        packaged = Path(autotune.__file__).with_name("megamoe_tuning.json")
        loaded = autotune.validate_tuning_input(json.loads(packaged.read_text()))
        self.assertEqual(loaded["resource_splits"], [{"route_sm_margin": 32, "shared_sms": 32}])

    def test_builtin_baseline_inserted_and_first(self):
        value = _input()
        value["kernel_candidates"] = [autotune.DEFAULT_KERNELS[1]]
        self.assertEqual(autotune.validate_tuning_input(value)["kernel_candidates"], autotune.DEFAULT_KERNELS[:2])
        value["kernel_candidates"] += [autotune.DEFAULT_KERNELS[0]]
        self.assertEqual(autotune.validate_tuning_input(value)["kernel_candidates"], autotune.DEFAULT_KERNELS[:2])

    def test_reject_invalid_schema(self):
        for update in (
            {"version": 2},
            {"version": True},
            {"status": "draft"},
            {"kernel_candidates": []},
            {"kernel_candidates": [autotune.DEFAULT_KERNELS[0]] * 2},
            {"kernel_candidates": [{"tile_n": 128, "load_stages": 4, "transform_stages": 7}]},
            {"kernel_candidates": [{"tile_n": True, "load_stages": 6, "transform_stages": 7}]},
            {"kernel_candidates": [{"tile_n": 32, "load_stages": 6, "transform_stages": 7, "tile_k": 96}]},
            {"resource_splits": []},
            {"resource_splits": [{"route_sm_margin": 16, "shared_sms": 32}]},
            {"resource_splits": [{"route_sm_margin": 31, "shared_sms": 16}]},
            {"resource_splits": [{"route_sm_margin": 32, "shared_sms": 3}]},
            {"token_buckets": []},
            {"token_buckets": [{"min": 0, "max": 0, "samples": [0]}]},
            {"token_buckets": [{"min": 0, "max": 32, "samples": [0, 32]}]},
            {"token_buckets": [{"min": 0, "max": 32, "samples": [33]}]},
            {"token_buckets": [{"min": 16, "max": 32, "samples": [15]}]},
            {"token_buckets": [{"min": 0, "max": 32, "samples": [1, 1]}]},
            {"token_buckets": [{"min": 0, "max": 32, "samples": [1]}, {"min": 32, "max": 64, "samples": [64]}]},
            {"workloads": []},
            {"workloads": [{"unknown": 2}]},
            {"workloads": [{"hidden": 129}]},
            {"workloads": [{"top_k": 33}]},
            {"workloads": [{"router_allow_tf32": 1}]},
            {"workloads": [{"router_weight_layout": "other"}]},
            {"workloads": [{"rms_eps": float("nan")}]},
            {"workloads": [{"rms_eps": True}]},
            {"workloads": [{"seed": -1}]},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                autotune.validate_tuning_input({**_input(), **update})

    def test_cli_reuses_benchmark_frontend_and_timing_arguments(self):
        args = autotune._parse_args(
            [
                "--profile-output",
                "profiles.json",
                "--topology-id",
                "fabric-A",
                "--no-post-norm",
                "--router-probability-order",
                "selected-logits",
                "--trials",
                "2",
                "--graph-batch",
                "3",
                "--cache-dir",
                "cache",
                "--route-sm-margin",
                "48",
                "--shared-sms",
                "48",
            ]
        )
        self.assertFalse(args.post_norm)
        self.assertEqual(args.router_probability_order, "selected-logits")
        self.assertEqual((args.trials, args.graph_batch, args.cache_dir), (2, 3, "cache"))
        self.assertTrue(args._cli_resource_split)

    def test_cli_requires_topology_and_rejects_replay_or_trace(self):
        for extra in (
            [],
            ["--topology-id", "fabric", "--tuned-profile", "other.json"],
            ["--topology-id", "fabric", "--trials", "0"],
            ["--topology-id", "fabric", "--tokens", "8"],
        ):
            with (
                self.subTest(extra=extra),
                patch("sys.stderr", new_callable=io.StringIO),
                self.assertRaises(SystemExit),
            ):
                autotune._parse_args(["--profile-output", "profiles.json", *extra])

    def test_explicit_capacity_is_used_before_capture(self):
        args = benchmark_shared._parse_args(["--tokens", "5"])
        args.max_tokens = 32
        routed, shared = benchmark_shared._configs(args, 0, 2, 152)
        self.assertEqual((routed.max_tokens, shared.max_tokens), (32, 32))
        args.max_tokens = 4
        with self.assertRaisesRegex(ValueError, "cover"):
            benchmark_shared._configs(args, 0, 2, 152)


class CollectiveTests(unittest.TestCase):
    def test_local_failure_reaches_all_ranks(self):
        with self.assertRaisesRegex(autotune.CollectiveTuningError, "rank 0: ValueError: compiler missing"):
            autotune.collective_call(
                lambda: (_ for _ in ()).throw(ValueError("compiler missing")),
                lambda error: [error, None],
                "compile",
            )

    def test_remote_failure_does_not_return_local_success(self):
        with self.assertRaisesRegex(autotune.CollectiveTuningError, "rank 1: compiler failed"):
            autotune.collective_call(lambda: object(), lambda error: [error, "compiler failed"], "compile")

    def test_success_returns_unserialized_local_result(self):
        value = object()
        self.assertIs(autotune.collective_call(lambda: value, lambda error: [error] * 2), value)

    def test_expected_preparation_and_correctness_errors_are_aggregated(self):
        for error in (
            ValueError("bad input"),
            TypeError("bad type"),
            RuntimeError("native failure"),
            OSError("cache failure"),
            TimeoutError("compiler timeout"),
            json.JSONDecodeError("bad JSON", "invalid", 0),
            AssertionError("incorrect result"),
            ImportError("missing dependency"),
            MemoryError("allocation failure"),
            subprocess.CalledProcessError(1, ["nvcc", "--version"]),
            subprocess.TimeoutExpired(["nvcc", "--version"], 30),
        ):
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(autotune.CollectiveTuningError, type(error).__name__),
            ):
                autotune.collective_call(lambda: (_ for _ in ()).throw(error), lambda value: [value])

    def test_unexpected_programming_errors_and_collective_failures_are_not_swallowed(self):
        with patch.object(autotune, "_all_gather") as gather, self.assertRaises(KeyError):
            autotune.collective_call(lambda: {}["missing"], gather)
        gather.assert_not_called()
        failure = RuntimeError("process group disconnected")
        with self.assertRaisesRegex(RuntimeError, "process group disconnected") as raised:
            autotune.collective_call(lambda: None, lambda value: (_ for _ in ()).throw(failure))
        self.assertIs(raised.exception, failure)

    def test_agreement_checks_variant_capacity_and_ctas(self):
        value = {"kernel": "builtin", "max_tokens": 32, "routed_ctas": 120, "shared_ctas": 32}
        self.assertEqual(
            autotune.collective_agree(value, lambda item: [item, dict(reversed(list(item.items())))]), value
        )
        for field in value:
            with self.subTest(field=field), self.assertRaisesRegex(autotune.CollectiveTuningError, "disagree"):
                autotune.collective_agree(value, lambda item: [item, {**item, field: "different"}])

    def test_collect_runtime_information_without_literal_hostname_key(self):
        args = benchmark_shared._parse_args(["--topology-id", "fabric"])
        with patch("mscclpp.ext.megamoe.jit.runtime_fingerprint", return_value={"gpu": "test"}):
            key = autotune.collect_selection_key(args, gather=lambda value: [value])
        self.assertEqual(key["topology"]["rank_hosts"], [0])
        self.assertEqual(key["topology"]["fabric"], "fabric")
        self.assertIn("offline_frontend_sha256", key["rank_fingerprints"][0])
        self.assertNotIn("host", key["topology"])

    def test_topology_required_and_rank_count_validated(self):
        args = benchmark_shared._parse_args([])
        for topology in (
            {"fabric": None, "rank_hosts": [0], "local_devices": [0]},
            {"fabric": "fabric", "rank_hosts": [0, 1], "local_devices": [0]},
        ):
            with self.assertRaises(ValueError):
                autotune.make_selection_key(args, [{}], topology)

    @staticmethod
    def _run_with_native_failures(constructors, synchronize=lambda device: None):
        def all_gather_object(output, value):
            output[:] = [value]

        distributed = SimpleNamespace(
            get_world_size=lambda: 1,
            all_gather_object=all_gather_object,
            barrier=lambda: None,
        )
        fake_torch = SimpleNamespace(
            distributed=distributed,
            cuda=SimpleNamespace(
                get_device_properties=lambda device: SimpleNamespace(multi_processor_count=152),
                synchronize=synchronize,
            ),
        )
        args = benchmark_shared._parse_args([])
        args.max_tokens = 32
        with (
            patch.dict(sys.modules, {"torch": fake_torch, "torch.distributed": distributed}),
            patch("mscclpp.ext.megamoe.api.MegaMoE", side_effect=constructors),
        ):
            return autotune._run_trial(
                args,
                {"min": 0, "max": 32, "samples": [1]},
                CompiledKernel(KernelConfig(), "", "builtin", True),
                object(),
                object(),
                ((), ()),
                "device",
                0,
                1,
                check=False,
            )

    def test_native_constructor_infeasibility_is_rejected_collectively(self):
        for constructors, phase in (
            ([ValueError("SMEM planning infeasible")], "routed native context preflight"),
            ([object(), RuntimeError("occupancy planning infeasible")], "shared native context preflight"),
        ):
            with self.subTest(phase=phase), self.assertRaisesRegex(autotune.CollectiveTuningError, phase):
                self._run_with_native_failures(constructors)

    def test_asynchronous_cuda_failure_during_cleanup_aborts_trial(self):
        failure = RuntimeError("asynchronous CUDA fault")
        with self.assertRaisesRegex(RuntimeError, "asynchronous CUDA fault") as raised:
            self._run_with_native_failures(
                [ValueError("native preflight failure")],
                synchronize=lambda device: (_ for _ in ()).throw(failure),
            )
        self.assertIs(raised.exception, failure)
        self.assertNotIsInstance(raised.exception, autotune.CollectiveTuningError)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        # Keep test scratch data in the checkout, never in a system temp directory.
        self.directory = Path.cwd() / f".megamoe-autotune-test-{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.path = self.directory / "profiles.json"
        self.entry = _entry()
        autotune.save_profile(self.path, self.entry)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_inclusive_edges_and_no_extrapolation(self):
        builtin = CompiledKernel(KernelConfig(), "", "builtin", True)
        with patch("mscclpp.ext.megamoe.jit.load_cached_kernel", return_value=builtin) as load:
            for tokens in (0, 1, 16, 31, 32):
                resolved = autotune.resolve_profile(self.path, _key(), tokens, cache_dir="local-cache")
                self.assertEqual((resolved.kernel, resolved.max_tokens), (builtin, 32))
            load.assert_called_with("builtin", cache_dir="local-cache")
            for tokens in (-1, 33, True):
                with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                    autotune.resolve_profile(self.path, _key(), tokens)

    def test_exact_scope_mismatches_are_explicit(self):
        for field in ("shape", "topology", "world_size", "frontend", "execution", "rank_fingerprints"):
            key = _key()
            key[field] = {"different": True}
            with self.subTest(field=field), self.assertRaisesRegex(autotune.ProfileMismatchError, field):
                autotune.resolve_profile(self.path, key, 16)

    def test_routed_only_or_stale_profiles_are_not_reused(self):
        for field, value in (("schedule", "routed-only"), ("graph_batch", 99)):
            key = _key()
            key["execution"][field] = value
            with self.assertRaises(autotune.ProfileMismatchError):
                autotune.resolve_profile(self.path, key, 16)
        key = _key()
        key["rank_fingerprints"][0]["source"] = "changed"
        with self.assertRaises(autotune.ProfileMismatchError):
            autotune.resolve_profile(self.path, key, 16)

    def test_missing_local_cache_is_not_compiled(self):
        with (
            patch("mscclpp.ext.megamoe.jit.load_cached_kernel", side_effect=FileNotFoundError("cache missing")),
            patch("mscclpp.ext.megamoe.jit.compile_kernel") as compile_kernel,
            self.assertRaises(FileNotFoundError),
        ):
            autotune.resolve_profile(self.path, _key(), 16)
        compile_kernel.assert_not_called()

    def test_jit_winner_loads_local_key_not_saved_path(self):
        entry = _entry("jit")
        entry["winner"].update(kernel_config=autotune.DEFAULT_KERNELS[1], kernel_key="a" * 64)
        autotune.save_profile(self.path, entry)
        kernel = CompiledKernel(KernelConfig(32, 6, 7), str(self.directory / "rank-local.so"), "a" * 64, True)
        with patch("mscclpp.ext.megamoe.jit.load_cached_kernel", return_value=kernel) as load:
            resolved = autotune.resolve_profile(self.path, _key(), 32)
            self.assertIs(resolved.kernel, kernel)
            load.assert_called_once_with("a" * 64, cache_dir=None)
        self.assertNotIn("rank-local.so", self.path.read_text())

    def test_mismatched_cached_module_rejected(self):
        wrong = CompiledKernel(KernelConfig(32, 6, 7), str(self.directory / "wrong.so"), "a" * 64, True)
        with (
            patch("mscclpp.ext.megamoe.jit.load_cached_kernel", return_value=wrong),
            self.assertRaisesRegex(autotune.ProfileMismatchError, "cached kernel"),
        ):
            autotune.resolve_profile(self.path, _key(), 16)

    def test_merge_preserves_other_shapes_buckets_and_previous_runs(self):
        autotune.save_profile(self.path, self.entry)
        second = _entry("second", 33, 64)
        autotune.save_profile(self.path, second)
        other_shape = _entry("shape")
        other_shape["selection_key"]["shape"]["hidden"] = 128
        autotune.save_profile(self.path, other_shape)
        replacement = _entry("replacement")
        autotune.save_profile(self.path, replacement)
        # A slow rank finishing an older save must not undo a newer completed run.
        autotune.save_profile(self.path, self.entry)
        profiles = autotune._read_profiles(self.path)
        self.assertEqual({item["id"] for item in profiles["entries"]}, {"replacement", "second", "shape"})
        self.assertEqual(profiles["history"], [self.entry])
        self.assertFalse(list(self.directory.glob("*.pending")))

    def test_conflicting_same_run_id_and_overlapping_buckets_rejected(self):
        entry = deepcopy(self.entry)
        entry["measurement"] = {"different": "data"}
        with self.assertRaisesRegex(ValueError, "same run id"):
            autotune.save_profile(self.path, entry)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            autotune.save_profile(self.path, _entry("overlap", 16, 64))
        self.assertEqual(autotune._read_profiles(self.path)["entries"], [self.entry])

    def test_concurrent_writers_do_not_lose_entries(self):
        entries = [_entry(f"parallel-{index}", 33 + index * 32, 64 + index * 32) for index in range(8)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda entry: autotune.save_profile(self.path, entry), entries))
        saved = autotune._read_profiles(self.path)
        self.assertEqual({item["id"] for item in saved["entries"]}, {item["id"] for item in entries + [self.entry]})

    def test_atomic_write_failure_retains_previous_json(self):
        with patch.object(autotune.os, "replace", side_effect=OSError("write failure")), self.assertRaises(OSError):
            autotune.save_profile(self.path, _entry("replacement"))
        self.assertEqual(autotune._read_profiles(self.path)["entries"], [self.entry])
        self.assertFalse(list(self.directory.glob("*.pending")))

    def test_profile_schema_rejects_draft_or_module_paths(self):
        self.path.write_text(json.dumps({"version": 1, "status": "draft"}))
        with self.assertRaises(ValueError):
            autotune.resolve_profile(self.path, _key(), 16)
        entry = _entry("bad")
        entry["winner"]["kernel_key"] = "/another-node/module.so"
        with self.assertRaises(ValueError):
            autotune.save_profile(self.path, entry)

    def test_benchmark_wiring_resolves_once_before_context_capacity(self):
        args = benchmark_shared._parse_args(
            ["--tokens", "7", "--tuned-profile", str(self.path), "--topology-id", "test-fabric"]
        )
        kernel = CompiledKernel(KernelConfig(), "", "builtin", True)
        with (
            patch.object(autotune, "collect_selection_key", return_value=_key()),
            patch("mscclpp.ext.megamoe.jit.load_cached_kernel", return_value=kernel) as load,
            patch("mscclpp.ext.megamoe.jit.compile_kernel") as compile_kernel,
        ):
            resolved = autotune.resolve_for_benchmark(args, gather=lambda value: [value, value])
        load.assert_called_once()
        compile_kernel.assert_not_called()
        self.assertEqual((args.max_tokens, args.route_sm_margin, args.shared_sms), (32, 32, 32))
        self.assertEqual(resolved.kernel.key, "builtin")
        route, shared = benchmark_shared._configs(args, 0, 2, 152)
        self.assertEqual((route.max_tokens, shared.max_tokens), (32, 32))
        self.assertEqual(args.tokens, 7)

    def test_mocked_cli_compiles_before_bootstraps_and_persists_winner(self):
        plan = {**_input(), "kernel_candidates": autotune.DEFAULT_KERNELS[:2]}
        config_path = self.directory / "input.json"
        config_path.write_text(json.dumps(plan))
        profile_path = self.directory / "cli-profiles.json"
        phases = []
        trials = []

        class Bootstrap:
            @staticmethod
            def create(rank, world):
                phases.append("bootstrap")
                return Bootstrap()

            @staticmethod
            def create_unique_id():
                return "local-id"

            def initialize(self, address):
                pass

        def all_gather_object(output, value):
            output[:] = [value]

        distributed = SimpleNamespace(
            get_world_size=lambda: 1,
            all_gather_object=all_gather_object,
            init_process_group=lambda *args, **kwargs: None,
            destroy_process_group=lambda: None,
        )
        fake_torch = SimpleNamespace(
            distributed=distributed,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                set_device=lambda device: None,
                get_device_capability=lambda device: (10, 0),
                get_device_properties=lambda device: SimpleNamespace(multi_processor_count=152),
                synchronize=lambda device: None,
            ),
            version=SimpleNamespace(hip=None),
            device=lambda *args: "device",
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
            manual_seed=lambda seed: None,
        )

        def compile_kernel(config, **kwargs):
            phases.append("compile")
            if config == KernelConfig():
                return CompiledKernel(config, "", "builtin", True)
            return CompiledKernel(config, str(self.directory / "local.so"), "a" * 64, True)

        def run_trial(args, bucket, kernel, *unused, check):
            trials.append((kernel.key, args.max_tokens, args.tokens, check))
            latency = 10 if kernel.key == "builtin" else 9
            return {
                "correctness": {"checked": check},
                "samples": {str(tokens): {"rank_samples_us": [[latency] * 3]} for tokens in bucket["samples"]},
            }

        with (
            patch.dict(sys.modules, {"torch": fake_torch, "torch.distributed": distributed}),
            patch.dict(
                autotune.os.environ,
                {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MASTER_ADDR": "host", "MASTER_PORT": "29500"},
            ),
            patch.object(sys.modules["mscclpp"], "TcpBootstrap", Bootstrap, create=True),
            patch.object(sys.modules["mscclpp"], "Communicator", lambda bootstrap: object(), create=True),
            patch("mscclpp.ext.megamoe.api.is_available", return_value=True),
            patch("mscclpp.ext.megamoe.jit.compile_kernel", side_effect=compile_kernel),
            patch("mscclpp.ext.megamoe.benchmark._weights", return_value=("weights",)),
            patch.object(autotune, "collect_selection_key", return_value=_key()),
            patch.object(autotune, "_run_trial", side_effect=run_trial),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            autotune.main(
                [
                    "--config",
                    str(config_path),
                    "--profile-output",
                    str(profile_path),
                    "--topology-id",
                    "test-fabric",
                    "--trials",
                    "2",
                ]
            )
        self.assertEqual(phases[:2], ["compile", "compile"])
        self.assertEqual(phases[2:], ["bootstrap", "bootstrap"])
        self.assertEqual([trial[0] for trial in trials], ["builtin", "a" * 64, "a" * 64, "builtin"])
        self.assertEqual([trial[3] for trial in trials], [True, True, False, False])
        self.assertTrue(all(trial[1:3] == (32, 32) for trial in trials))
        entry = autotune._read_profiles(profile_path)["entries"][0]
        self.assertEqual(entry["winner"]["kernel_key"], "a" * 64)
        self.assertEqual(entry["measurement"]["scores"][1]["worst_ratio"], 0.9)
        self.assertEqual(entry["provenance"]["reference_split"], autotune.DEFAULT_RESOURCE_SPLIT)
        self.assertEqual(entry["provenance"]["reference_candidate"], 0)


class SelectionTests(unittest.TestCase):
    def test_worst_rank_is_aggregated_before_median(self):
        baseline = _measurement(0, autotune.DEFAULT_KERNELS[0], {1: [[1, 100, 1], [100, 1, 100]]})
        candidate = _measurement(1, autotune.DEFAULT_KERNELS[1], {1: [[60, 60, 60], [60, 60, 60]]})
        winner, scores = autotune.select_winner([baseline, candidate], [1])
        self.assertEqual(winner["candidate"], 1)
        self.assertEqual(scores[0]["median_us"]["1"], 100)
        self.assertEqual(scores[1]["worst_ratio"], 0.6)

    def test_worst_sample_ratio_not_average_or_component_latency(self):
        baseline = _measurement(0, autotune.DEFAULT_KERNELS[0], {1: [[10]], 32: [[100]]})
        faster_average = _measurement(1, autotune.DEFAULT_KERNELS[1], {1: [[11]], 32: [[50]]})
        balanced = _measurement(2, autotune.DEFAULT_KERNELS[2], {1: [[9]], 32: [[90]]})
        winner, scores = autotune.select_winner([baseline, faster_average, balanced], [1, 32])
        self.assertEqual(winner["candidate"], 2)
        self.assertEqual([score["worst_ratio"] for score in scores], [1, 1.1, 0.9])

    def test_common_reference_compares_complete_layer_latency_across_splits(self):
        split = {"route_sm_margin": 48, "shared_sms": 48}
        first = _measurement(0, autotune.DEFAULT_KERNELS[0], {1: [[100]]})
        fast = _measurement(1, autotune.DEFAULT_KERNELS[1], {1: [[90]]})
        second = _measurement(2, autotune.DEFAULT_KERNELS[0], {1: [[200]]}, split)
        slow = _measurement(3, autotune.DEFAULT_KERNELS[1], {1: [[160]]}, split)
        rejected = _measurement(4, autotune.DEFAULT_KERNELS[2], {1: [[1]]})
        rejected["status"] = "rejected"
        # Default 32/32 is the reference even if another split occurs first.
        winner, scores = autotune.select_winner([second, slow, first, fast, rejected], [1])
        self.assertEqual(winner["candidate"], 1)
        self.assertEqual([score["worst_ratio"] for score in scores], [2, 1.6, 1, 0.9])
        self.assertTrue(all(score["reference_split"] == first["resource_split"] for score in scores))
        self.assertTrue(all(score["reference_median_us"] == {"1": 100} for score in scores))
        self.assertTrue(all(score["reference_candidate"] == 0 for score in scores))

    def test_feasible_variant_does_not_require_its_own_split_builtin(self):
        split = {"route_sm_margin": 48, "shared_sms": 48}
        baseline = _measurement(0, autotune.DEFAULT_KERNELS[0], {1: [[100]]})
        rejected = _measurement(1, autotune.DEFAULT_KERNELS[0], {1: [[200]]}, split)
        rejected["status"] = "rejected"
        tuned = _measurement(2, autotune.DEFAULT_KERNELS[1], {1: [[90]]}, split)
        winner, scores = autotune.select_winner([baseline, rejected, tuned], [1])
        self.assertEqual(winner["candidate"], 2)
        self.assertEqual(scores[1]["ratio_to_reference_builtin"], {"1": 0.9})
        self.assertEqual(scores[1]["reference_split"], baseline["resource_split"])

    def test_first_feasible_builtin_is_fallback_when_default_unavailable(self):
        rejected = _measurement(0, autotune.DEFAULT_KERNELS[0], {1: [[100]]})
        rejected["status"] = "rejected"
        split = {"route_sm_margin": 48, "shared_sms": 48}
        baseline = _measurement(1, autotune.DEFAULT_KERNELS[0], {1: [[200]]}, split)
        other = _measurement(2, autotune.DEFAULT_KERNELS[0], {1: [[160]]}, {"route_sm_margin": 16, "shared_sms": 16})
        winner, scores = autotune.select_winner([rejected, baseline, other], [1])
        self.assertEqual(winner["candidate"], 2)
        self.assertEqual([score["worst_ratio"] for score in scores], [1, 0.8])
        self.assertTrue(all(score["reference_split"] == split for score in scores))
        with self.assertRaisesRegex(ValueError, "baseline"):
            autotune.select_winner([rejected], [1])

    def test_empty_ragged_inputs_are_correctness_only(self):
        baseline = _measurement(0, autotune.DEFAULT_KERNELS[0], {1: [[10]]})
        baseline["trials"][0]["correctness"] = {"empty": {"latency": 0}, "ragged": {"latency": 0}}
        winner, _ = autotune.select_winner([baseline], [1])
        self.assertEqual(winner["candidate"], 0)
        with self.assertRaises(ValueError):
            autotune.select_winner([baseline], [0])

    def test_invalid_measurements_cannot_win(self):
        for ranks in ([], [[]], [[1], [1, 2]], [[float("nan")]], [[0]], [[-1]]):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                autotune.select_winner([_measurement(0, autotune.DEFAULT_KERNELS[0], {1: ranks})], [1])

    def test_repeated_trials_are_aggregated_not_best_run_selected(self):
        baseline = _measurement(0, autotune.DEFAULT_KERNELS[0], {1: [[10]]})
        candidate = _measurement(1, autotune.DEFAULT_KERNELS[1], {1: [[1]]})
        candidate["trials"].extend([{"samples": {"1": {"rank_samples_us": [[20]]}}}] * 2)
        winner, scores = autotune.select_winner([baseline, candidate], [1])
        self.assertEqual(winner["candidate"], 0)
        self.assertEqual(scores[1]["worst_ratio"], 2)


if __name__ == "__main__":
    unittest.main()
