# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Real-GPU regression for the old MoECommunicator expanded-layout API.

GPU suite (requires a built/installed mscclpp with the expanded LL backend):
    mpirun -np 8 python test/python/ep/test_rank_major_topk_expanded.py
    mpirun -np 8 python -m pytest -q test/python/ep/test_rank_major_topk_expanded.py

Independent, standard-library-only oracle tests:
    python test/python/ep/test_rank_major_topk_expanded.py --cpu-only
Plain pytest also runs the CPU tests and skips the GPU test outside mpirun.
Importing this module does not import torch, mpi4py, or mscclpp.

Contracts: row = (source_rank * capacity + source_token) * topk + k;
tokens/expert outputs [R * capacity * K, H], flat int32 IDs/FP32 weights,
and int32 counts [R] counting local selections, including zero-weight ones.
Only valid local dispatch payloads are specified. All other metadata must be
sentinel/zero, even after shrinking. Combine reads the exact registered expert
buffer and applies original weights once in source FP32, then converts to BF16.

Expert results are unweighted input copies plus small, exact integer slot tags
to expose duplicate-slot aliasing. Inactive and valid zero-weight expert rows
are deliberately NaN. A separate identity-expert case distinguishes source FP32
from per-destination BF16 rounding using exactly representable quarter weights.
Eight paired operations, including route/shape changes, share one CUDA graph;
snapshots of every pair are checked outside capture after every replay.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import struct
import sys
import traceback
from types import SimpleNamespace
import unittest

CASES = (
    "dense_signed",
    "changed_routes",
    "shrunk_mixed",
    "all_invalid",
    "empty_all",
    "implicit_unit",
    "source_fp32",
)
GRAPH_PAIRS = 8
LOCAL_EXPERTS = 4


@dataclass
class Sample:
    name: str
    x: list[list[float]]
    ids: list[list[int]]
    weights: list[list[float]] | None
    tagged: bool = True


@dataclass
class DispatchReference:
    ids: list[int]
    weights: list[float]
    counts: list[int]
    # Payloads exist only for valid local rows, INCLUDING zero-weight rows.
    payloads: dict[int, list[float]]


def _fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bf16(value):
    """Round a scalar FP32 value to BF16, nearest-even, without GPU packages."""
    value = _fp32(value)
    if not math.isfinite(value):
        return value
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _slot_bias(expert, k):
    # Repeated expert IDs still get distinct per-slot results. Never weight here.
    return (expert % LOCAL_EXPERTS) * 2 + k - 4


def _dispatch_reference(samples, destination, capacity, topk, num_experts, sentinel):
    ranks = len(samples)
    local_experts = num_experts // ranks
    rows = ranks * capacity * topk
    result = DispatchReference([sentinel] * rows, [0.0] * rows, [0] * ranks, {})
    for source, sample in enumerate(samples):
        assert len(sample.x) == len(sample.ids) < capacity
        for token, selections in enumerate(sample.ids):
            assert len(selections) == topk
            for k, expert in enumerate(selections):
                if not 0 <= expert < num_experts or expert // local_experts != destination:
                    continue
                row = (source * capacity + token) * topk + k
                result.ids[row] = expert
                result.weights[row] = 1.0 if sample.weights is None else sample.weights[token][k]
                result.counts[source] += 1
                result.payloads[row] = sample.x[token]
    return result


def _combine_reference(sample, num_experts):
    """Scalar source oracle: no received metadata, counts, or destination sums.

    Fixture inputs/biases are exact BF16 integers and weights are quarters, so
    each product is exactly representable in FP32. Explicit FP32 accumulation
    therefore also agrees with fused multiply-add, without host fma support.
    """
    result = []
    for token, values in enumerate(sample.x):
        total = [0.0] * len(values)
        for k, expert in enumerate(sample.ids[token]):
            if not 0 <= expert < num_experts:
                continue
            weight = 1.0 if sample.weights is None else sample.weights[token][k]
            if weight == 0:
                continue  # Skip before reading even a potentially NaN payload.
            bias = _slot_bias(expert, k) if sample.tagged else 0
            for column, value in enumerate(values):
                expert_value = _bf16(value + bias)
                total[column] = _fp32(total[column] + expert_value * weight)
        result.append([_bf16(value) for value in total])
    return result


def _make_sample(name, rank, ranks, capacity, topk, hidden):
    phase = CASES.index(name)
    experts = ranks * LOCAL_EXPERTS
    tokens = capacity - 1
    if name == "shrunk_mixed":
        tokens = 0 if rank == ranks - 1 else 1
    elif name == "all_invalid":
        tokens = capacity - 2
    elif name == "empty_all":
        tokens = 0
    elif name == "source_fp32":
        tokens = 1

    x, ids, weights = [], [], []
    signed_weights = (0.5, -0.25, 1.5, -1.0, 0.0, 1.25, -0.5, 0.75, 2.0)
    for token in range(tokens):
        values = [float((rank * 13 + token * 7 + column * 3 + phase * 5) % 31 - 15) for column in range(hidden)]
        # Exact source/token anchors, plus varying payload across the whole row.
        values[:4] = [float(rank), float(token % 128), float(token // 128), float(phase)]
        owner = (rank + token + phase) % ranks
        a = owner * LOCAL_EXPERTS
        other = ((owner + 1) % ranks) * LOCAL_EXPERTS
        route = [((owner + k // 4) % ranks) * LOCAL_EXPERTS + (k // 2) % LOCAL_EXPERTS for k in range(topk)]
        w = [signed_weights[(k + token + phase) % len(signed_weights)] for k in range(topk)]
        if name in ("shrunk_mixed", "implicit_unit"):
            route = [a, a, a + 1, -1, experts, experts + 7, other, -7, other + 1][:topk]
            w = [0.5, -0.25, 0.0, 1.25, -1.0, 0.75, 1.5, -0.5, 2.0][:topk]
        elif name == "all_invalid":
            invalid = (-1, experts, experts + 7, -7)
            route = [invalid[(token + k) % len(invalid)] for k in range(topk)]
        elif name == "source_fp32":
            # 65 * 1.25 - 65 == 16.25; BF16(81.25) - 65 == 16.
            # Use a distant rank, crossing a typical two-node rank partition.
            values = [65.0] * hidden
            route = [rank * LOCAL_EXPERTS, ((rank + ranks // 2) % ranks) * LOCAL_EXPERTS]
            route += [rank * LOCAL_EXPERTS + k % LOCAL_EXPERTS for k in range(2, topk)]
            w = [1.25, -1.0] + [0.0] * (topk - 2)
        x.append(values)
        ids.append(route)
        weights.append(w)
    return Sample(name, x, ids, None if name == "implicit_unit" else weights, name != "source_fp32")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, default=7168, choices=(4096, 6656, 7168, 8192, 8704, 9216))
    parser.add_argument(
        "--capacity", type=int, default=8, help="Fixed allocation capacity, strictly larger than every batch"
    )
    parser.add_argument(
        "--topk", type=int, default=8, choices=range(2, 10), help="At least two slots to test duplicates/FP32"
    )
    parser.add_argument("--graph-replays", type=int, default=5)
    parser.add_argument("--cpu-only", action="store_true", help="Run only the independent stdlib unittest oracle tests")
    args = parser.parse_args(argv)
    if args.capacity < 3:
        parser.error("--capacity must be >= 3 to test nonempty shrinking batches")
    if args.graph_replays < 1:
        parser.error("--graph-replays must be >= 1")
    return args


def _prepare_case(torch, sample, reference, expected_combine, args, device, num_experts):
    """Stage every tensor and reference before capture; allgather is already done."""
    tokens = len(sample.x)
    rows = len(reference.ids)
    valid_rows = sorted(reference.payloads)
    biases = [
        _slot_bias(expert, row % args.topk) if sample.tagged and expert != num_experts else 0
        for row, expert in enumerate(reference.ids)
    ]
    poison = [expert == num_experts or weight == 0 for expert, weight in zip(reference.ids, reference.weights)]
    return SimpleNamespace(
        sample=sample,
        x=torch.tensor(sample.x, dtype=torch.bfloat16, device=device).reshape(tokens, args.hidden),
        ids=torch.tensor(sample.ids, dtype=torch.int64, device=device).reshape(tokens, args.topk),
        weights=(
            None
            if sample.weights is None
            else torch.tensor(sample.weights, dtype=torch.float32, device=device).reshape(tokens, args.topk)
        ),
        bias=torch.tensor(biases, dtype=torch.bfloat16, device=device).reshape(rows, 1),
        poison=torch.tensor(poison, dtype=torch.bool, device=device).reshape(rows, 1),
        valid_rows=torch.tensor(valid_rows, dtype=torch.int64, device=device),
        expected_tokens=torch.tensor(
            [reference.payloads[row] for row in valid_rows], dtype=torch.bfloat16, device="cpu"
        ).reshape(len(valid_rows), args.hidden),
        expected_ids=torch.tensor(reference.ids, dtype=torch.int32, device="cpu"),
        expected_weights=torch.tensor(reference.weights, dtype=torch.float32, device="cpu"),
        expected_counts=torch.tensor(reference.counts, dtype=torch.int32, device="cpu"),
        expected_combine=torch.tensor(expected_combine, dtype=torch.bfloat16, device="cpu").reshape(
            tokens, args.hidden
        ),
    )


def _make_step(torch, prepared, expert_output, ranks):
    # Public dispatch buffers alias across calls. Save EVERY captured pair, not
    # just the final result, so transient corruption cannot be hidden by reuse.
    device = expert_output.device
    return SimpleNamespace(
        case=prepared,
        tokens=torch.empty_like(expert_output),
        ids=torch.empty(expert_output.shape[0], dtype=torch.int32, device=device),
        weights=torch.empty(expert_output.shape[0], dtype=torch.float32, device=device),
        counts=torch.empty(ranks, dtype=torch.int32, device=device),
        combined=torch.empty_like(prepared.x),
        dispatch_out=None,
        handle=None,
        returned=None,
    )


def _execute_step(moe, expert_output, step):
    """Capture-safe: GPU ops only; no MPI, .cpu(), .item(), or synchronization."""
    case = step.case
    dispatch_out, handle = moe.dispatch(case.x, case.ids, case.weights, output_buffer=None)
    step.tokens.copy_(dispatch_out.tokens)
    step.ids.copy_(dispatch_out.topk_ids)
    step.weights.copy_(dispatch_out.weights)
    step.counts.copy_(dispatch_out.layout.num_tokens_per_rank)
    expert_output.copy_(dispatch_out.tokens)
    expert_output.add_(case.bias)
    expert_output.masked_fill_(case.poison, float("nan"))
    step.returned = moe.combine(expert_output, handle, out=step.combined)
    step.dispatch_out, step.handle = dispatch_out, handle


def _reset_snapshots(steps):
    # Outside capture: stale correct snapshots must not make a no-op replay pass.
    for step in steps:
        step.tokens.fill_(float("nan"))
        step.ids.fill_(-12345)
        step.weights.fill_(float("nan"))
        step.counts.fill_(-1)
        step.combined.fill_(float("nan"))


def _validate_step(torch, ep, step, expert_output, args, rank, ranks, label):
    case = step.case
    dispatch_out = step.dispatch_out
    rows = ranks * args.capacity * args.topk
    prefix = f"rank={rank} {label} case={case.sample.name}"
    assert dispatch_out.layout.kind == ep.DispatchLayout.RANK_MAJOR_TOPK_EXPANDED, prefix
    assert dispatch_out.quant is None and dispatch_out.layout.offsets is None, prefix
    assert dispatch_out.layout.num_tokens_per_expert is None, prefix
    for tensor, shape, dtype in (
        (dispatch_out.tokens, (rows, args.hidden), torch.bfloat16),
        (dispatch_out.topk_ids, (rows,), torch.int32),
        (dispatch_out.weights, (rows,), torch.float32),
        (dispatch_out.layout.num_tokens_per_rank, (ranks,), torch.int32),
        (expert_output, (rows, args.hidden), torch.bfloat16),
        (step.returned, (len(case.sample.x), args.hidden), torch.bfloat16),
    ):
        assert tensor is not None, prefix
        assert tuple(tensor.shape) == shape and tensor.dtype == dtype, prefix
        assert tensor.device == case.x.device and tensor.is_contiguous(), prefix
    assert dispatch_out.tokens.data_ptr() != expert_output.data_ptr(), prefix
    assert step.returned.data_ptr() == step.combined.data_ptr(), prefix
    context = step.handle.combine_context
    assert context.max_tokens_per_rank == args.capacity, prefix
    assert context.num_tokens == len(case.sample.x) < args.capacity, prefix
    assert context.topk_ids is case.ids and context.weights is case.weights, prefix

    for field, actual, expected in (
        ("IDs (including all padding/nonlocal rows)", step.ids.cpu(), case.expected_ids),
        ("weights (including all padding/nonlocal rows)", step.weights.cpu(), case.expected_weights),
        ("counts (slots, irrespective of weights)", step.counts.cpu(), case.expected_counts),
        ("valid payloads", step.tokens.index_select(0, case.valid_rows).cpu(), case.expected_tokens),
        ("source FP32 combine", step.combined.cpu(), case.expected_combine),
    ):
        # Exact comparison also rejects NaN, including nonempty all-invalid output.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=False, msg=f"{prefix}: {field}")


def run(args):
    # Initialize MPI before importing GPU packages so any later failure can
    # abort the world rather than leaving peers waiting in a collective/kernel.
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, ranks = comm.Get_rank(), comm.Get_size()
    try:
        if not 2 <= ranks <= 64:
            raise ValueError("GPU regression requires 2..64 MPI ranks (one GPU per rank)")
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        num_experts = ranks * LOCAL_EXPERTS
        samples = [_make_sample(name, rank, ranks, args.capacity, args.topk, args.hidden) for name in CASES]
        # Python object collective: the oracle has the actual source input,
        # routing, and weights before any CUDA allocation or dispatch is made.
        gathered = comm.allgather(samples)
        references = [
            _dispatch_reference(
                [source[index] for source in gathered], rank, args.capacity, args.topk, num_experts, num_experts
            )
            for index in range(len(CASES))
        ]
        expected_combines = [_combine_reference(sample, num_experts) for sample in samples]
        # Check the discriminator itself independently of any GPU result.
        assert expected_combines[-1] == [[16.25] * args.hidden]
        assert _bf16(_bf16(65.0 * 1.25) + _bf16(-65.0)) == 16.0

        import torch

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        import mscclpp
        import mscclpp.ep as ep

        group = mscclpp.CommGroup(mpi_comm=MPI.COMM_WORLD)
        moe = ep.MoECommunicator(
            comm=group,
            num_experts=num_experts,
            num_local_experts=LOCAL_EXPERTS,
            hidden_size=args.hidden,
            topk=args.topk,
            max_tokens_per_rank=args.capacity,
            mode=ep.MoEMode.LOW_LATENCY,
            output_layout=ep.DispatchLayout.RANK_MAJOR_TOPK_EXPANDED,
            low_latency_num_blocks=130,
            low_latency_combine_mode=ep.CombineMode.RANK_LOCAL_REDUCE,
            invalid_token_expert_id=num_experts,
            enable_overlap=False,
            quant=None,
        )
        assert moe.is_available(), "expanded low-latency backend is unavailable"
        expert_output = moe.get_expert_output_buffer()
        registered_pointer = expert_output.data_ptr()
        prepared = [
            _prepare_case(torch, sample, reference, expected, args, device, num_experts)
            for sample, reference, expected in zip(samples, references, expected_combines)
        ]
        # Dense -> rerouted -> shrinking -> invalid -> empty -> repopulated ->
        # rounding witness -> dense again, all on the SAME communicator/storage.
        steps = [_make_step(torch, case, expert_output, ranks) for case in prepared]
        steps.append(_make_step(torch, prepared[0], expert_output, ranks))
        assert len(steps) == GRAPH_PAIRS

        for index, step in enumerate(steps):
            _reset_snapshots([step])
            torch.cuda.synchronize()
            comm.Barrier()
            _execute_step(moe, expert_output, step)
            torch.cuda.synchronize()
            _validate_step(torch, ep, step, expert_output, args, rank, ranks, f"eager[{index}]")
            assert moe.get_expert_output_buffer().data_ptr() == registered_pointer
            comm.Barrier()
            if rank == 0:
                print(f"[EXPANDED_TEST] pass eager[{index}] {step.case.sample.name}", flush=True)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())
        # Exactly one warm dispatch + expert copy + combine on the capture stream.
        comm.Barrier()
        with torch.cuda.stream(stream):
            _execute_step(moe, expert_output, steps[0])
        torch.cuda.synchronize()
        _validate_step(torch, ep, steps[0], expert_output, args, rank, ranks, "graph warmup")
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for step in steps:
                _execute_step(moe, expert_output, step)

        for replay in range(args.graph_replays):
            _reset_snapshots(steps)
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for index, step in enumerate(steps):
                _validate_step(torch, ep, step, expert_output, args, rank, ranks, f"graph[{replay}] pair[{index}]")
            assert moe.get_expert_output_buffer().data_ptr() == registered_pointer
            comm.Barrier()
            if rank == 0:
                print(f"[EXPANDED_TEST] pass graph replay={replay + 1} pairs={GRAPH_PAIRS}", flush=True)
        if rank == 0:
            print(
                f"[EXPANDED_TEST] pass ranks={ranks} hidden={args.hidden} capacity={args.capacity} "
                f"topk={args.topk} graph_replays={args.graph_replays}",
                flush=True,
            )
    except BaseException:
        print(f"[EXPANDED_TEST] FAIL rank={rank}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        comm.Abort(1)
        raise


class CpuReferenceTests(unittest.TestCase):
    """Oracle checks only: these do not emulate or validate the native kernels."""

    def test_defaults(self):
        args = parse_args([])
        self.assertEqual((args.hidden, args.capacity, args.topk, args.graph_replays), (7168, 8, 8, 5))

    def test_bf16_rounds_nearest_even(self):
        self.assertEqual(_bf16(81.25), 81.0)
        self.assertEqual(_bf16(81.75), 82.0)
        self.assertEqual(_bf16(-81.25), -81.0)
        self.assertEqual(_bf16(16.25), 16.25)

    def test_fixed_rows_duplicates_zero_weight_and_global_ids(self):
        samples = [
            Sample("a", [[2.0]], [[2, 2, 3, -1, 4]], [[0.5, -0.25, 0.0, 7.0, 9.0]]),
            Sample("b", [[7.0]], [[0, 3, 5, -7, 2]], None),
        ]
        ref = _dispatch_reference(samples, 1, 3, 5, 4, -99)
        self.assertEqual(ref.counts, [3, 2])
        self.assertEqual(sorted(ref.payloads), [0, 1, 2, 16, 19])
        expected_ids = [-99] * 30
        expected_weights = [0.0] * 30
        for row, expert, weight in ((0, 2, 0.5), (1, 2, -0.25), (2, 3, 0.0), (16, 3, 1.0), (19, 2, 1.0)):
            expected_ids[row], expected_weights[row] = expert, weight
        self.assertEqual(ref.ids, expected_ids)
        self.assertEqual(ref.weights, expected_weights)
        self.assertEqual(ref.payloads[0], ref.payloads[1])

    def test_combine_applies_original_signed_weights_once(self):
        sample = Sample("signed", [[8.0, -4.0]], [[0, 0, 3, -1, 4]], [[0.5, -0.25, 1.5, 9.0, 9.0]], False)
        self.assertEqual(_combine_reference(sample, 4), [[14.0, -7.0]])
        sample.weights = None
        self.assertEqual(_combine_reference(sample, 4), [[24.0, -12.0]])

    def test_slot_tags_distinguish_duplicate_expert_rows(self):
        sample = Sample("duplicates", [[8.0]], [[0, 0]], [[0.5, -0.25]])
        # Slot results are 8-4=4 and 8-3=5, not a deduplicated expert row.
        self.assertEqual(_combine_reference(sample, 4), [[0.75]])

    def test_invalid_and_zero_weight_nan_rows_are_skipped(self):
        sample = Sample(
            "poison",
            [[math.nan], [math.nan]],
            [[-1, 4, 7, -7], [0, 1, 2, 3]],
            [[1.0, -1.0, 2.0, 0.5], [0.0, 0.0, 0.0, 0.0]],
        )
        self.assertEqual(_combine_reference(sample, 4), [[0.0], [0.0]])
        self.assertEqual(_combine_reference(Sample("empty", [], [], None), 4), [])

    def test_source_fp32_not_destination_bf16(self):
        for rank in range(8):
            sample = _make_sample("source_fp32", rank, 8, 8, 8, 8)
            self.assertNotEqual(sample.ids[0][0] // LOCAL_EXPERTS, sample.ids[0][1] // LOCAL_EXPERTS)
            self.assertEqual(_combine_reference(sample, 32), [[16.25] * 8])
        legacy = _bf16(_bf16(65.0 * 1.25) + _bf16(65.0 * -1.0))
        self.assertEqual(legacy, 16.0)
        self.assertNotEqual(legacy, 16.25)

    def test_fixture_transitions_and_all_destination_metadata(self):
        ranks, capacity, topk, hidden = 8, 8, 8, 8
        experts = ranks * LOCAL_EXPERTS
        all_cases = {
            name: [_make_sample(name, rank, ranks, capacity, topk, hidden) for rank in range(ranks)] for name in CASES
        }
        self.assertEqual(len(all_cases["shrunk_mixed"][-1].x), 0)
        self.assertTrue(all(sample.weights is None for sample in all_cases["implicit_unit"]))
        for rank in range(ranks):
            self.assertNotEqual(all_cases["dense_signed"][rank].ids, all_cases["changed_routes"][rank].ids)
        cleared_rows = 0
        for destination in range(ranks):
            previous = None
            for name in CASES:
                samples = all_cases[name]
                ref = _dispatch_reference(samples, destination, capacity, topk, experts, experts)
                self.assertEqual(len(ref.ids), ranks * capacity * topk)
                self.assertEqual(sum(ref.counts), len(ref.payloads))
                for row in range(len(ref.ids)):
                    source, rem = divmod(row, capacity * topk)
                    token, k = divmod(rem, topk)
                    sample = samples[source]
                    self.assertLess(len(sample.x), capacity)
                    expert = sample.ids[token][k] if token < len(sample.x) else -1
                    local = 0 <= expert < experts and expert // LOCAL_EXPERTS == destination
                    self.assertEqual(ref.ids[row], expert if local else experts)
                    if not local:
                        self.assertEqual(ref.weights[row], 0.0)
                        self.assertNotIn(row, ref.payloads)
                        if name == "shrunk_mixed" and previous.ids[row] != experts:
                            cleared_rows += 1
                if name in ("all_invalid", "empty_all"):
                    self.assertEqual(ref.counts, [0] * ranks)
                previous = ref
        self.assertGreater(cleared_rows, 0, "shrinking must exercise formerly live rows")


@unittest.skipUnless(
    int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1,
    "real GPU test requires mpirun with at least two ranks; the standalone script accepts tuning flags",
)
class MultirankGpuTests(unittest.TestCase):
    def test_rank_major_topk_expanded(self):
        run(parse_args([]))


def main():
    args = parse_args()
    if args.cpu_only:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(CpuReferenceTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            return 1
        print("[EXPANDED_TEST] pass CPU oracle only; GPU/MPI/graph tests NOT RUN", flush=True)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
