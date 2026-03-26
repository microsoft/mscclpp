# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Correctness test for FP8 allreduce with different accumulation types.
#
# Verifies that FP8 allreduce with float32 accumulation produces more accurate
# results than native FP8 accumulation, by comparing against a float32 reference.
#
# Usage:
#   MSCCLPP_MASTER_ADDR=<ip> MSCCLPP_MASTER_PORT=<port> \
#     torchrun --nnodes=1 --nproc_per_node=8 fp8_accum_correctness.py

import os
import torch
import mscclpp
import mscclpp.ext
import mscclpp.utils as mscclpp_utils
import netifaces as ni
import ipaddress


def interfaces_for_ip_netifaces(ip: str):
    target = ipaddress.ip_address(ip)
    for interface in ni.interfaces():
        addresses = ni.ifaddresses(interface)
        if ni.AF_INET in addresses:
            for link in addresses[ni.AF_INET]:
                if "addr" in link:
                    addr = ipaddress.ip_address(link["addr"])
                    if addr == target:
                        return interface
    return None


def init_dist() -> mscclpp.CommGroup:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MSCCLPP_MASTER_ADDR"]
    master_port = os.environ["MSCCLPP_MASTER_PORT"]
    interface = interfaces_for_ip_netifaces(master_addr)
    if interface is None:
        raise ValueError(f"Cannot find network interface for IP address {master_addr}")
    interfaceIpPortTrio = f"{interface}:{master_addr}:{master_port}"
    return mscclpp.CommGroup(interfaceIpPortTrio=interfaceIpPortTrio, rank=rank, size=world)


def run_allreduce(algo, comm, tensor, dtype_override=None, accum_dtype=None, nblocks=0, nthreads_per_block=0):
    """Run a single allreduce on the tensor (in-place) and return a clone of the result."""
    dtype = dtype_override if dtype_override is not None else mscclpp_utils.torch_dtype_to_mscclpp_dtype(tensor.dtype)
    ret = algo.execute(
        comm=comm.communicator,
        input_buffer=tensor.data_ptr(),
        output_buffer=tensor.data_ptr(),
        input_size=tensor.nbytes,
        output_size=tensor.nbytes,
        dtype=dtype,
        op=mscclpp.ReduceOp.SUM,
        stream=torch.cuda.current_stream().cuda_stream,
        nblocks=nblocks,
        nthreads_per_block=nthreads_per_block,
        symmetric_memory=True,
        accum_dtype=accum_dtype,
    )
    torch.cuda.synchronize()
    if ret != 0:
        raise RuntimeError(f"Allreduce failed with error code {ret}")
    return tensor.clone()


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    comm = init_dist()

    # Build algorithms
    scratch_dlpack = mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(torch.float16))
    scratch = torch.utils.dlpack.from_dlpack(scratch_dlpack)
    builder = mscclpp.ext.AlgorithmCollectionBuilder()
    algorithms = builder.build_default_algorithms(
        scratch_buffer=scratch.data_ptr(), scratch_buffer_size=scratch.nbytes, rank=rank
    )

    # Use packet algorithm - works for FP8 at all sizes
    algo_packet = [a for a in algorithms if a.name == "default_allreduce_packet"][0]

    fp8_dtype = torch.float8_e4m3fn
    test_sizes = [1024, 4096, 16384, 65536, 262144, 1048576]

    if rank == 0:
        print(f"FP8 Accumulation Correctness Test ({world_size} GPUs, packet algorithm)")
        print(f"{'='*90}")
        print(f"{'Size':>10} {'AccumType':>12} " f"{'MaxAbsErr':>12} {'MeanAbsErr':>12} {'MaxRelErr':>12}")
        print(f"{'-'*90}")

    # Allocate symmetric FP8 buffer
    max_bytes = max(test_sizes)
    buf_dlpack = mscclpp.RawGpuBuffer(max_bytes).to_dlpack(data_type=str(fp8_dtype))
    buf = torch.utils.dlpack.from_dlpack(buf_dlpack)

    accum_configs = [
        ("fp8 (native)", mscclpp.DataType.float8_e4m3),
        ("float16", mscclpp.DataType.float16),
    ]

    for size in test_sizes:
        n_elements = size  # 1 byte per FP8 element
        tensor_view = buf[:n_elements]

        results = {}
        for accum_label, accum_dtype in accum_configs:
            # Generate deterministic per-rank data
            torch.manual_seed(42 + rank)
            src_f32 = torch.randn(n_elements, dtype=torch.float32, device="cuda")
            src_f32 = src_f32.clamp(-240.0, 240.0)
            src_fp8 = src_f32.to(fp8_dtype)

            # Copy into symmetric buffer
            tensor_view.copy_(src_fp8)
            torch.cuda.synchronize()

            # Run allreduce
            result = run_allreduce(algo_packet, comm, tensor_view, accum_dtype=accum_dtype)
            result_f32 = result.float()

            # Compute float32 reference: sum all ranks' quantized FP8 inputs in float32
            ref_f32 = torch.zeros(n_elements, dtype=torch.float32, device="cuda")
            for r in range(world_size):
                torch.manual_seed(42 + r)
                rank_data = torch.randn(n_elements, dtype=torch.float32, device="cuda")
                rank_data = rank_data.clamp(-240.0, 240.0)
                rank_data_fp8 = rank_data.to(fp8_dtype)
                ref_f32 += rank_data_fp8.float()

            # Compute errors
            abs_err = (result_f32 - ref_f32).abs()
            max_abs_err = abs_err.max().item()
            mean_abs_err = abs_err.mean().item()
            denom = ref_f32.abs().clamp(min=1e-6)
            max_rel_err = (abs_err / denom).max().item()

            results[accum_label] = (max_abs_err, mean_abs_err, max_rel_err)

            if rank == 0:
                print(
                    f"{size:>10} {accum_label:>12} " f"{max_abs_err:>12.4f} {mean_abs_err:>12.6f} {max_rel_err:>12.6f}"
                )

            # Reset between runs
            algo_packet.reset()

        # Compare and print verdict
        if rank == 0:
            fp8_err = results["fp8 (native)"][1]  # mean abs error
            f32_err = results["float16"][1]
            if f32_err < fp8_err:
                improvement = (fp8_err - f32_err) / fp8_err * 100
                print(f"{'':>10} >> float32 accum is {improvement:.1f}% more accurate (mean abs err)")
            elif f32_err == fp8_err:
                print(f"{'':>10} >> identical results (sum of 8 ranks fits in FP8 precision)")
            else:
                print(f"{'':>10} >> WARNING: float32 accum is worse (unexpected)")
            print()

    if rank == 0:
        print(f"{'='*90}")
        print(
            "Note: With only 8 GPUs, the sum of 8 FP8 values may still fit in FP8 range,\n"
            "so differences can be small. The benefit of float32 accumulation grows with\n"
            "more GPUs or when intermediate sums exceed FP8 precision."
        )

    # ---- fp8_e4m3b15 test section ----
    # fp8_e4m3b15 has no native torch dtype. We store data as uint8 and pass
    # DataType.float8_e4m3b15 explicitly. The format has bias=15 so max finite
    # value is only 0.9375 — we generate valid bit patterns directly.

    def e4m3b15_to_float(bits_tensor):
        """Decode uint8 tensor of fp8_e4m3b15 bit patterns to float32 on GPU."""
        bits = bits_tensor.to(torch.int32)
        sign = (bits >> 7) & 1
        exp = (bits >> 3) & 0xF
        mant = bits & 0x7

        # Normal: value = (-1)^s * 2^(exp-15) * (1 + mant/8)
        normal_val = (2.0 ** (exp.float() - 15.0)) * (1.0 + mant.float() / 8.0)
        # Subnormal (exp==0): value = (-1)^s * 2^(-14) * (mant/8)
        subnormal_val = (2.0 ** (-14.0)) * (mant.float() / 8.0)

        result = torch.where(exp == 0, subnormal_val, normal_val)
        result = torch.where(sign == 1, -result, result)
        # Zero
        result = torch.where((exp == 0) & (mant == 0), torch.zeros_like(result), result)
        # NaN: exp==15 or negative zero (0x80)
        nan_mask = (exp == 15) | (bits == 0x80)
        result = torch.where(nan_mask, torch.full_like(result, float("nan")), result)
        return result

    def float_to_e4m3b15(f32_tensor):
        """Encode float32 tensor to uint8 tensor of fp8_e4m3b15 bit patterns on GPU.
        Simple element-wise: clamp to [-0.9375, 0.9375], quantize to nearest."""
        val = f32_tensor.clamp(-0.9375, 0.9375)
        sign = (val < 0).to(torch.int32)
        absval = val.abs()

        # Find best (exp, mant) for each value.
        # For normal: value = 2^(e-15) * (1 + m/8), e in [1..14], m in [0..7]
        # For subnormal: value = 2^(-14) * (m/8), m in [1..7]
        result_bits = torch.zeros_like(f32_tensor, dtype=torch.uint8)

        for e in range(15):
            for m in range(8):
                if e == 0 and m == 0:
                    continue  # skip zero
                if e == 0:
                    fval = (2.0 ** (-14.0)) * (m / 8.0)
                else:
                    fval = (2.0 ** (e - 15.0)) * (1.0 + m / 8.0)
                # Closest representable
                bits_val = (e << 3) | m
                # This is brute force but works for correctness testing
                candidate = torch.full_like(absval, fval)
                curr_best = e4m3b15_to_float(result_bits & 0x7F).abs()
                closer = (absval - candidate).abs() < (absval - curr_best).abs()
                result_bits = torch.where(closer, torch.full_like(result_bits, bits_val), result_bits)

        # Add sign
        result_bits = result_bits | (sign.to(torch.uint8) << 7)
        # Handle exact zero
        result_bits = torch.where(absval == 0, torch.zeros_like(result_bits), result_bits)
        return result_bits

    if rank == 0:
        print(f"\n{'='*90}")
        print(f"FP8 E4M3B15 Accumulation Correctness Test ({world_size} GPUs)")
        print(f"{'='*90}")

    # Allocate symmetric uint8 buffer for fp8_e4m3b15 data.
    buf_b15_dlpack = mscclpp.RawGpuBuffer(max_bytes).to_dlpack(data_type=str(torch.uint8))
    buf_b15 = torch.utils.dlpack.from_dlpack(buf_b15_dlpack)

    e4m3b15_accum_configs = [
        ("e4m3b15 (native)", mscclpp.DataType.float8_e4m3b15),
        ("float16", mscclpp.DataType.float16),
        ("float32", mscclpp.DataType.float32),
    ]

    e4m3b15_test_sizes = [1024, 4096, 65536]

    # Gather algorithms to test: packet, nvls_packet, and rsag_zero_copy.
    algo_names_to_test = [
        "default_allreduce_packet",
        "default_allreduce_nvls_packet",
        "default_allreduce_rsag_zero_copy",
    ]
    algo_map = {a.name: a for a in algorithms}

    # rsag_zero_copy needs larger sizes for proper work division
    rsag_min_size = world_size * 16 * 16  # at least 16 int4s per rank

    for algo_name in algo_names_to_test:
        if algo_name not in algo_map:
            if rank == 0:
                print(f"\n  Skipping {algo_name} (not available)")
            continue
        algo = algo_map[algo_name]

        if rank == 0:
            print(f"\n  Algorithm: {algo_name}")
            print(f"  {'Size':>10} {'AccumType':>16} " f"{'MaxAbsErr':>12} {'MeanAbsErr':>12} {'MaxRelErr':>12}")
            print(f"  {'-'*80}")

        for size in e4m3b15_test_sizes:
            # rsag_zero_copy needs larger buffers for proper work division
            if "rsag" in algo_name and size < rsag_min_size:
                continue
            n_elements = size
            tensor_view = buf_b15[:n_elements]

            results = {}
            for accum_label, accum_dtype in e4m3b15_accum_configs:
                # Generate deterministic per-rank random uint8 values in the valid
                # e4m3b15 range. Avoid exp==15 (NaN) and 0x80 (negative zero = NaN).
                torch.manual_seed(42 + rank)
                # Generate random bytes, mask off exp==15 rows
                raw = torch.randint(0, 0x78, (n_elements,), dtype=torch.uint8, device="cuda")
                # Randomly add sign bit
                signs = torch.randint(0, 2, (n_elements,), dtype=torch.uint8, device="cuda") << 7
                src_uint8 = raw | signs
                # Fix negative zero → positive zero
                src_uint8 = torch.where(src_uint8 == 0x80, torch.zeros_like(src_uint8), src_uint8)

                # Copy into symmetric buffer
                tensor_view.copy_(src_uint8)
                torch.cuda.synchronize()

                # Run allreduce with explicit dtype override
                # rsag_zero_copy doesn't auto-select block/thread counts (unlike packet/nvls_packet),
                # so we must provide explicit values to avoid launching 0-block kernels that hang.
                nb = 32 if "rsag" in algo_name else 0
                nt = 1024 if "rsag" in algo_name else 0
                result = run_allreduce(
                    algo,
                    comm,
                    tensor_view,
                    dtype_override=mscclpp.DataType.float8_e4m3b15,
                    accum_dtype=accum_dtype,
                    nblocks=nb,
                    nthreads_per_block=nt,
                )

                # Decode result (uint8 → float32 via e4m3b15 interpretation)
                result_f32 = e4m3b15_to_float(result)

                # Compute float32 reference: decode each rank's e4m3b15 bits → float32, sum
                ref_f32 = torch.zeros(n_elements, dtype=torch.float32, device="cuda")
                for r in range(world_size):
                    torch.manual_seed(42 + r)
                    raw_r = torch.randint(0, 0x78, (n_elements,), dtype=torch.uint8, device="cuda")
                    signs_r = torch.randint(0, 2, (n_elements,), dtype=torch.uint8, device="cuda") << 7
                    bits_r = raw_r | signs_r
                    bits_r = torch.where(bits_r == 0x80, torch.zeros_like(bits_r), bits_r)
                    ref_f32 += e4m3b15_to_float(bits_r)

                # Clamp reference to e4m3b15 representable range before comparing.
                ref_f32 = ref_f32.clamp(-0.9375, 0.9375)

                # Compute errors
                valid = ~result_f32.isnan() & ~ref_f32.isnan()
                abs_err = (result_f32[valid] - ref_f32[valid]).abs()
                max_abs_err = abs_err.max().item() if abs_err.numel() > 0 else 0.0
                mean_abs_err = abs_err.mean().item() if abs_err.numel() > 0 else 0.0
                denom = ref_f32[valid].abs().clamp(min=1e-8)
                max_rel_err = (abs_err / denom).max().item() if abs_err.numel() > 0 else 0.0

                results[accum_label] = (max_abs_err, mean_abs_err, max_rel_err)

                if rank == 0:
                    print(
                        f"  {size:>10} {accum_label:>16} "
                        f"{max_abs_err:>12.6f} {mean_abs_err:>12.8f} {max_rel_err:>12.6f}"
                    )

                algo.reset()

            if rank == 0:
                if len(results) >= 3:
                    native_err = results["e4m3b15 (native)"][1]
                    f16_err = results["float16"][1]
                    f32_err = results["float32"][1]
                    if f16_err < native_err:
                        improvement = (native_err - f16_err) / native_err * 100
                        print(f"  {'':>10} >> float16 accum is {improvement:.1f}% more accurate than native")
                    elif f16_err == native_err:
                        print(f"  {'':>10} >> float16 vs native: identical results")
                    else:
                        print(f"  {'':>10} >> WARNING: float16 accum is worse than native (unexpected)")
                    if f32_err < native_err:
                        improvement = (native_err - f32_err) / native_err * 100
                        print(f"  {'':>10} >> float32 accum is {improvement:.1f}% more accurate than native")
                    elif f32_err == native_err:
                        print(f"  {'':>10} >> float32 vs native: identical results")
                    else:
                        print(f"  {'':>10} >> WARNING: float32 accum is worse than native (unexpected)")
                    if f16_err == f32_err:
                        print(f"  {'':>10} >> float16 and float32 accum produce identical results")
                    elif f16_err < f32_err:
                        print(f"  {'':>10} >> float16 accum is slightly better than float32 (rounding)")
                    else:
                        diff_pct = (f16_err - f32_err) / f32_err * 100
                        print(f"  {'':>10} >> float32 accum is {diff_pct:.2f}% better than float16")
                print()

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
