# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Conservative tensor memory estimates for the fixed benchmark workload."""


def estimate_tensor_bytes(config, counts):
    """Conservative tensor budget, excluding native runtime and library allocations."""
    model, inputs, environment = config["model"], config["input"], config["environment"]
    hidden, intermediate = model["d_model"], model["intermediate_size"]
    local_experts = model["num_experts"] // environment["num_gpus"]
    weights = 6 * local_experts * hidden * intermediate
    capacity = environment["num_gpus"] * inputs["tokens_per_rank"]
    # Native runtime also owns a dispatch view; reserve one extra padded tensor.
    dispatch_and_output = 3 * 2 * local_experts * capacity * hidden
    mode, layout = environment.get("mode", "latency"), environment.get("layout", "expert_major")
    if mode == "throughput" or layout != "expert_major":
        rows = capacity
        if environment["communication_backend"] == "nccl" and layout == "expert_major":
            rows *= min(model["top_k"], local_experts)
        # Native receive/combine, FP32 reduction, gather/weighted temporaries,
        # IDs/weights, per-expert sort indices and graph-private allocations.
        adapter_bytes = 24 * rows * hidden + 48 * rows * model["top_k"] * local_experts
        if environment["cuda_graph"] == "mlp":
            adapter_bytes *= 2
        dispatch_and_output += adapter_bytes
        if environment["communication_backend"] == "mscclpp" and mode == "throughput":
            # RecvPoolConfig: fixed 65536 rows, 16384 payload + 768 metadata bytes.
            dispatch_and_output += 65536 * (16384 + 768) + 4 * 65536 * environment["num_gpus"] + 1024**2
    workspace = 4 * max(counts, default=0) * intermediate
    inputs_and_output = 4 * inputs["tokens_per_rank"] * hidden
    if environment["compute_backend"] == "cutlass_flashinfer":
        # Both dimensions are multiples of 128, so weight scales need no padding.
        elements = 3 * local_experts * hidden * intermediate
        weights = elements // 2 + elements // 32
        # Only one expert's original BF16 matrices and packing/quantization
        # temporaries are live at once, never an expanded BF16 expert bank.
        staging = 14 * hidden * intermediate
        rows = sum(counts)
        padded = sum((count + 127) // 128 * 128 for count in counts)
        workspace = 4 * rows * hidden  # BF16 packed input and output
        workspace += 2 * (rows * hidden + rows * hidden // 32)  # quant outputs + stable copies
        workspace += padded * intermediate + padded * intermediate // 32
        workspace += 32 * (padded + rows + local_experts)  # schedules, indices, unit weights
        if environment["cuda_graph"] == "mlp":
            workspace *= 2  # graph-private quant/elementwise allocations and warmup pool
        # Reserve compiler/kernel and allocator overhead; native comm remains extra.
        workspace += 64 * 1024**2
        return weights + staging + dispatch_and_output + workspace + inputs_and_output
    return weights + dispatch_and_output + workspace + inputs_and_output
