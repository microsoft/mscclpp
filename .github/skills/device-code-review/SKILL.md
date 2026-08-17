---
name: device-code-review
description: Review MSCCL++ CUDA/HIP device-side changes for correctness, memory safety, races, synchronization, memory ordering, communication semantics, data types, portability, and concrete GPU performance issues. Use when reviewing pull requests or code that changes kernels, device functions, low-level GPU communication paths, or their host launch context.
---

# Device Code Review

Perform a read-only review of MSCCL++ device-side code. Review CUDA and HIP kernels, device functions, low-level communication paths, and the host launch code needed to establish their execution context. Do not modify code as part of the review.

## Review Priorities

1. **Correctness and memory safety**
   - Check indexing, bounds, pointer arithmetic, alignment, aliasing, lifetime, and data layout assumptions.
   - Trace ownership of every shared or remotely visible buffer, flag, counter, and queue entry.
   - Identify races, deadlocks, hangs, early publication, duplicate publication, and stale-data hazards.

2. **Thread and synchronization semantics**
   - Map work across lanes, warps, warp groups, blocks, and cooperative grids.
   - Verify that every barrier is reached convergently by the required participants.
   - Check that warp-, block-, device-, and system-scoped operations use the appropriate scope.
   - Verify release/acquire ordering, fences, atomics, asynchronous copies, and completion waits.

3. **Communication correctness**
   - Confirm payload writes complete before readiness signals are published.
   - Check local, IPC/peer-mapped, and PortChannel paths independently.
   - Verify single-writer contracts, remote offsets, source/destination rank mapping, and epoch or buffer reuse.
   - Inspect the relevant MSCCL++ primitive semantics before recommending lower-level replacements.

4. **Performance**
   - Look for load imbalance, poor coalescing, unnecessary serialization, excessive atomics, divergence, redundant synchronization, and avoidable global-memory traffic.
   - Check launch geometry, occupancy constraints, register/shared-memory pressure, and warp utilization.
   - Review asynchronous pipelines for overlap, dependency, and buffer-reuse hazards.
   - Report performance concerns only when the mechanism and likely impact are concrete.

5. **Structure and boundary design**
   - Flag redundant structures that duplicate fields, mirror another type, or only forward state without defining a distinct abstraction.
   - Require each structure to have one coherent responsibility, explicit ownership or borrowing semantics, and invariants that it can maintain within its own boundary.
   - Check that host configuration, device views, transport state, synchronization state, and workspace metadata have clear, non-overlapping roles.
   - Identify designs that require callers to coordinate hidden invariants across multiple structures.
   - Prefer reusing, merging, or removing structures when they do not have a distinct semantic, ownership, or lifetime boundary.

6. **Data types and conversions**
   - Prefer the portable scalar types, vector types, `DataType`, `mscclpp::to`, `mscclpp::bit_cast`, and other helpers from `include/mscclpp/gpu_data_types.hpp` over raw CUDA/HIP types or locally duplicated conversion code.
   - Search `gpu_data_types.hpp` before recommending a new packed type, alias, bit representation, clipping operation, or numeric conversion.
   - If a required reusable type or conversion helper is missing, recommend adding a documented CUDA/HIP-compatible definition or function to `include/mscclpp/gpu_data_types.hpp` instead of an ad hoc helper in a kernel file.
   - Keep unavoidable backend-native types and intrinsics isolated at hardware or ABI boundaries, and convert to MSCCL++ types at the boundary.

7. **Portability and maintainability**
   - Check CUDA/HIP portability and architecture guards when the code is not intentionally backend-specific.
   - Prefer existing MSCCL++ helpers and established patterns when they preserve the required low-level semantics.
   - Flag confusing naming or duplication only when it increases correctness or maintenance risk.

## Review Method

1. Establish the kernel's purpose, launch configuration, input/output layout, and synchronization contract.
2. Inspect call sites, configuration code, related helpers, and producer/consumer kernels before drawing conclusions.
3. Trace representative execution paths, including empty work, partial warps, uneven expert/token distributions, local peers, remote peers, and buffer rollover.
4. For each suspected issue, identify the exact failing interleaving, input, architecture, or configuration.
5. Check whether existing barriers, atomics, or API guarantees already prevent the issue.
6. Report only findings that are actionable and supported by the code.

## Avoid Incorrect Review Heuristics

- Do not reject raw kernels or device intrinsics merely because a higher-level API exists.
- Do not flag a single-thread control loop when its cost is negligible or parallelization would add synchronization.
- Do not recommend removing a barrier without proving that all dependent reads and writes remain ordered.
- Do not assume CUDA behavior applies to HIP, or vice versa.
- Do not enforce naming or style preferences that are not established by the repository.
- Do not merge structures solely to reduce the type count; preserve types that enforce a meaningful invariant or boundary.
- Do not replace a raw backend type when it is required by an intrinsic, assembly constraint, hardware format, or external ABI; isolate and document that boundary instead.
- Do not produce speculative performance claims without explaining the bottleneck.

## Output Format

List findings in descending severity. For each finding provide:

- `severity - file:line - concise title`
- The concrete correctness or performance impact.
- The execution scenario that triggers it.
- A specific fix or safer design direction.

If there are no high-confidence findings, say so directly and note only material residual risks or validation gaps. Do not create empty category sections, repeat the code, or include style-only noise.

Ask for clarification only when correctness depends on an undocumented external contract, target architecture, transport capability, or performance requirement that cannot be established from the repository.
