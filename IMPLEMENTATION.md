# ETP (expert-tensor-parallelism) in MSCCL++ `feature/ep`

Branch: `feature/etp`, based on `feature/ep` @ `2df3af01207b49fa91a9f84b09e1fa61e1c4727a`.
Nothing was pushed; there is no `origin` remote in this clone.

Target topology: `N=16, EP=4, ETP=4` — each expert is owned by an EP group of 4 ranks that
tensor-shard its weights along the FFN intermediate dimension. `etpSize=1` is the pre-existing
expert-parallel behavior and must stay unchanged.

Implements the agreed design in `../mscclpp-etp-investigation/ETP-design-report.md`:
single-send to the same-`tpIndex` leader (option B), fused peer pull inside `dispatchRecvWorker`
(B2), TP-group reduce-scatter on the combine side, duplicate-send (option A) behind a flag.

---

## 1. Commits (one per stage)

| Commit | Stage | Summary |
|---|---|---|
| `821e5fdb` | 0/1 | `MoETopology`, `EtpRankOrder`, `EtpReduceMode`, `EtpDispatchMode`; plumbed through `MoERuntime`, contexts, `DeviceContext`, nanobind, Python; buffer sizing keyed on the EP group; host unit tests for the index math. |
| `c6b779ec` | 2 | `ExpertMap` — one shared host/device helper replacing every open-coded `nExperts / nRanks` and `expert / nLocalExperts`; asserts relaxed to `% epSize`; `SWITCH_RANKS` and the throughput combine switch keyed on the a2a (EP) degree. |
| `dd4ac2a9` | 3 | Dispatch under ETP: leader single-send, `RecvTask::tpPeer_`, scheduler reads peer metadata, fused peer pull in `dispatchRecvWorker`, rank-private combine row offsets. |
| `b539c806` | 4 | Combine under ETP: `GROUP_REDUCE_SCATTER` (default, 3-phase) and `SOURCE_SIDE` fallback; `GROUP_NVLS` rejected stub. |
| `cfee2e95` | 5 | CPU dataflow model, Python multirank test ETP cases, benchmark `--etp*` flags. |
| `df7283a7` | — | Restore the send-side TMA/atomic overlap for the default path; add nvcc-free check tooling. |
| `b57a1f39` | — | First IMPLEMENTATION.md. |
| `5b0b1e37` | post-HW | Unit test uses the repo test framework instead of the unvendored gtest (fixes `-DMSCCLPP_BUILD_TESTS=ON`). |
| `35a989ff` | post-HW | FP8+ETP fix: the test's FP8 reference double-counted EP-group ranks; CPU model extended to FP8-style scales (§4.3). |
| this commit | post-HW | Document the measured A-vs-B2 inversion at the API level and refresh the verification status. |

---

## 2. What changed, per file

### Public headers
* `include/mscclpp/ext/ep/types.hpp` — new `MoETopology{numRanks, epSize, etpSize, order}` with
  `epIndex/tpIndex/rankOf` plus `numExpertsPerGroup/expertGroup/localExpert/leaderRank/ownsExpert`;
  new enums `EtpRankOrder{EP_MAJOR, TP_MAJOR}`, `EtpReduceMode{SOURCE_SIDE, GROUP_REDUCE_SCATTER,
  GROUP_NVLS}`, `EtpDispatchMode{LEADER_SINGLE_SEND, DUPLICATE_SEND}`. All host+device inline.
* `include/mscclpp/ext/ep/moe_runtime.hpp` — `MoERuntime(...)` and `createMoERuntime(...)` take
  `etpSize = 1, etpRankOrder = EP_MAJOR, etpReduceMode = GROUP_REDUCE_SCATTER,
  etpDispatchMode = LEADER_SINGLE_SEND`; accessors `epSize()/etpSize()/epIndex()/tpIndex()/topology()`.

### Runtime / host
* `src/ext/ep/moe_runtime.cc` — builds the topology, asserts `numRanks % etpSize == 0`, requires an
  IPC-complete world and rejects `GROUP_NVLS`; rejects `etpSize > 1` in THROUGHPUT mode.
* `src/ext/ep/include/moe_runtime_context.hpp`, `latency.cc`, `throughput.cc` — contexts carry the
  topology/reduce/dispatch modes; latency asserts `numExperts % epSize == 0` and (for ETP)
  EXPERT_MAJOR + RANK_LOCAL_REDUCE and a fixed `maxTokensPerRank`; the ETP staging buffer pointer is
  published in `DeviceContext`; throughput availability and template selection use `epSize`.
* `src/ext/ep/include/config.hpp` — `LatencyStorageLayout` takes the topology/modes;
  `numLocalExperts = numExperts / epSize`; metadata slots =
  `(duplicateSend ? numRanks : epSize) * numLocalExperts`; new `etpReduceBuffer_` region sized
  `numRanks * maxTokensPerRank * hidden` BF16 (0 unless `etpSize > 1 && GROUP_REDUCE_SCATTER`).
* `src/ext/ep/include/device_context.hpp` — `topology_`, `etpReduceMode_`, `etpDispatchMode_`,
  `etpReduceBuffer_`.
* `src/ext/ep/include/expert_map.hpp` (new) — `ExpertMap`: `group/localExpert/globalExpertBase/
  ownsExpert/rankOwnsExpert/leaderRank/numSendCopies/destinationRank/sourcePeer/groupLeaderFor/
  leaderInGroupOf/metadataExpertSlot(s)`.
* `src/ext/ep/common/latency.cuh` — `dispatchMetadataBytes(map)`; `RecvTask` gains `tpPeer_`;
  `WorkspaceView` gains `dispatchCombineRowOffsets_` and is parameterized by the map + capacity;
  `workspaceBytes(..., epSize)`.

### Dispatch kernels (`src/ext/ep/dispatch/common.cuh`)
* Send path: payload metadata staging split from slot reservation; a `numSendCopies()` loop makes
  LEADER_SINGLE_SEND (1 copy) and DUPLICATE_SEND (`etpSize` copies) share one path. Copy 0 keeps the
  original overlap with the payload TMA load.
* `countDispatchRoutes` counts per destination rank across copies; `writeDispatchMetadata` writes rank
  counts only into buffers this rank actually sends to and keys the per-expert slot on
  `metadataExpertSlot(src, localExpert)`; `publishDispatchPayloads` drains completions for the ranks it
  wrote to, then (after `__syncthreads()` and, under ETP, `__threadfence_system()`) releases *every*
  rank of each destination EP group, because the non-leader peers pull the rows.
* `dispatchRecvScheduler` reads every source's rank/expert LL8 counts from
  `mappedBuffer(recvBuffer, sourcePeer(src))` and stores `tpPeer_` in each `RecvTask`.
* `dispatchRecvWorker` TMA-loads payloads from `mappedBuffer(recvBuffer, task.tpPeer_)` — the fused
  pull: no extra buffer, kernel, or copy. `globalExpertBase = epIndex * numExpertsPerGroup`.
* `dispatchRecvExpertMajorOutput` writes the combine input row offset into the rank-private
  `dispatchCombineRowOffsets_` instead of overwriting the (now shared) payload's `topKIndices`.

### Combine kernels (`src/ext/ep/combine/common.cuh`)
* `sendRankReducedPartials<..., EtpStage>` reads the payload from the ETP peer and the row offsets
  from the workspace; with `EtpStage` it stores the weighted partial into
  `etpReduceBuffer[(myTp * epSize + epIndex(src)) * maxTokens + srcToken]` of `groupLeaderFor(src)`.
* New `reduceEtpGroupPartials` (phase B): each leader sums the `etpSize` staged rows for the sources
  it owns and pushes one row per token to the source — the existing combine a2a over the EP sub-world.
* `combineBody` runs A → ready-exchange → B → ready-exchange → source reduction when
  `etpSize > 1 && GROUP_REDUCE_SCATTER`; otherwise the original single-phase send.
* `reduceRankPartialsBf16x8` takes a contributor count so `SOURCE_SIDE` sums the `etpSize` ranks of
  each routed group (`topology.rankOf(epIndex(partialRank), c)`).
* Rank-major/direct-send paths use `ExpertMap` but are still EP-only (rejected for `etpSize > 1`).

### Throughput (EP-only, de-hardcoded)
* `throughput_prepare.cu` / `throughput_dispatch.cu` / `throughput_reduce_combine.cu` /
  `throughput.cc` — expert→rank uses `topology.numExpertsPerGroup()`; the rank axis of
  `isTokenInRank`, the control buffer, the launch geometry and `SWITCH_RANKS`/the combine switch are
  keyed on `topology_.epSize`. With `etpSize == 1` (the only accepted value there) `epSize == numRanks`.

### Python
* `types.py` — `MoECommunicatorConfig.etp_size / ep_size / etp_rank_order / etp_reduce_mode /
  etp_dispatch_mode`.
* `utils.py` — new `resolve_etp_topology()`; `resolve_expert_placement()` keys on the EP group.
* `latency.py` — resolves the topology, validates ETP restrictions, passes the four new arguments to
  `create_moe_runtime`; buffer shapes follow `num_local_experts = num_experts // ep_size`.
* `throughput.py` — raises `NotImplementedError` for `etp_size > 1`.
* `communicator.py`, `__init__.py`, `_cpp.py` — `ep_size/etp_size/ep_index/tp_index` properties and
  the new enums re-exported.
* `src/ext/ep/bindings.cpp` — three new `nb::enum_`s, `etp_*` keyword arguments,
  `ep_size/etp_size/ep_index/tp_index` read-only properties.

### Tests / tools
* `test/unit/ep_topology_tests.cc` (+ CMakeLists) — gtest for the topology index math.
* `test/ep_cpu_model/etp_model_test.cc` + `run_etp_model_test.sh` — CPU dataflow model (see §4).
* `test/python/ep/test_latency_multirank.py` — `--etp-size`, `--etp-reduce-mode`,
  `--etp-dispatch-mode` (or `MSCCLPP_EP_ETP`); expert ownership keyed on the EP group; the simulated
  expert MLP is scaled by `1 / etp_size` to model an intermediate-dim shard; the RANK_LOCAL_REDUCE
  reference rounds per EP *group*.
* `test/python/ep/test_intranode_multirank.py` — asserts `ep_size`/`etp_size` and that THROUGHPUT
  rejects `etp_size > 1`.
* `test/python/ep/run_ep_bench_python.py`, `ep_bench_mscclpp.py` — `--etp`, `--etp-reduce-mode`,
  `--etp-dispatch-mode`; divisibility keyed on `ep_size`; throughput backend rejects `--etp > 1`.
* `tools/host_syntax_check.sh`, `tools/host_syntax_shim.hpp` — nvcc-free host syntax/type check.

---

## 3. Why `etpSize == 1` is unchanged

Proven on hardware: this branch at ETP=1 is **bit-exact against the base commit** (max diff 0.0) across
the 6 base configurations. The algebraic argument below is why.

Every ETP expression degenerates algebraically:

| Expression | `etpSize == 1` |
|---|---|
| `epSize = numRanks / etpSize` | `numRanks` |
| `numExpertsPerGroup = numExperts / epSize` | `numExperts / numRanks` (old `nLocalExperts`) |
| `leaderRank(e) = rankOf(group(e), myTp)` | `e / nLocalExperts` (old `dstRank`) |
| `numSendCopies()` | 1 (single send, one destination per token/rank) |
| `sourcePeer(s)`, `groupLeaderFor(s)`, `leaderInGroupOf(d)` | `rank_`, `rank_`, `d` |
| `metadataExpertSlot(s, le)` | `s * nLocalExperts + le`; slots = `numExperts` |
| `metadataOf(s)` / `mappedBuffer(buf, tpPeer_)` | the local buffer |
| `etpReduceRows(...)` | 0 → the staging buffer is not allocated |
| `numContributors` in the source reduction | 1 → the original single row |
| `SWITCH_RANKS(epSize)` | `SWITCH_RANKS(numRanks)` |
| `globalExpertBase = epIndex * nEPG` | `rank * nLocalExperts` |

Three deliberate, non-semantic differences remain at `etpSize == 1`:

1. **Workspace grows** by `numRanks * maxTokensPerRank * numTopk` ints (the new
   `dispatchCombineRowOffsets_`) and by 4 bytes per `RecvTask`. The symmetric buffer is byte-identical.
2. **The combine input row offset moved** from the dispatch payload's `topKIndices` into that
   workspace array (the payload can be shared with ETP peers, so it must not be mutated). Same values,
   same consumers; dispatch no longer clobbers the received payload's top-k ids.
3. **`publishDispatchPayloads` gained a `__syncthreads()`** between the completion drain and the
   signalling loop (the `__threadfence_system()` is ETP-only). Same set of signals, same order of
   effects.

Numerics for `etpSize == 1` are untouched: no accumulation order changed.

---

## 4. Verification status

### 4.1 Verified on hardware (16x GB200, 4 nodes, one NVLink domain, sm_100a)

Run by the GPU worker; evidence in `/home/changhohwang/one/mscclpp-etp-runs/`
(`raw/matrix.tsv`, `BASELINE.md`, `ETP-BUILD.md`, `logs/`).

| Item | Result |
|---|---|
| Compile for sm_100a | clean, zero warnings, zero register spills |
| Base `feature/ep`, EP=16, 6 configurations | 6/6 pass |
| This branch at ETP=1 vs base | **bit-exact, max diff 0.0** — the "`etpSize=1` unchanged" claim is proven |
| ETP=1 performance | dispatch 36.8 µs vs base 38.1, combine 39.4 vs 38.3 — within repeat spread; the +8…+13 register increase costs nothing measurable |
| EP=4 / ETP=4, BF16 | pass: default, `--etp-reduce-mode source_side`, `--etp-dispatch-mode duplicate_send` |
| ETP=2 and ETP=8, BF16 | pass |
| Declared gaps (THROUGHPUT ETP, RANK_MAJOR ETP, DIRECT_SEND ETP) | all rejected cleanly |
| **FP8 + ETP>1** | **failed** (`max diff=1422.0`) — root-caused as a test-reference bug and fixed, see §4.3; **needs re-validation** |

Measured cost of ETP=4 vs EP=16 at the same total token count: dispatch 2.2×,
combine 3.1×, with per-rank bandwidth up 1.7–1.8× (each rank handles `ETP×` the
token volume for its group).

### 4.2 Verified without a GPU (this host: no GPU, no `nvcc`)

* **Topology index math** — `test/unit/ep_topology_tests.cc`: EP×ETP is a bijection for both rank
  orders; the leader of an expert is in the expert's group and shares the sender's `tpIndex`; expert
  ownership is group-wide; `etpSize == 1` reproduces `expert / nLocalExperts`. (The GPU worker also
  ran this after the gtest include fix.)
* **CPU dataflow model** — `test/ep_cpu_model/etp_model_test.cc` replays the kernels' index algebra
  with the real `ExpertMap` / `MoETopology` / `WorkspaceView` / `LatencyStorageLayout` headers for 16
  ranks over **48** configurations: `{bf16-style, fp8-style}` × `etpSize ∈ {2,4,8}` ×
  `{EP_MAJOR, TP_MAJOR}` × `{GROUP_REDUCE_SCATTER, SOURCE_SIDE}` ×
  `{LEADER_SINGLE_SEND, DUPLICATE_SEND}`. Checks: ETP output == `etpSize=1` output == dense
  single-GPU reference (≤1.5e-6 fp32); every ETP rank of a group receives the same token multiset;
  each dispatched token is held by exactly `numGroups * etpSize` ranks; no slot / metadata slot /
  workspace row-offset / expert-major row escapes its buffer; and, for the FP8-style payload, every
  replicated scale belongs to its own row's source token and ETP peers dequantize a shared token
  identically. **Passes.** This model caught the `groupLeaderFor` bug before hardware.
* **Host syntax/type check** — `tools/host_syntax_check.sh`: `config.hpp`, `expert_map.hpp`,
  `common/latency.cuh`, `combine/common.cuh` clean. `dispatch/common.cuh` fails only on the
  device-only `mscclpp::to<>` in the untouched `common/quantization.cuh` — identical on the base
  commit. Device bodies guarded by `MSCCLPP_BULK_AVAILABLE` are compiled out in this mode.

### 4.3 The FP8 + ETP>1 failure: root cause and fix

Reported as `AssertionError: LL rank-local combine mismatch; max diff=1422.0` for every FP8 run with
`etpSize > 1`.

**It is a bug in the test's reference, not in the kernels.** `reconstruct_expert_major_reference()`
(used only on the FP8 path; BF16 compares against `x` directly) rebuilds each source token's
dispatched payload per rank and sums the per-rank reconstructions with `dist.all_reduce`. Under ETP
all `etp_size` ranks of an EP group own the same experts and hold the same rows, so each row was
counted `etp_size` times and the *reference* was `etp_size×` too large.

Evidence:

1. `diff = (etp_size - 1) * max|true|`. Measured 474.0 at ETP=2 and 1422.0 at ETP=4 — exactly 3.0×,
   which is `(4-1)/(2-1)` with the same `max|true| = 474.0` in both runs (same seed, same routing).
   No kernel defect produces that exact algebraic signature.
2. Identical failure value for `source_side` / `group_reduce_scatter` / `duplicate_send`, and BF16
   passing: all three follow from the defect living in the shared FP8 reference path.
3. Dispatch validation passes, and it compares the received FP8 block scales at `rtol=1e-6` per rank.

**The reported hypothesis (combine dequantizes group-peer rows with the wrong rank's scales) is
refuted**: combine never touches scales — its input is the caller's BF16 expert output — and
`dispatchRecvExpertMajorOutput` replicates the scale vector from the payload it is copying into that
same rank's own output row, so it cannot pick up a peer's scales. The extended CPU model now covers
exactly this path (FP8-style payload, replicated scales, dequantization before the expert compute)
across all 24 ETP configurations and passes.

Fix: only `tp_index == 0` of each EP group contributes to the all-reduced reference (no-op at
`etp_size == 1`), plus the FP8 coverage in the CPU model. Commit `35a989ff`.

### 4.4 Still unverified

* **FP8 + ETP after the fix** — needs a hardware re-run (`--dispatch-dtype fp8_e4m3 --etp-size 4`).
* **Shape coverage** — everything above was measured at a single shape: 128 tokens / hidden 7168 /
  top-k 8 / 256 experts. No token-count, hidden, top-k or expert-count sweep exists.
* **Backend comparison** — `run_ep_bench_python.py` never ran: the sglang container image lacks
  `mpi4py`. DeepEP / NCCL / FlashInfer comparisons are unmeasured, as are the `--etp*` benchmark
  flags themselves.
* **Internode ETP** — blocked by the missing IB path for EP; these nodes have no IB path for EP either,
  so the B2 volume argument is untested in the regime where it should matter.
* **`GROUP_NVLS`** — not implemented.
* **CUDA-graph capture under ETP** — only exercised at ETP=1.

---

## 5. Design finding: duplicate-send (option A) beats the fused peer pull (B2) intra-domain

The design report recommends option B2 (single send to the same-`tpIndex` leader plus a fused NVLink
peer pull) over option A (duplicate send). **The hardware says the opposite for a domain-local world.**

16× GB200, one NVLink domain, EP=4/ETP=4, 128 tokens / hidden 7168 / top-k 8 / 256 experts, medians of
3 repeats:

| Mode | dispatch µs | combine µs | total µs |
|---|---|---|---|
| `LEADER_SINGLE_SEND` (B2, default) | **82.0** | 117.7 | 199.7 |
| `DUPLICATE_SEND` (A) | 87.6 | **93.1** | **180.7** |

Option A loses 7 % on dispatch (it writes `etpSize×` the bytes) and wins 21 % on combine, for ~10 %
end to end. The report predicted the two would be roughly equal on intra-node byte count; it did not
price the receive-side cost of B2.

Likely mechanism, from the code rather than from a profile: under B2 every combine phase-A iteration
reads the token's top-k weights and `srcTokenGlobalIdx` **from the ETP peer's dispatch receive
buffer** (`sendRankReducedPartials`, per token, on the critical path), and the receive scheduler waits
on `etpSize×` more ready epochs. Under A all of that is local. This is a latency effect, not a
bandwidth effect, which is why it shows up in combine rather than dispatch.

**Recommendation (for Pione to decide, not applied):** keep `LEADER_SINGLE_SEND` as the default.
Reasons: it is the only mode whose all-to-all volume stays at 1× — decisive the moment any part of the
a2a leaves the NVLink domain, which is the stated direction of travel; the measurement is a single
shape on one topology; and the gap looks addressable rather than intrinsic. Concrete follow-up that
would likely erase most of the 21 %: cache the per-token `srcTokenGlobalIdx` and top-k weights into
the local workspace during `dispatchRecvWorker` (which already reads them) next to
`dispatchCombineRowOffsets_`, so combine phase A becomes fully local under B2 too. That is a contained
change, but it is unvalidatable here, so it is not in this branch.

Meanwhile the tradeoff is now explicit at the API level: the measured numbers and the selection criterion are
documented on `EtpDispatchMode` (`include/mscclpp/ext/ep/types.hpp`) and on
`MoECommunicatorConfig.etp_dispatch_mode` (`python/mscclpp/ep/types.py`), and both modes are selectable
from the tests and the benchmark.

---

## 6. Known gaps / TODO

1. **THROUGHPUT mode has no ETP.** `etpSize > 1` is rejected in `MoERuntime`, `throughput.py`, the
   benchmark and the intranode test (verified rejecting on hardware). Only the de-hardcoding landed there.
2. **LATENCY `RANK_MAJOR` and `DIRECT_SEND` have no ETP** (report stage 3): rejected by host asserts
   and by `latency.py` (verified rejecting on hardware).
3. **`EtpReduceMode::GROUP_NVLS` is a stub** — rejected at construction. Needs
   `connectNvlsCollective(comm, tpPeerRanks, bytes)` per TP group in `LatencyContext::initialize`, a
   `SwitchChannelDeviceHandle` in `DeviceContext`, and `multimem.ld_reduce` in place of the phase-B sum.
4. **FP8 + ETP is fixed but not re-validated on hardware** (§4.3). The kernel path is covered by the
   CPU model; the hardware run is pending.
5. **ETP requires a fixed `maxTokensPerRank`** between dispatch and combine (asserted), because the
   row-offset workspace array is laid out with the active capacity. Expert-major already forbids a
   per-call capacity override.
6. **ETP requires the whole world in one IPC domain** (asserted). Internode ETP is blocked by the same
   missing IB path as internode EP.
7. **`num_blocks` lower bound is still `world_size + 2`**, not `ep_size + 2`: the receive scheduler
   still enumerates all `numRanks` sources under design B2, so worker blocks ≥ `numRanks` is required.
   The report's suggested relaxation does not apply to B2.
8. **Dispatch metadata under `DUPLICATE_SEND` costs `etpSize×` slots** (`numRanks * nEPG`), reflected
   in the symmetric buffer size at construction time. Measured peak memory at ETP=4 duplicate_send:
   ~9.2 GiB per rank at this shape.
9. **B2 combine reads peer payload metadata per token** — the likely cause of the A-vs-B2 gap (§5) and
   the recommended follow-up optimization.
10. `MoECommunicatorConfig.num_local_experts` / `local_expert_start` are still advisory on the C++ side
    (report open question 7); they are now validated against the EP group.

---

## 7. Commands for the GPU worker (16 GPUs)

Build (any node with the 16-GPU harness):

```bash
cd mscclpp-etp-impl
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMSCCLPP_BYPASS_GPU_CHECK=OFF -DMSCCLPP_BUILD_EXT_EP=ON
cmake --build build -j
# unit tests include the host-side topology test (repo framework, not gtest):
cmake -B build-tests -DCMAKE_BUILD_TYPE=Release -DMSCCLPP_BUILD_TESTS=ON -DMSCCLPP_BUILD_EXT_EP=ON
cmake --build build-tests -j --target unit_tests
./build-tests/test/unit_tests
```

Nvcc-free pre-checks (also runnable anywhere):

```bash
CUDA_INCLUDE=/usr/local/cuda/include ./tools/host_syntax_check.sh          # dispatch/common.cuh fails pre-existing, see §4.2
CUDA_INCLUDE=/usr/local/cuda/include ./test/ep_cpu_model/run_etp_model_test.sh
```

### Regression: EP=16, ETP=1 (must match the pre-change baseline)

```bash
torchrun --nproc_per_node=16 test/python/ep/test_latency_multirank.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256
torchrun --nproc_per_node=16 test/python/ep/test_latency_multirank.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 --dispatch-dtype fp8_e4m3
torchrun --nproc_per_node=16 test/python/ep/test_latency_multirank.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 --output-layout rank_major
torchrun --nproc_per_node=16 test/python/ep/test_latency_multirank.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 --combine-mode direct_send
torchrun --nproc_per_node=16 test/python/ep/test_latency_multirank.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 --cuda-graph
torchrun --nproc_per_node=16 test/python/ep/test_intranode_multirank.py            # throughput
MSCCLPP_EP_OUTPUT_LAYOUT=rank_major torchrun --nproc_per_node=16 test/python/ep/test_intranode_multirank.py
MSCCLPP_EP_ETP=4 torchrun --nproc_per_node=16 test/python/ep/test_intranode_multirank.py  # must reject ETP
```

Compare the latency numbers against the `feature/ep` baseline; ETP=1 should be within noise.

### ETP: EP=4, ETP=4 (the target configuration)

The FP8 case below is the re-validation of the §4.3 fix and is the highest-priority run.

```bash
# default: leader single-send + TP-group reduce-scatter
torchrun --nproc_per_node=16 test/python/ep/test_latency_multirank.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 --etp-size 4
# fallbacks / option A, same expected output
torchrun --nproc_per_node=16 test/python/ep/test_latency_multirank.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 --etp-size 4 \
    --etp-reduce-mode source_side
torchrun --nproc_per_node=16 test/python/ep/test_latency_multirank.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 --etp-size 4 \
    --etp-dispatch-mode duplicate_send
# sweep the other legal degrees
for etp in 2 8; do
  torchrun --nproc_per_node=16 test/python/ep/test_latency_multirank.py \
      --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 --etp-size $etp
done
# FP8 under ETP (untested here)
torchrun --nproc_per_node=16 test/python/ep/test_latency_multirank.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256 --etp-size 4 \
    --dispatch-dtype fp8_e4m3
```

### Benchmarks

```bash
mpirun -n 16 python test/python/ep/run_ep_bench_python.py --backends mscclpp \
    -t 128 -d 7168 -k 8 -e 256 --validate                         # EP=16, ETP=1 baseline
mpirun -n 16 python test/python/ep/run_ep_bench_python.py --backends mscclpp \
    -t 128 -d 7168 -k 8 -e 256 --etp 4 --validate                 # EP=4, ETP=4
mpirun -n 16 python test/python/ep/run_ep_bench_python.py --backends mscclpp \
    -t 128 -d 7168 -k 8 -e 256 --etp 4 --etp-dispatch-mode duplicate_send --validate
```

The A-vs-B comparison of the report (§2.4) is the last two commands: identical results, different
NVLink traffic shape (single send + peer pull vs `etpSize` writes).

### What to report back (second round)

* **FP8 + ETP=2/4/8 after the fix** — the primary question;
* that `-DMSCCLPP_BUILD_TESTS=ON` builds and `unit_tests` runs the `EpTopology` cases;
* that ETP=1 is still bit-exact against the base;
* any shape sweep that can be afforded (token count, hidden, top-k, expert count) — everything so far
  is one shape, 128/7168/8/256;
* if `mpi4py` can be added to the image, `run_ep_bench_python.py` with and without `--etp` for the
  DeepEP/NCCL/FlashInfer comparison.
