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

## 4. Verification performed (and what was *not* possible)

**No GPU and no `nvcc` on this host** (`nvidia-smi` absent; the PyPI `nvidia-cuda-nvcc-cu12` wheels
ship only `ptxas`, no compiler driver). Therefore **the CUDA device code has not been compiled**.
That is the single largest gap in this handoff — treat a clean `nvcc` build as the first GPU-side step.

What *was* verified:

1. **Topology index math** — `test/unit/ep_topology_tests.cc` (gtest, host-only) plus an equivalent
   standalone run here: EP×ETP is a bijection for both rank orders, the leader of an expert is always
   in the expert's group and shares the sender's `tpIndex` (so the a2a is a permutation and each
   group member leads exactly `numRanks/etpSize` sources), expert ownership is group-wide, and
   `etpSize == 1` reproduces `expert / nLocalExperts`. **Passed.**
2. **CPU dataflow model** — `test/ep_cpu_model/etp_model_test.cc` replays the kernels' index algebra
   using the *real* `ExpertMap` / `MoETopology` / `WorkspaceView` / `LatencyStorageLayout` headers for
   16 ranks and 24 configurations (`etpSize ∈ {2,4,8}` × `{EP_MAJOR, TP_MAJOR}` ×
   `{GROUP_REDUCE_SCATTER, SOURCE_SIDE}` × `{LEADER_SINGLE_SEND, DUPLICATE_SEND}`), checking:
   ETP output == `etpSize=1` output == dense single-GPU reference (max abs diff ≤ 1.5e-6 in fp32),
   every ETP rank of a group receives the same token multiset, and no slot / metadata slot /
   workspace row-offset / expert-major row escapes its buffer. **Passed** (`all ETP model checks
   passed`). This model already caught one real bug: `DUPLICATE_SEND + GROUP_REDUCE_SCATTER` staged
   partials to `tpPeer_` (= self) instead of the group's reduce-scatter leader; fixed by
   `ExpertMap::groupLeaderFor()`.
3. **Host syntax/type check** — `tools/host_syntax_check.sh`: `config.hpp`, `expert_map.hpp`,
   `common/latency.cuh`, `combine/common.cuh` compile clean with `g++ -fsyntax-only`.
   `dispatch/common.cuh` fails on `mscclpp::to<>` inside the untouched `common/quantization.cuh`
   (device-only API) — reproduced identically on the unmodified base commit, i.e. pre-existing.
   Note that device bodies guarded by `MSCCLPP_BULK_AVAILABLE` are compiled *out* in this mode, so
   this check covers host paths and signatures, not the device bodies.
4. **Python** — all touched modules parse (`ast.parse`); no runtime execution (torch is not installed
   here).
5. **Code review** of every changed site against the report's design.

Unverified (needs a GPU):
* that the CUDA sources compile at all (see above);
* every runtime/synchronization property: the third-party visibility of `sender → leader` payload
  writes observed by a non-leader ETP peer after the sender's `signal()` (guarded by
  `__threadfence_system()` before the release, mirroring the existing peer-read pattern in
  `recvRankMajorRemotePartialsTma`), TMA reads of a peer's receive buffer, the extra
  `exchangeCombineReady` round, shared-memory budgets, occupancy, and performance;
* BF16 accumulation tolerances of the new group reduction on real data.

---

## 5. Known gaps / TODO

1. **THROUGHPUT mode has no ETP.** `etpSize > 1` is rejected in `MoERuntime`, `throughput.py`, the
   benchmark and the intranode test. Only the de-hardcoding (report stage 2) landed there.
2. **LATENCY `RANK_MAJOR` and `DIRECT_SEND` have no ETP** (report stage 3): rejected by host asserts
   and by `latency.py`.
3. **`EtpReduceMode::GROUP_NVLS` is a stub** — rejected at construction. Implementing it needs
   `connectNvlsCollective(comm, tpPeerRanks, bytes)` per TP group in `LatencyContext::initialize`, a
   `SwitchChannelDeviceHandle` in `DeviceContext`, and `multimem.ld_reduce` in place of the phase-B sum.
4. **FP8 dispatch under ETP is untested.** The code path is shared with BF16 (scales live in the
   payload and are replicated by the receiver), so it should work, but the CPU model only covers the
   BF16 data flow, and the Python ETP case defaults to BF16.
5. **ETP requires a fixed `maxTokensPerRank`** between dispatch and combine (asserted), because the
   row-offset workspace array is laid out with the active capacity. Expert-major already forbids a
   per-call capacity override.
6. **ETP requires the whole world in one IPC domain** (asserted). Internode ETP is blocked by the
   same missing IB path as internode EP.
7. **`num_blocks` lower bound is still `world_size + 2`**, not `ep_size + 2`: the receive scheduler
   still enumerates all `numRanks` sources under design B2, so worker blocks ≥ `numRanks` is still
   required. The report's suggested relaxation does not apply to B2.
8. **Dispatch metadata under `DUPLICATE_SEND` costs `etpSize×` slots** (`numRanks * nEPG`), which is
   reflected in the symmetric buffer size at construction time.
9. `MoECommunicatorConfig.num_local_experts` / `local_expert_start` are still advisory on the C++
   side (open question 7 of the report is unresolved); they are now validated against the EP group.

---

## 6. Commands for the GPU worker (16 GPUs)

Build (any node with the 16-GPU harness):

```bash
cd mscclpp-etp-impl
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMSCCLPP_BYPASS_GPU_CHECK=OFF -DMSCCLPP_BUILD_EXT_EP=ON
cmake --build build -j
# unit tests include the new host-side topology test:
./build/test/unit_tests --gtest_filter='EpTopology*'
```

Nvcc-free pre-checks (also runnable anywhere):

```bash
CUDA_INCLUDE=/usr/local/cuda/include ./tools/host_syntax_check.sh          # dispatch/common.cuh fails pre-existing, see §4
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

### What to report back

* whether `nvcc` compiles the branch cleanly (and any errors — they are unverified here);
* ETP=1 regression pass/fail plus latency deltas vs the `feature/ep` baseline;
* ETP=4 correctness (the tests assert `max|got-expected|` bounds and print it);
* whether `--etp-reduce-mode source_side` and `--etp-dispatch-mode duplicate_send` agree with the
  default within tolerance;
* dispatch/combine latency for EP=4/ETP=4 vs EP=16/ETP=1 at equal token counts.
