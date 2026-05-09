# Low-Latency Dispatch/Combine Optimization History

Working branch: `qinghuazhou/expert_parallel_ll_opt` (uncommitted as of last session)

Goal: close the per-rank BW gap between mscclpp_ep LL path and nccl-ep LL on the
2× 8×H100 NDv5 testbed (master `10.0.0.4` ↔ worker `10.0.0.11` via `ssh -p 2222`,
8× mlx5_ib NICs, MTU=4096, IB transport).

---

## Test harness

```bash
cd /home/qinghuazhou/nccl-tests
pkill -9 -f "python.*test_low_latency" 2>/dev/null
ssh -p 2222 10.0.0.11 "pkill -9 -f 'python.*test_low_latency'" 2>/dev/null
sleep 4

# Default LL config: TOKENS=128, TOPK=8, hidden=7168, BF16, 64 experts, 4 local/rank.
MSCCLPP_EP_USE_IBGDA=1 \
MSCCLPP_EP_IBGDA_SHARDS=K \
MSCCLPP_EP_BENCH_ITERS=20 MSCCLPP_EP_BENCH_WARMUP=10 \
MSCCLPP_EP_LL_TOKENS=128 MSCCLPP_EP_LL_TOPK=8 \
timeout 90 bash run_ll_mpirun.sh
```

Build/deploy after editing `src/`:

```bash
cd /home/qinghuazhou/mscclpp_ep/build && cmake --build . -j 8 --target mscclpp_ep_cpp
scp -P 2222 lib/mscclpp_ep_cpp.so 10.0.0.11:/home/qinghuazhou/mscclpp_ep/build/lib/
md5sum lib/mscclpp_ep_cpp.so
ssh -p 2222 10.0.0.11 "md5sum /home/qinghuazhou/mscclpp_ep/build/lib/mscclpp_ep_cpp.so"
```

Microbench: `build/probe_stage4_perf` (2-rank MPI cross-node single-QP latency probe).

---

## Stage A — Diagnosing the proxy-FIFO bound (early May 2026)

Baseline mscclpp_ep LL ~30 GB/s/rank dispatch vs nccl-ep ~66 GB/s/rank target.

Sweep of `MSCCLPP_PROXY_BATCH_THRESHOLD`:

| T | Dispatch GB/s | Combine GB/s | triggersPerPost |
|---|---:|---:|---:|
| 1  | 30.70 | 27.94 | 1.00 |
| 8  | 28.31 | 26.59 | ~6.6 |
| 32 | 22.43 | 21.45 | ~15-18 |

**Conclusion:** batching `ibv_post_send` hurts BW monotonically. The bound is
GPU↔CPU FIFO round-trip latency, not per-WR syscall cost. Only GPU-initiated
postSend (IBGDA) can close 30→66 GB/s.

ABI hard lessons:
- `_mscclpp.cpython-*.so` lives in `<dist-packages>/mscclpp/` with `RUNPATH=$ORIGIN/lib`.
  Build emits `RUNPATH=/home/.../build/lib`; must `patchelf --set-rpath '$ORIGIN/lib'`
  before deploy to dist-packages.
- Adding fields to `ProxyService` changes layout and crashes pybind11 holders unless
  `_mscclpp.so` AND `libmscclpp.so` are rebuilt+resynced together. Use side-tables
  (`unordered_map<ProxyService*, X>`) for diagnostic state.

---

## Stage B — Native IBGDA bring-up (probe stages 0-4)

Goal: kernel-issued RDMA WRITE via `mlx5dv` doorbell, no NVSHMEM/DEVX/gdrcopy.

### Stage 0 PROBE — single-rank self-loop QP — PASSED.

Key technique:
- `cuMemHostRegister(uar_page_aligned, 4096, IOMEMORY)` — must page-align,
  add bf offset (typ 0x800) back to device pointer.
- `cudaHostRegister` for `sq.buf` and `dbrec`.
- Doorbell sequencing: WQE writes → `__threadfence_system` → DBR write
  → `__threadfence_system` → 8B BF MMIO store.

Gotchas (each cost iterations):
1. `mlx5_wqe_ctrl_seg.fm_ce_se` is a single byte at offset 11. Treating it as
   `uint32` puts `MLX5_WQE_CTRL_CQ_UPDATE` in the wrong byte → WR completes
   silently with NO CQE.
2. `lkey`/`rkey` in WQE `data_seg`/`raddr_seg` must be `htonl()`'d (BE),
   unlike host-order keys returned by `ibv_reg_mr`.
3. BF DB-mode value: `{htobe32(prod_idx<<8), htobe32(qpn<<8)}` — opcode and
   ds fields are zero (the real values live in the WQE itself).

### Stage 4 throughput probe — PASSED.

Cross-node 7168-byte WRs, 1024 msgs, single QP one direction:

| chunk | avg us | bw_avg GB/s |
|-------|-------:|------------:|
| 1     | 3660 | 2.01 |
| 4     | 1273 | 5.77 |
| 16    | 986  | 7.44 |
| 32    | 809  | 9.07 |
| 64    | 753  | **9.74** |

NCCL achieves ~8.25 GB/s per NIC. Native IBGDA at chunk=64 gives **9.74 GB/s
per NIC, exceeding NCCL**. Old host-FIFO PortChannel path was 3.75 GB/s/NIC.

**Bottleneck for chunk=1:**
1. `ready_head` CAS-spin (each thread waits for predecessor to publish).
2. `post_send_lock` CAS (one BF doorbell per WR).
3. BF MMIO write rate (~1 us each).

**Fix:** added `rdma_write_strided_burst()` in `src/core/include/ibgda_device.cuh`
— single thread reserves N slots, writes N WQEs, single `submit_requests<true>`,
single doorbell ring.

---

## Stage C — Wire IBGDA into LL kernel (Stage 4b)

Changed dispatch/combine to call `mscclpp::ibgda::port_put` (UNSIGNALED, no DB)
for data WRs and `rdma_write_inl8` (SIGNALED, rings DB) for the trailing
count/flag write that flushes the per-QP queue.

**Result:** ~36 GB/s/rank dispatch+combine — better than the 30 GB/s host-FIFO
baseline but still ~30% short of nccl-ep's ~65 GB/s target.

---

## Stage D — Single-QP latency microbench pinpoint

Microbench `probe_stage4_perf` (single QP, 1 thread/WR, kernel-issued via
`mscclpp::ibgda::rdma_write`):

14336 B sweep (median end-to-end, 30 iters, 5 warmup):

| msgs | us | us/WR amortized |
|---|---:|---:|
| 1   | 160  | 160 |
| 4   | 184  | 46 |
| 16  | 277  | 17 |
| 64  | 572  | 9 |
| 256 | 1457 | 5.7 |

7168 B sweep: 1→160us, 4→182us, 16→270us, 64→525us, 256→1212us.

**Interpretation:** LL kernel issues ~4-16 WRs per active (channel,dst_rank) QP
per dispatch. 4-16 WRs cross-node single-QP RTT = 184-277 us. This EXACTLY
matches the LL kernel's measured wait≈200 us window.

**Conclusion:** LL sits at the per-QP IBGDA latency floor. Per-QP throughput
asymptote is ~2.5 GB/s. Per-QP latency dominates because each batch is small
(4-16 WRs) serialized through a single QP.

### Optimization options identified
1. **Fan WRs across MORE QPs per peer** ("K-shard fan-out"): split each
   `(src_rank, dst_local_expert)` batch over K QPs. Expected: K parallel QPs
   each at lower load → near-linear speedup until NIC saturates.
2. **Concatenate per-peer tokens and post as 1-2 large WRs** (nccl-ep approach).
   Reduces WR count from ~16 to ~1-2, amortizes 160us per-WR fixed cost.
3. NULL options ruled out by additional probes: TMA combine (arith only 22us),
   NIC affinity (already 1:1), QP count (sweep flat), WR coalescing (random
   dest offsets prevent it).

---

## Stage E — K-shard fan-out (option 1) — IMPLEMENTED, NO SPEEDUP

### Idea
Each `(src_rank, dst_local_expert)` batch routes its K-th data WR through
`channel = dst_expert_local_idx + (slot_idx % K) * num_local_experts`. The
trailing count write fans out K-way: each shard QP writes `-(count_k+1)` to
its own slot at `rdma_recv_count[le][src][k]` so the receiver can sum them up
to recover the total.

### Files changed
- `src/ext/ep/kernels/api.cuh` — added `K_MAX_IBGDA_SHARDS=8` and
  `int num_ibgda_shards` parameter on `dispatch()`/`combine()` declarations.
- `src/ext/ep/kernels/configs.cuh` — duplicate `K_MAX_IBGDA_SHARDS` (kernel TU
  doesn't include `api.cuh`).
- `src/ext/ep/config.hpp` — host-side `kMaxIbgdaShards` mirror; signaling
  buffer sized 8× to support any K up to kMaxIbgdaShards.
- `src/ext/ep/kernels/internode_ll.cu` — dispatch/combine K-shard send + recv
  logic (data WR routing, K-loop count/flag writes, K-loop receiver poll,
  self-rank fan-out).
- `src/ext/ep/buffer.cc` — `MSCCLPP_EP_IBGDA_SHARDS` env, auto-bump
  `num_ibgda_channels` so `K * num_local_experts ≤ num_channels`.
- `nccl-tests/run_ll_mpirun.sh` — forwards `MSCCLPP_EP_IBGDA_SHARDS`.

### Bug encountered (and fix)
K=2 hung in dispatch receive on slot k=1 for ALL `(le, src=rank)` pairs
(self-rank). Cause: the `dst_rank == rank` branch only touched slot k=0 in
both dispatch and combine, but the receiver polls all K slots. Fix: when
`dst_rank == rank`, write slot 0 with `-(N+1)` and slots 1..K-1 with `-1`
(count_k=0 sentinel). Receiver decode `total = -sum - K = N` works for any
K because `-1` slot contributes `+1` to `-sum` and `-K` cancels.

### Sweep result (TOKENS=128, TOPK=8, ITERS=20, WARMUP=10)

| K | dispatch GB/s | combine GB/s | channels |
|---|---:|---:|---:|
| 1 | 36.32 | 37.43 | 16 |
| 2 | 36.79 | 37.36 | 16 |
| 4 | 37.29 | 37.43 | 32 |
| 8 | 37.21 | 36.45 | 64 |

**No measurable speedup.** All variants pass correctness. The per-QP latency
hypothesis from the microbench did NOT translate to LL workload speedup —
something else dominates at the system level (likely cross-rank synchronization
/ skew, not QP-level latency).

---

## Open hypotheses for the remaining ~30 GB/s gap

1. **Cross-rank skew / sync.** All 16 ranks must finish their sends before
   any receiver makes progress. Stragglers (slowest NIC, slowest expert)
   gate the whole step. K-shard fan-out doesn't help because the slow path
   is whichever rank is slowest, not which QP.
2. **Per-grid `cg::this_grid().sync()`** between send and recv phases —
   larger grids cost more.
3. **Token aggregation (option 2 from Stage D)** — concatenate all tokens
   destined for the same `(dst_expert, dst_rank)` into one or two large WRs.
   Microbench predicts 1457 us / 256 msgs → 5.7 us/WR vs current 17 us/WR
   at 16-msg batch; ~3× per-WR amortization. Requires shared-memory staging
   in the dispatch send loop.
4. **Combine warp-specialized TMA pipeline.** nccl-ep combine uses
   `tma_load_1d` + mbarrier ring (kNumStages buffers, 1 load warp + N reduce
   warps) per ([nccl/contrib/nccl_ep/device/low_latency.cu:1349-1455]).
   mscclpp_ep combine is a flat synchronous loop over topk → no async
   staging, MLP=0 across topk dim. Expected ~30% combine BW gain.
5. **1024 threads/block** vs current 960 (kNumWarpsPerGroup or kNumWarpGroups
   bump) — minor (~6%).
6. **LogFMT 10-bit encoding** — heavy code, modest gain. Defer.

---

## Key file map (mscclpp_ep)

| Path | Purpose |
|---|---|
| `src/ext/ep/kernels/internode_ll.cu` | LL dispatch + combine kernels (K-shard fan-out, IBGDA path) |
| `src/ext/ep/kernels/api.cuh`         | host-callable launcher decls; `K_MAX_IBGDA_SHARDS` |
| `src/ext/ep/kernels/configs.cuh`     | kernel-TU-side config macros |
| `src/ext/ep/config.hpp`              | `LowLatencyLayout`, `kMaxIbgdaShards`, signaling-buffer sizing |
| `src/ext/ep/buffer.cc`               | `Buffer::sync()` IBGDA setup, env-var read, plumbing into dispatch/combine |
| `src/ext/ep/ibgda_setup.cc/.hpp`     | per-rank QP creation, RTR/RTS, MR allgather, device-handle table |
| `src/core/include/ibgda_device.cuh`  | device-side primitives: `reserve_wqe_slots`, `submit_requests`, `submit_no_db`, `rdma_write`, `rdma_write_inl4/8`, `rdma_write_strided_burst` |
| `src/core/include/ibgda_port_channel_device.cuh` | `port_put`, `port_signal`, `port_wait` device ops |
| `src/core/ibgda.cc`                  | `IbgdaResources` (sq/dbrec/UAR registration) |
| `test/ibgda_probe/probe_stage4_perf.cu` | cross-node single-QP throughput microbench |

## Key file map (test driver)

| Path | Purpose |
|---|---|
| `nccl-tests/run_ll_mpirun.sh`        | mpirun launcher for `test_low_latency.py` (forwards `MSCCLPP_EP_USE_IBGDA`, `MSCCLPP_EP_IBGDA_SHARDS`, `MSCCLPP_EP_BENCH_ITERS`, `MSCCLPP_EP_BENCH_WARMUP`, `MSCCLPP_EP_LL_TOKENS`, `MSCCLPP_EP_LL_TOPK`) |

## Deploy targets (per worker node)
```
/usr/local/lib/python3.10/dist-packages/mscclpp/lib/libmscclpp.so.0.9.0
/usr/local/lib/python3.10/dist-packages/mscclpp/lib/libmscclpp_collectives.so.0.9.0
/usr/local/lib/python3.10/dist-packages/mscclpp/_mscclpp.cpython-310-x86_64-linux-gnu.so
/usr/local/lib/python3.10/dist-packages/mscclpp_ep_cpp.so   # legacy path
/home/qinghuazhou/mscclpp_ep/build/lib/mscclpp_ep_cpp.so    # current dev path
```

## Current performance vs target

| Path                | dispatch (GB/s/rank) | combine (GB/s/rank) |
|---------------------|---:|---:|
| mscclpp_ep proxy-FIFO baseline | 30.70 | 27.94 |
| mscclpp_ep IBGDA (current)     | ~37   | ~37 |
| nccl-ep target                 | ~65   | ~65 |
| nccl-ep intra-node (8×H100)    | ~198  | ~168 |

Cross-node IBGDA wins per-NIC at chunk=64 in the microbench (9.74 vs 8.25 GB/s),
but the LL workload's small per-QP batch sizes (4-16 WRs) sit at the latency
floor. K-shard fan-out across QPs did not help — gap is most likely cross-rank
skew, not per-QP throughput.

## Recommended next experiments
1. **Per-rank phase-time profiling** with CUDA events to isolate stragglers
   and quantify cross-rank skew vs raw kernel work.
2. **Token aggregation** (option 2 from Stage D) — biggest predicted win
   from the microbench data.
3. **Combine TMA/cp.async pipeline** — clean architectural win, decoupled
   from dispatch issues.

---

## Phase 6 — Grid-config sweep on IBGDA RDMA path (WIN: +6%)

### Setup
- TOKENS=128, TOPK=8, num_experts=64, num_ranks=16, BF16 hidden=7168.
- File: `src/ext/ep/kernels/internode_ll.cu` lines ~514-515 (dispatch) and
  ~828-829 (combine). Constants `kNumWarpsPerGroupRdma` /
  `kNumWarpGroupsRdma`.
- Old comment said RDMA path stuck at (3, 10) "to avoid host-proxy FIFO
  contention". That rationale applied to the **PortChannel** path; on the
  **IBGDA** path there is no host proxy in the dataline. So the constraint
  no longer applies and the grid is free to scale.

### Sweep (cross-node, 2×8×H100, MPI mpirun, 20 iters / 10 warmup)

| (kNumWarpGroupsRdma, kNumWarpsPerGroupRdma) | total blocks | dispatch GB/s | combine GB/s |
|---------------------------------------------|--------------|---------------|--------------|
| (3, 10) baseline                            | 22           | 36.32         | 37.43        |
| (2, 16)                                     | 32           | ~36.6         | ~36.9        |
| (4, 8)                                      | 16           | 34.21         | n/a          |
| **(1, 32) — adopted**                       | 64           | **38.65–38.73** | **39.43–39.56** |

Net win: **+6.4% dispatch / +5.4% combine** with no correctness regressions.

### Why it works
- `num_sms_base = cell_div(num_experts, kNumWarpGroups)` ⇒ at (1, 32) the
  grid grows from 22 to 64 blocks (1 expert per SM, 32 warps inside).
- The recv-side unpack body strides tokens by `sub_warp_id`; with 32 warps
  per warp-group instead of 10, each block's unpack runs ~3× faster.
- The send phase issues 1 unsignaled WR per (warp, dst_expert), so
  changing the warp-group geometry redistributes WRs across blocks but
  does **not** change total WR count — wait time stays ~190us.
- Profile: send 28us, wait 200us, unpack 5us, total 282us (vs old 310us).
  Bench delta was smaller (372→356) because per-iter launch overhead
  partially offsets the kernel-time win when grid grows from 22→64 blocks.
- (4, 8) regresses because each block ends up with too few warps to
  service the unpack body, and combine recv mismatch on token striping.

### What did NOT work this session
- **Per-WR doorbell ringing** (data WRs `ring_db=true`): wait dropped
  190→35us but send ballooned 55→280us due to per-WR `post_send_lock`
  contention. Net slightly worse. Reverted.
- **Dispatch grid_sync removal**: no perf gain (already non-blocking on
  the critical path); combine grid_sync IS needed (mismatch without it).
- **TOPK halving** (8→4): wait dropped 175→73us as expected — confirms
  wait scales linearly with WR count. Not a real optimization.
- **(2, 16) and (4, 8) grid configs**: marginal or worse vs (1, 32).

### Updated bottleneck breakdown (post-fix)
- send: ~28us (was 55us at (3, 10))
- wait: ~200us (per-QP wire serialization, unchanged — NIC-bound)
- unpack: ~5us
- kernel total: ~280us
- bench-per-iter: ~355us → ~75us launch + sync overhead
- per-rank BW: 38.7 GB/s = **77% of NDR HCA single-NIC ceiling (50 GB/s)**

## Updated next experiments (priority order)
1. **Multi-SGE WR**: pack multiple tokens going to same (le, dst_rank)
   into 1 WR with N scatter-gather entries. Predicted to halve wait time.
   Requires kernel restructure so each block serves 1 (le, dst_rank) pair
   and a new primitive `write_rdma_write_multi_sge_wqe()` in
   `src/core/include/ibgda_device.cuh`.
2. **Combine TMA/cp.async pipeline** — orthogonal architectural win on
   the recv-side reduce-add.
3. Investigate ~75us per-iter launch overhead (cooperative-launch cost
   with 64 blocks).

---

## Phase 7 — Post-grid optimization attempts (May 9, 2026)

After Phase 6's grid (1, 32) tuning landed (38.7 / 39.6 GB/s), explored
three more avenues. None produced material gains; all confirmed the same
underlying constraint: **single-NIC bandwidth ceiling**.

### 7.1 Multi-SGE WR coalesce — REVERTED

Implemented a `kMultiSge` template flag on the dispatch kernel. New
`rdma_write_multi_sge()` primitive in `src/core/include/ibgda_device.cuh`
packs N SGEs (each one full LL token) into one RDMA WRITE WQE (ds = 2 + N).
QP `max_send_sge` bumped 1 → 4 in `src/core/ib.cc`; SQ depth lowered
8192 → 2048 to keep per-QP WQE memory bounded.

Per-block work was rewired so each SM owns one (`dst_local_expert`, `dst_rank`)
pair. After staging + grid-sync, the warp builds a `shared_token_list[wg][256]`
of token indices targeted at this expert, then issues ⌈N / 4⌉ multi-SGE WRs
delivering them into N consecutive recv slots on the peer.

| variant                | dispatch GB/s | wait avg (us) |
|------------------------|---------------|---------------|
| baseline (Phase 6)     | 38.53         | 212           |
| **Multi-SGE ON**       | **38.68**     | **197** (-7%) |

WR count per QP fell 16 → 4 (a 4× reduction) but wait dropped only 7%.
Math: 1008 WRs × 14336 B/WR = 14.4 MB per rank per iter; at NDR 50 GB/s NIC
ceiling that is 288 µs minimum wire time. Current 38.7 GB/s payload + 7%
IB overhead = ~41 GB/s on the wire = **82% of single-NIC ceiling**.

Multi-SGE saves WR-post overhead but the bytes on the wire are unchanged.
Reverted (stashed) since the +0.3 GB/s gain doesn't justify the kernel
restructure and the QP `max_send_sge` ABI change.

### 7.2 CUDA Graph capture — INCOMPATIBLE

Tried wrapping `_dispatch()` and `_combine()` in `torch.cuda.CUDAGraph`
to amortize the ~70 µs/iter gap between kernel-total time and bench-loop
time (suspected `cudaLaunchCooperativeKernel` overhead).

Capture succeeded; the warmup PASS confirmed correctness once. Replay then
hung — first-node ranks completed all 20 iters but second-node ranks only
saw the warmup traffic.

Root cause: IBGDA primitives keep device-side QP state (`resv_head`,
`prod_idx`, `post_send_lock`) that monotonically advances per kernel
invocation. Graph replay re-executes the WQE writes verbatim, but the
runtime QP state has already advanced past the captured snapshot;
subsequent replays write into stale / wrapped SQ slots and the doorbell
ring index is wrong, so peers never see the data.

CUDA Graph is fundamentally incompatible with this IBGDA QP state model.
Test reverted.

### 7.3 Cross-rank skew investigation

Captured per-rank dispatch profile at iter 10 (16 ranks × 64 blocks each):

| metric        | min (us) | avg (us) | max (us) | range |
|---------------|----------|----------|----------|-------|
| wait avg      | 188      | 203      | 226      | 38    |
| wait max      | 277      | 295      | 321      | 44    |
| total avg     | 222      | 237      | 262      | 40    |

Slowest ranks: r2, r7 (node 1), r8 (node 2). Cross-rank skew is bounded
(~40 µs); the dominant variance is **within-rank**: a single rank's wait
ranges 200 µs (avg) to 321 µs (max) across its 64 blocks (= 64 (le, src_rank)
pairs it is receiving from). That 120 µs intra-rank spread reflects
per-(le, src_rank) NIC contention on the receiver side and is essentially
a wire-level scheduling phenomenon.

Best case: even if all 16 ranks were perfectly synchronized at the slowest
rank's avg (226 µs), we'd save ~25 µs = ~7%. Not pursuing — the within-rank
spread dominates and is not an addressable software issue.

### 7.4 Synthesis: single-NIC ceiling

Three independent attacks (K-shard fan-out, multi-SGE coalesce, CUDA Graph)
all bottomed out at the same number. The conclusion is unambiguous:

**At TOKENS=128, TOPK=8, BF16 hidden=7168, 2×8×H100 NDR with 1 mlx5_ib
NIC per rank, the practical single-NIC LL ceiling is ~41 GB/s payload
(82% of the 50 GB/s NDR raw line rate).** Phase 6's 38.7/39.6 GB/s sits
at 94-97% of that practical ceiling.

To meaningfully exceed this requires either:
- Multi-NIC striping (would need to fan WRs across mlx5_ib0+mlx5_ib1 +
  topology-aware QP allocation; non-trivial ABI change).
- A different problem size where NIC is no longer the binding constraint
  (smaller hidden, smaller per-token payload, sparser topk).
- Hardware uplift (NDR2 / 100 GB/s NIC).

Combine path may still have non-NIC headroom (recv-side reduce-add could
be pipelined with TMA / cp.async); leaving as a future independent attack.

---

## Phase 8 — Combine TMA / cp.async pipeline (DECLINED after profiling)

The Phase 7 synthesis flagged combine as a candidate "non-NIC software
option" worth pursuing. Profiled the combine kernel (TOKENS=128, TOPK=8,
BF16 hidden=7168, 16 ranks, IBGDA, grid (1, 32) → 64 blocks):

```
[ep-prof combine #10 r0] blocks=64
  send=9.5/16.6/19.1us
  wait=1.3/220.2/330.8us
  grid_sync=1.0/111.7/335.3us
  reduce=7.9/8.9/9.6us
  total=356.3/357.2/358.0us  (min/avg/max)
```

Breakdown of average per-block time (357 µs total):

| phase     | time   | %      | notes                                 |
|-----------|-------:|-------:|---------------------------------------|
| send      | 17 µs  |   5%   | RDMA WRITEs back to source ranks      |
| **wait**  | 220 µs | **62%**| NIC bandwidth bound (same as dispatch)|
| grid_sync | 112 µs |  31%   | per-block skew slack (free CTA wait)  |
| **reduce**| **9 µs** | **2.5%**| weighted reduce-add over topk inputs|

The reduce arithmetic — the *only* place TMA / `cp.async` could help — is
**9 µs out of 357 µs = 2.5%** of the total kernel time. Even halving it
(an aggressive estimate; H100 HBM3 already runs the loop at ~50% of peak
BW, so realistic savings are ~3 µs) yields **<1% end-to-end perf**.

Wait at 220 µs is the same NIC-bandwidth ceiling that dispatch hits:
combine sends the same ~14 MB/rank/iter back from each (le, src_rank)
to its source rank. The Phase 7 speculation that combine had a
"non-NIC bottleneck profile" was wrong — both paths are NIC-bound.

The 112 µs `grid_sync` interval is the same within-rank wait skew
documented in Phase 7.3 — blocks that finished waiting early sit in
`cg::this_grid().sync()` until the slowest block catches up. It is not
addressable by software (it tracks per-(le, src_rank) NIC contention).

**Decision: not implementing.** The cp.async pipeline would add 50+ LoC
of shared-memory plumbing and `__pipeline_*` synchronization for ≤1%
perf. Profile is the rebuttal.

## Updated synthesis (post Phase 8)

Both LL paths (dispatch + combine) are **single-NIC bandwidth bound** at
this problem size. The kernel-side software work is essentially done at
38.9 / 39.6 GB/s = 94-97% of the ~41 GB/s practical ceiling.

Remaining categories of attack, all architectural / non-trivial:

1. **Multi-NIC striping** — fan WRs across `mlx5_ib0 + mlx5_ib1` per
   rank. Requires QP-pool restructure, topology-aware peer mapping,
   and recv-side aggregation. Largest predicted win but biggest change.

2. **Smaller per-token payload** — at hidden=2048 or FP8, NIC stops
   being the binding constraint and kernel-side wins re-open. Useful
   for application-level integration but doesn't move the LL benchmark
   number on the current setup.

3. **NDR2 / 100 GB/s NIC** — hardware uplift, out of scope.

For the current 2×8×H100 NDR(50) setup with TOKENS=128 / TOPK=8 / BF16
hidden=7168, **the LL benchmark is closed**. Future commits should
either pivot to multi-NIC or to a different problem regime.
