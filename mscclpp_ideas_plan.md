# 02 - App-Level Design: Expert Rebalance & Dynamic All-to-All-v

Proposal doc. How to build an adaptive MoE load balancer **on top of** MSCCL++'s one-sided channels. Core principle: **the app owns _demand_** (which experts are hot, how many tokens go where), not physical network paths. It never manages QPs and never measures network congestion.

Grounded in two reference implementations cloned under `/home/malekmusleh/repos`: **DeepEP** (the primary port target) and **Tutel** (kernel + capacity techniques).

---

## 1. Why this layer exists

MSCCL++ has **no MoE/dispatch/combine code** and **no working all-to-all-v** (`ncclAllToAllv` and the NCCL-compat `ncclSend`/`ncclRecv` are stubs in `src/ext/nccl/nccl.cc:774-808`). The executor/DSL layer is structurally static: plans scale total size uniformly but cannot express **data-dependent per-peer counts** (`src/core/executor/execution_plan.cc:198-299`). So the MoE dynamic all-to-all-v must be built in **custom kernels + host control** over the native `MemoryChannel`/`PortChannel` primitives.

NCCL itself has **no** native all-to-all-v. It is assembled from grouped `ncclSend`/`ncclRecv`, so it is not a direct reference.

---

## 2. The dynamic all-to-all-v: follow DeepEP's one-sided two-phase algorithm

**DeepEP** (`/home/malekmusleh/repos/DeepEP`) is the near-exact reference: a fully **one-sided, dropless, two-phase** dispatch/combine whose primitives map approximately 1:1 onto MSCCL++ channels.

1. **Local layout** (no comms): histogram `topk_idx` into per-rank / per-expert counts plus an `is_token_in_rank` mask (`csrc/kernels/legacy/layout.cu`, `get_dispatch_layout`).
2. **Count exchange + offset build**: each rank **puts** its count row into peers' buffers, barriers, then **locally reduces + prefix-sums** to exact offsets (`notify_dispatch` in `intranode.cu` / `internode.cu`). **No collective.**
3. **Dispatch**: sender **writes tokens directly into receiver-owned slots** at the precomputed prefix offsets, and **signals** with a released tail flag or a **sign-encoded remote atomic-add** (`-count-1` disambiguates "zero tokens" from "not arrived"). The receiver polls and compacts.
4. **Combine**: reverse path. Receiver **reduces across the top-k source copies** (weighted sum + optional bias) using the `send_head` slot map saved during dispatch.

### Primitive mapping to MSCCL++

Directly from DeepEP's `ibgda_device.cuh`:

| DeepEP primitive | MSCCL++ equivalent |
| --- | --- |
| `nvshmemi_ibgda_put_nbi_warp` / `gin.put` | `PortChannel::put` |
| direct P2P store / `gin.get_sym_ptr` | `MemoryChannel` |
| `nvshmemi_ibgda_amo_nonfetch_add` / `ptx::red_add` | MSCCL++ remote atomic / signal |
| `nvshmemi_ibgda_quiet` / `gin.flush` | `flush` |
| **`nvshmemi_get_p2p_ptr(...) == 0`** (`ibgda_device.cuh:448`) | **the per-peer `MemoryChannel`-vs-`PortChannel` decision** |

DeepEP's **scaleup (NVLink) / scaleout (RDMA)** split is our **inference / training** transport split. Its **LL (low-latency) vs HT (high-throughput)** modes are our **inference vs training** modes. The **LL** path is the cleaner one to port first: count is a sign-encoded remote atomic, data is a put into a fixed max-slot region, and the receiver polls a flag. There is **no CPU round-trip**.

### Dispatch/combine kernel technique from Tutel

Use Tutel's **fused location-scatter** instead of the dense one-hot einsum: write each token directly to `expert * capacity + location` by its precomputed integer slot (`tutel/jit_kernels/sparse.py:32`), which is **O(S * M)** rather than **O(S * E * M)**. Combine is the symmetric gather plus gate-weighted reduce.

---

## 3. Actuators: adaptive, escalating, covers both workloads

| Stage | Actuator | Workload | Notes |
| --- | --- | --- | --- |
| **v1** | capacity cap / **token drop** | **training** only | bounded safety valve; **inference disallows drop** (quality + non-determinism) |
| **v2** | **dropless** dispatch + **hot-expert replication** | both | inference paths start here |
| **v3** | **expert placement / migration** | both | flatten load by relocating experts; see section 6 for the WideEP-scale design |

The controller escalates **drop -> replicate -> place** as skew persists.

**Grounded in the references:**

- Tutel's `capacity_factor` **tri-mode** formalizes v1 <-> v2 (`fast_dispatch.py:188-202`): `>0` = padded + drop; `==0` = **dropless** (capacity = runtime `max(location)`, all-reduced across ranks); `<0` = adaptive-but-capped.
- Drop **lowest-gate-score tokens first** (`batch_prioritized_routing`), not by position. This avoids positional bias.
- **DeepEP is the dropless reference:** worst-case buffer sizing + ring back-pressure, no capacity concept at all.

Whatever the actuator, **log dropped-token counts. Never drop silently.**

---

## 4. Transport split: same app logic, two substrates

The histogram/rebalance logic is **transport-agnostic**. Only the channel and the feed-forward target differ:

- **Training** MoE all-to-all (inter-node EP): over `PortChannel` (IB/RoCE). MRC EV profiles apply, feeding forward to doc 03.
- **Inference** MoE decode (intra-rack): over `MemoryChannel` / `SwitchChannel` (NVLink/NVSwitch). The NVSwitch crossbar is **non-blocking full-bisection**. There is **no path to steer**, so there is **no MRC/EV feed-forward here**; the only lever is app-level **expert placement** (section 6).

`ProxyTrigger`'s 32-bit size/offset (about 4 GB, `include/mscclpp/fifo_device.hpp:23-24`) bounds a single `PortChannel` transfer. That is fine per MoE step, but design the tables around it.

---

## 5. The histogram is dual-purpose: the link to doc 03's feed-forward (c4)

The **same** per-expert count / prefix-sum the app computes for dispatch also yields the **incast degree** and **byte volume** per destination. This is a **predictive** demand signal, available _before_ traffic flows. The app **feeds this forward** to the transport as:

- MRC **hints** (`num_remote_recv_peers` = "incast experienced by the target", `num_send_peers`, `mrc.h:293`), and
- an **EV/CC profile choice** per connection (via a `demand -> tier` table; see `03-shim-middleware-steering`).

So **one histogram drives three things**: the token-drop decision, the dispatch offsets, and the feed-forward demand hint. **No congestion measurement, no privilege needed.** The app stops at _demand_; it never defines paths or reads network telemetry. `ProxyHandler` byte tallies (`include/mscclpp/proxy.hpp:26`) are demand/accounting signals, not congestion.

---

## 6. WideEP: expert placement at NVL72 scale

WideEP (wide expert parallelism) spreads a model's full expert set across many GPUs inside one scale-up domain, such as GB300 NVL72, then dispatches tokens to the owning GPU and combines the results. In other words, section 2's dispatch/combine runs at rack scale, on every MoE layer, every forward pass. This section concretizes the **v3 actuator** (expert placement/migration, section 3) for that regime.

### 6.1 Why WideEP is the binding case for v3

Kimi-K3-class MoE models (reported, not verified: about 2.8T params, **about 896 experts**, KDA linear attention in 3 of 4 layers, about 1.5 TB HBM traffic per forward pass at MXFP4) cannot be served from one GPU's HBM. Serving them profitably at reasonable interactivity **requires aggregating many chips over a scale-up fabric**, i.e. WideEP.

The traffic asymmetry vs. KV-cache transfer:

| Traffic | Frequency | Effect of KDA linear attention |
| --- | --- | --- |
| KV-cache transfer (prefill -> decode) | **once per turn**; incremental transfer shrinks it further | up to **about 10x less** bandwidth |
| WideEP dispatch + combine | **2x per MoE layer per forward pass**, yielding 120+ all-to-all-v pairs per output token, multiplied by 500+ output tokens per turn | **unchanged**; grows with expert count |

More-efficient attention does **not** reduce scale-up all-to-all demand; the massive expert count _increases_ it. WideEP is exactly section 4's **inference intra-rack** row (`MemoryChannel`/`SwitchChannel` over NVLink/NVSwitch) at maximum intensity. Since the NVSwitch crossbar leaves **no path to steer** (`03-shim-middleware-steering`, section 1), **expert placement is the only lever**.

### 6.2 Two-layer API: app <-> transport/WideEP

Same core principle as the doc header, restated for this layer split:

- **App layer** talks in **experts, tokens, demand profiles**: never GPUs, NVLink, QPs, or NCCL.
- **Transport/WideEP layer** talks in **GPU IDs, the NVL72 fabric, all-to-all-v, resharding**.

App-facing sketch (illustrative C++, not tied to MSCCL++ headers):

```cpp
// Demand for this batch/turn: the SAME dual-purpose histogram of section 5.
struct ExpertDemandProfile {
  std::vector<uint64_t> tokens_per_expert;  // size = num_experts, e.g. 896
};

enum class QoSMode { LatencySensitive, ThroughputSensitive };

// App -> transport: update demand histogram, aggregated over a window.
void wideep_update_demand(const ExpertDemandProfile& demand);

// App -> transport: request an OPTIONAL rebalance. Non-blocking; returns a
// plan id. The plan may be applied later; see section 6.3.
ReshardPlanId wideep_request_reshard(QoSMode qos_hint);

// App -> transport: one WideEP forward pass. App provides token-to-expert
// routing plus activations; transport does dispatch/combine + expert execution.
void wideep_forward(const RoutingMap& routing,
                    const Activations& inputs,
                    Activations& outputs);

// Telemetry/debug only; never used for routing decisions by the app.
ExpertPlacement wideep_get_placement();
```

Under the hood:

- `wideep_update_demand`: aggregate `tokens_per_expert` over N batches/turns; detect hot experts, skew, underused experts, and overloaded GPUs. This is **the same histogram** the app already computes for dispatch offsets and the c4 feed-forward (section 5). It is not a new measurement.
- `wideep_request_reshard`: build an expert-to-GPU plan from current demand. **`LatencySensitive`** spreads hot experts to reduce local overload; **`ThroughputSensitive`** packs experts to maximize GEMM efficiency. The plan is stored and applied lazily (section 6.3).
- `wideep_forward`: token packing (static shapes for CUDA Graph capture) -> dispatch (all-to-all-v over `MemoryChannel`/`SwitchChannel`, section 4) -> expert GEMMs -> combine. This **is** section 2's DeepEP-derived one-sided two-phase algorithm: the LL path, fused and CUDA-graph-captured. Nothing new is introduced at the transfer level.

### 6.3 Resharding without global barriers

A **global barrier** would be required only if placement changed **mid-forward-pass**, routing tables changed **with kernels in flight**, or an **actively-used** expert migrated. Avoid all three by construction:

| Technique | Rule |
| --- | --- |
| **Epoch-based** | reshard plans apply only **between serving epochs** (every N turns/seconds); routing + placement are consistent within an epoch |
| **Lazy migration** | copy expert weights to the new GPU **in the background**; flip the routing table once, after copies complete and no in-flight work uses the old placement; **per-expert** locks, never cluster-wide |
| **Per-expert quiescence** | mark expert "migrating" -> stop assigning new tokens -> drain in-flight -> copy -> update placement -> resume. **Local, not global** |
| **Background pacing** | migrate **1-3 experts per window**, never all 896 at once |

**Real costs**: extra HBM + NVLink bandwidth for weight copies, routing-table updates, and control-plane complexity. **Avoided**: full-cluster stalls, large synchronization events, and broken CUDA Graph capture. The epoch boundary is also the natural point where section 3's controller escalates **drop -> replicate -> place**. Migration and hot-expert replication share the same lazy-copy machinery.

---

## 7. Prototype

- **Runnable PoC:** `examples/torch-integration/moe_feedforward_poc.py` demonstrates the app-side flow end-to-end: skewed gate -> histogram -> capacity/drop -> dynamic all-to-all-v -> expert FFN -> combine, plus the c4 feed-forward profile selection. It uses `torch.distributed.all_to_all_single` as a **stand-in** for MSCCL++ channels, and a **mocked** profile sink because no MRC hardware is present.
- **Reusable proxy:** `examples/torch-integration/mrc_feedforward_proxy.py`: `MRCFeedForwardProxy` factors the feed-forward out so the identical hint path serves MoE dispatch **and** KV-cache transfer.
- **MSCCL++ follow-on:** replace the torch transport with native channels + a custom `moe_dispatch.cu` modeled on `test/mscclpp-test/alltoall_test.cu` + `python/mscclpp_benchmark/mscclpp_op.py`, following DeepEP's LL path.
- **WideEP follow-on (deferred):** extend `moe_feedforward_poc.py` with an epoch-based reshard simulation: demand histogram -> reshard plan -> lazy apply at the epoch boundary (section 6.3), with the torch stand-in transport.

---

## Prior Art & Positioning

**Not a novelty claim.** App-level MoE load balancing is standard practice: **MoETuner** (balanced expert placement + token routing to cut all-to-all congestion), **Expert Choice routing** (perfect balance, no drops), **Tutel**, **DeepEP**, and **FasterMoE**. WideEP-style wide expert-parallel serving (section 6) is likewise established practice: DeepSeek-V3's wide-EP deployment over DeepEP, SGLang/vLLM large-scale EP, and the section 6.3 resharding scheme as standard epoch/lazy-migration technique.

This doc _reuses_ those techniques. The only possibly-new element is the feed-forward binding to MRC described in `03-shim-middleware-steering`.
