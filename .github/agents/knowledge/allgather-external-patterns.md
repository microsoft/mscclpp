# AllGather External Pattern Catalog (distilled)

Distilled patterns from the external reference repository, encoded so the AllGather DSL agent does **not** need live access to the source repo.

## Source

- **Repo:** `msccl-users` (`https://msazure.visualstudio.com/One/_git/msccl-users`)
- **Branch:** `t-ekoww/mscclpp_benchmark`
- **Folder in scope:** `/algos/mscclpp_new_DSL/allgather/` (single-node and multi-node subfolders)
- **Snapshot commit:** `5eeaa06152e8bd034c5d00938970be9ce0ffa26c`
- **Snapshot date:** 2026-05-14

> If the upstream branch advances past this commit, re-snapshot before relying on the catalog for new work.

## Catalog conventions

Each entry below captures **only** what's needed to recognize, reproduce, or vary the pattern in MSCCL++ DSL — not the full source. Source code is intentionally *not* mirrored; if full source is needed, fetch it from the upstream repo via the Azure DevOps MCP server (`azure-devops-repo_file`).

Per-entry fields:

- **Source file** — upstream path, for citation.
- **Intent** — what regime / objective the algorithm targets.
- **Topology assumptions** — node/GPU layout, divisibility constraints.
- **Channels** — `MemoryChannel` / `PortChannel` / `SwitchChannel` usage and pairing.
- **Algorithm phases** — high-level structure (sync pacing, fanout, ring/fullmesh, etc.).
- **TB layout** — thread-block role assignment.
- **Parameters** — `protocol`, `chunk_factor`, `inplace`, `use_double_scratch_buffer`, defaults for `instances` / `num_threads_per_block`.
- **Quirks** — anything surprising or easy to get wrong on recreation.
- **Reconstruction inputs** — the minimal spec to hand the AllGather agent to recreate this pattern.

---

## Multi-node patterns

### `allgather_hierarchical_ring` (despite filename "allpairs")

- **Source file:** `/algos/mscclpp_new_DSL/allgather/multinode/allgather_hierarchical_allpairs.py`
- **Intent:** Multi-node AllGather, bandwidth-bound, `Simple` protocol, medium-to-large messages.
- **Topology assumptions:**
  - `size = num_gpus`, user-supplied `gpus_per_node`, `nodes = size / gpus_per_node`. Homogeneous nodes (same `gpus_per_node` everywhere).
  - Rank `r` is on node `r // gpus_per_node` with local index `r % gpus_per_node`.
- **Channels:**
  - **`MemoryChannel`** — full intra-node bipartite mesh: every pair `(src, dst)` within a node. `gpus_per_node × (gpus_per_node − 1)` channels per node.
  - **`PortChannel`** — **inter-node ring paired by local rank index `g`**: each rank `(n, g)` has IB channels to `(n±1, g)` only. *Not* an inter-node all-pairs — the filename is misleading.
- **Algorithm phases:**
  1. **Intra-node fullmesh broadcast (setup).** `signal(data_sync=none)` → `wait(data_sync=after)` → `put`. Every rank publishes its own chunk to every same-node peer.
  2. **Inter-node ring fused with intra-node fanout.** For `step in 0..nodes-2`:
     - Each rank puts the chunk at `offset = g + ((n - step) % nodes) * gpus_per_node` to its `next_rank` over IB.
     - Receives the chunk at `recv_offset = g + ((n - 1 - step) % nodes) * gpus_per_node` from `prev_rank` over IB.
     - Cross-TB `Rank.barrier()` between the IB-receiving TB and the intra-node TBs.
     - Fans the received chunk to all intra-node peers via `MemoryChannel.put`.
     - On the final step, signals intra-node peers with `data_sync=before`.
  3. **Final wait** on the last round of intra-node puts.
- **TB layout:**
  - `tb_offset = gpus_per_node - 1` is reserved for the IB hop (`PortChannel` signal/wait/put).
  - Intra-node ops use TB `peer if peer < g else peer - 1` — packs the `gpus_per_node - 1` non-self peers into TB ids `0..gpus_per_node - 2`.
  - Net: 1 IB TB + `gpus_per_node - 1` intra-node TBs, running concurrently.
- **Parameters:**
  - `protocol="Simple"`, `chunk_factor=1` (`chunksperloop=1`), `inplace=True`, `use_double_scratch_buffer=False`.
  - `instances`, `num_threads_per_block` (default 1024), `min_message_size`, `max_message_size` are CLI-parameterized.
- **Sync pacing:**
  - Initial intra-node `signal(data_sync=none)` → `wait(data_sync=after)`.
  - Final intra-node `signal(data_sync=before)` on `step == nodes - 2`.
  - `Rank.barrier(tb_list=[0..tb_offset])` between inter-node IB recv and intra-node fanout each step.
- **Quirks:**
  - Filename says "allpairs" but the inter-node phase is a **ring**. Preserve this if recreating for parity.
  - `if src_rank != next_rank` guard handles the degenerate `nodes == 1` case.
  - Uses `Rank(r).get_output_buffer()[i:i+1]` slicing — relies on `chunk_factor=1` so each rank's output buffer has exactly `num_gpus` chunks.
  - No NVLS / `SwitchChannel`.
- **Reconstruction inputs (paste into the AllGather agent):**
  ```
  collective:        allgather
  topology:          multi-node, <NODES> nodes × <GPUS_PER_NODE> GPUs
  algorithm_hint:    hierarchical — fullmesh intra-node + ring inter-node
                     (paired by local rank index; NOT inter-node all-pairs)
  size:              <BW-bound regime, e.g. 1MB–64MB per-rank>
  inplace:           yes
  chunk_factor:      1
  protocol:          Simple
  channels:          MemoryChannel intra-node fullmesh
                     + PortChannel inter-node ring (next/prev only, paired by local g)
                     no NVLS
  use_double_scratch_buffer: no
  tb_layout:         (gpus_per_node - 1) intra-node TBs + 1 IB TB at id = gpus_per_node - 1
  sync_pacing:       signal(none) -> wait(after); final intra-node signal(before);
                     Rank.barrier between IB recv and intra-node fanout each step
  instances:         sweep 1..4
  num_threads_per_block: 1024
  constraint:        homogeneous gpus_per_node
  ```

---

### `allgather_hierarchical-copilot` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/multinode/allgather_hierarchical-copilot.py`
- **Status:** TBD. Fetch and distill into the same field structure as the entry above.

### `allgather_hierarchicalNew-ring` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/multinode/allgather_hierarchicalNew-ring.py`
- **Status:** TBD.

### `allgather_hierarchical_pkt` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/multinode/allgather_hierarchical_pkt.py`
- **Status:** TBD. Name suggests packet (LL) protocol variant of the hierarchical pattern — likely targets small-message latency.

### `allgather_oop` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/multinode/allgather_oop.py`
- **Status:** TBD. Name suggests out-of-place AllGather.

### `allgather_tbg-medium_nozerocopy` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/multinode/allgather_tbg-medium_nozerocopy.py`
- **Status:** TBD. Name suggests `ThreadBlockGroup` variant for medium messages, explicitly without zero-copy.

---

## Single-node patterns

### `allgather_large_H100` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_large_H100.py`
- **Status:** TBD. Likely H100-tuned bandwidth-bound design.

### `allgather_large_alternate` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_large_alternate.py`
- **Status:** TBD.

### `allgather_large_get` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_large_get.py`
- **Status:** TBD. Name suggests pull-based (`get`) variant.

### `allgather_large_get_alternate` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_large_get_alternate.py`
- **Status:** TBD.

### `allgather_large_new` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_large_new.py`
- **Status:** TBD.

### `allgather_large_nonfused` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_large_nonfused.py`
- **Status:** TBD. Name suggests deliberately disabled op-fusion variant — useful for measuring fusion benefit.

### `allgather_large_put1` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_large_put1.py`
- **Status:** TBD.

### `allgather_large_put2` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_large_put2.py`
- **Status:** TBD.

### `allgather_single_node_large_put` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_single_node_large_put.py`
- **Status:** TBD.

### `allgather_single_node_pkt` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_single_node_pkt.py`
- **Status:** TBD. Likely packet (LL) AllGather for small messages.

### `allgather_single_node_put` — _not yet distilled_

- **Source file:** `/algos/mscclpp_new_DSL/allgather/singlenode/allgather_single_node_put.py`
- **Status:** TBD.

---

## Maintenance

- **When the upstream branch advances**, re-snapshot the commit SHA at the top of this file and re-distill any entries whose source has changed.
- **Adding a new entry**: copy the field structure from the `allgather_hierarchical_ring` entry above. Keep entries focused on *what would be needed to recreate the pattern*, not on summarizing the code.
- **Stub entries** ("not yet distilled") are intentional — they record the existence of a pattern and its source path so the agent knows what to fetch on demand. They are not a substitute for a real distillation.
