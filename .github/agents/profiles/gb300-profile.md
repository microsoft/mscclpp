# Hardware Profile: NVIDIA GB300 (placeholder)

> **Not yet active.** This profile is a stub. Before using it, the agent must confirm the specifics below with the user and fill them in. Do not assume GB300 capabilities until this file is populated and explicitly selected as the active profile.

## Topology (to confirm)
- **num_gpus:** _TBD_ (depends on NVL domain size)
- NVL domain size (GPUs per coherent NVLink domain): _TBD_
- Per-GPU NVLink bandwidth and generation: _TBD_
- Intra-domain switch fabric (NVSwitch generation, multimem semantics): _TBD_
- Inter-domain interconnect (Quantum / IB / Ethernet): _TBD_
- Grace CPU ↔ Blackwell GPU coherent memory characteristics: _TBD_

## Compute (to confirm)
- CUDA capability: _TBD_
- SM count, shared memory per SM, register file size: _TBD_
- Supported dtypes (fp8 variants, fp6/fp4 if applicable): _TBD_

## NVLink SHARP (NVLS) on GB300 (to confirm)
- Availability and supported reduce ops / dtypes: _TBD_
- Any semantic changes from H100 NVLS (e.g., wider multimem, new collectives): _TBD_

## Channels (DSL mapping, to confirm)
- `MemoryChannel` behavior across an extended NVL domain: _TBD_
- `SwitchChannel` semantics on the new switch generation: _TBD_
- `PortChannel` transport for inter-domain traffic: _TBD_

## AllReduce — Recommended starting points by message size
| Regime | Size | Suggested algorithm family | Notes |
| --- | --- | --- | --- |
| Latency-bound | _TBD_ | _TBD_ | confirm with user |
| Medium / one-shot | _TBD_ | _TBD_ | confirm with user |
| Bandwidth-bound | _TBD_ | _TBD_ | confirm with user |
| Cross-domain | _TBD_ | _TBD_ | confirm with user |

## AllGather — Recommended starting points by message size
| Regime | Size | Suggested algorithm family | Notes |
| --- | --- | --- | --- |
| Latency-bound | _TBD_ | _TBD_ | confirm with user |
| Medium | _TBD_ | _TBD_ | confirm with user |
| Bandwidth-bound | _TBD_ | _TBD_ | confirm with user |
| Cross-domain | _TBD_ | _TBD_ | confirm with user |

## Recommended starting parameters
- `num_threads_per_block`: _TBD_
- `instances`: _TBD_
- Thread blocks: _TBD_
- `protocol`: _TBD_
- `use_double_scratch_buffer`: _TBD_

## Known pitfalls
- _TBD — capture as we learn._

## Baselines
- _TBD — link to results once measured._

---

### Activation checklist (for the agent)

Before treating this profile as authoritative:
1. Ask the user for each `_TBD_` field above (batch the questions).
2. Update this file in-place with confirmed values.
3. Confirm the recommended algorithm families against actual GB300 hardware behavior, not extrapolation from H100.
4. Note in the generated DSL file's header comment that the design was tuned for GB300 per this profile revision.
