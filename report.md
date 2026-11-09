# DG versus MSCCL++ strict MegaMoE decode tuning

## Scope and method

- Harness provenance: `harness-tuning/full_campaign.py` is a custom strict campaign harness, not the stock DG/`bench_model.py` timing entry point.
- Reused from DG: the DG benchmark `Fixture`, model configuration, canonical inputs/routes, packed weights, rank placement, workspace setup and native `fp8_fp4_mega_moe` invocation.
- Changed from the DG benchmark: orchestration and the timing boundary were replaced so DG and MSCCL++ use the same prepared-input, fused-kernel-only CUDA Graph method.
- Hardware: 32 NVIDIA GB300 GPUs, DP32/EP32/TP1.
- Workload: H4096, I4352, 512 experts, 16 experts/rank, top-7.
- Matrix: balanced, hot_experts, lognormal, zipf and zipf_skewed at 8, 16, 32 and 64 tokens/rank.
- Boundary: exactly one fused MegaMoE kernel per invocation; 102 captured invocations rotating 16 independent weight banks.
- Excluded: eager timing, input quantization/staging, routing, shared experts, preparation, output copies and all pre/post kernels.
- Timing: maximum rank, queued CUDA events over CUDA Graph replay. Winners pool two independent 15-sample confirmation cohorts.
- DG search: complete frozen 190-candidate W4A8/decode manifest.
- MSCCL++ search: bounded K128/K256 compile-time policies followed by resident-CTA margin refinement.

## Confirmed winners

| Pattern | Tokens/rank | DG µs/kernel | DG config | MSCCL++ µs/kernel | MSCCL++ config | Faster backend | Advantage |
|---|---:|---:|---|---:|---|---|---:|
| balanced | 8 | 92.115134 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 97.359371 | `k128-epilogue16-warps8@margin2` | **dg** | 5.693% |
| balanced | 16 | 92.463425 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 97.732235 | `k128-epilogue16-warps8@margin2` | **dg** | 5.698% |
| balanced | 32 | 94.176470 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 98.801466 | `k128-epilogue16-warps8@margin4` | **dg** | 4.911% |
| balanced | 64 | 98.701178 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 100.649410 | `k128-profile@margin4` | **dg** | 1.974% |
| hot_experts | 8 | 75.966486 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 77.285960 | `k128-epilogue16-warps8@margin2` | **dg** | 1.737% |
| hot_experts | 16 | 79.666929 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 79.409151 | `k128-epilogue16-warps8@margin2` | **mscclpp** | 0.325% |
| hot_experts | 32 | 89.131137 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 85.569880 | `k256-warps16@margin2` | **mscclpp** | 4.162% |
| hot_experts | 64 | 101.291766 | `generic-m64-mma64-sm152-pull4096-stages8` | 89.532076 | `k128-epilogue16-warps8@margin0` | **mscclpp** | 13.135% |
| lognormal | 8 | 91.473361 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 96.104472 | `k128-epilogue16-warps8@margin2` | **dg** | 5.063% |
| lognormal | 16 | 94.682669 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 97.390586 | `k128-cache-expert-major@margin2` | **dg** | 2.860% |
| lognormal | 32 | 102.286065 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 99.190170 | `k128-cache-weights@margin4` | **mscclpp** | 3.121% |
| lognormal | 64 | 115.775108 | `generic-m64-mma64-sm152-pull4096-stages8` | 103.529830 | `k128-adaptive-cache@margin0` | **mscclpp** | 11.828% |
| zipf | 8 | 78.773330 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 82.855895 | `k128-epilogue16-warps8@margin4` | **dg** | 5.183% |
| zipf | 16 | 82.300495 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 84.194405 | `k128-epilogue16-warps8@margin4` | **dg** | 2.301% |
| zipf | 32 | 89.391372 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 86.172185 | `k128-expert-major@margin2` | **mscclpp** | 3.736% |
| zipf | 64 | 100.166900 | `generic-m64-mma64-sm152-pull4096-stages8` | 91.646902 | `k128-expert-major@margin4` | **mscclpp** | 9.297% |
| zipf_skewed | 8 | 80.225882 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 78.539138 | `k128-epilogue16-warps8@margin0` | **mscclpp** | 2.148% |
| zipf_skewed | 16 | 90.917438 | `generic-m64-mma64-sm152-pull4096-stages8` | 82.743582 | `k128-epilogue16-warps8@margin2` | **mscclpp** | 9.879% |
| zipf_skewed | 32 | 111.089619 | `generic-m64-mma64-sm152-pull4096-stages8` | 92.438274 | `k256-warps16@margin2` | **mscclpp** | 20.177% |
| zipf_skewed | 64 | 153.104003 | `generic-m64-mma64-sm152-pull4096-stages8` | 116.139343 | `k128-dispatch-stages2@margin0` | **mscclpp** | 31.828% |

## Winning configuration catalog

### dg

**`generic-m64-mma64-sm152-pull4096-stages8`**

- Configuration: `algorithm=generic, block_m=64, dispatch_pull_bytes=4096, input_scale=static_unit, mma_block_m=64, num_sms=152, pipeline_stages=8`
- Extension SHA-256: `d327449601148e9e9bbee7d6bfda7ad0db8f9b6fb8de8cc39fd4a4e02e1e6c30`
- Winning cases: n16-zipf-skewed, n32-zipf-skewed, n64-hot-experts, n64-lognormal, n64-zipf, n64-zipf-skewed

**`sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8`**

- Configuration: `algorithm=sfa_l2_prefetch, block_m=32, dispatch_pull_bytes=4096, input_scale=static_unit, mma_block_m=32, num_sms=144, pipeline_stages=8`
- Extension SHA-256: `d327449601148e9e9bbee7d6bfda7ad0db8f9b6fb8de8cc39fd4a4e02e1e6c30`
- Winning cases: n16-lognormal, n16-zipf

**`sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8`**

- Configuration: `algorithm=sfa_l2_prefetch, block_m=32, dispatch_pull_bytes=4096, input_scale=static_unit, mma_block_m=32, num_sms=152, pipeline_stages=8`
- Extension SHA-256: `d327449601148e9e9bbee7d6bfda7ad0db8f9b6fb8de8cc39fd4a4e02e1e6c30`
- Winning cases: n16-balanced, n16-hot-experts, n32-balanced, n32-hot-experts, n32-lognormal, n32-zipf, n64-balanced, n8-balanced, n8-hot-experts, n8-lognormal, n8-zipf, n8-zipf-skewed

### mscclpp

**`k128-adaptive-cache@margin0`**

- Configuration: `adaptive_cache_weights=True, cache_weights=False, cluster_size=2, cta_count=152, dispatch_chunk=4096, dispatch_stages=1, dispatch_warps=2, epilogue_registers=224, epilogue_tokens=32, epilogue_warps=4, expert_major_tasks=False, fixed_token_count=True, load_stages=8, load_warps=2, num_warps=16, sm_margin=0, split_pipelines=False, tile_k=128, tile_n=128, transfer_registers=128`
- JIT key: `5b1b36925a2e29be4bbd61dcfc94ec796b5731268fef2c66f0e48b50be7a5109`
- Module SHA-256: `53ae571819c1a09655b3f6216564655ed1522f6666a5cc2d07c2207dd90a4762`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n64-lognormal

**`k128-cache-expert-major@margin2`**

- Configuration: `adaptive_cache_weights=False, cache_weights=True, cluster_size=2, cta_count=150, dispatch_chunk=4096, dispatch_stages=1, dispatch_warps=2, epilogue_registers=224, epilogue_tokens=32, epilogue_warps=4, expert_major_tasks=True, fixed_token_count=True, load_stages=8, load_warps=2, num_warps=16, sm_margin=2, split_pipelines=False, tile_k=128, tile_n=128, transfer_registers=128`
- JIT key: `b97e6a6c548ce587c2a82af5b5937b1cf7d5da1a18275af140dccdaa292322c9`
- Module SHA-256: `fa904d32adcf9fcb78c7881bc9d6ed97bbe16b8203f42c4e221efbd4e57cc1bf`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n16-lognormal

**`k128-cache-weights@margin4`**

- Configuration: `adaptive_cache_weights=False, cache_weights=True, cluster_size=2, cta_count=148, dispatch_chunk=4096, dispatch_stages=1, dispatch_warps=2, epilogue_registers=224, epilogue_tokens=32, epilogue_warps=4, expert_major_tasks=False, fixed_token_count=True, load_stages=8, load_warps=2, num_warps=16, sm_margin=4, split_pipelines=False, tile_k=128, tile_n=128, transfer_registers=128`
- JIT key: `c06ddcd71f837d12ddd2eeb39efabf242d759d176b42246b562b6a84191b6e85`
- Module SHA-256: `a51cdf0810e3904a10d0012868ccc5648014f8ec2be1669b2e6c295b8c99f49f`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n32-lognormal

**`k128-dispatch-stages2@margin0`**

- Configuration: `adaptive_cache_weights=False, cache_weights=False, cluster_size=2, cta_count=152, dispatch_chunk=4096, dispatch_stages=2, dispatch_warps=2, epilogue_registers=224, epilogue_tokens=32, epilogue_warps=4, expert_major_tasks=False, fixed_token_count=True, load_stages=8, load_warps=2, num_warps=16, sm_margin=0, split_pipelines=False, tile_k=128, tile_n=128, transfer_registers=128`
- JIT key: `581c25e55c13390661cd8dc8c18ecb03ef2c994059abf1d4b0111b0d226638aa`
- Module SHA-256: `61eb66cf7eede5a4bd9ebe4d066d198277806f0e2c02d77df6cf7c0707f1ea5b`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n64-zipf-skewed

**`k128-epilogue16-warps8@margin0`**

- Configuration: `adaptive_cache_weights=False, cache_weights=False, cluster_size=2, cta_count=152, dispatch_chunk=4096, dispatch_stages=1, dispatch_warps=2, epilogue_registers=160, epilogue_tokens=16, epilogue_warps=8, expert_major_tasks=False, fixed_token_count=True, load_stages=8, load_warps=2, num_warps=16, sm_margin=0, split_pipelines=False, tile_k=128, tile_n=128, transfer_registers=128`
- JIT key: `1955dc4fad24cb5a9b7e372b6aea5918fa6bdd41ee25ef60b8a8f38de6ebd14d`
- Module SHA-256: `65eb648713965e73c5cba8b40554928e6628752e0cd951d59abd5828a8acebb1`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n64-hot-experts, n8-zipf-skewed

**`k128-epilogue16-warps8@margin2`**

- Configuration: `adaptive_cache_weights=False, cache_weights=False, cluster_size=2, cta_count=150, dispatch_chunk=4096, dispatch_stages=1, dispatch_warps=2, epilogue_registers=160, epilogue_tokens=16, epilogue_warps=8, expert_major_tasks=False, fixed_token_count=True, load_stages=8, load_warps=2, num_warps=16, sm_margin=2, split_pipelines=False, tile_k=128, tile_n=128, transfer_registers=128`
- JIT key: `1955dc4fad24cb5a9b7e372b6aea5918fa6bdd41ee25ef60b8a8f38de6ebd14d`
- Module SHA-256: `65eb648713965e73c5cba8b40554928e6628752e0cd951d59abd5828a8acebb1`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n16-balanced, n16-hot-experts, n16-zipf-skewed, n8-balanced, n8-hot-experts, n8-lognormal

**`k128-epilogue16-warps8@margin4`**

- Configuration: `adaptive_cache_weights=False, cache_weights=False, cluster_size=2, cta_count=148, dispatch_chunk=4096, dispatch_stages=1, dispatch_warps=2, epilogue_registers=160, epilogue_tokens=16, epilogue_warps=8, expert_major_tasks=False, fixed_token_count=True, load_stages=8, load_warps=2, num_warps=16, sm_margin=4, split_pipelines=False, tile_k=128, tile_n=128, transfer_registers=128`
- JIT key: `1955dc4fad24cb5a9b7e372b6aea5918fa6bdd41ee25ef60b8a8f38de6ebd14d`
- Module SHA-256: `65eb648713965e73c5cba8b40554928e6628752e0cd951d59abd5828a8acebb1`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n16-zipf, n32-balanced, n8-zipf

**`k128-expert-major@margin2`**

- Configuration: `adaptive_cache_weights=False, cache_weights=False, cluster_size=2, cta_count=150, dispatch_chunk=4096, dispatch_stages=1, dispatch_warps=2, epilogue_registers=224, epilogue_tokens=32, epilogue_warps=4, expert_major_tasks=True, fixed_token_count=True, load_stages=8, load_warps=2, num_warps=16, sm_margin=2, split_pipelines=False, tile_k=128, tile_n=128, transfer_registers=128`
- JIT key: `26e741f44a56f55e4364dec7d9ceaef6d1fe00dd95dd9462561d726ecd0a9ea9`
- Module SHA-256: `0af9df688fa4e267de8b6891628e57d192d218c2df8c03e6978c1a8979cd3810`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n32-zipf

**`k128-expert-major@margin4`**

- Configuration: `adaptive_cache_weights=False, cache_weights=False, cluster_size=2, cta_count=148, dispatch_chunk=4096, dispatch_stages=1, dispatch_warps=2, epilogue_registers=224, epilogue_tokens=32, epilogue_warps=4, expert_major_tasks=True, fixed_token_count=True, load_stages=8, load_warps=2, num_warps=16, sm_margin=4, split_pipelines=False, tile_k=128, tile_n=128, transfer_registers=128`
- JIT key: `26e741f44a56f55e4364dec7d9ceaef6d1fe00dd95dd9462561d726ecd0a9ea9`
- Module SHA-256: `0af9df688fa4e267de8b6891628e57d192d218c2df8c03e6978c1a8979cd3810`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n64-zipf

**`k128-profile@margin4`**

- Configuration: `adaptive_cache_weights=False, cache_weights=False, cluster_size=2, cta_count=148, dispatch_chunk=4096, dispatch_stages=1, dispatch_warps=2, epilogue_registers=224, epilogue_tokens=32, epilogue_warps=4, expert_major_tasks=False, fixed_token_count=True, load_stages=8, load_warps=2, num_warps=16, sm_margin=4, split_pipelines=False, tile_k=128, tile_n=128, transfer_registers=128`
- JIT key: `23408943a6583de20eb30300b4c3dfe9098c3db47d574bab7ce2e48c3b4c1122`
- Module SHA-256: `24c9b6396b9ab99b3919891515557970a7bd73d1c7f1aca0ad7cd169193927ac`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n64-balanced

**`k256-warps16@margin2`**

- Configuration: `adaptive_cache_weights=False, cache_weights=False, cluster_size=2, cta_count=150, dispatch_chunk=4096, dispatch_stages=1, dispatch_warps=4, epilogue_registers=224, epilogue_tokens=32, epilogue_warps=4, expert_major_tasks=False, fixed_token_count=True, load_stages=4, load_warps=2, num_warps=16, sm_margin=2, split_pipelines=False, tile_k=256, tile_n=128, transfer_registers=128`
- JIT key: `34300d3a31d5abe82eaed020e36219755b0d223e8ac3beefc79b2e1af8a66405`
- Module SHA-256: `476d2462f363239edf42059203ca3479f15cd716f28edc8eac503032ac3f5ba9`
- Extension SHA-256: `258ec67740a60e72b8d8bf5c609aaff094c3252cd6bda7472e25b2b0369c4d7e`
- Winning cases: n32-hot-experts, n32-zipf-skewed

## Screening top three by case

| Backend | Case | Rank | Candidate | Pooled median µs | Samples |
|---|---|---:|---|---:|---:|
| dg | n8-balanced | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 92.114093 | 5 |
| dg | n8-balanced | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 92.135423 | 5 |
| dg | n8-balanced | 3 | `sfa_l2_prefetch-m32-mma64-sm144-pull4096-stages8` | 93.591322 | 5 |
| dg | n16-balanced | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 92.483658 | 5 |
| dg | n16-balanced | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 92.502482 | 5 |
| dg | n16-balanced | 3 | `sfa_l2_prefetch-m32-mma64-sm144-pull4096-stages8` | 94.026142 | 5 |
| dg | n32-balanced | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 94.159999 | 5 |
| dg | n32-balanced | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 94.226931 | 5 |
| dg | n32-balanced | 3 | `sfa_l2_prefetch-m32-mma32-sm152-pull2048-stages8` | 95.667658 | 5 |
| dg | n64-balanced | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 98.734955 | 5 |
| dg | n64-balanced | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 98.754721 | 5 |
| dg | n64-balanced | 3 | `sfa_l2_prefetch-m32-mma32-sm152-pull2048-stages8` | 99.505780 | 5 |
| dg | n8-hot-experts | 1 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 75.972395 | 5 |
| dg | n8-hot-experts | 2 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 75.979083 | 5 |
| dg | n8-hot-experts | 3 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 77.109231 | 5 |
| dg | n16-hot-experts | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 79.716913 | 5 |
| dg | n16-hot-experts | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 79.749650 | 5 |
| dg | n16-hot-experts | 3 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 80.058247 | 5 |
| dg | n32-hot-experts | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 89.151370 | 5 |
| dg | n32-hot-experts | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 89.177306 | 5 |
| dg | n32-hot-experts | 3 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 89.672787 | 5 |
| dg | n64-hot-experts | 1 | `generic-m64-mma64-sm152-pull4096-stages8` | 101.284495 | 5 |
| dg | n64-hot-experts | 2 | `generic-m64-mma64-sm144-pull4096-stages8` | 101.706249 | 5 |
| dg | n64-hot-experts | 3 | `generic-m64-mma64-sm152-pull2048-stages8` | 103.814381 | 5 |
| dg | n8-lognormal | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 91.478691 | 5 |
| dg | n8-lognormal | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 91.533075 | 5 |
| dg | n8-lognormal | 3 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 93.324655 | 5 |
| dg | n16-lognormal | 1 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 94.682251 | 5 |
| dg | n16-lognormal | 2 | `sfa_l2_prefetch-m32-mma64-sm144-pull4096-stages8` | 94.702222 | 5 |
| dg | n16-lognormal | 3 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 94.722093 | 5 |
| dg | n32-lognormal | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 102.243031 | 5 |
| dg | n32-lognormal | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 102.342063 | 5 |
| dg | n32-lognormal | 3 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 103.868023 | 5 |
| dg | n64-lognormal | 1 | `generic-m64-mma64-sm152-pull4096-stages8` | 115.734487 | 5 |
| dg | n64-lognormal | 2 | `generic-m64-mma64-sm152-pull2048-stages8` | 118.257572 | 5 |
| dg | n64-lognormal | 3 | `generic-m64-mma64-sm144-pull4096-stages8` | 118.394665 | 5 |
| dg | n8-zipf | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 78.746665 | 5 |
| dg | n8-zipf | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 78.840892 | 5 |
| dg | n8-zipf | 3 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 80.587605 | 5 |
| dg | n16-zipf | 1 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 82.287059 | 5 |
| dg | n16-zipf | 2 | `sfa_l2_prefetch-m32-mma64-sm144-pull4096-stages8` | 82.306195 | 5 |
| dg | n16-zipf | 3 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 82.414116 | 5 |
| dg | n32-zipf | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 89.384783 | 5 |
| dg | n32-zipf | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 89.438220 | 5 |
| dg | n32-zipf | 3 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 90.187815 | 5 |
| dg | n64-zipf | 1 | `generic-m64-mma64-sm152-pull4096-stages8` | 100.133441 | 5 |
| dg | n64-zipf | 2 | `generic-m64-mma64-sm144-pull4096-stages8` | 101.172079 | 5 |
| dg | n64-zipf | 3 | `generic-m64-mma64-sm152-pull2048-stages8` | 103.373802 | 5 |
| dg | n8-zipf-skewed | 1 | `sfa_l2_prefetch-m32-mma32-sm152-pull4096-stages8` | 80.219188 | 5 |
| dg | n8-zipf-skewed | 2 | `sfa_l2_prefetch-m32-mma64-sm152-pull4096-stages8` | 80.259030 | 5 |
| dg | n8-zipf-skewed | 3 | `sfa_l2_prefetch-m32-mma32-sm144-pull4096-stages8` | 82.048940 | 5 |
| dg | n16-zipf-skewed | 1 | `generic-m64-mma64-sm152-pull4096-stages8` | 90.877178 | 5 |
| dg | n16-zipf-skewed | 2 | `generic-m64-mma64-sm144-pull4096-stages8` | 93.965911 | 5 |
| dg | n16-zipf-skewed | 3 | `generic-m64-mma64-sm152-pull2048-stages8` | 94.133333 | 5 |
| dg | n32-zipf-skewed | 1 | `generic-m64-mma64-sm152-pull4096-stages8` | 111.039480 | 5 |
| dg | n32-zipf-skewed | 2 | `generic-m64-mma64-sm152-pull2048-stages8` | 114.256940 | 5 |
| dg | n32-zipf-skewed | 3 | `generic-m64-mma64-sm144-pull4096-stages8` | 114.302018 | 5 |
| dg | n64-zipf-skewed | 1 | `generic-m64-mma64-sm152-pull4096-stages8` | 153.023377 | 5 |
| dg | n64-zipf-skewed | 2 | `generic-m64-mma64-sm152-pull2048-stages8` | 156.580607 | 5 |
| dg | n64-zipf-skewed | 3 | `generic-m64-mma64-sm144-pull4096-stages8` | 158.320732 | 5 |
| mscclpp | n8-balanced | 1 | `k128-epilogue16-warps8@margin2` | 97.192471 | 5 |
| mscclpp | n8-balanced | 2 | `k128-epilogue16-warps8@margin4` | 97.218613 | 5 |
| mscclpp | n8-balanced | 3 | `k128-expert-major@margin4` | 97.440003 | 5 |
| mscclpp | n16-balanced | 1 | `k128-epilogue16-warps8@margin2` | 97.547400 | 5 |
| mscclpp | n16-balanced | 2 | `k128-epilogue16-warps8@margin4` | 97.613490 | 5 |
| mscclpp | n16-balanced | 3 | `k128-cache-expert-major@margin4` | 97.730612 | 5 |
| mscclpp | n32-balanced | 1 | `k128-epilogue16-warps8@margin4` | 98.553414 | 5 |
| mscclpp | n32-balanced | 2 | `k128-epilogue16-warps8@margin2` | 98.587610 | 5 |
| mscclpp | n32-balanced | 3 | `k128-expert-major@margin4` | 98.761515 | 5 |
| mscclpp | n64-balanced | 1 | `k128-cache-weights@margin4` | 100.268133 | 5 |
| mscclpp | n64-balanced | 2 | `k128-adaptive-cache@margin4` | 100.287375 | 5 |
| mscclpp | n64-balanced | 3 | `k128-cache-expert-major@margin4` | 100.294063 | 5 |
| mscclpp | n8-hot-experts | 1 | `k128-epilogue16-warps8@margin2` | 77.049829 | 5 |
| mscclpp | n8-hot-experts | 2 | `k128-epilogue16-warps8@margin0` | 77.153673 | 5 |
| mscclpp | n8-hot-experts | 3 | `k128-adaptive-expert-major@margin2` | 77.340968 | 5 |
| mscclpp | n16-hot-experts | 1 | `k128-epilogue16-warps8@margin2` | 79.315553 | 5 |
| mscclpp | n16-hot-experts | 2 | `k128-epilogue16-warps8@margin0` | 79.382273 | 5 |
| mscclpp | n16-hot-experts | 3 | `k128-cache-weights@margin2` | 79.395762 | 5 |
| mscclpp | n32-hot-experts | 1 | `k256-warps16@margin2` | 85.546562 | 5 |
| mscclpp | n32-hot-experts | 2 | `k256-warps16@margin0` | 85.626770 | 5 |
| mscclpp | n32-hot-experts | 3 | `k128-adaptive-expert-major@margin2` | 85.767322 | 5 |
| mscclpp | n64-hot-experts | 1 | `k128-epilogue16-warps8@margin0` | 89.565489 | 5 |
| mscclpp | n64-hot-experts | 2 | `k128-adaptive-cache@margin0` | 89.699346 | 5 |
| mscclpp | n64-hot-experts | 3 | `k128-cache-expert-major@margin0` | 89.726118 | 5 |
| mscclpp | n8-lognormal | 1 | `k128-epilogue16-warps8@margin2` | 95.993728 | 5 |
| mscclpp | n8-lognormal | 2 | `k128-epilogue16-warps8@margin4` | 96.040782 | 5 |
| mscclpp | n8-lognormal | 3 | `k128-adaptive-expert-major@margin2` | 96.316444 | 5 |
| mscclpp | n16-lognormal | 1 | `k128-epilogue16-warps8@margin2` | 97.419502 | 5 |
| mscclpp | n16-lognormal | 2 | `k128-cache-expert-major@margin2` | 97.547400 | 5 |
| mscclpp | n16-lognormal | 3 | `k128-profile@margin2` | 97.553359 | 5 |
| mscclpp | n32-lognormal | 1 | `k128-dispatch-warps4@margin4` | 99.149280 | 5 |
| mscclpp | n32-lognormal | 2 | `k128-cache-weights@margin4` | 99.169569 | 5 |
| mscclpp | n32-lognormal | 3 | `k128-expert-major@margin4` | 99.169775 | 5 |
| mscclpp | n64-lognormal | 1 | `k128-epilogue16-warps8@margin0` | 103.433098 | 5 |
| mscclpp | n64-lognormal | 2 | `k128-adaptive-expert-major@margin0` | 103.507553 | 5 |
| mscclpp | n64-lognormal | 3 | `k128-adaptive-cache@margin0` | 103.520001 | 5 |
| mscclpp | n8-zipf | 1 | `k128-epilogue16-warps8@margin4` | 82.875919 | 5 |
| mscclpp | n8-zipf | 2 | `k128-epilogue16-warps8@margin0` | 82.896102 | 5 |
| mscclpp | n8-zipf | 3 | `k128-epilogue16-warps8@margin2` | 82.918067 | 5 |
| mscclpp | n16-zipf | 1 | `k128-epilogue16-warps8@margin4` | 84.247951 | 5 |
| mscclpp | n16-zipf | 2 | `k128-epilogue16-warps8@margin2` | 84.395296 | 5 |
| mscclpp | n16-zipf | 3 | `k128-profile@margin4` | 84.430015 | 5 |
| mscclpp | n32-zipf | 1 | `k128-expert-major@margin2` | 86.249096 | 5 |
| mscclpp | n32-zipf | 2 | `k128-cache-weights@margin2` | 86.249202 | 5 |
| mscclpp | n32-zipf | 3 | `k128-adaptive-cache@margin2` | 86.249308 | 5 |
| mscclpp | n64-zipf | 1 | `k128-adaptive-cache@margin4` | 91.652808 | 5 |
| mscclpp | n64-zipf | 2 | `k128-adaptive-expert-major@margin4` | 91.674562 | 5 |
| mscclpp | n64-zipf | 3 | `k128-cache-weights@margin4` | 91.687109 | 5 |
| mscclpp | n8-zipf-skewed | 1 | `k128-epilogue16-warps8@margin2` | 78.559165 | 5 |
| mscclpp | n8-zipf-skewed | 2 | `k128-epilogue16-warps8@margin0` | 78.605801 | 5 |
| mscclpp | n8-zipf-skewed | 3 | `k128-epilogue-reg192@margin2` | 78.659556 | 5 |
| mscclpp | n16-zipf-skewed | 1 | `k128-epilogue16-warps8@margin2` | 82.735685 | 5 |
| mscclpp | n16-zipf-skewed | 2 | `k128-expert-major@margin2` | 82.842353 | 5 |
| mscclpp | n16-zipf-skewed | 3 | `k128-profile@margin2` | 82.842565 | 5 |
| mscclpp | n32-zipf-skewed | 1 | `k256-warps16@margin2` | 92.459922 | 5 |
| mscclpp | n32-zipf-skewed | 2 | `k128-dispatch-warps4@margin0` | 92.476652 | 5 |
| mscclpp | n32-zipf-skewed | 3 | `k128-dispatch-warps4@margin2` | 92.636551 | 5 |
| mscclpp | n64-zipf-skewed | 1 | `k128-dispatch-stages2@margin0` | 116.002089 | 5 |
| mscclpp | n64-zipf-skewed | 2 | `k128-cache-weights@margin0` | 116.042143 | 5 |
| mscclpp | n64-zipf-skewed | 3 | `k128-adaptive-expert-major@margin0` | 116.054896 | 5 |

## Validation and limitations

- Result files: 132; canonical cases: 20.
- Every successful candidate passed ordinary-versus-prepared, eager-versus-graph and timed-versus-graph exactness with finite output.
- Inputs, routes and scores match across both backends for every rank and case; outputs match across candidates within each backend/case.
- DG and MSCCL++ outputs are intentionally not compared bitwise because their intermediate rounding, activation requantization, route-weight placement, partial precision and combine arithmetic differ.
- This is a fixed synthetic decode workload and a kernel-only result, not end-to-end model latency.
- MSCCL++ coverage is a bounded policy search, not an exhaustive search over every legal compile-time combination.
- Full per-run timings and configurations are in `all-runs.csv`; machine-readable winners and pooled samples are in `final-summary.json`.

## Tuning parameters and commands

The workload, routes, weights and arithmetic policy were fixed while tuning. In particular, both backends remained W4A8 with no shared experts; H4096, I4352, EP32, top-7, capacity 64, 16 weight banks and graph depth 102 did not change. Only the kernel scheduling and geometry controls below varied. The listed values are the union of tested policies, not a claim that every possible Cartesian combination was run.

### DeepGEMM parameters

DeepGEMM screened all 190 legal entries in `manifests/dg-candidates.json`: 110 `generic` and 80 `sfa_l2_prefetch` candidates.

| Parameter | Values tested | Superficial meaning |
|---|---|---|
| `algorithm` | `generic`, `sfa_l2_prefetch` | Selects the experimental W4A8 scheduling/kernel family. |
| `block_m` | 16, 32, 64 | Number of routed-token rows grouped in the M tile. |
| `mma_block_m` | 16, 32, 64 | M extent used by the MMA path. Legal pairs were generic 16/16, 32/32 and 64/64, plus SFA 32/32 and 32/64. |
| `num_sms` | 64, 96, 128, 144, 152 | Limits how many of the 152 physical SMs participate in the persistent schedule. |
| `dispatch_pull_bytes` | 1024, 2048, 4096 | Dispatch-transfer pull granularity. |
| `pipeline_stages` | 4, 6, 8 | Mainloop software-pipeline depth. |
| `input_scale` | `static_unit` only | Fixed input-scaling contract; recorded in every candidate but not tuned. |

The manifest contains only admitted combinations; shared-memory and geometry constraints prune the naive cross-product. The configuration was passed to the same configurable DG extension through `--dg-candidate-manifest`, so the extension and W4A8/decode arithmetic stayed fixed.

### MSCCL++ parameters

MSCCL++ first attempted 30 bounded compile-time policies at `sm_margin=2` (150 resident CTAs). Twenty-six ran successfully. `k128-cluster1`, `k128-dispatch-warps8`, `k128-load9` and `k128-load10` were rejected by native preflight because their specialization exceeded the device shared-memory limit. The union of the initial top-five policies was then retested at margins 0, 4 and 8; margin 2 reused the screening result.

| Parameter | Values tested | Superficial meaning |
|---|---|---|
| `tile_k` | 128, 256 | Reduction-dimension tile; this is the K128 versus K256 split. |
| `tile_n` | 128 only | Output-dimension tile, fixed for this campaign. |
| `load_stages` | 4, 6, 7, 8, 9, 10 | Asynchronous weight/input load-pipeline depth. |
| `num_warps` | 12, 16 | Total worker warps in a CTA. |
| `load_warps` | 2 only | Dedicated loading warps, held fixed. |
| `transfer_registers` | 96, 128 | Register budget for the transfer path. |
| `split_pipelines` | false, true | Selects the combined or split pipeline implementation. |
| `epilogue_tokens` | 16, 32 | Token tile handled by an epilogue task. |
| `epilogue_warps` | 4, 8 | Number of epilogue worker warps. |
| `epilogue_registers` | 160, 192, 224 | Epilogue register budget. |
| `dispatch_chunk` | 512, 1024, 2048, 3072, 4096 | Routed dispatch chunk size. |
| `dispatch_warps` | 2, 4, 8 | Number of dispatch worker warps. |
| `dispatch_stages` | 1, 2 | Dispatch software-pipeline depth. |
| `expert_major_tasks` | false, true | Toggles expert-major task ordering. |
| `cache_weights` | false, true | Enables the explicit weight-cache policy. |
| `adaptive_cache_weights` | false, true | Enables route-sensitive adaptive weight caching. |
| `cluster_size` | 1, 2 | CTA cluster size. |
| `fixed_token_count` | true only | Keeps the fixed decode-token specialization enabled. |
| `sm_margin` | 0, 2, 4, 8 | Leaves that many SM slots unused: CTA counts 152, 150, 148 and 144 respectively. This is a runtime residency control, not a different JIT module. |

Most policies changed one field from the K128 or K256 seed; a few intentionally combined related fields, such as `k128-epilogue16-warps8`. This was a targeted policy search rather than an exhaustive cross-product.

### Scripts and commands used

- `tuning/compile_candidates.py` defines the first K128/K256 policy set.
- `harness-tuning/compile_full_candidates.py` defines the expanded route/cache/epilogue set; `tuning/compile_full_candidates.py` is its identical analysis-side copy.
- `harness-tuning/compile-full-mscclpp.sh` compiles those policies into the prepared MSCCL++ JIT cache.
- `harness-tuning/full_campaign.py` is the strict 20-case kernel-only benchmark.
- `harness-tuning/run-full-dg-slice.sh` and `run-full-mscclpp-candidate.sh` launch individual DG slices and MSCCL++ policies.
- `harness-tuning/run-full-screen.sh` performs five-sample, three-replay screening and writes the initial top-five shortlist.
- `harness-tuning/run-refine-confirm.sh` refines margins, reselects the top three per case, runs two 15-sample/three-replay confirmation cohorts in opposite order, analyzes the results and renders this report.

The top-level commands used were:

```bash
ROOT=/home/caiorocha/test/run-dg-mscclpp/kernel-only-tuning-n8-64-20261103T224544Z
cd "$ROOT"

# Compile the expanded MSCCL++ policy set on each host, then screen both backends.
./harness-tuning/compile-full-mscclpp.sh
./harness-tuning/run-full-screen.sh

# Refine resident-CTA margins and run both confirmation cohorts.
./harness-tuning/run-refine-confirm.sh

# Standalone regeneration of the aggregate files and this report.
python3 tuning/analyze_full_campaign.py \
  --campaign "$ROOT/harness-tuning/full-tuning" --shortlist 3
python3 tuning/render_full_report.py \
  --campaign "$ROOT/harness-tuning/full-tuning"
```

For example, the screen driver expands to helper calls shaped like:

```bash
# DG manifest entries 0 through 9, all 20 cases, 5 samples x 3 graph replays.
./harness-tuning/run-full-dg-slice.sh \
  slice-000-010 0 10 screen-config 5 3

# One MSCCL++ policy at margin 2 with the same screen sampling.
./harness-tuning/run-full-mscclpp-candidate.sh \
  k128-profile \
  23408943a6583de20eb30300b4c3dfe9098c3db47d574bab7ce2e48c3b4c1122 \
  2 screen-config 5 3
```

Every distributed invocation saves its fully expanded MPI command in the corresponding `full-tuning/**/command.txt`. During the temporary host-SSH interruption, the resumable confirmation driver was restarted with `FULL_TUNING_LAUNCHER=$ROOT/harness-tuning/mpi-container-ssh-3020.sh`; completed result files were skipped, and no benchmark parameter, binary, route, timing method or sample count changed.
