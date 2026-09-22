# Vendored: DOCA GPUNetIO (GDAKI) device verbs

This directory contains a vendored copy of the DOCA GPUNetIO GPU-initiated
networking sources, imported from the NCCL GIN `gdaki` backend
(`nccl/src/transport/net_ib/gdaki/doca-gpunetio`).

- **License:** BSD-3-Clause (NVIDIA CORPORATION & AFFILIATES). See the SPDX
  headers in each file; the license text is reproduced in every source/header.
- **Why vendored:** like NCCL, MSCCL++ compiles these sources directly rather
  than linking a DOCA installation. The code reaches the NIC purely through
  `dlopen` of `libibverbs`/`libmlx5`/`libcuda`, so it adds **no new NEEDED
  shared libraries** to `libmscclpp`.
- **Build:** gated behind the `MSCCLPP_USE_GPUNETIO` CMake option. Host sources
  compile into the `mscclpp_gpunetio_obj` object library with
  `-DDOCA_VERBS_USE_NET_WRAPPER` (selects the dlopen ibverbs/mlx5dv wrappers).
  Device headers (`include/device/*.cuh`) are consumed only by the GPUNetIO
  PortChannel backend implementation
  (`include/mscclpp/internal/port_channel_gpunetio_device_impl.hpp`).

Do not hand-edit these files; re-import from upstream to update.
