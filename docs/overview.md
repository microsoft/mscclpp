# MSCCL++ Overview

MSCCL++ (Microsoft Collective Communication Library ++, pronounced *em-sickle-plus-plus*) is a GPU communication library that provides **multiple levels of abstraction** for writing high-performance distributed GPU applications.

- **Primitive API:** At the lowest level, MSCCL++ provides boilerplate-free C++ API (which we call *primitives*) for writing highly flexible GPU communication kernels.
- **DSL API:** Over the primitive layer, MSCCL++ provides a Python-based domain-specific language (DSL) that helps users quickly develop large-scale collective communication algorithms.
- **NCCL API:** At the highest level, MSCCL++ reimplements the NCCL API, allowing users to replace NCCL with MSCCL++ in their existing applications without any code changes.

