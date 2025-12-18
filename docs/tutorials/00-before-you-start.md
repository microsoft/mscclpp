# Before You Start

This tutorial introduces how to use the MSCCL++ Primitive API to write highly flexible and optimized GPU communication kernels from the lowest level. If you are looking for the high-level APIs, please refer to the [DSL API](../guide/mscclpp-dsl.md) or the [NCCL API](../quickstart.md#nccl-benchmark).

## Hardware Requirements

To run example code in this tutorial, you may need a system with at least two NVIDIA or AMD GPUs. For multi-node examples, you will need RDMA Network Interface Cards (NICs) and a network setup that allows communication between nodes. See the {ref}`prerequisites` for details.

## Environment Setup

We provide {ref}`docker-images` and a {ref}`vscode-dev-container` to simplify the environment setup.

## Prior Knowledge

This tutorial assumes that readers have a basic understanding of C++ and GPU programming (CUDA). If you are unfamiliar with the following concepts, we recommend reviewing the relevant documentation or tutorials:
- **C++ Basics:** STL containers, smart pointers, templates, futures, etc.
- **CUDA Basics:** thread blocks, warps, shared memory, etc.
- **(Optional) RDMA Basics:** If you are interested in multi-node communication, understanding RDMA concepts (`ibverbs` library) will be helpful.

In the next page, we will introduce a few basic concepts of the MSCCL++ Primitive API by a simple ping-pong example between two GPUs.
