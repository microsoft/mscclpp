// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NCCL_COMMON_HPP_
#define NCCL_COMMON_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/switch_channel.hpp>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#define __syncwarp() __builtin_amdgcn_wave_barrier()
#else
#define WARP_SIZE 32
#endif

constexpr int NUM_NVLS_CONNECTION = 16;
constexpr int NUM_SEMAPHORES = 64;

constexpr int MAX_NRANKS_PER_NODE = 8;

constexpr int SCRATCH_SIZE = 2 * 1024 * 1024 * 70;  // double buffer * 35 thread-blocks * 8 ranks * 256KB = 70MB
static bool mscclppDisableChannelCache = mscclpp::env()->disableChannelCache;

__device__ mscclpp::DeviceSyncer deviceSyncer;
__constant__ mscclpp::DeviceSemaphore deviceSemaphore[NUM_SEMAPHORES];

std::vector<mscclpp::RegisteredMemory> setupRemoteMemories(std::shared_ptr<mscclpp::Communicator> comm, int rank,
                                                           mscclpp::RegisteredMemory localMemory);

std::vector<mscclpp::MemoryChannel> setupMemoryChannels(
    const std::vector<mscclpp::Connection>& connections,
    const std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>& memorySemaphores,
    const std::vector<mscclpp::RegisteredMemory>& remoteMemories, mscclpp::RegisteredMemory localMemory,
    int nChannelsPerConnection);

std::vector<mscclpp::Connection> setupConnections(std::shared_ptr<mscclpp::Communicator> comm);

std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> setupMemorySemaphores(
    std::shared_ptr<mscclpp::Communicator> comm, const std::vector<mscclpp::Connection>& connections,
    int nChannelsPerConnection);

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::MemoryChannel>> setupMemoryChannelDeviceHandles(
    const std::vector<mscclpp::MemoryChannel>& memoryChannels);

std::vector<std::shared_ptr<mscclpp::NvlsConnection>> setupNvlsConnections(std::shared_ptr<mscclpp::Communicator> comm,
                                                                           size_t size, int numConnections);

std::vector<mscclpp::SwitchChannel> setupNvlsChannels(std::vector<std::shared_ptr<mscclpp::NvlsConnection>> conns,
                                                      void* buffer, size_t bufferSize, int nSwitchChannels);

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> setupNvlsChannelDeviceHandles(
    const std::vector<mscclpp::SwitchChannel>& nvlsChannels);

std::vector<mscclpp::BaseMemoryChannel> setupBaseMemoryChannels(
    const std::vector<mscclpp::Connection>& connections,
    const std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>& memorySemaphores,
    int nChannelsPerConnection);

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>> setupBaseMemoryChannelDeviceHandles(
    const std::vector<mscclpp::BaseMemoryChannel>& baseMemoryChannels);

#endif  // NCCL_COMMON_HPP_
