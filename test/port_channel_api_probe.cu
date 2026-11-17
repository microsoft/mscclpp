// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Also compile directly against an installed prefix, with no USE_IBVERBS/GPUNetIO definitions.
#include <cstdio>
#include <cstdlib>
#include <mscclpp/fifo.hpp>
#include <mscclpp/port_channel.hpp>
#include <type_traits>

using Backend = mscclpp::PortChannelBackend;
using Host = mscclpp::PortChannelDeviceHandle;
using Device = mscclpp::PortChannelDeviceHandleT<Backend::Device>;
using DeviceBase = mscclpp::BasePortChannelDeviceHandleT<Backend::Device>;
using C = Host::Concurrency;
static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::PortChannel>, Host>);
static_assert(std::is_same_v<decltype(std::declval<mscclpp::PortChannel>().deviceBackendHandle()), Device>);
static_assert(std::is_trivially_copyable_v<Host> && std::is_trivially_copyable_v<Device>);
static_assert(!std::is_assignable_v<mscclpp::BasePortChannelDeviceHandle&, DeviceBase>);
static_assert(!std::is_assignable_v<DeviceBase&, mscclpp::BasePortChannelDeviceHandle>);
constexpr Device defaultPortHandle;
__constant__ Device constantPortHandle;

template <class T>
concept ExposesBackend = requires(T t) { t.device_; } || requires(T t) { t.state_; };
template <class T>
concept ExposesDescriptors = requires(T t) { t.dstMemory_; } || requires(T t) { t.bound_; };
static_assert(!ExposesBackend<Host> && !ExposesBackend<Device>);
static_assert(!ExposesDescriptors<Host> && !ExposesDescriptors<Device>);

// One algorithm source, distinct concrete kernel instantiations and explicit host extraction.
template <class Handle>
__global__ void portTutorial(Handle channel, size_t bytes) {
  if (threadIdx.x != 0) return;
  channel.signal();
  channel.wait();
  channel.putWithSignal(0, 0, bytes);
  channel.wait();
  channel.flush();
}

void deviceTutorialFlow(mscclpp::Communicator& comm, const mscclpp::Connection& connection, int peer,
                        mscclpp::RegisteredMemory remote, mscclpp::RegisteredMemory local, size_t bytes) {
  auto sema = comm.buildSemaphore(connection, peer).get();
  mscclpp::PortChannel channel(sema, remote, local);
  portTutorial<<<1, 1>>>(channel.deviceBackendHandle(), bytes);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
}

void hostTutorialFlow(mscclpp::Communicator& comm, const mscclpp::Connection& connection, int peer,
                      mscclpp::RegisteredMemory remote, mscclpp::RegisteredMemory local, size_t bytes) {
  auto sema = comm.buildSemaphore(connection, peer).get();
  mscclpp::ProxyService proxy;
  auto sid = proxy.addSemaphore(sema);
  auto src = proxy.addMemory(local);
  auto dst = proxy.addMemory(remote);
  auto channel = proxy.portChannel(sid, dst, src);
  proxy.startProxy();
  portTutorial<<<1, 1>>>(channel.deviceHandle(), bytes);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  proxy.stopProxy();
}

template <C Mode, uint32_t Slots>
__global__ void operationScratch(Device channel) {
  channel.signal<Mode, Slots>();
  channel.flush<Mode, Slots>();
}
template __global__ void operationScratch<C::Exclusive, 0>(Device);
template __global__ void operationScratch<C::Exclusive, 8>(Device);
template __global__ void operationScratch<C::Exclusive, 32>(Device);
template __global__ void operationScratch<C::Block, 32>(Device);
template __global__ void operationScratch<C::Device, 1024>(Device);
#ifdef PROBE_INVALID_SLOTS
template __global__ void operationScratch<C::Device, 1025>(Device);
#endif

// Each macro selects an independent expected compilation failure.
#ifdef PROBE_HOST_CONCURRENCY
__global__ void invalidHostConcurrency(Host channel) { channel.signal<C::Exclusive>(); }
#endif
#ifdef PROBE_HOST_BLOCK
__global__ void invalidHostBlock(Host channel) { channel.flush<C::Block>(); }
#endif
#ifdef PROBE_HOST_ACCUMULATE
__global__ void invalidHostCapability(Host channel) { channel.putWithAccumulate(0, 0, 8, 0, 1); }
#endif
#ifdef PROBE_DEVICE_IDS
__global__ void invalidDeviceCapability(DeviceBase channel) { channel.put(0, 0, 0, 0, 8); }
#endif
#ifdef PROBE_CROSS_BACKEND
void invalidBackendAssignment(mscclpp::BasePortChannelDeviceHandle& host, DeviceBase device) { host = device; }
#endif

// Optional isolated negative executions, never run within the unit-test CUDA context.
// 0: default Host; 1: default Device; 2: Device on an IB-off installation; 3: Host base-only assignment.
#ifdef PROBE_PORT_FAILURE
static_assert(PROBE_PORT_FAILURE >= 0 && PROBE_PORT_FAILURE <= 3);
#if PROBE_PORT_FAILURE == 0 || PROBE_PORT_FAILURE == 3
using FailureHandle = Host;
#else
using FailureHandle = Device;
#endif
#if PROBE_PORT_FAILURE == 3
#ifndef PROBE_HOST_BOUND_OPERATION
#define PROBE_HOST_BOUND_OPERATION 0
#endif
__global__ void expectedPortFailure(FailureHandle channel) {
#if PROBE_HOST_BOUND_OPERATION == 0
  channel.put(0, 0, 8);
#elif PROBE_HOST_BOUND_OPERATION == 1
  channel.putWithSignal(0, 0, 8);
#elif PROBE_HOST_BOUND_OPERATION == 2
  channel.putWithSignalAndFlush(0, 0, 8);
#elif PROBE_HOST_BOUND_OPERATION == 3
  channel.accumulate(0, 1);
#else
#error "Unknown Host bound-operation probe"
#endif
}
#else
__global__ void expectedPortFailure(FailureHandle channel) { channel.signal(); }
#endif
[[noreturn]] void finishPortFailure(int status) {
  // These are isolated failure executables. A trapped context cannot safely run FIFO CUDA destructors.
  std::fflush(stderr);
  std::_Exit(status);
}
int main() {
#if PROBE_PORT_FAILURE == 2
  static_assert(!MSCCLPP_HAS_GPUNETIO, "Run the unavailable-Device probe against the IB-off installation");
#endif
  FailureHandle handle;
#if PROBE_PORT_FAILURE == 3
  // A usable posting FIFO and completed flush counter ensure missing bound checks cannot hide behind base errors.
  mscclpp::Fifo fifo(16);
  uint64_t* done = nullptr;
  MSCCLPP_CUDATHROW(cudaMalloc(reinterpret_cast<void**>(&done), sizeof(uint64_t)));
  MSCCLPP_CUDATHROW(cudaMemset(done, 0xff, sizeof(uint64_t)));
  static_cast<mscclpp::BasePortChannelDeviceHandle&>(handle) =
      mscclpp::BasePortChannelDeviceHandle(0, {}, fifo.deviceHandle(), done);
#endif
  expectedPortFailure<<<1, 1>>>(handle);
  if (cudaGetLastError() != cudaSuccess) finishPortFailure(2);
  auto result = cudaDeviceSynchronize();
  if (result == cudaSuccess) finishPortFailure(1);
  std::fprintf(stderr, "Expected PortChannel failure: %s\n", cudaGetErrorString(result));
#if defined(MSCCLPP_DEVICE_CUDA)
  if (result != cudaErrorIllegalInstruction && result != cudaErrorLaunchFailure) finishPortFailure(3);
#endif
  finishPortFailure(0);
}
#endif
