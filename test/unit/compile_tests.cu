// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../framework.hpp"

#undef NDEBUG
#ifndef DEBUG_BUILD
#define DEBUG_BUILD
#endif  // DEBUG_BUILD
#include <assert.h>

#include <mscclpp/memory_channel.hpp>
#include <mscclpp/poll_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/switch_channel.hpp>
#include <type_traits>
#include <utility>

TEST(CompileTest, Assert) { assert(true); }

namespace generic_handle_compile_test {
using Backend = mscclpp::PortChannelBackend;
enum class OtherBackend { Host };

// Compile-only regression: instantiate aliases inside a kernel, not just at namespace scope.
__global__ void kernelScopeHandleTypes() {
  __shared__ mscclpp::DeviceHandle<mscclpp::MemoryChannel> channels[1];
  static_assert(
      std::is_same_v<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>, mscclpp::BaseMemoryChannelDeviceHandle>);
  static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::SwitchChannel>, mscclpp::SwitchChannelDeviceHandle>);
  static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::BasePortChannel, Backend::Host>,
                               mscclpp::BasePortChannelDeviceHandle<Backend::Host>>);
  static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::BasePortChannel, Backend::Device>,
                               mscclpp::BasePortChannelDeviceHandle<Backend::Device>>);
  static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::PortChannel, Backend::Host>,
                               mscclpp::PortChannelDeviceHandle<Backend::Host>>);
  static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::PortChannel, Backend::Device>,
                               mscclpp::PortChannelDeviceHandle<Backend::Device>>);
  (void)channels;
}

template <class T, auto... Selectors>
concept HasAlias = requires { typename mscclpp::DeviceHandle<T, Selectors...>; };
template <class T, auto... Selectors>
concept HasExtraction = requires(T& owner) { mscclpp::deviceHandle<Selectors...>(owner); };
template <class T, auto... Selectors>
constexpr bool rejectsExtraction = !HasAlias<T, Selectors...> && !HasExtraction<T, Selectors...>;
template <class T, class Expected, auto... Selectors>
constexpr bool matchesExtraction =
    std::is_same_v<mscclpp::DeviceHandle<T, Selectors...>, Expected> &&
    std::is_same_v<decltype(mscclpp::deviceHandle<Selectors...>(std::declval<T&>())), Expected>;

static_assert(matchesExtraction<mscclpp::BaseMemoryChannel, mscclpp::BaseMemoryChannelDeviceHandle>);
static_assert(matchesExtraction<mscclpp::MemoryChannel, mscclpp::MemoryChannelDeviceHandle>);
static_assert(matchesExtraction<const mscclpp::MemoryChannel&, mscclpp::MemoryChannelDeviceHandle>);
static_assert(matchesExtraction<mscclpp::SwitchChannel, mscclpp::SwitchChannelDeviceHandle>);
static_assert(matchesExtraction<const mscclpp::SwitchChannel&&, mscclpp::SwitchChannelDeviceHandle>);
static_assert(
    matchesExtraction<mscclpp::BasePortChannel, mscclpp::BasePortChannelDeviceHandle<Backend::Host>, Backend::Host>);
static_assert(matchesExtraction<const mscclpp::BasePortChannel&, mscclpp::BasePortChannelDeviceHandle<Backend::Device>,
                                Backend::Device>);
static_assert(matchesExtraction<mscclpp::PortChannel, mscclpp::PortChannelDeviceHandle<Backend::Host>, Backend::Host>);
static_assert(matchesExtraction<const mscclpp::PortChannel&&, mscclpp::PortChannelDeviceHandle<Backend::Device>,
                                Backend::Device>);

template <class T>
constexpr bool rejectsFixedSelectors =
    rejectsExtraction<T, Backend::Host> && rejectsExtraction<T, Backend::Device> && rejectsExtraction<T, 0> &&
    rejectsExtraction<T, OtherBackend::Host> && rejectsExtraction<T, Backend::Host, Backend::Device>;
template <class T>
constexpr bool rejectsPortSelectors =
    rejectsExtraction<T> && rejectsExtraction<T, 0> && rejectsExtraction<T, true> &&
    rejectsExtraction<T, OtherBackend::Host> && rejectsExtraction<T, static_cast<Backend>(-1)> &&
    rejectsExtraction<T, static_cast<Backend>(2)> && rejectsExtraction<T, Backend::Host, Backend::Device>;
static_assert(rejectsFixedSelectors<mscclpp::BaseMemoryChannel>);
static_assert(rejectsFixedSelectors<mscclpp::MemoryChannel>);
static_assert(rejectsFixedSelectors<mscclpp::SwitchChannel>);
static_assert(rejectsPortSelectors<mscclpp::BasePortChannel>);
static_assert(rejectsPortSelectors<mscclpp::PortChannel>);
static_assert(rejectsExtraction<int> && rejectsExtraction<void>);
static_assert(rejectsExtraction<volatile mscclpp::MemoryChannel>);
static_assert(rejectsExtraction<volatile mscclpp::PortChannel, Backend::Host>);

// These fixtures test expression/cv/ref semantics, not production owner or GPU behavior.
struct FixedExpressionOwner {
  using DeviceHandle = void;  // The chosen expression, not this associated type, determines the result.
  int value;
  constexpr int& deviceHandle() & { return value; }
  constexpr const int& deviceHandle() const& { return value; }
  constexpr int&& deviceHandle() && { return std::move(value); }
};
struct SelectedExpressionOwner {
  int host;
  long device;
  template <Backend B>
    requires(B == Backend::Host || B == Backend::Device)
  constexpr auto& deviceHandle() & {
    if constexpr (B == Backend::Host)
      return host;
    else
      return device;
  }
  template <Backend B>
    requires(B == Backend::Host || B == Backend::Device)
  constexpr const auto& deviceHandle() const& {
    if constexpr (B == Backend::Host)
      return host;
    else
      return device;
  }
  template <Backend B>
    requires(B == Backend::Host || B == Backend::Device)
  constexpr auto&& deviceHandle() && {
    if constexpr (B == Backend::Host)
      return std::move(host);
    else
      return std::move(device);
  }
};
struct RvalueOnlyOwner {
  int deviceHandle() &&;
};
static_assert(matchesExtraction<FixedExpressionOwner, int&>);
static_assert(matchesExtraction<const FixedExpressionOwner&, const int&>);
static_assert(matchesExtraction<FixedExpressionOwner&&, int&>);
static_assert(std::is_same_v<decltype(mscclpp::deviceHandle(std::declval<FixedExpressionOwner&&>())), int&>);
static_assert(matchesExtraction<SelectedExpressionOwner, int&, Backend::Host>);
static_assert(matchesExtraction<SelectedExpressionOwner&&, long&, Backend::Device>);
static_assert(matchesExtraction<const SelectedExpressionOwner&, const long&, Backend::Device>);
static_assert(
    std::is_same_v<decltype(mscclpp::deviceHandle<Backend::Device>(std::declval<SelectedExpressionOwner&&>())), long&>);
static_assert(rejectsExtraction<RvalueOnlyOwner>);
static_assert(rejectsFixedSelectors<FixedExpressionOwner>);
static_assert(rejectsPortSelectors<SelectedExpressionOwner>);

void fixedExtraction(const mscclpp::BaseMemoryChannel& base, const mscclpp::MemoryChannel& memory,
                     const mscclpp::SwitchChannel& sw) {
  (void)mscclpp::deviceHandle(base);
  (void)mscclpp::deviceHandle(memory);
  (void)mscclpp::deviceHandle(sw);
}
}  // namespace generic_handle_compile_test

TEST(CompileTest, NamedLvalueExtraction) {
  using namespace generic_handle_compile_test;
  FixedExpressionOwner fixed{7};
  auto& fixedHandle = mscclpp::deviceHandle(std::move(fixed));
  EXPECT_TRUE(&fixedHandle == &fixed.value);
  const auto& constFixed = fixed;
  EXPECT_TRUE(&mscclpp::deviceHandle(constFixed) == &fixed.value);
  SelectedExpressionOwner selected{11, 13};
  EXPECT_TRUE(&mscclpp::deviceHandle<Backend::Host>(selected) == &selected.host);
  EXPECT_TRUE(&mscclpp::deviceHandle<Backend::Device>(std::move(selected)) == &selected.device);
  const auto& constSelected = selected;
  EXPECT_TRUE(&mscclpp::deviceHandle<Backend::Device>(constSelected) == &selected.device);
}

namespace port_channel_compile_test {
using Backend = mscclpp::PortChannelBackend;
using HostHandle = mscclpp::PortChannelDeviceHandle<>;
using DeviceHandle = mscclpp::PortChannelDeviceHandle<Backend::Device>;
using DeviceBase = mscclpp::BasePortChannelDeviceHandle<Backend::Device>;
using C = HostHandle::Concurrency;
constexpr HostHandle defaultHost;
constexpr DeviceHandle defaultDevice;
constexpr mscclpp::BasePortChannelDeviceHandle<> defaultHostBase;
constexpr DeviceBase defaultDeviceBase;
static_assert(defaultHost.src_ == 0 && defaultDevice.srcMemory_.addr == 0);
static_assert(defaultHostBase.semaphoreId_ == 0 && defaultDeviceBase.device_.qp == nullptr);
static_assert(!std::is_assignable_v<mscclpp::BasePortChannelDeviceHandle<>&, DeviceBase>);
static_assert(!std::is_assignable_v<DeviceBase&, mscclpp::BasePortChannelDeviceHandle<>>);

template <class Handle>
concept HasCombinedPut = requires(Handle channel) { channel.putWithAccumulate(0, 0, 8, 0, 1); };
template <class Handle>
concept HasMemoryIdPut = requires(Handle channel) {
  channel.put(mscclpp::MemoryId{}, uint64_t{}, mscclpp::MemoryId{}, uint64_t{}, uint64_t{});
};
static_assert(!HasCombinedPut<HostHandle> && HasCombinedPut<DeviceHandle>);
static_assert(HasMemoryIdPut<mscclpp::BasePortChannelDeviceHandle<>> && !HasMemoryIdPut<DeviceBase>);

__constant__ HostHandle constantHost;
__constant__ DeviceHandle constantDevice;
__constant__ mscclpp::BasePortChannelDeviceHandle<> constantHostBase;
__constant__ DeviceBase constantDeviceBase;

// Compiled but not launched by the test runner; these operations require live channels.
template <class Handle>
__global__ void portAlgorithm(Handle channel) {
  channel.putWithSignal(0, 8);
  channel.flush();
}
void explicitExtraction(const mscclpp::PortChannel& host, const mscclpp::PortChannel& device) {
  portAlgorithm<<<1, 1>>>(host.deviceHandle<Backend::Host>());
  portAlgorithm<<<1, 1>>>(device.deviceHandle<Backend::Device>());
}
void genericExtraction(const mscclpp::PortChannel& host, const mscclpp::PortChannel& device) {
  portAlgorithm<<<1, 1>>>(mscclpp::deviceHandle<Backend::Host>(host));
  portAlgorithm<<<1, 1>>>(mscclpp::deviceHandle<Backend::Device>(device));
}

template <C Mode, uint32_t Capacity>
__global__ void deviceCapabilities(DeviceBase base, DeviceHandle bound, mscclpp::DeviceMemoryHandle dst,
                                   mscclpp::DeviceMemoryHandle src) {
  base.put<Mode, Capacity>(dst, 0, src, 0, 8);
  base.putWithSignal<Mode, Capacity>(dst, src, 0, 8);
  bound.putWithAccumulate<Mode, Capacity>(0, 0, 8, 0, 1);
  bound.signal<Mode, Capacity>();
  bound.flush<Mode>();
}
template __global__ void deviceCapabilities<C::Exclusive, 0>(DeviceBase, DeviceHandle, mscclpp::DeviceMemoryHandle,
                                                             mscclpp::DeviceMemoryHandle);
template __global__ void deviceCapabilities<C::Exclusive, 8>(DeviceBase, DeviceHandle, mscclpp::DeviceMemoryHandle,
                                                             mscclpp::DeviceMemoryHandle);
template __global__ void deviceCapabilities<C::Block, 32>(DeviceBase, DeviceHandle, mscclpp::DeviceMemoryHandle,
                                                          mscclpp::DeviceMemoryHandle);
template __global__ void deviceCapabilities<C::Device, 1024>(DeviceBase, DeviceHandle, mscclpp::DeviceMemoryHandle,
                                                             mscclpp::DeviceMemoryHandle);
}  // namespace port_channel_compile_test
