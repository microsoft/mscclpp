// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <vector>

#include "../framework.hpp"

namespace {

constexpr int kConversionPaths = 3;

template <class T, class... Args>
std::array<T, sizeof...(Args)> makeArray(Args... args) {
  return {static_cast<T>(args)...};
}

__device__ uint32_t floatToBitsDevice(float value) {
  union {
    float f;
    uint32_t u;
  } cvt = {value};
  return cvt.u;
}

uint32_t floatToBitsHost(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

__global__ void kernelE4m3b15TypeConvert(const float* input, int encodeCases, const uint8_t* raw, int decodeCases,
                                         uint8_t* encoded, uint32_t* decodedBits) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  for (int offset = 0; offset < encodeCases; offset += 4) {
    mscclpp::f32x4 inputX4;
    for (int i = 0; i < 4; ++i) {
      inputX4.data[i] = input[offset + i];
    }

    mscclpp::f8_e4m3b15x4 encodedX4 = mscclpp::to<mscclpp::f8_e4m3b15x4>(inputX4);
    for (int i = 0; i < 4; ++i) {
      encoded[offset + i] = encodedX4.data[i].__x;
    }

    for (int pair = 0; pair < 2; ++pair) {
      mscclpp::f32x2 inputX2;
      inputX2.data[0] = input[offset + pair * 2];
      inputX2.data[1] = input[offset + pair * 2 + 1];
      mscclpp::f8_e4m3b15x2 encodedX2 = mscclpp::to<mscclpp::f8_e4m3b15x2>(inputX2);
      encoded[encodeCases + offset + pair * 2] = encodedX2.data[0].__x;
      encoded[encodeCases + offset + pair * 2 + 1] = encodedX2.data[1].__x;
    }
  }

  for (int i = 0; i < encodeCases; ++i) {
    encoded[2 * encodeCases + i] = __fp8_e4m3b15(input[i]).__x;
  }

  for (int offset = 0; offset < decodeCases; offset += 4) {
    mscclpp::f8_e4m3b15x4 rawX4;
    for (int i = 0; i < 4; ++i) {
      rawX4.data[i] = __fp8_e4m3b15::fromRaw(raw[offset + i]);
    }

    mscclpp::f32x4 decodedX4 = mscclpp::to<mscclpp::f32x4>(rawX4);
    for (int i = 0; i < 4; ++i) {
      decodedBits[offset + i] = floatToBitsDevice(decodedX4.data[i]);
    }

    for (int pair = 0; pair < 2; ++pair) {
      mscclpp::f8_e4m3b15x2 rawX2;
      rawX2.data[0] = __fp8_e4m3b15::fromRaw(raw[offset + pair * 2]);
      rawX2.data[1] = __fp8_e4m3b15::fromRaw(raw[offset + pair * 2 + 1]);
      mscclpp::f32x2 decodedX2 = mscclpp::to<mscclpp::f32x2>(rawX2);
      decodedBits[decodeCases + offset + pair * 2] = floatToBitsDevice(decodedX2.data[0]);
      decodedBits[decodeCases + offset + pair * 2 + 1] = floatToBitsDevice(decodedX2.data[1]);
    }
  }

  for (int i = 0; i < decodeCases; ++i) {
    decodedBits[2 * decodeCases + i] = floatToBitsDevice(float(__fp8_e4m3b15::fromRaw(raw[i])));
  }
}

}  // namespace

TEST(GpuDataTypesTest, E4m3b15TypeConvert) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float maxFloat = std::numeric_limits<float>::max();

  // Each input value maps to the byte at the same index in expectedEncoded. The fp8_e4m3b15 format has no
  // NaN/Inf encoding, so NaN, Inf, and overflow inputs saturate to +/-1.875 (max byte 0x7f/0xff).
  const auto input = makeArray<float>(0.0f, -0.0f,                // +/-0
                                      0x1.0p-19f, -0x1.0p-19f,    // +/-2^-19: underflows to signed 0
                                      0x1.0p-18f, -0x1.0p-18f,    // +/-2^-18: rounds to min subnormal
                                      0x1.0p-17f, -0x1.0p-17f,    // +/-2^-17: min subnormal
                                      0x1.0p-14f, -0x1.0p-14f,    // +/-2^-14: min normal
                                      0x1.0fcp-2f, -0x1.0fcp-2f,  // Boundary rounds down in magnitude
                                      0x1.0fep-2f, -0x1.0fep-2f,  // Boundary rounds up in magnitude
                                      0x1.cfep-2f, -0x1.cfep-2f,  // Boundary rounds to +/-0.46875
                                      0x1.cp0f, -0x1.cp0f,        // +/-1.75: max finite
                                      2.0f, -2.0f,                // Overflow saturation
                                      inf, -inf,                  // +/-Inf saturation
                                      nan, -maxFloat);            // NaN / large negative saturation

  const auto expectedEncoded = makeArray<uint8_t>(0x00, 0x80,   // +/-0
                                                  0x00, 0x80,   // Underflow to signed zero
                                                  0x01, 0x81,   // Round to min signed subnormal
                                                  0x01, 0x81,   // Min signed subnormal
                                                  0x08, 0x88,   // Min signed normal
                                                  0x68, 0xe8,   // Boundary rounds to +/-0.25
                                                  0x69, 0xe9,   // Boundary rounds to +/-0.28125
                                                  0x6f, 0xef,   // Boundary rounds to +/-0.46875
                                                  0x7e, 0xfe,   // Max finite at fp16 grid (1.75)
                                                  0x7f, 0xff,   // Overflow saturation (1.875)
                                                  0x7f, 0xff,   // Inf saturation (1.875)
                                                  0x7f, 0xff);  // NaN / large negative saturation (1.875)

  // Raw bytes to decode, with expectedDecoded giving the exact float value at the same index.
  const auto raw = makeArray<uint8_t>(0x00, 0x80,                         // +/-0
                                      0x01, 0x81,                         // +/-2^-17: min subnormal
                                      0x08, 0x88,                         // +/-2^-14: min normal
                                      0x68, 0xe8,                         // +/-0.25
                                      0x69, 0xe9,                         // +/-0.28125
                                      0x7e, 0xfe);                        // +/-1.75: max finite
  const auto expectedDecoded = makeArray<float>(0.0f, -0.0f,              // +/-0
                                                0x1.0p-17f, -0x1.0p-17f,  // +/-2^-17: min subnormal
                                                0x1.0p-14f, -0x1.0p-14f,  // +/-2^-14: min normal
                                                0x1.0p-2f, -0x1.0p-2f,    // +/-0.25
                                                0x1.2p-2f, -0x1.2p-2f,    // +/-0.28125
                                                0x1.cp0f, -0x1.cp0f);     // +/-1.75: max finite

  ASSERT_EQ(input.size(), expectedEncoded.size());
  ASSERT_EQ(raw.size(), expectedDecoded.size());
  ASSERT_EQ(input.size() % 4, size_t(0));
  ASSERT_EQ(raw.size() % 4, size_t(0));

  auto inputDev = mscclpp::detail::gpuCallocShared<float>(input.size());
  auto rawDev = mscclpp::detail::gpuCallocShared<uint8_t>(raw.size());
  auto encodedDev = mscclpp::detail::gpuCallocShared<uint8_t>(input.size() * kConversionPaths);
  auto decodedBitsDev = mscclpp::detail::gpuCallocShared<uint32_t>(raw.size() * kConversionPaths);

  mscclpp::gpuMemcpy(inputDev.get(), input.data(), input.size(), cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy(rawDev.get(), raw.data(), raw.size(), cudaMemcpyHostToDevice);

  kernelE4m3b15TypeConvert<<<1, 1>>>(inputDev.get(), static_cast<int>(input.size()), rawDev.get(),
                                     static_cast<int>(raw.size()), encodedDev.get(), decodedBitsDev.get());
  MSCCLPP_CUDATHROW(cudaGetLastError());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  std::vector<uint8_t> encoded(input.size() * kConversionPaths);
  std::vector<uint32_t> decodedBits(raw.size() * kConversionPaths);
  mscclpp::gpuMemcpy(encoded.data(), encodedDev.get(), encoded.size(), cudaMemcpyDeviceToHost);
  mscclpp::gpuMemcpy(decodedBits.data(), decodedBitsDev.get(), decodedBits.size(), cudaMemcpyDeviceToHost);

  for (int path = 0; path < kConversionPaths; ++path) {
    for (size_t i = 0; i < input.size(); ++i) {
      EXPECT_EQ(static_cast<int>(encoded[path * input.size() + i]), static_cast<int>(expectedEncoded[i]));
    }
  }

  for (int path = 0; path < kConversionPaths; ++path) {
    for (size_t i = 0; i < raw.size(); ++i) {
      EXPECT_EQ(decodedBits[path * raw.size() + i], floatToBitsHost(expectedDecoded[i]));
    }
  }
}
