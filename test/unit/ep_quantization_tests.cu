// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "../framework.hpp"
#include "common/quantization.cuh"

namespace {

constexpr int Mxfp8BlockSize = 32;
constexpr int VectorsPerBlock = Mxfp8BlockSize / mscclpp::bf16x8::Size;
constexpr uint16_t Bf16SignBit = 0x8000;
constexpr uint16_t Bf16MaxFiniteMagnitude = 0x7F7F;
constexpr uint16_t Bf16Fp8E4M3Max = 0x43E0;  // 448

__global__ void quantizeMxfp8Blocks(const mscclpp::bf16x8* input, mscclpp::f8_e4m3x8* output, uint8_t* scales,
                                    int numVectors) {
  const int vectorIdx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  // The four lanes of a block exit together because numVectors is a multiple of four.
  if (vectorIdx >= numVectors) return;
  uint8_t scale;
  output[vectorIdx] = mscclpp::ep::quantizeBf16x8ToMxfp8(input[vectorIdx], scale, mscclpp::ep::getLaneId());
  if (vectorIdx % VectorsPerBlock == 0) scales[vectorIdx / VectorsPerBlock] = scale;
}

__global__ void quantizeStaticFp8Vectors(const mscclpp::bf16x8* input, mscclpp::f8_e4m3x8* output, int numVectors,
                                         float scale) {
  const int vectorIdx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vectorIdx < numVectors) {
    output[vectorIdx] = mscclpp::ep::quantizeBf16x8ToStaticFp8(input[vectorIdx], scale);
  }
}

double bf16ToDouble(uint16_t bits) {
  const uint32_t floatBits = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &floatBits, sizeof(value));
  return value;
}

// Smallest scale byte s in [1, 254] with maxAbs <= 448 * 2^(s - 127); all-zero blocks use 127.
uint8_t referenceScale(double maxAbs) {
  if (maxAbs == 0.0) return 127;
  int scale = 1;
  while (scale < 254 && maxAbs > std::ldexp(448.0, scale - 127)) ++scale;
  return static_cast<uint8_t>(scale);
}

// Nearest finite E4M3 code with ties to the even code; magnitudes above 448 saturate.
uint8_t referenceFp8E4M3(double value) {
  static const std::array<double, 0x7F> codeValues = [] {
    std::array<double, 0x7F> values;
    for (int code = 0; code < 0x7F; ++code) {
      const int exponent = code >> 3;
      const int mantissa = code & 7;
      values[code] = exponent == 0 ? std::ldexp(mantissa, -9) : std::ldexp(8 + mantissa, exponent - 10);
    }
    return values;
  }();
  const double magnitude = std::fabs(value);
  const int upper =
      static_cast<int>(std::lower_bound(codeValues.begin(), codeValues.end(), magnitude) - codeValues.begin());
  int code = std::min(upper, 0x7E);
  if (upper > 0 && upper < 0x7F) {
    const double below = magnitude - codeValues[upper - 1];
    const double above = codeValues[upper] - magnitude;
    if (below < above || (below == above && (upper - 1) % 2 == 0)) code = upper - 1;
  }
  return static_cast<uint8_t>((std::signbit(value) ? 0x80 : 0) | code);
}

// Quantizes 32-element BF16 blocks on the GPU and compares every scale and payload byte with the host reference.
std::string checkMxfp8Quantization(const std::vector<uint16_t>& input) {
  const size_t numBlocks = input.size() / Mxfp8BlockSize;
  const int numVectors = static_cast<int>(numBlocks * VectorsPerBlock);
  mscclpp::GpuBuffer<uint16_t> deviceInput(input.size());
  mscclpp::GpuBuffer<uint8_t> deviceOutput(input.size());
  mscclpp::GpuBuffer<uint8_t> deviceScales(numBlocks);
  MSCCLPP_CUDATHROW(
      cudaMemcpy(deviceInput.data(), input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  constexpr int Threads = 256;
  quantizeMxfp8Blocks<<<(numVectors + Threads - 1) / Threads, Threads>>>(
      reinterpret_cast<const mscclpp::bf16x8*>(deviceInput.data()),
      reinterpret_cast<mscclpp::f8_e4m3x8*>(deviceOutput.data()), deviceScales.data(), numVectors);
  MSCCLPP_CUDATHROW(cudaGetLastError());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  std::vector<uint8_t> output(input.size());
  std::vector<uint8_t> scales(numBlocks);
  MSCCLPP_CUDATHROW(cudaMemcpy(output.data(), deviceOutput.data(), output.size(), cudaMemcpyDeviceToHost));
  MSCCLPP_CUDATHROW(cudaMemcpy(scales.data(), deviceScales.data(), scales.size(), cudaMemcpyDeviceToHost));

  size_t mismatches = 0;
  std::ostringstream firstMismatch;
  for (size_t block = 0; block < numBlocks; ++block) {
    const uint16_t* blockInput = input.data() + block * Mxfp8BlockSize;
    double maxAbs = 0.0;
    for (int element = 0; element < Mxfp8BlockSize; ++element) {
      maxAbs = std::max(maxAbs, std::fabs(bf16ToDouble(blockInput[element])));
    }
    const uint8_t scale = referenceScale(maxAbs);
    if (scales[block] != scale && mismatches++ == 0) {
      firstMismatch << "scale of block " << block << " (max " << maxAbs << "): expected " << int{scale} << ", got "
                    << int{scales[block]};
    }
    for (int element = 0; element < Mxfp8BlockSize; ++element) {
      const uint8_t expected = referenceFp8E4M3(std::ldexp(bf16ToDouble(blockInput[element]), 127 - scale));
      const uint8_t actual = output[block * Mxfp8BlockSize + element];
      if (actual != expected && mismatches++ == 0) {
        firstMismatch << "payload of block " << block << ", element " << element << " (bf16 0x" << std::hex
                      << blockInput[element] << ", scale " << std::dec << int{scale} << "): expected 0x" << std::hex
                      << int{expected} << ", got 0x" << int{actual};
      }
    }
  }
  return mismatches == 0 ? std::string() : std::to_string(mismatches) + " mismatches, first: " + firstMismatch.str();
}

std::string checkStaticFp8Quantization() {
  std::vector<uint16_t> input;
  input.reserve(static_cast<size_t>(Bf16MaxFiniteMagnitude + 1) * 2);
  for (uint32_t magnitude = 0; magnitude <= Bf16MaxFiniteMagnitude; ++magnitude) {
    input.push_back(static_cast<uint16_t>(magnitude));
    input.push_back(static_cast<uint16_t>(magnitude | Bf16SignBit));
  }

  const int numVectors = static_cast<int>(input.size() / mscclpp::bf16x8::Size);
  mscclpp::GpuBuffer<uint16_t> deviceInput(input.size());
  mscclpp::GpuBuffer<uint8_t> deviceOutput(input.size());
  MSCCLPP_CUDATHROW(
      cudaMemcpy(deviceInput.data(), input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  constexpr int Threads = 256;
  std::vector<uint8_t> output(input.size());
  for (const float scale : {0.125f, 0.75f, 1.0f}) {
    quantizeStaticFp8Vectors<<<(numVectors + Threads - 1) / Threads, Threads>>>(
        reinterpret_cast<const mscclpp::bf16x8*>(deviceInput.data()),
        reinterpret_cast<mscclpp::f8_e4m3x8*>(deviceOutput.data()), numVectors, scale);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    MSCCLPP_CUDATHROW(cudaMemcpy(output.data(), deviceOutput.data(), output.size(), cudaMemcpyDeviceToHost));

    size_t mismatches = 0;
    std::ostringstream firstMismatch;
    for (size_t element = 0; element < input.size(); ++element) {
      const float scaled = static_cast<float>(bf16ToDouble(input[element])) * scale;
      const uint8_t expected = referenceFp8E4M3(scaled);
      if (output[element] != expected && mismatches++ == 0) {
        firstMismatch << "element " << element << " (bf16 0x" << std::hex << input[element] << ", scale " << std::dec
                      << scale << "): expected 0x" << std::hex << int{expected} << ", got 0x"
                      << int{output[element]};
      }
    }
    if (mismatches != 0) {
      return std::to_string(mismatches) + " mismatches, first: " + firstMismatch.str();
    }
  }
  return {};
}

// One block per finite nonzero BF16 magnitude, which is the block maximum. A coprime stride scrambles the order so
// adjacent blocks have unrelated scales. The maximum moves across lanes and signs. Other elements have random signs
// and are up to 20 binades smaller, reaching E4M3 subnormals and underflow.
std::vector<uint16_t> everyMaximumBlocks() {
  constexpr uint32_t NumMagnitudes = Bf16MaxFiniteMagnitude;
  std::vector<uint16_t> input(static_cast<size_t>(NumMagnitudes) * Mxfp8BlockSize);
  uint32_t random = 12345;
  for (uint32_t block = 0; block < NumMagnitudes; ++block) {
    const uint32_t maxBits = 1 + block * 7919 % NumMagnitudes;
    const uint32_t maxElement = block % Mxfp8BlockSize;
    for (uint32_t element = 0; element < Mxfp8BlockSize; ++element) {
      random = random * 1664525u + 1013904223u;
      const uint32_t offset = element == maxElement ? 0 : (random >> 8) % (20 * 128 + 1);
      const bool negative = element == maxElement ? block % 2 == 1 : (random >> 31) != 0;
      const uint32_t bits = maxBits > offset ? maxBits - offset : 0;
      input[block * Mxfp8BlockSize + element] = static_cast<uint16_t>(bits | (negative ? Bf16SignBit : 0));
    }
  }
  return input;
}

// A +/-448 element pins every block's scale byte to 127. The remaining elements are every BF16 magnitude up to 448 with
// both signs, covering each E4M3 rounding tie, the subnormal range, and underflow to zero.
std::vector<uint16_t> unitScaleBlocks() {
  std::vector<uint16_t> values;
  for (uint32_t bits = 0; bits <= Bf16Fp8E4M3Max; ++bits) {
    values.push_back(static_cast<uint16_t>(bits));
    values.push_back(static_cast<uint16_t>(bits | Bf16SignBit));
  }
  std::vector<uint16_t> input;
  for (size_t next = 0; next < values.size();) {
    const size_t block = input.size() / Mxfp8BlockSize;
    for (size_t element = 0; element < Mxfp8BlockSize; ++element) {
      if (element == block % Mxfp8BlockSize) {
        input.push_back(block % 2 == 0 ? Bf16Fp8E4M3Max : Bf16Fp8E4M3Max | Bf16SignBit);
      } else {
        input.push_back(next < values.size() ? values[next++] : 0);
      }
    }
  }
  return input;
}

// Positive, negative, and mixed signed-zero blocks alternate with blocks near 2^31 (scale byte 150), so a maximum
// leaking from a neighbor would change the zero blocks' scale byte from 127.
std::vector<uint16_t> zeroBlocks() {
  constexpr uint16_t Bf16TwoPow31 = 0x4F00;
  std::vector<uint16_t> input;
  for (const uint16_t zeroSigns : {0x0000, 0xFFFF, 0xAAAA}) {
    for (int element = 0; element < Mxfp8BlockSize; ++element) {
      input.push_back(static_cast<uint16_t>(((zeroSigns >> (element % 16)) & 1) * Bf16SignBit));
    }
    for (int element = 0; element < Mxfp8BlockSize; ++element) {
      input.push_back(static_cast<uint16_t>((Bf16TwoPow31 + element) | (element % 3 == 0 ? Bf16SignBit : 0)));
    }
  }
  return input;
}

void requireSm90() {
  int device;
  int major;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  if (major < 9) SKIP_TEST() << "EP quantization requires SM90 or newer";
}

}  // namespace

TEST(EpQuantizationTest, Mxfp8EveryBf16Maximum) {
  requireSm90();
  const std::string error = checkMxfp8Quantization(everyMaximumBlocks());
  if (!error.empty()) FAIL() << error;
}

TEST(EpQuantizationTest, Mxfp8UnitScaleRounding) {
  requireSm90();
  const std::string error = checkMxfp8Quantization(unitScaleBlocks());
  if (!error.empty()) FAIL() << error;
}

TEST(EpQuantizationTest, Mxfp8ZeroBlocks) {
  requireSm90();
  const std::string error = checkMxfp8Quantization(zeroBlocks());
  if (!error.empty()) FAIL() << error;
}

TEST(EpQuantizationTest, StaticFp8EveryFiniteBf16Value) {
  requireSm90();
  const std::string error = checkStaticFp8Quantization();
  if (!error.empty()) FAIL() << error;
}
