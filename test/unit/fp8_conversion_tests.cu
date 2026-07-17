// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <vector>

#include "../framework.hpp"

template <typename Fp8T>
__global__ void scalarRoundtripKernel(const float* in, float* outFloat, uint8_t* outBits, int nElem) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nElem) return;
  Fp8T v(in[i]);
  outFloat[i] = static_cast<float>(v);
  outBits[i] = *reinterpret_cast<const uint8_t*>(&v);
}

template <typename Fp8VecT, typename FloatVecT>
__global__ void vectorRoundtripKernel(const float* in, float* outFloat, uint8_t* outBits) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  Fp8VecT enc = mscclpp::to<Fp8VecT, FloatVecT>(*reinterpret_cast<const FloatVecT*>(in));
  *reinterpret_cast<Fp8VecT*>(outBits) = enc;
  *reinterpret_cast<FloatVecT*>(outFloat) = mscclpp::to<FloatVecT, Fp8VecT>(enc);
}

template <typename Fp8T>
static void scalarRoundtrip(const std::vector<float>& inputs, std::vector<float>& outFloat,
                            std::vector<uint8_t>& outBits) {
  const int nElem = static_cast<int>(inputs.size());
  auto devIn = mscclpp::detail::gpuCallocUnique<float>(nElem);
  auto devOutFloat = mscclpp::detail::gpuCallocUnique<float>(nElem);
  auto devOutBits = mscclpp::detail::gpuCallocUnique<uint8_t>(nElem);

  mscclpp::gpuMemcpy<float>(devIn.get(), inputs.data(), nElem, cudaMemcpyHostToDevice);
  const int nThreads = 32;
  const int nBlocks = (nElem + nThreads - 1) / nThreads;
  scalarRoundtripKernel<Fp8T><<<nBlocks, nThreads>>>(devIn.get(), devOutFloat.get(), devOutBits.get(), nElem);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  outFloat.resize(nElem);
  outBits.resize(nElem);
  mscclpp::gpuMemcpy<float>(outFloat.data(), devOutFloat.get(), nElem, cudaMemcpyDeviceToHost);
  mscclpp::gpuMemcpy<uint8_t>(outBits.data(), devOutBits.get(), nElem, cudaMemcpyDeviceToHost);
}

template <typename Fp8VecT, typename FloatVecT>
static void vectorRoundtrip(const std::vector<float>& inputs, std::vector<float>& outFloat,
                            std::vector<uint8_t>& outBits) {
  constexpr int nElem = FloatVecT::Size;
  static_assert(Fp8VecT::Size == nElem, "vector sizes must match");

  auto devIn = mscclpp::detail::gpuCallocUnique<float>(nElem);
  auto devOutFloat = mscclpp::detail::gpuCallocUnique<float>(nElem);
  auto devOutBits = mscclpp::detail::gpuCallocUnique<uint8_t>(nElem);

  mscclpp::gpuMemcpy<float>(devIn.get(), inputs.data(), nElem, cudaMemcpyHostToDevice);
  vectorRoundtripKernel<Fp8VecT, FloatVecT><<<1, 1>>>(devIn.get(), devOutFloat.get(), devOutBits.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  outFloat.resize(nElem);
  outBits.resize(nElem);
  mscclpp::gpuMemcpy<float>(outFloat.data(), devOutFloat.get(), nElem, cudaMemcpyDeviceToHost);
  mscclpp::gpuMemcpy<uint8_t>(outBits.data(), devOutBits.get(), nElem, cudaMemcpyDeviceToHost);
}

// E4M3B15: software type, max finite = 0.9375 (exp==15 reserved for NaN).
constexpr float kE4M3B15Max = 0.9375f;
constexpr float kE4M3B15MinNormal = 6.103515625e-05f;       // 2^-14
constexpr float kE4M3B15MinSubnormal = 7.62939453125e-06f;  // 2^-17

TEST(Fp8ConversionTest, E4M3B15Scalar) {
  const std::vector<float> inputs = {
      0.0f,  -0.0f,  kE4M3B15Max, -kE4M3B15Max, kE4M3B15MinNormal, kE4M3B15MinSubnormal, 1.0f, 1e10f,
      -1.0f, -1e10f, 1e-10f,      1e-30f};
  const std::vector<float> expectedFloat = {0.0f,
                                            0.0f,
                                            kE4M3B15Max,
                                            -kE4M3B15Max,
                                            kE4M3B15MinNormal,
                                            kE4M3B15MinSubnormal,
                                            kE4M3B15Max,
                                            kE4M3B15Max,
                                            -kE4M3B15Max,
                                            -kE4M3B15Max,
                                            0.0f,
                                            0.0f};
  const std::vector<uint8_t> expectedBits = {0x00, 0x00, 0x77, 0xF7, 0x08, 0x01, 0x77, 0x77, 0xF7, 0xF7, 0x00, 0x00};

  std::vector<float> outFloat;
  std::vector<uint8_t> outBits;
  scalarRoundtrip<__fp8_e4m3b15>(inputs, outFloat, outBits);

  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(outFloat[i], expectedFloat[i]);
    EXPECT_EQ(outBits[i], expectedBits[i]);
  }
}

TEST(Fp8ConversionTest, E4M3B15ScalarNaN) {
  std::vector<float> outFloat;
  std::vector<uint8_t> outBits;
  scalarRoundtrip<__fp8_e4m3b15>({std::nanf("")}, outFloat, outBits);
  EXPECT_EQ(outBits[0], 0x80);
  EXPECT_TRUE(std::isnan(outFloat[0]));
}

TEST(Fp8ConversionTest, E4M3B15Vector) {
  const std::vector<float> inputs = {kE4M3B15Max,       -kE4M3B15Max,         1.0f,   -1e3f,
                                     kE4M3B15MinNormal, kE4M3B15MinSubnormal, 1e-20f, 0.0f};
  const std::vector<float> expectedFloat = {kE4M3B15Max,       -kE4M3B15Max,         kE4M3B15Max, -kE4M3B15Max,
                                            kE4M3B15MinNormal, kE4M3B15MinSubnormal, 0.0f,        0.0f};
  const std::vector<uint8_t> expectedBits = {0x77, 0xF7, 0x77, 0xF7, 0x08, 0x01, 0x00, 0x00};

  std::vector<float> outFloat;
  std::vector<uint8_t> outBits;
  vectorRoundtrip<mscclpp::f8_e4m3b15x2, mscclpp::f32x2>(inputs, outFloat, outBits);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(outFloat[i], expectedFloat[i]);
    EXPECT_EQ(outBits[i], expectedBits[i]);
  }
  vectorRoundtrip<mscclpp::f8_e4m3b15x4, mscclpp::f32x4>(inputs, outFloat, outBits);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(outFloat[i], expectedFloat[i]);
    EXPECT_EQ(outBits[i], expectedBits[i]);
  }
}

#if defined(__FP8_TYPES_EXIST__)

#if defined(MSCCLPP_DEVICE_HIP) && defined(HIP_VERSION_MAJOR) && \
    ((HIP_VERSION_MAJOR == 6) || (HIP_VERSION_MAJOR > 6 && HIP_FP8_TYPE_FNUZ && !HIP_FP8_TYPE_OCP))
// E4M3FNUZ: bias=8, max = 240, single zero, single NaN.
constexpr float kE4M3Max = 240.0f;
constexpr float kE4M3MinNormal = 7.8125e-3f;       // 2^-7
constexpr float kE4M3MinSubnormal = 9.765625e-4f;  // 2^-10
constexpr bool kE4M3IsFnuz = true;
#else
// E4M3FN: bias=7, max = 448, signed zero, NaN at S.1111.111.
constexpr float kE4M3Max = 448.0f;
constexpr float kE4M3MinNormal = 1.5625e-2f;       // 2^-6
constexpr float kE4M3MinSubnormal = 1.953125e-3f;  // 2^-9
constexpr bool kE4M3IsFnuz = false;
#endif

TEST(Fp8ConversionTest, E4M3Scalar) {
  // The float ctor uses sat-to-finite mode: overflow saturates to ±max-finite,
  // not NaN. (This differs from torch.float8_e4m3fn's default cast behavior.)
  const std::vector<float> inputs = {0.0f,  kE4M3Max, -kE4M3Max, kE4M3MinNormal, kE4M3MinSubnormal, kE4M3Max * 2.0f,
                                     1e10f, -1e10f,   1e-10f,    1e-20f};
  const std::vector<float> expectedFloat = {0.0f,     kE4M3Max, -kE4M3Max, kE4M3MinNormal, kE4M3MinSubnormal,
                                            kE4M3Max, kE4M3Max, -kE4M3Max, 0.0f,           0.0f};

  std::vector<float> outFloat;
  std::vector<uint8_t> outBits;
  scalarRoundtrip<__fp8_e4m3>(inputs, outFloat, outBits);
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(outFloat[i], expectedFloat[i]);
  }

  // FNUZ has only one zero: -0.0 must encode to 0x00.
  if (kE4M3IsFnuz) {
    scalarRoundtrip<__fp8_e4m3>({-0.0f}, outFloat, outBits);
    EXPECT_EQ(outBits[0], 0x00);
  }
}

// Vector conversions via mscclpp::to<>. Specializations exist at x2/x4; the
// primary template auto-decomposes N>4 into x4 chunks, so x8/x16 reuse the
// same leaf converters. We cross-check encoded bytes against the scalar path.
template <typename Fp8VecT, typename FloatVecT>
static void checkE4M3Vector(const std::vector<float>& inputs, const std::vector<float>& expectedFloat) {
  constexpr int nElem = FloatVecT::Size;
  std::vector<float> outFloat, scalarFloat;
  std::vector<uint8_t> outBits, scalarBits;
  vectorRoundtrip<Fp8VecT, FloatVecT>(inputs, outFloat, outBits);
  scalarRoundtrip<__fp8_e4m3>(inputs, scalarFloat, scalarBits);
  for (int i = 0; i < nElem; ++i) {
    EXPECT_EQ(outFloat[i], expectedFloat[i]);
    EXPECT_EQ(outBits[i], scalarBits[i]);
  }
}

TEST(Fp8ConversionTest, E4M3Vector) {
  const std::vector<float> inputs = {
      kE4M3Max, -kE4M3Max, kE4M3Max * 2.0f, -1e10f, kE4M3MinNormal, kE4M3MinSubnormal, 1e-20f, 0.0f,
      kE4M3Max, -kE4M3Max, kE4M3Max * 2.0f, -1e10f, kE4M3MinNormal, kE4M3MinSubnormal, 1e-20f, 0.0f};
  const std::vector<float> expectedFloat = {
      kE4M3Max, -kE4M3Max, kE4M3Max, -kE4M3Max, kE4M3MinNormal, kE4M3MinSubnormal, 0.0f, 0.0f,
      kE4M3Max, -kE4M3Max, kE4M3Max, -kE4M3Max, kE4M3MinNormal, kE4M3MinSubnormal, 0.0f, 0.0f};
  checkE4M3Vector<mscclpp::f8_e4m3x2, mscclpp::f32x2>(inputs, expectedFloat);
  checkE4M3Vector<mscclpp::f8_e4m3x4, mscclpp::f32x4>(inputs, expectedFloat);
  checkE4M3Vector<mscclpp::f8_e4m3x8, mscclpp::VectorType<float, 8>>(inputs, expectedFloat);
  checkE4M3Vector<mscclpp::f8_e4m3x16, mscclpp::VectorType<float, 16>>(inputs, expectedFloat);
}

#endif  // __FP8_TYPES_EXIST__
