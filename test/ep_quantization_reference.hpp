// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_TEST_EP_QUANTIZATION_REFERENCE_HPP_
#define MSCCLPP_TEST_EP_QUANTIZATION_REFERENCE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mscclpp {
namespace test {

// Smallest scale byte s in [1, 254] with maxAbs <= 448 * 2^(s - 127); all-zero blocks use 127.
inline uint8_t referenceScale(double maxAbs) {
  if (maxAbs == 0.0) return 127;
  int scale = 1;
  while (scale < 254 && maxAbs > std::ldexp(448.0, scale - 127)) ++scale;
  return static_cast<uint8_t>(scale);
}

inline double referenceFp8E4M3Value(uint8_t code) {
  const int magnitude = code & 0x7F;
  if (magnitude == 0x7F) return std::numeric_limits<double>::quiet_NaN();
  const int exponent = magnitude >> 3;
  const int mantissa = magnitude & 7;
  const double value = exponent == 0 ? std::ldexp(mantissa, -9) : std::ldexp(8 + mantissa, exponent - 10);
  return code & 0x80 ? -value : value;
}

// Nearest finite E4M3 code with ties to the even code; magnitudes above 448 saturate.
inline uint8_t referenceFp8E4M3(double value) {
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

}  // namespace test
}  // namespace mscclpp

#endif  // MSCCLPP_TEST_EP_QUANTIZATION_REFERENCE_HPP_
