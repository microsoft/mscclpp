// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_DATATYPE_HPP_
#define MSCCLPP_DATATYPE_HPP_

#include <mscclpp/executor.hpp>
#include <cstddef>

namespace mscclpp {

// Get the size in bytes of a data type
inline size_t getDataTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::INT8:
    case DataType::UINT8:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
      return 1;
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
      return 2;
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::FLOAT32:
      return 4;
    case DataType::INT64:
    case DataType::UINT64:
    case DataType::FLOAT64:
      return 8;
    default:
      return 0;
  }
}

}  // namespace mscclpp

#endif  // MSCCLPP_DATATYPE_HPP_