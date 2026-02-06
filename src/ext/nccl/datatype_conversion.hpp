// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_DATATYPE_CONVERSION_HPP_
#define MSCCLPP_DATATYPE_CONVERSION_HPP_

#include <mscclpp/ext/nccl/nccl.h>

#include <cassert>
#include <cstddef>
#include <mscclpp/gpu_data_types.hpp>

// Convert ncclDataType_t to mscclpp::DataType
inline mscclpp::DataType ncclDataTypeToMscclpp(ncclDataType_t dtype) {
  switch (dtype) {
    case ncclInt32:
      return mscclpp::DataType::INT32;
    case ncclUint32:
      return mscclpp::DataType::UINT32;
    case ncclFloat16:
      return mscclpp::DataType::FLOAT16;
    case ncclFloat32:
      return mscclpp::DataType::FLOAT32;
    case ncclBfloat16:
      return mscclpp::DataType::BFLOAT16;
#ifdef __FP8_TYPES_EXIST__
    case ncclFloat8e4m3:
      return mscclpp::DataType::FP8_E4M3;
    case ncclFloat8e5m2:
      return mscclpp::DataType::FP8_E5M2;
#endif
    default:
      throw mscclpp::Error("Unsupported ncclDataType_t: " + std::to_string(dtype), mscclpp::ErrorCode::InvalidUsage);
  }
}

// Get the size in bytes of a data type
inline size_t getDataTypeSize(mscclpp::DataType dtype) {
  switch (dtype) {
    case mscclpp::DataType::FP8_E4M3:
    case mscclpp::DataType::FP8_E5M2:
      return 1;
    case mscclpp::DataType::FLOAT16:
    case mscclpp::DataType::BFLOAT16:
      return 2;
    case mscclpp::DataType::INT32:
    case mscclpp::DataType::UINT32:
    case mscclpp::DataType::FLOAT32:
      return 4;
    default:
      return 0;
  }
}

static inline ncclDataType_t mscclppToNcclDataType(mscclpp::DataType dtype) {
  switch (dtype) {
    case mscclpp::DataType::INT32:
      return ncclInt32;
    case mscclpp::DataType::UINT32:
      return ncclUint32;
    case mscclpp::DataType::FLOAT16:
      return ncclFloat16;
    case mscclpp::DataType::FLOAT32:
      return ncclFloat32;
    case mscclpp::DataType::BFLOAT16:
      return ncclBfloat16;
#ifdef __FP8_TYPES_EXIST__
    case mscclpp::DataType::FP8_E4M3:
      return ncclFloat8e4m3;
    case mscclpp::DataType::FP8_E5M2:
      return ncclFloat8e5m2;
#endif
    default:
      assert(false && "Unsupported mscclpp::DataType");
      return ncclNumTypes;
  }
}

inline mscclpp::ReduceOp ncclRedOpToMscclpp(ncclRedOp_t op) {
  switch (op) {
    case ncclSum:
      return mscclpp::ReduceOp::SUM;
    case ncclMin:
      return mscclpp::ReduceOp::MIN;
    default:
      return mscclpp::ReduceOp::NOP;
  }
}

#endif  // MSCCLPP_DATATYPE_CONVERSION_HPP_