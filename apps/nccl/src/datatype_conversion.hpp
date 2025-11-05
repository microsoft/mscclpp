// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef DATATYPE_CONVERSION_HPP_
#define DATATYPE_CONVERSION_HPP_

#include <mscclpp/executor.hpp>
#include <mscclpp/nccl.h>

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
      return mscclpp::DataType::FLOAT32;  // fallback
  }
}

// Convert mscclpp::DataType to ncclDataType_t
inline ncclDataType_t mscclppDataTypeToNccl(mscclpp::DataType dtype) {
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
      return ncclFloat32;  // fallback
  }
}

#endif  // DATATYPE_CONVERSION_HPP_