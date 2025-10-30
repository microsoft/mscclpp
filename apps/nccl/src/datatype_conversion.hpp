// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef DATATYPE_CONVERSION_HPP_
#define DATATYPE_CONVERSION_HPP_

#include <mscclpp/executor.hpp>
#include <nccl.h>

// Convert ncclDataType_t to mscclpp::DataType
inline mscclpp::DataType ncclDataTypeToMscclpp(ncclDataType_t dtype) {
  switch (dtype) {
    case ncclInt8:      // ncclChar is an alias for ncclInt8
      return mscclpp::DataType::INT8;
    case ncclUint8:
      return mscclpp::DataType::UINT8;
    case ncclInt32:     // ncclInt is an alias for ncclInt32
      return mscclpp::DataType::INT32;
    case ncclUint32:
      return mscclpp::DataType::UINT32;
    case ncclInt64:
      return mscclpp::DataType::INT64;
    case ncclUint64:
      return mscclpp::DataType::UINT64;
    case ncclFloat16:   // ncclHalf is an alias for ncclFloat16
      return mscclpp::DataType::FLOAT16;
    case ncclFloat32:   // ncclFloat is an alias for ncclFloat32
      return mscclpp::DataType::FLOAT32;
    case ncclFloat64:   // ncclDouble is an alias for ncclFloat64
      return mscclpp::DataType::FLOAT64;
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
    case mscclpp::DataType::INT8:
      return ncclInt8;
    case mscclpp::DataType::UINT8:
      return ncclUint8;
    case mscclpp::DataType::INT32:
      return ncclInt32;
    case mscclpp::DataType::UINT32:
      return ncclUint32;
    case mscclpp::DataType::INT64:
      return ncclInt64;
    case mscclpp::DataType::UINT64:
      return ncclUint64;
    case mscclpp::DataType::FLOAT16:
      return ncclFloat16;
    case mscclpp::DataType::FLOAT32:
      return ncclFloat32;
    case mscclpp::DataType::FLOAT64:
      return ncclFloat64;
    case mscclpp::DataType::BFLOAT16:
      return ncclBfloat16;
    case mscclpp::DataType::FP8_E4M3:
#ifdef __FP8_TYPES_EXIST__
      return ncclFloat8e4m3;
#else
      return ncclFloat32;  // fallback
#endif
    case mscclpp::DataType::FP8_E5M2:
#ifdef __FP8_TYPES_EXIST__
      return ncclFloat8e5m2;
#else
      return ncclFloat32;  // fallback
#endif
    default:
      return ncclFloat32;  // fallback
  }
}

#endif  // DATATYPE_CONVERSION_HPP_