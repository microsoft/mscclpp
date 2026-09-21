// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_JIT_HPP_
#define MSCCLPP_EXT_MEGAMOE_JIT_HPP_

#include <stddef.h>
#include <stdint.h>

// Internal, versioned C ABI. No C++ objects, CUDA headers, ownership transfer of
// caller buffers, or exceptions cross the module boundary. Changing any layout
// or its semantics requires a new ABI version and exported entrypoint.
// Compile megamoe.cu, megamoe_launch.cu, and megamoe_jit.cu (not runtime.cc)
// with C++20, PIC/hidden visibility,
// -gencode=arch=compute_100a,code=sm_100a, --expt-relaxed-constexpr,
// --expt-extended-lambda, MSCCLPP_USE_CUDA, MSCCLPP_MEGAMOE_JIT_MODULE=1,
// the three specialization definitions, and a quoted MSCCLPP_MEGAMOE_JIT_ID.
// Link the module against mscclpp, cudart, and the CUDA driver.
#define MSCCLPP_MEGAMOE_JIT_ABI_VERSION 1
#define MSCCLPP_MEGAMOE_JIT_ENTRYPOINT "mscclpp_megamoe_jit_get_api_v1"
#define MSCCLPP_MEGAMOE_JIT_ID_CAPACITY 65

typedef struct MegaMoeJitConfigV1 {
  int32_t rank;
  int32_t worldSize;
  int32_t maxTokens;
  int32_t hidden;
  int32_t intermediate;
  int32_t numExperts;
  int32_t topK;
  int32_t smMargin;
  int32_t weightE5M2;
  float gateUpClamp;
} MegaMoeJitConfigV1;

typedef struct MegaMoeJitWeightsV1 {
  uint8_t* fc1;
  uint8_t* fc1Scale;
  uint8_t* fc2;
  uint8_t* fc2Scale;
} MegaMoeJitWeightsV1;

typedef struct MegaMoeJitLayoutV1 {
  uint64_t symmetricBytes;
  uint64_t input;
  uint64_t topkIds;
  uint64_t topkWeights;
  uint64_t partialOutput;
  uint64_t epoch;
  uint64_t peerSignals;
  uint64_t expectedPeerSignals;
  uint64_t tokenCount;
  uint64_t privateBytes;
  uint64_t sharedBytes;
  int32_t ctaCount;
  int32_t reserved;
} MegaMoeJitLayoutV1;

// Status 0 is success, 1 is invalid input/configuration, 2 is a runtime error.
// On failure, callbacks write a NUL-terminated diagnostic to the caller's error
// buffer when its capacity is nonzero. A failed create leaves *plan == NULL.
// Preflight checks device support and cluster occupancy without allocating GPU
// workspace. Streams are CUDA cudaStream_t values passed as opaque pointers.
typedef struct MegaMoeJitApiV1 {
  uint32_t abiVersion;
  uint32_t structBytes;
  uint32_t configBytes;
  uint32_t weightsBytes;
  uint32_t layoutBytes;
  int32_t tileM;
  int32_t tileN;
  int32_t tileK;
  int32_t loadStages;
  int32_t transformStages;
  int32_t clusterSize;
  int32_t accumulatorStages;
  int32_t architecture;
  char kernelId[MSCCLPP_MEGAMOE_JIT_ID_CAPACITY];
  int (*preflight)(const MegaMoeJitConfigV1* config, MegaMoeJitLayoutV1* layout, char* error, size_t capacity);
  int (*packWeights)(const MegaMoeJitConfigV1* config, const MegaMoeJitWeightsV1* source,
                     const MegaMoeJitWeightsV1* destination, void* stream, char* error, size_t capacity);
  int (*createPlan)(const MegaMoeJitConfigV1* config, void* symmetric, const uint64_t* peerBases, void* workspace,
                    const MegaMoeJitWeightsV1* weights, void** plan, char* error, size_t capacity);
  void (*destroyPlan)(void* plan);
  int (*launch)(void* plan, int32_t numTokens, void* output, void* stream, uint32_t* startSignal, int32_t shared,
                char* error, size_t capacity);
} MegaMoeJitApiV1;

typedef const MegaMoeJitApiV1* (*MegaMoeJitGetApiV1)(void);

#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) const MegaMoeJitApiV1* mscclpp_megamoe_jit_get_api_v1(void);
#ifdef __cplusplus
}
#endif

#endif  // MSCCLPP_EXT_MEGAMOE_JIT_HPP_
