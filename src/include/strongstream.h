/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_STRONGSTREAM_H_
#define MSCCLPP_STRONGSTREAM_H_

#include "mscclpp.h"
#include "checks.h"

#include <stdint.h>

// /* mscclppCudaGraph: Wraps a cudaGraph_t so that we can support pre-graph CUDA runtimes
//  * easily.
//  */
// struct mscclppCudaGraph {
// #if CUDART_VERSION >= 11030
//   cudaGraph_t graph;
//   unsigned long long graphId;
// #endif
// };

// inline struct mscclppCudaGraph mscclppCudaGraphNone() {
//   struct mscclppCudaGraph tmp;
//   #if CUDART_VERSION >= 11030
//     tmp.graph = nullptr;
//     tmp.graphId = ULLONG_MAX;
//   #endif
//   return tmp;
// }

// inline bool mscclppCudaGraphValid(struct mscclppCudaGraph graph) {
//   #if CUDART_VERSION >= 11030
//     return graph.graph != nullptr;
//   #else
//     return false;
//   #endif
// }

// inline bool mscclppCudaGraphSame(struct mscclppCudaGraph a, struct mscclppCudaGraph b) {
//   #if CUDART_VERSION >= 11030
//     return a.graphId == b.graphId;
//   #else
//     return true;
//   #endif
// }

// mscclppResult_t mscclppCudaGetCapturingGraph(struct mscclppCudaGraph* graph, cudaStream_t stream);
// mscclppResult_t mscclppCudaGraphAddDestructor(struct mscclppCudaGraph graph, cudaHostFn_t fn, void* arg);

// /* mscclppStrongStream: An abstraction over CUDA streams that do not lose their
//  * identity while being captured. Regular streams have the deficiency that the
//  * captured form of a stream in one graph launch has no relation to the
//  * uncaptured stream or to the captured form in other graph launches. This makes
//  * streams unfit for the use of serializing access to a persistent resource.
//  * Strong streams have been introduced to address this need.
//  *
//  * - All updates to a strong stream must be enclosed by a Acquire/Release pair.
//  *
//  * - The Acquire, Release, and all updates take a mscclppCudaGraph parameter
//  *   indicating the currently capturing graph (or none). This parameter must be
//  *   the same for the entire sequence of {Acquire; ...; Release}.
//  *
//  * - An {Acquire; ...; Release} sequence must not be concurrent with any
//  *   other operations against the strong stream including graph launches which
//  *   reference this stream.
//  */
// struct mscclppStrongStream;

// mscclppResult_t mscclppStrongStreamConstruct(struct mscclppStrongStream* ss);
// mscclppResult_t mscclppStrongStreamDestruct(struct mscclppStrongStream* ss);

// // Acquire-fence the strong stream.
// mscclppResult_t mscclppStrongStreamAcquire(
//   struct mscclppCudaGraph graph, struct mscclppStrongStream* ss
// );

// // Acquire-fence the strong stream assuming no graph is capturing. This permits
// // the caller to enqueue directly to the `ss->cudaStream` member using native CUDA
// // calls. Strong stream still must be released via:
// //   mscclppStrongStreamRelease(mscclppCudaGraphNone(), ss);
// mscclppResult_t mscclppStrongStreamAcquireUncaptured(struct mscclppStrongStream* ss);

// // Release-fence of the strong stream.
// mscclppResult_t mscclppStrongStreamRelease(struct mscclppCudaGraph graph, struct mscclppStrongStream* ss);

// // Add a host launch to the stream.
// mscclppResult_t mscclppStrongStreamLaunchHost(
//   struct mscclppCudaGraph graph, struct mscclppStrongStream* ss,
//   cudaHostFn_t fn, void* arg
// );
// // Add a kernel launch to the stream.
// mscclppResult_t mscclppStrongStreamLaunchKernel(
//   struct mscclppCudaGraph graph, struct mscclppStrongStream* ss,
//   void* fn, dim3 grid, dim3 block, void** args, size_t sharedMemBytes
// );

// // Cause `a` to wait for the current state `b`. Both `a` and `b` must be acquired.
// // `b_subsumes_a` indicates that all work in `a` is already present in `b`, thus
// // we want to fast-forward `a` to be a clone of `b`. Knowing this permits the
// // implementation to induce few graph dependencies.
// mscclppResult_t mscclppStrongStreamWaitStream(
//   struct mscclppCudaGraph graph, struct mscclppStrongStream* a, struct mscclppStrongStream* b, bool b_subsumes_a=false
// );
// // `b` must be capturing within `graph`.
// mscclppResult_t mscclppStrongStreamWaitStream(
//   struct mscclppCudaGraph graph, struct mscclppStrongStream* a, cudaStream_t b, bool b_subsumes_a=false
// );
// // `a` must be capturing within `graph`.
// mscclppResult_t mscclppStrongStreamWaitStream(
//   struct mscclppCudaGraph graph, cudaStream_t a, struct mscclppStrongStream* b, bool b_subsumes_a=false
// );

// // Synchrnoization does not need the strong stream to be acquired.
// mscclppResult_t mscclppStrongStreamSynchronize(struct mscclppStrongStream* ss);

// ////////////////////////////////////////////////////////////////////////////////

// struct mscclppStrongStreamGraph; // internal to mscclppStrongStream

struct mscclppStrongStream {
  // Used when not graph capturing.
  cudaStream_t cudaStream;
// #if CUDART_VERSION >= 11030
//   // The event used to establish order between graphs and streams. During acquire
//   // this event is waited on, during release it is recorded to.
//   cudaEvent_t serialEvent;
//   // This stream ever appeared in a graph capture.
//   bool everCaptured;
//   // Tracks whether serialEvent needs to be recorded to upon Release().
//   bool serialEventNeedsRecord;
//   struct mscclppStrongStreamGraph* graphHead;
// #else
//   cudaEvent_t scratchEvent;
// #endif
};

#endif
