// Host-only syntax-check shim: provides the few nvcc builtins used by host-side
// paths of the EP headers. Device code is compiled out (MSCCLPP_BULK_AVAILABLE == 0).
#include <algorithm>
template <typename T> constexpr const T& min(const T& a, const T& b) { return std::min(a, b); }
template <typename T> constexpr const T& max(const T& a, const T& b) { return std::max(a, b); }
// warp / block intrinsics (declarations only; bodies are never emitted)
extern "C" {
void __syncthreads();
void __syncwarp(unsigned mask = 0xffffffffu);
void __threadfence_system();
void __threadfence();
int __ffs(int);
}
template <typename T> T __shfl_sync(unsigned, T, int, int width = 32);
template <typename T> T __shfl_up_sync(unsigned, T, unsigned, int width = 32);
template <typename T> T __shfl_xor_sync(unsigned, T, int, int width = 32);
template <typename T> unsigned __match_any_sync(unsigned, T);
unsigned __ballot_sync(unsigned, bool);
int __any_sync(unsigned, bool);
template <typename T> T __ldg(const T*);
template <typename T> T atomicAdd(T*, T);
template <typename T> T atomicAdd_block(T*, T);
#include <cmath>
inline float fabsf_(float x){return x<0?-x:x;}
inline float fabsf(float x){return fabsf_(x);}
inline float fmaxf(float a,float b){return a>b?a:b;}
inline float fminf(float a,float b){return a<b?a:b;}
inline unsigned long long __cvta_generic_to_shared(const void*) { return 0; }
