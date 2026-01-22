#include <cstdint>

__device__ __forceinline__ void probe(const void* ptr) {
  uint32_t w0, w1, w2, w3;
  unsigned long long addr = reinterpret_cast<unsigned long long>(ptr);

  asm volatile(
    "multimem.ld_reduce.relaxed.sys.global.add.v4.e4m3x4 {%0,%1,%2,%3}, [%4];"
    : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
    : "l"(addr)
    : "memory"
  );

  (void)w0; (void)w1; (void)w2; (void)w3;
}

__global__ void k(const void* p) { probe(p); }
int main() { k<<<1,1>>>(nullptr); return 0; }

