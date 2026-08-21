// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cassert>
#include <type_traits>
#include <unordered_set>

#include <mscclpp/algorithm.hpp>

int main() {
  using mscclpp::Algorithm;
  using mscclpp::AlgorithmCtxKey;
  using mscclpp::DataType;

  AlgorithmCtxKey base{reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000), 128, 2048, 7,
                       0x3000, 0, 64, static_cast<int>(DataType::BFLOAT16), false};
  AlgorithmCtxKey exact = base;
  assert(base == exact);
  assert(std::hash<AlgorithmCtxKey>{}(base) == std::hash<AlgorithmCtxKey>{}(exact));

  std::unordered_set<AlgorithmCtxKey> keys{base};
  auto addDifference = [&](auto mutate) {
    AlgorithmCtxKey key = base;
    mutate(key);
    assert(!(key == base));
    keys.insert(key);
  };
  addDifference([](auto& key) { key.baseSendBuff = reinterpret_cast<void*>(0x1001); });
  addDifference([](auto& key) { key.baseRecvBuff = reinterpret_cast<void*>(0x2001); });
  addDifference([](auto& key) { key.baseSendSize += 16; });
  addDifference([](auto& key) { key.baseRecvSize += 16; });
  addDifference([](auto& key) { key.tag += 1; });
  addDifference([](auto& key) { key.communicatorIdentity += 1; });
  addDifference([](auto& key) { key.device += 1; });
  addDifference([](auto& key) { key.elementCount += 1; });
  addDifference([](auto& key) { key.dtype = static_cast<int>(DataType::FLOAT16); });
  addDifference([](auto& key) { key.symmetricMemory = true; });
  assert(keys.size() == 11);

  static_assert(std::is_member_function_pointer_v<decltype(&Algorithm::prepare)>);
  static_assert(std::is_member_function_pointer_v<decltype(&Algorithm::executePrepared)>);
  static_assert(std::is_member_function_pointer_v<decltype(&Algorithm::releasePrepared)>);
  return 0;
}
