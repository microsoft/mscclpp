// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_net_io_topology.hpp"

namespace {
using mscclpp::detail::gpunetio::HcaTopology;
using mscclpp::detail::gpunetio::selectClosestHcas;

void expect(const std::vector<std::string>& actual, const std::vector<std::string>& wanted) {
  if (actual != wanted) {
    std::cerr << "Expected:";
    for (const auto& name : wanted) std::cerr << ' ' << name;
    std::cerr << "; got:";
    for (const auto& name : actual) std::cerr << ' ' << name;
    std::cerr << '\n';
    throw std::runtime_error("HCA selection mismatch");
  }
}
}  // namespace

int main() {
  try {
    const std::vector<std::string> gpuPaths = {
        "/sys/devices/pci0004:00/0004:00:00.0/0004:01:00.0", "/sys/devices/pci0005:00/0005:00:00.0/0005:01:00.0",
        "/sys/devices/pci0014:00/0014:00:00.0/0014:01:00.0", "/sys/devices/pci0015:00/0015:00:00.0/0015:01:00.0"};
    std::vector<HcaTopology> hcas = {{"mlx5_ib0", "/sys/devices/pci0000:00/0000:00:00.0/0000:01:00.0", 0},
                                     {"mlx5_ib1", "/sys/devices/pci0002:00/0002:00:00.0/0002:01:00.0", 0},
                                     {"mlx5_ib2", "/sys/devices/pci0010:00/0010:00:00.0/0010:01:00.0", 1},
                                     {"mlx5_ib3", "/sys/devices/pci0012:00/0012:00:00.0/0012:01:00.0", 1}};
    const std::vector<std::vector<std::string>> expected = {
        {"mlx5_ib0", "mlx5_ib1"}, {"mlx5_ib0", "mlx5_ib1"}, {"mlx5_ib2", "mlx5_ib3"}, {"mlx5_ib2", "mlx5_ib3"}};
    int checked = 0;
    do {
      for (int gpu : {3, 0, 2, 1, 0, 3}) {
        expect(selectClosestHcas(gpuPaths[gpu], gpu / 2, hcas), expected[gpu]);
        ++checked;
      }
    } while (std::next_permutation(hcas.begin(), hcas.end(), [](const HcaTopology& left, const HcaTopology& right) {
      return left.name < right.name;
    }));

    const std::string gpu = "/sys/devices/pci0000:00/0000:00:00.0/0000:01:00.0";
    const std::string near = "/sys/devices/pci0000:00/0000:00:00.0/0000:01:00.1";
    const std::string far = "/sys/devices/pci0002:00/0002:00:00.0/0002:01:00.0";
    expect(selectClosestHcas(gpu, 0, {{"near", near, 0}, {"far", far, 0}}), {"near"});
    expect(selectClosestHcas(gpu, 0, {{"remote-numa", near, 1}, {"local-numa", far, 0}}), {"local-numa"});
    expect(selectClosestHcas(gpu, -1, {{"near", near, -1}, {"far", far, -1}}), {"near"});
    expect(selectClosestHcas(gpu, 0, {{"only", far, 1}}), {"only"});
    expect(selectClosestHcas(gpu, 0, {}), {});
    expect(selectClosestHcas("", 0, {{"local-b", "", 0}, {"remote", "", 1}, {"local-a", "", 0}}),
           {"local-a", "local-b"});
    expect(selectClosestHcas(gpu, 0, {{"unknown-path", "", 0}, {"near", near, 0}}), {"near"});
    expect(selectClosestHcas(gpu, 0, {{"mlx5_10", near, 0}, {"mlx5_2", near, 0}}), {"mlx5_10", "mlx5_2"});
    checked += 8;
    std::cout << "PASS: " << checked << " topology selection checks\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}