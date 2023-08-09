#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace mscclpp {
int getDeviceNumaNode(int cudaDev);
void numaBind(int node);
};

void register_numa(nb::module_ &m) {
  nb::module_ _sub_m = m.def_submodule("numa", "numa functions");
  _sub_m.def("get_device_numa_node", &mscclpp::getDeviceNumaNode);
  _sub_m.def("numa_bind", &mscclpp::numaBind);
}