// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/semaphore.hpp>
#include <vector>

namespace nb = nanobind;

class MyProxyService {
 private:
  int my_rank_, nranks_, dataSize_;
  std::vector<mscclpp::RegisteredMemory> allRegMem_;
  std::vector<mscclpp::Host2DeviceSemaphore> semaphores_;
  mscclpp::Proxy proxy_;

 public:
  MyProxyService(int my_rank, int nranks, int dataSize, nb::list allRegMemList, nb::list semaphoreList)
      : my_rank_(my_rank),
        nranks_(nranks),
        dataSize_(dataSize),
        proxy_([&](mscclpp::ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); }) {
    allRegMem_.reserve(allRegMemList.size());
    for (size_t i = 0; i < allRegMemList.size(); ++i) {
      auto regMem = nb::cast<mscclpp::RegisteredMemory>(allRegMemList[i]);
      allRegMem_.emplace_back(regMem);
    }
    semaphores_.reserve(semaphoreList.size());
    for (size_t i = 0; i < semaphoreList.size(); ++i) {
      auto sema = nb::cast<mscclpp::Semaphore>(semaphoreList[i]);
      semaphores_.emplace_back(sema);
    }
  }

  mscclpp::ProxyHandlerResult handleTrigger(mscclpp::ProxyTrigger) {
    int dataSizePerRank = dataSize_ / nranks_;
    for (int r = 1; r < nranks_; ++r) {
      int nghr = (my_rank_ + r) % nranks_;
      auto& sema = semaphores_[nghr];
      auto& conn = sema.connection();
      conn.write(allRegMem_[nghr], my_rank_ * (uint64_t)dataSizePerRank, allRegMem_[my_rank_],
                 my_rank_ * (uint64_t)dataSizePerRank, dataSizePerRank);
      sema.signal();
      conn.flush();
    }
    return mscclpp::ProxyHandlerResult::Continue;
  }

  void start() { proxy_.start(); }

  void stop() { proxy_.stop(); }

  mscclpp::FifoDeviceHandle fifoDeviceHandle() { return proxy_.fifo()->deviceHandle(); }
};

NB_MODULE(_ext, m) {
  #ifdef MSCCLPP_DISABLE_NB_LEAK_WARNINGS
    nb::set_leak_warnings(false);
  #endif
  nb::class_<MyProxyService>(m, "MyProxyService")
      .def(nb::init<int, int, int, nb::list, nb::list>(), nb::arg("rank"), nb::arg("nranks"), nb::arg("data_size"),
           nb::arg("reg_mem_list"), nb::arg("sem_list"))
      .def("fifo_device_handle", &MyProxyService::fifoDeviceHandle)
      .def("start", &MyProxyService::start)
      .def("stop", &MyProxyService::stop);
}
