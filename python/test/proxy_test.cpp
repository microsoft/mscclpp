#include <cuda.h>
#include <nanobind/nanobind.h>

#include <iostream>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/cuda_utils.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/semaphore.hpp>
#include <vector>

// #include "common.h"
// #include "p2p.h"

namespace nb = nanobind;

class MyProxyService {
 private:
  int deviceNumaNode_;
  mscclpp::Proxy proxy_;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections_;
  std::vector<std::shared_ptr<mscclpp::RegisteredMemory>> allRegMem_;
  std::vector<std::shared_ptr<mscclpp::Host2DeviceSemaphore>> semaphores_;
  int my_rank_, nranks_, dataSize_;

 public:
  MyProxyService(int my_rank, int nranks, int dataSize, std::vector<std::shared_ptr<mscclpp::Connection>> conns,
                 std::vector<std::shared_ptr<mscclpp::RegisteredMemory>> allRegMem,
                 std::vector<std::shared_ptr<mscclpp::Host2DeviceSemaphore>> semaphores)
      : my_rank_(my_rank),
        nranks_(nranks),
        dataSize_(dataSize),
        connections_(conns),
        allRegMem_(allRegMem),
        semaphores_(semaphores),
        proxy_([&](mscclpp::ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); }, [&]() { bindThread(); }) {
    int cudaDevice;
    cudaGetDevice(&cudaDevice);
    deviceNumaNode_ = mscclpp::getDeviceNumaNode(cudaDevice);
  }

  void bindThread() {
    if (deviceNumaNode_ >= 0) {
      mscclpp::numaBind(deviceNumaNode_);
    }
  }

  mscclpp::ProxyHandlerResult handleTrigger(mscclpp::ProxyTrigger triggerRaw) {
    int dataSizePerRank = dataSize_ / nranks_;
    for (int r = 1; r < nranks_; ++r) {
      int nghr = (my_rank_ + r) % nranks_;
      connections_[nghr]->write(*allRegMem_[nghr], my_rank_ * dataSizePerRank, *allRegMem_[my_rank_],
                                my_rank_ * dataSizePerRank, dataSizePerRank);
      semaphores_[nghr]->signal();
      connections_[nghr]->flush();
    }
    return mscclpp::ProxyHandlerResult::FlushFifoTailAndContinue;
  }

  void start() { proxy_.start(); }

  void stop() { proxy_.stop(); }

  mscclpp::FifoDeviceHandle fifoDeviceHandle() { return proxy_.fifo().deviceHandle(); }
};

void init_mscclpp_proxy_test_module(nb::module_ &m) {
  nb::class_<MyProxyService>(m, "MyProxyService")
      .def(nb::init<int, int, int, std::vector<std::shared_ptr<mscclpp::Connection>>,
                    std::vector<std::shared_ptr<mscclpp::RegisteredMemory>>,
                    std::vector<std::shared_ptr<mscclpp::Host2DeviceSemaphore>>>(),
           nb::arg("rank"), nb::arg("nranks"), nb::arg("data_size"), nb::arg("conn_vec"), nb::arg("reg_mem_vec"),
           nb::arg("h2d_sem_vec"))
      .def("fifo_device_handle", &MyProxyService::fifoDeviceHandle)
      .def("start", &MyProxyService::start)
      .def("stop", &MyProxyService::stop);
}
