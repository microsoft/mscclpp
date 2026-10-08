// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_net_io_service.hpp>

#include "api.h"

#if !defined(MSCCLPP_HAS_GPUNETIO)
namespace mscclpp {
namespace {
[[noreturn]] void rejectGpuNetIo() { throw Error("MSCCL++ was built without GPUNetIO", ErrorCode::InvalidUsage); }
}  // namespace

MSCCLPP_API_CPP GpuNetIoService::GpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& ibDeviceName,
                                                 int cudaDeviceId)
    : GpuNetIoService(bootstrap, ibDeviceName, cudaDeviceId, 1) {}

MSCCLPP_API_CPP GpuNetIoService::GpuNetIoService(std::shared_ptr<Bootstrap>, const std::string&, int, int) {
  rejectGpuNetIo();
}

MSCCLPP_API_CPP GpuNetIoService::~GpuNetIoService() = default;

MSCCLPP_API_CPP void GpuNetIoService::setup() { rejectGpuNetIo(); }

MSCCLPP_API_CPP void GpuNetIoService::setup(void*, size_t) { rejectGpuNetIo(); }

MSCCLPP_API_CPP void GpuNetIoService::setup(const std::vector<int>&, int) { rejectGpuNetIo(); }

MSCCLPP_API_CPP GpuNetIoConnection GpuNetIoService::connect(int, int) const { rejectGpuNetIo(); }

MSCCLPP_API_CPP GpuNetIoMemory GpuNetIoService::registerMemory(void*, size_t, std::shared_ptr<void>) const {
  rejectGpuNetIo();
}

MSCCLPP_API_CPP GpuNetIoMemory GpuNetIoService::exchangeMemory(const GpuNetIoConnection&, const GpuNetIoMemory&,
                                                               int) const {
  rejectGpuNetIo();
}

MSCCLPP_API_CPP GpuNetIoSemaphore GpuNetIoService::buildSemaphore(const GpuNetIoConnection&, int) const {
  rejectGpuNetIo();
}

MSCCLPP_API_CPP GpuNetIoDeviceContext* GpuNetIoService::deviceContext() const { rejectGpuNetIo(); }

MSCCLPP_API_CPP GpuNetIoDeviceContext* GpuNetIoService::deviceContext(int) const { rejectGpuNetIo(); }

}  // namespace mscclpp
#endif