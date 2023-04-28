#include "registered_memory.hpp"
#include "api.h"
#include "checks.hpp"
#include <algorithm>
#include <cuda.h>

namespace mscclpp {

RegisteredMemory::Impl::Impl(void* data, size_t size, int rank, TransportFlags transports, Communicator::Impl& commImpl)
  : data(data), dataInitialized(true), size(size), rank(rank), transports(transports)
{
  if (transports.has(Transport::CudaIpc)) {
    TransportInfo transportInfo;
    transportInfo.transport = Transport::CudaIpc;
    cudaIpcMemHandle_t handle;

    void* baseDataPtr;
    size_t baseDataSize; // dummy
    CUTHROW(cuMemGetAddressRange((CUdeviceptr*)&baseDataPtr, &baseDataSize, (CUdeviceptr)data));
    CUDATHROW(cudaIpcGetMemHandle(&handle, baseDataPtr));
    transportInfo.cudaIpcHandle = handle;
    this->transportInfos.push_back(transportInfo);
  }
  if ((transports & AllIBTransports).any()) {
    auto addIb = [&](Transport ibTransport) {
      TransportInfo transportInfo;
      transportInfo.transport = ibTransport;
      const IbMr* mr = commImpl.getIbContext(ibTransport)->registerMr(data, size);
      transportInfo.ibMr = mr;
      transportInfo.ibLocal = true;
      this->transportInfos.push_back(transportInfo);
    };
    if (transports.has(Transport::IB0))
      addIb(Transport::IB0);
    if (transports.has(Transport::IB1))
      addIb(Transport::IB1);
    if (transports.has(Transport::IB2))
      addIb(Transport::IB2);
    if (transports.has(Transport::IB3))
      addIb(Transport::IB3);
    if (transports.has(Transport::IB4))
      addIb(Transport::IB4);
    if (transports.has(Transport::IB5))
      addIb(Transport::IB5);
    if (transports.has(Transport::IB6))
      addIb(Transport::IB6);
    if (transports.has(Transport::IB7))
      addIb(Transport::IB7);
  }
}

MSCCLPP_API_CPP RegisteredMemory::RegisteredMemory(std::shared_ptr<Impl> pimpl) : pimpl(pimpl)
{
}

MSCCLPP_API_CPP RegisteredMemory::~RegisteredMemory() = default;

void* RegisteredMemory::data()
{
  if (!pimpl->dataInitialized) {
    if (pimpl->transports.has(Transport::CudaIpc)) {
      auto entry = pimpl->getTransportInfo(Transport::CudaIpc);
      CUDATHROW(cudaIpcOpenMemHandle(&pimpl->data, entry.cudaIpcHandle, cudaIpcMemLazyEnablePeerAccess));
      INFO(MSCCLPP_P2P, "Opened CUDA IPC handle for base point of %p", data);
    }
    else
    {
      pimpl->data = nullptr;
    }
    pimpl->dataInitialized = true;
  }
  return pimpl->data;
}

size_t RegisteredMemory::size()
{
  return pimpl->size;
}

int RegisteredMemory::rank()
{
  return pimpl->rank;
}

TransportFlags RegisteredMemory::transports()
{
  return pimpl->transports;
}

MSCCLPP_API_CPP std::vector<char> RegisteredMemory::serialize()
{
  std::vector<char> result;
  std::copy_n(reinterpret_cast<char*>(&pimpl->size), sizeof(pimpl->size), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&pimpl->rank), sizeof(pimpl->rank), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&pimpl->transports), sizeof(pimpl->transports), std::back_inserter(result));
  if (pimpl->transportInfos.size() > std::numeric_limits<int8_t>::max()) {
    throw std::runtime_error("Too many transport info entries");
  }
  int8_t transportCount = pimpl->transportInfos.size();
  std::copy_n(reinterpret_cast<char*>(&transportCount), sizeof(transportCount), std::back_inserter(result));
  for (auto& entry : pimpl->transportInfos) {
    std::copy_n(reinterpret_cast<char*>(&entry.transport), sizeof(entry.transport), std::back_inserter(result));
    if (entry.transport == Transport::CudaIpc) {
      std::copy_n(reinterpret_cast<char*>(&entry.cudaIpcHandle), sizeof(entry.cudaIpcHandle),
                  std::back_inserter(result));
    } else if (AllIBTransports.has(entry.transport)) {
      std::copy_n(reinterpret_cast<char*>(&entry.ibMrInfo), sizeof(entry.ibMrInfo), std::back_inserter(result));
    } else {
      throw std::runtime_error("Unknown transport");
    }
  }
  return result;
}

MSCCLPP_API_CPP RegisteredMemory RegisteredMemory::deserialize(const std::vector<char>& data)
{
  return RegisteredMemory(std::make_shared<Impl>(data));
}

RegisteredMemory::Impl::Impl(const std::vector<char>& serialization)
{
  auto it = serialization.begin();
  std::copy_n(it, sizeof(this->size), reinterpret_cast<char*>(&this->size));
  it += sizeof(this->size);
  std::copy_n(it, sizeof(this->rank), reinterpret_cast<char*>(&this->rank));
  it += sizeof(this->rank);
  std::copy_n(it, sizeof(this->transports), reinterpret_cast<char*>(&this->transports));
  it += sizeof(this->transports);
  int8_t transportCount;
  std::copy_n(it, sizeof(transportCount), reinterpret_cast<char*>(&transportCount));
  it += sizeof(transportCount);
  for (int i = 0; i < transportCount; ++i) {
    TransportInfo transportInfo;
    std::copy_n(it, sizeof(transportInfo.transport), reinterpret_cast<char*>(&transportInfo.transport));
    it += sizeof(transportInfo.transport);
    if (transportInfo.transport == Transport::CudaIpc) {
      cudaIpcMemHandle_t handle;
      std::copy_n(it, sizeof(handle), reinterpret_cast<char*>(&handle));
      it += sizeof(handle);
      transportInfo.cudaIpcHandle = handle;
    } else if (AllIBTransports.has(transportInfo.transport)) {
      IbMrInfo info;
      std::copy_n(it, sizeof(info), reinterpret_cast<char*>(&info));
      it += sizeof(info);
      transportInfo.ibMrInfo = info;
      transportInfo.ibLocal = false;
    } else {
      throw std::runtime_error("Unknown transport");
    }
    this->transportInfos.push_back(transportInfo);
  }
  if (it != serialization.end()) {
    throw std::runtime_error("Deserialization failed");
  }

  dataInitialized = false;
}

} // namespace mscclpp
