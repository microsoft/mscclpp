#include "registered_memory.hpp"
#include "checks.hpp"
#include <algorithm>

namespace mscclpp {

RegisteredMemory::Impl::Impl(void* data, size_t size, int rank, TransportFlags transports, Communicator::Impl& commImpl) : data(data), size(size), rank(rank), transports(transports) {
  if (transports & TransportCudaIpc) {
    TransportInfo transportInfo;
    transportInfo.transport = TransportCudaIpc;
    cudaIpcMemHandle_t handle;
    CUDATHROW(cudaIpcGetMemHandle(&handle, data));
    transportInfo.cudaIpcHandle = handle;
    this->transportInfos.push_back(transportInfo);
  }
  if (transports & TransportAllIB) {
    auto addIb = [&](TransportFlags ibTransport) {
      TransportInfo transportInfo;
      transportInfo.transport = ibTransport;
      mscclppIbMr* mr;
      MSCCLPPTHROW(mscclppIbContextRegisterMr(commImpl.getIbContext(ibTransport), data, size, &mr));
      transportInfo.ibMr = mr;
      transportInfo.ibLocal = true;
      this->transportInfos.push_back(transportInfo);
    };
    if (transports & TransportIB0) addIb(TransportIB0);
    if (transports & TransportIB1) addIb(TransportIB1);
    if (transports & TransportIB2) addIb(TransportIB2);
    if (transports & TransportIB3) addIb(TransportIB3);
    if (transports & TransportIB4) addIb(TransportIB4);
    if (transports & TransportIB5) addIb(TransportIB5);
    if (transports & TransportIB6) addIb(TransportIB6);
    if (transports & TransportIB7) addIb(TransportIB7);
  }
}

RegisteredMemory::RegisteredMemory(std::shared_ptr<Impl> pimpl) : pimpl(pimpl) {}

RegisteredMemory::~RegisteredMemory() = default;

void* RegisteredMemory::data() {
  return pimpl->data;
}

size_t RegisteredMemory::size() {
  return pimpl->size;
}

int RegisteredMemory::rank() {
  return pimpl->rank;
}

TransportFlags RegisteredMemory::transports() {
  return pimpl->transports;
}

std::vector<char> RegisteredMemory::serialize() {
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
    if (entry.transport == TransportCudaIpc) {
      std::copy_n(reinterpret_cast<char*>(&entry.cudaIpcHandle), sizeof(entry.cudaIpcHandle), std::back_inserter(result));
    } else if (entry.transport & TransportAllIB) {
      std::copy_n(reinterpret_cast<char*>(&entry.ibMrInfo), sizeof(entry.ibMrInfo), std::back_inserter(result));
    } else {
      throw std::runtime_error("Unknown transport");
    }
  }
  return result;
}

RegisteredMemory RegisteredMemory::deserialize(const std::vector<char>& data) {
  return RegisteredMemory(std::make_shared<Impl>(data));
}

RegisteredMemory::Impl::Impl(const std::vector<char>& serialization) {
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
    if (transportInfo.transport & TransportCudaIpc) {
      cudaIpcMemHandle_t handle;
      std::copy_n(it, sizeof(handle), reinterpret_cast<char*>(&handle));
      it += sizeof(handle);
      transportInfo.cudaIpcHandle = handle;
    } else if (transportInfo.transport & TransportAllIB) {
      mscclppIbMrInfo info;
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

  if (transports & TransportCudaIpc) {
    auto entry = getTransportInfo(TransportCudaIpc);
    CUDATHROW(cudaIpcOpenMemHandle(&data, entry.cudaIpcHandle, cudaIpcMemLazyEnablePeerAccess));
  }
}

} // namespace mscclpp
