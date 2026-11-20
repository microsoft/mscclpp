// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_RDMA_PROVIDER_HPP_
#define MSCCLPP_RDMA_PROVIDER_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <utility>
#include <vector>

namespace mscclpp {

class RdmaProvider;

class RdmaMemory {
 public:
  virtual ~RdmaMemory() = default;
  virtual bool isLocal() const = 0;
  virtual void serialize(std::vector<char>& data) const = 0;

 protected:
  explicit RdmaMemory(std::shared_ptr<RdmaProvider> provider = {}) : provider_(std::move(provider)) {}

 private:
  std::shared_ptr<RdmaProvider> provider_;
};

class RdmaEndpoint {
 public:
  virtual ~RdmaEndpoint() = default;
  virtual bool isLocal() const = 0;
  virtual void serialize(std::vector<char>& data) const = 0;
  virtual std::shared_ptr<BaseConnection> createConnection(std::shared_ptr<Context> context, const Endpoint& local,
                                                           const Endpoint& remote) const = 0;

 protected:
  explicit RdmaEndpoint(std::shared_ptr<RdmaProvider> provider = {}) : provider_(std::move(provider)) {}

 private:
  std::shared_ptr<RdmaProvider> provider_;
};

class RdmaProvider : public std::enable_shared_from_this<RdmaProvider> {
 public:
  virtual ~RdmaProvider() = default;
  virtual std::shared_ptr<RdmaEndpoint> createEndpoint(EndpointConfig& config) = 0;
  virtual std::shared_ptr<RdmaMemory> registerMemory(void* data, size_t size) = 0;
};

TransportFlags getRdmaTransports();
bool areRdmaTransportsCompatible(Transport local, Transport remote);

std::shared_ptr<RdmaProvider> createRdmaProvider(Transport transport);
std::shared_ptr<RdmaEndpoint> deserializeRdmaEndpoint(const EndpointConfig& config,
                                                      std::vector<char>::const_iterator& it);
std::shared_ptr<RdmaMemory> deserializeRdmaMemory(Transport transport, std::vector<char>::const_iterator& it);

}  // namespace mscclpp

#endif  // MSCCLPP_RDMA_PROVIDER_HPP_
