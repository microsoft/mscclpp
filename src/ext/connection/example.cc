#include "connection.hpp"
#include "ext/connection/connection.hpp"

void test() {
  auto context = mscclpp::Context::create();
  auto localEndpoint = context->createEndpoint({mscclpp::Transport::CudaIpc});
  auto remoteEndpoint = context->createEndpoint({mscclpp::Transport::CudaIpc});
  mscclpp::Device fwd(mscclpp::DeviceType::GPU, 2);
  std::shared_ptr<mscclpp::ConnectionScheduler> scheduler = std::make_shared<mscclpp::DefaultConnectionScheduler>(context, fwd);
  context->set("scheduler", scheduler);
  mscclpp::ConnectionFactory::registerConnection(
      "indirect", [context](std::shared_ptr<mscclpp::Context> ctx, mscclpp::Endpoint local, mscclpp::Endpoint remote) {
        std::shared_ptr<mscclpp::ConnectionScheduler> scheduler = std::static_pointer_cast<mscclpp::ConnectionScheduler>(context->get("scheduler"));
        return std::make_shared<mscclpp::IndirectConnection>(ctx, local, scheduler);
      });
  auto connection = mscclpp::ConnectionFactory::createConnection("indirect", context, localEndpoint, remoteEndpoint);
}