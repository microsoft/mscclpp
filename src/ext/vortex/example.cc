#include "connection.hpp"
#include "ext/vortex/connection.hpp"

void test() {
    auto context = mscclpp::Context::create();
    auto localEndpoint = context->createEndpoint({mscclpp::Transport::CudaIpc});
    auto remoteEndpoint = context->createEndpoint({mscclpp::Transport::CudaIpc});
    mscclpp::Device fwd(mscclpp::DeviceType::GPU, 2);
    uint64_t granularity = 20'000'000;
    std::shared_ptr<mscclpp::Scheduler> scheduler = std::make_shared<mscclpp::VortexScheduler>(context, granularity, fwd);
    mscclpp::ConnectionFactory::registerConnection(
        "indirect", [scheduler](std::shared_ptr<mscclpp::Context> ctx, mscclpp::Endpoint local, mscclpp::Endpoint remote) {
            return std::make_shared<mscclpp::IndirectConnection>(ctx, local, scheduler);
        });
    auto connection = mscclpp::ConnectionFactory::createConnection("indirect", context, localEndpoint, remoteEndpoint);
}