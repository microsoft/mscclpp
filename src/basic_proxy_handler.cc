#include "basic_proxy_handler.hpp"

namespace mscclpp {

ProxyHandler makeBasicProxyHandler(Communicator::Impl& comm)
{
  return [&comm](ProxyTrigger triggerRaw) {
    ChannelTrigger* trigger = reinterpret_cast<ChannelTrigger*>(&triggerRaw);
    HostConnection& conn = *comm.connections.at(trigger->fields.connId);

    auto result = ProxyHandlerResult::Continue;

    if (trigger->fields.type & mscclppData) {
      conn.put(trigger->fields.dstBufferHandle, trigger->fields.dstOffset, trigger->fields.srcBufferHandle,
               trigger->fields.srcOffset, trigger->fields.size);
    }

    if (trigger->fields.type & mscclppFlag) {
      conn.signal();
    }

    if (trigger->fields.type & mscclppSync) {
      conn.flush();
      result = ProxyHandlerResult::FlushFifoTailAndContinue;
    }

    return result;
  };
}

} // namespace mscclpp
