#include "mscclpp/core.hpp"
#include "mscclpp/gpu_utils.hpp"

namespace mscclpp {

class ConnectionScheduler {
 public:
  virtual void sched(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) = 0;
  virtual void sync() = 0;
};

class DefaultConnectionScheduler : public ConnectionScheduler {
 public:
  DefaultConnectionScheduler(std::shared_ptr<Context> context, Device device) : context_(context), device_(device) {}

  void sched(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override {
    // Implementation for scheduling tasks
  }

  void sync() override {
    // Implementation for synchronizing tasks
  }

 private:
  std::shared_ptr<Context> context_;
  Device device_;
};

class IndirectConnection : public Connection {
  std::shared_ptr<ConnectionScheduler> scheduler_ptr_;

 public:
  IndirectConnection(std::shared_ptr<Context> context, Endpoint localEndpoint,
                     std::shared_ptr<ConnectionScheduler> scheduler)
      : Connection(context, localEndpoint), scheduler_ptr_(scheduler) {
    if (scheduler_ptr_ == nullptr) {
      throw std::runtime_error("Scheduler not set in context");
    }
  }
  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void flush(int64_t timeoutUsec = -1) override;
  Transport transport() const override;
  Transport remoteTransport() const override;

  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t *src, uint64_t newValue) override {
    throw std::runtime_error("IndirectConnection does not support updateAndSync");
  }
};
}  // namespace mscclpp