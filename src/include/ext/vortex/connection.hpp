#include "mscclpp/core.hpp"
#include "mscclpp/gpu_utils.hpp"

namespace mscclpp {

class BufferResource {
 public:
  virtual ~BufferResource() = default;
  virtual RegisteredMemory next_get() = 0;
  virtual RegisteredMemory next_put() = 0;
  virtual void produce() = 0;
  virtual void consume() = 0;
};

class DoubleBuffer : public BufferResource {
  std::array<RegisteredMemory, 2> bufs_;
  int cur_{0};

 public:
  DoubleBuffer(RegisteredMemory buf1, RegisteredMemory buf2) : bufs_({buf1, buf2}) {}
  RegisteredMemory next_get() override { return bufs_[cur_]; }
  RegisteredMemory next_put() override { return bufs_[cur_ ^ 1]; }
  void produce() override { cur_ ^= 1; }
  void consume() override {}
};

class IOTask {
 public:
  void *dst, *src;
  uint64_t size;
  IOTask(void *dst_, void *src_, uint64_t size_) : dst(dst_), src(src_), size(size_) {}
};

class Scheduler {
 public:
  virtual std::vector<IOTask> produce_tasks(void *dst, void *src, uint64_t size) = 0;
  virtual void launch(const std::vector<IOTask> &tasks) = 0;
  virtual void sync() = 0;
};

class VortexScheduler : public Scheduler {
  std::shared_ptr<DoubleBuffer> buf_ptr_;
  uint64_t granularity_;
  std::array<std::shared_ptr<CudaStreamWithFlags>, 2> streams_;
  Device forwarding_device_;

 public:
  VortexScheduler(std::shared_ptr<Context> context, uint64_t granularity, Device device);
  ~VortexScheduler();
  std::vector<IOTask> produce_tasks(void *dst, void *src, uint64_t size) override;
  void launch(const std::vector<IOTask> &tasks) override;
  void sync() override;
};

class IndirectConnection : public Connection {
  std::shared_ptr<Scheduler> scheduler_ptr_;

 public:
  IndirectConnection(std::shared_ptr<Context> context, Endpoint localEndpoint, std::shared_ptr<Scheduler> scheduler)
      : Connection(context, localEndpoint), scheduler_ptr_(scheduler) {
    if (!scheduler_ptr_) {
      throw std::runtime_error("Scheduler pointer cannot be null");
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