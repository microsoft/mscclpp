#include "ext/vortex/connection.hpp"

#include "connection.hpp"

namespace mscclpp {

// IndirectConnection

VortexScheduler::VortexScheduler(std::shared_ptr<Context> context, uint64_t granularity, Device device)
    : granularity_(granularity),
      streams_({std::make_shared<CudaStreamWithFlags>(0), std::make_shared<CudaStreamWithFlags>(0)}),
      forwarding_device_(device) {
  if (device.type != DeviceType::GPU) {
    throw std::runtime_error("The forwarding device must be a GPU");
  }
  int origin_device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&origin_device));
  MSCCLPP_CUDATHROW(cudaSetDevice(device.id));
  void *buf1, *buf2;
  MSCCLPP_CUDATHROW(cudaMalloc((void**)&buf1, granularity));
  MSCCLPP_CUDATHROW(cudaMalloc((void**)&buf2, granularity));
  buf_ptr_ =
      std::make_shared<mscclpp::DoubleBuffer>(context->registerMemory(buf1, granularity, mscclpp::Transport::CudaIpc),
                                              context->registerMemory(buf2, granularity, mscclpp::Transport::CudaIpc));
  MSCCLPP_CUDATHROW(cudaSetDevice(origin_device));
}

VortexScheduler::~VortexScheduler() {
  if (buf_ptr_) {
    int origin_device;
    cudaGetDevice(&origin_device);
    cudaSetDevice(forwarding_device_.id);
    cudaFree(buf_ptr_->next_get().data());
    cudaFree(buf_ptr_->next_put().data());
    cudaSetDevice(origin_device);
  }
}

std::vector<IOTask> VortexScheduler::produce_tasks(void* dst, void* src, uint64_t size) {
  std::vector<IOTask> tasks_;
  for (uint64_t i = 0; i < size; i += granularity_) {
    tasks_.push_back({dst + i, src + i, std::min(granularity_, size - i)});
  }
  return tasks_;
}

void VortexScheduler::launch(const std::vector<IOTask>& tasks) {
  if (tasks.empty()) {
    return;
  }

  cudaEvent_t event0, event1;
  MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&event0, cudaEventBlockingSync | cudaEventDisableTiming));
  MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&event1, cudaEventBlockingSync | cudaEventDisableTiming));

  MSCCLPP_CUDATHROW(cudaMemcpyAsync(buf_ptr_->next_put().data(), tasks.front().src, tasks.front().size,
                                    cudaMemcpyDefault, *streams_[0]));
  MSCCLPP_CUDATHROW(cudaEventRecord(event0, *streams_[0]));
  buf_ptr_->produce();

  for (uint64_t i = 1; i < tasks.size(); ++i) {
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(*streams_[1], event0, 0));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(*streams_[0], event1, 0));
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(tasks[i - 1].dst, buf_ptr_->next_get().data(), tasks[i - 1].size,
                                      cudaMemcpyDefault, *streams_[1]));
    MSCCLPP_CUDATHROW(cudaEventRecord(event1, *streams_[1]));
    MSCCLPP_CUDATHROW(
        cudaMemcpyAsync(buf_ptr_->next_put().data(), tasks[i].src, tasks[i].size, cudaMemcpyDefault, *streams_[0]));
    MSCCLPP_CUDATHROW(cudaEventRecord(event0, *streams_[0]));

    buf_ptr_->consume();
    buf_ptr_->produce();
  }

  MSCCLPP_CUDATHROW(cudaStreamWaitEvent(*streams_[1], event0, 0));
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(tasks.back().dst, buf_ptr_->next_get().data(), tasks.back().size, cudaMemcpyDefault,
                                    *streams_[1]));
  MSCCLPP_CUDATHROW(cudaEventRecord(event1, *streams_[1]));
  buf_ptr_->consume();

  MSCCLPP_CUDATHROW(cudaEventDestroy(event0));
  MSCCLPP_CUDATHROW(cudaEventDestroy(event1));
}

void VortexScheduler::sync() {
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(*streams_[0]));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(*streams_[1]));
}

void IndirectConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                               uint64_t size) {
  if (dstOffset + size > dst.size() || srcOffset + size > src.size()) {
    throw Error("IndirectionConnection::write out of bounds", ErrorCode::InvalidUsage);
  }
  auto tasks = scheduler_ptr_->produce_tasks(dst.data() + dstOffset, src.data() + srcOffset, size);
  scheduler_ptr_->launch(tasks);
}

void IndirectConnection::flush(int64_t timeoutUsec) {
  if (timeoutUsec != -1) {
    throw std::runtime_error("IndirectConnection does not support timeout in flush");
  }
  scheduler_ptr_->sync();
}
Transport IndirectConnection::transport() const { return Transport::CudaIpc; }
Transport IndirectConnection::remoteTransport() const { return Transport::CudaIpc; }

}  // namespace mscclpp