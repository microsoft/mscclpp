// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_plan.hpp"

#include <cassert>
#include <fstream>
#include <set>

namespace {
template <typename T, typename Predicate>
std::vector<T> filter(const std::vector<T>& vec, Predicate pred) {
  std::vector<T> filtered;
  std::copy_if(vec.begin(), vec.end(), std::back_inserter(filtered), pred);
  return filtered;
}

auto getOpType = [](const std::string& str) {
  if (str == "nop") {
    return mscclpp::OperationType::NOP;
  } else if (str == "barrier") {
    return mscclpp::OperationType::BARRIER;
  } else if (str == "put") {
    return mscclpp::OperationType::PUT;
  } else if (str == "pws") {
    return mscclpp::OperationType::PUT_WITH_SIGNAL;
  } else if (str == "pwsf") {
    return mscclpp::OperationType::PUT_WITH_SIGNAL_AND_FLUSH;
  } else if (str == "get") {
    return mscclpp::OperationType::GET;
  } else if (str == "copy") {
    return mscclpp::OperationType::COPY;
  } else if (str == "signal") {
    return mscclpp::OperationType::SIGNAL;
  } else if (str == "wait") {
    return mscclpp::OperationType::WAIT;
  } else if (str == "flush") {
    return mscclpp::OperationType::FLUSH;
  } else if (str == "re") {
    return mscclpp::OperationType::REDUCE;
  } else if (str == "rs") {
    return mscclpp::OperationType::REDUCE_SEND;
  } else if (str == "rrc") {
    return mscclpp::OperationType::READ_REDUCE_COPY;
  } else if (str == "rrcs") {
    return mscclpp::OperationType::READ_REDUCE_COPY_SEND;
  } else if (str == "ppkt") {
    return mscclpp::OperationType::PUT_PACKET;
  } else if (str == "rppkt") {
    return mscclpp::OperationType::READ_PUT_PACKET;
  } else if (str == "rspkt") {
    return mscclpp::OperationType::REDUCE_SEND_PACKET;
  } else if (str == "cpkt") {
    return mscclpp::OperationType::COPY_PACKET;
  } else if (str == "tpkt") {
    return mscclpp::OperationType::TRANSFORM_TO_PACKET;
  } else if (str == "rpkt") {
    return mscclpp::OperationType::REDUCE_PACKET;
  } else if (str == "glres") {
    return mscclpp::OperationType::MULTI_LOAD_REDUCE_STORE;
  } else if (str == "rsignal") {
    return mscclpp::OperationType::RELAXED_SIGNAL;
  } else if (str == "rwait") {
    return mscclpp::OperationType::RELAXED_WAIT;
  } else if (str == "pipeline") {
    return mscclpp::OperationType::PIPELINE;
  } else if (str == "sem_acquire") {
    return mscclpp::OperationType::SEM_ACQUIRE;
  } else if (str == "sem_release") {
    return mscclpp::OperationType::SEM_RELEASE;
  } else {
    throw mscclpp::Error("Invalid operation type", mscclpp::ErrorCode::ExecutorError);
  }
};

auto convertToBufferType = [](const std::string& str) {
  if (str == "i") {
    return mscclpp::BufferType::INPUT;
  } else if (str == "o") {
    return mscclpp::BufferType::OUTPUT;
  } else if (str == "s") {
    return mscclpp::BufferType::SCRATCH;
  } else {
    throw mscclpp::Error("Invalid buffer type", mscclpp::ErrorCode::ExecutorError);
  }
};

auto convertToChannelType = [](const std::string& str) {
  if (str == "memory") {
    return mscclpp::ChannelType::MEMORY;
  } else if (str == "port") {
    return mscclpp::ChannelType::PORT;
  } else if (str == "none") {
    return mscclpp::ChannelType::NONE;
  } else if (str == "switch") {
    return mscclpp::ChannelType::SWITCH;
  } else {
    throw mscclpp::Error("Invalid channel type", mscclpp::ErrorCode::ExecutorError);
  }
};

std::set groupChannelType{mscclpp::ChannelType::SWITCH};

}  // namespace

namespace mscclpp {
using json = nlohmann::json;

ExecutionPlan::Impl::Impl(const std::string& planPath, int rank)
    : planPath(planPath), isUsingPacket(false), rank(rank) {
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  this->name = obj["name"];
  this->collective = obj["collective"];
  this->isInPlace = obj["inplace"];
  this->bufferAlignment = obj.value("buffer_alignment", 16);
  this->minMessageSize = obj.value("min_message_size", 0);
  this->maxMessageSize = obj.value("max_message_size", std::numeric_limits<uint64_t>::max());
}

std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfos(int rank, ChannelType channelType) const {
  auto pred = [channelType](const ChannelInfo& info) { return info.channelType == channelType; };
  return filter(this->channelInfos.at(rank), pred);
}

std::vector<ChannelInfo> ExecutionPlan::Impl::getUnpairedChannelInfos(int rank, int worldSize,
                                                                      ChannelType channelType) {
  std::vector<ChannelInfo> unpaired;
  for (int peer = 0; peer < worldSize; peer++) {
    if (peer == rank) {
      continue;
    }
    if (this->channelCountMap[{rank, channelType}][peer] < this->channelCountMap[{peer, channelType}][rank]) {
      int count = this->channelCountMap[{peer, channelType}][rank] - this->channelCountMap[{rank, channelType}][peer];
      for (int i = 0; i < count; i++) {
        ChannelInfo info;
        info.channelType = channelType;
        info.connectedPeers.push_back(peer);
        unpaired.push_back(info);
      }
    }
  }
  return unpaired;
}

std::vector<NvlsInfo> ExecutionPlan::Impl::getNvlsInfos(int rank, size_t sendBuffserSize, size_t recvBufferSize) const {
  if (sendBuffserSize == 0 && recvBufferSize == 0) {
    return this->nvlsInfos.at(rank);
  }
  size_t chunkSize = this->getUpperBoundChunkSize(sendBuffserSize, recvBufferSize);
  std::vector<NvlsInfo> infos = this->nvlsInfos.at(rank);
  for (auto& info : infos) {
    info.bufferSize = info.bufferSize * chunkSize;
  }
  return infos;
}

std::vector<int> ExecutionPlan::Impl::getConnectedPeers(int rank) const {
  std::set<int> peers;
  for (const auto& info : this->channelInfos.at(rank)) {
    for (int peer : info.connectedPeers) {
      peers.insert(peer);
    }
  }
  for (const auto& info : this->channelInfosByDstRank.at(rank)) {
    for (int peer : info.connectedPeers) {
      peers.insert(peer);
    }
  }
  return std::vector<int>(peers.begin(), peers.end());
}

std::vector<BufferInfo> ExecutionPlan::Impl::getRemoteBufferInfos(int rank) const {
  if (this->remoteBufferInfos.find(rank) == this->remoteBufferInfos.end()) {
    return std::vector<BufferInfo>();
  }
  return this->remoteBufferInfos.at(rank);
}

std::vector<BufferInfo> ExecutionPlan::Impl::getLocalBufferToSend(int rank) const {
  if (this->localBufferToSend.find(rank) == this->localBufferToSend.end()) {
    return std::vector<BufferInfo>();
  }
  return this->localBufferToSend.at(rank);
}

size_t ExecutionPlan::Impl::calScratchBufferSize(size_t inputSize, size_t outputSize) const {
  size_t sizePerChunk = 0;
  size_t size = 0;
  if (this->inputChunks != 0) {
    sizePerChunk = (inputSize + this->inputChunks - 1) / this->inputChunks;
  } else if (this->outputChunks != 0) {
    sizePerChunk = (outputSize + this->outputChunks - 1) / this->outputChunks;
  } else {
    throw mscclpp::Error("Output or Input chunks must be greater than 0", mscclpp::ErrorCode::ExecutorError);
  }

  if (this->isUsingPacket) {
    size = sizePerChunk * this->scratchChunks * 2 /* data + flag*/ * 2 /*double buffer*/;
  } else {
    size = sizePerChunk * this->scratchChunks;
  }
  return (size + this->bufferAlignment - 1) / this->bufferAlignment * this->bufferAlignment;
}

size_t ExecutionPlan::Impl::calScratchChunkSize(size_t scratchSize) const {
  if (this->scratchChunks == 0) {
    return 0;
  }
  size_t size = (scratchSize + this->scratchChunks - 1) / this->scratchChunks;
  return (size + this->bufferAlignment - 1) / this->bufferAlignment * this->bufferAlignment;
}

std::vector<Operation> ExecutionPlan::Impl::getOperations(int threadblock) const {
  return this->operations[threadblock];
}

int ExecutionPlan::Impl::getThreadblockCount() const { return this->operations.size(); }

int ExecutionPlan::Impl::getNThreadsPerBlock() const { return this->nThreadsPerBlock; }

// TODO: setup OPs only for current rank
void ExecutionPlan::Impl::loadExecutionPlan(int rank, size_t inputSize, size_t outputSize, size_t contsSrcOffset,
                                            size_t constDstOffset) {
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  if (this->name != obj["name"]) {
    throw Error("Plan name does not match", ErrorCode::ExecutorError);
  }
  this->collective = obj["collective"];
  std::string protocol = obj["protocol"];
  if (protocol == "LL") {
    this->isUsingPacket = true;
  }
  this->inputSize = inputSize;
  this->outputSize = outputSize;
  this->nThreadsPerBlock = obj.value("num_threads_per_block", 1024);
  this->minMessageSize = obj.value("min_message_size", 0);
  this->maxMessageSize = obj.value("max_message_size", std::numeric_limits<uint64_t>::max());
  if (!isMessageSizeValid(inputSize, outputSize)) {
    throw Error("Input or output size is not valid", ErrorCode::ExecutorError);
  }

  this->isInPlace = obj["inplace"];
  const auto& gpus = obj["gpus"];

  auto& gpu = gpus[rank];
  if (gpu["id"] != rank) {
    throw Error("GPU rank does not match", ErrorCode::ExecutorError);
  }
  this->inputChunks = gpu["input_chunks"];
  this->outputChunks = gpu["output_chunks"];
  this->scratchChunks = gpu["scratch_chunks"];
  this->setupChannels(gpus);
  this->setupRemoteBuffers(gpus);
  this->setupSemaphores(gpu);
  this->setupOperations(gpu, contsSrcOffset, constDstOffset);
}

void ExecutionPlan::Impl::lightLoadExecutionPlan(int rank, size_t inputSize, size_t outputSize, size_t contsSrcOffset,
                                                 size_t constDstOffset) {
  if (!isMessageSizeValid(inputSize, outputSize)) {
    throw Error("Input or output size is not valid", ErrorCode::ExecutorError);
  }
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  if (this->name != obj["name"]) {
    throw Error("Plan name does not match", ErrorCode::ExecutorError);
  }
  std::string protocol = obj["protocol"];
  if (protocol == "LL") {
    this->isUsingPacket = true;
  }
  const auto& gpus = obj["gpus"];
  const auto& gpu = gpus[rank];
  if (gpu["id"] != rank) {
    throw Error("GPU rank does not match", ErrorCode::ExecutorError);
  }

  this->inputChunks = gpu["input_chunks"];
  this->outputChunks = gpu["output_chunks"];
  this->scratchChunks = gpu["scratch_chunks"];

  this->inputSize = inputSize;
  this->outputSize = outputSize;
  this->setupOperations(gpus, contsSrcOffset, constDstOffset);
}

bool ExecutionPlan::Impl::isMessageSizeValid(size_t inputSize, size_t outputSize) const {
  size_t size = inputSize;
  if (this->collective == "allgather") {
    size = outputSize;
  }
  if (size < this->minMessageSize || size > this->maxMessageSize) {
    return false;
  }
  return true;
}

void ExecutionPlan::Impl::parseChannels(const json& gpu, std::vector<ChannelInfo>& channelInfos,
                                        std::vector<NvlsInfo>& nvlsInfos,
                                        std::map<std::pair<int, ChannelType>, std::vector<int>>& chanConnectedPeersMap,
                                        int rank) {
  for (const auto& channel : gpu["channels"]) {
    ChannelType chanType = convertToChannelType(channel["type"]);

    if (chanType == ChannelType::SWITCH) {
      NvlsInfo info;
      info.bufferType = convertToBufferType(channel["buff"]);
      for (const auto& group : channel["rank_groups"]) {
        info.bufferSize = (int)group["size"];
        info.ranks.clear();
        for (int rank : group["ranks"]) {
          info.ranks.push_back(rank);
        }
        nvlsInfos.push_back(info);
      }
    } else {
      ChannelInfo info;
      info.channelType = convertToChannelType(channel["type"]);
      for (const auto& peer : channel["connected_to"]) {
        info.connectedPeers.push_back(peer);
        chanConnectedPeersMap[{peer, info.channelType}].push_back(rank);
        this->channelCountMap[{rank, info.channelType}][peer]++;
      }
      channelInfos.push_back(info);
    }
  }
}

void ExecutionPlan::Impl::parseRemoteBuffer(const nlohmann::json& gpu, int rank) {
  auto& bufferInfos = this->remoteBufferInfos[rank];
  auto& bufferIndexMap = this->bufferIndexMap_[rank];
  std::unordered_map<ChannelType, int> channelCountMap;
  for (auto& remoteBuffer : gpu["remote_buffers"]) {
    int bufferId = bufferInfos.size();
    int oriRank = remoteBuffer["rank"];
    BufferType bufferType = convertToBufferType(remoteBuffer["type"]);
    std::vector<ChannelType> accessChannels;
    for (const auto& channel : remoteBuffer["access_channel_types"]) {
      ChannelType chanType = convertToChannelType(channel);
      accessChannels.push_back(chanType);
      bufferIndexMap[{bufferId, chanType}] = channelCountMap[chanType]++;
    }
    BufferInfo info{oriRank, rank, bufferType, accessChannels};
    bufferInfos.push_back(info);
    this->localBufferToSend[oriRank].push_back(info);
  }
}

// Construct the channel info. Step 1. Flatten MEMORY and PORT channels into separate vectors.
// Step 2. For each threadblock, construct a vector of channel indexes and keys.
void ExecutionPlan::Impl::setupChannels(const json& gpus) {
  using mapKey = std::pair<int, ChannelType>;
  std::map<mapKey, std::vector<int>> chanConnectedPeersMap;
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    std::vector<ChannelInfo> channelInfos;
    std::vector<NvlsInfo> nvlsInfos;
    this->parseChannels(gpu, channelInfos, nvlsInfos, chanConnectedPeersMap, rank);
    this->channelInfos[rank] = channelInfos;
    this->nvlsInfos[rank] = nvlsInfos;
  }

  for (const auto& [key, connectedFrom] : chanConnectedPeersMap) {
    auto [peer, channelType] = key;
    ChannelInfo info;
    info.channelType = channelType;
    info.connectedPeers = connectedFrom;
    this->channelInfosByDstRank[peer].push_back(info);
  }

  // setup threadblockChannelMap
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    auto channelTypes = {ChannelType::MEMORY, ChannelType::PORT, ChannelType::SWITCH};
    std::unordered_map<ChannelKey, std::vector<int>> channelMap;
    for (auto channelType : channelTypes) {
      const std::vector<ChannelInfo> channelInfos = this->getChannelInfos(rank, channelType);
      int index = 0;
      if (channelType == ChannelType::SWITCH) {
        const std::vector<NvlsInfo> nvlsInfos = this->getNvlsInfos(rank);
        for (const auto& info : nvlsInfos) {
          ChannelKey key = {info.bufferType, ChannelType::SWITCH};
          channelMap[key].push_back(index++);
        }
      } else {
        for (const auto& info : channelInfos) {
          ChannelKey key = {BufferType::NONE, info.channelType};
          for (size_t i = 0; i < info.connectedPeers.size(); i++) {
            channelMap[key].push_back(index++);
          }
        }
      }
    }
    int nthreadblocks = gpu["threadblocks"].size();
    this->threadblockMemoryChannelMap[rank].resize(nthreadblocks);
    this->threadblockPortChannelMap[rank].resize(nthreadblocks);
    this->threadblockNvlsChannelMap[rank].resize(nthreadblocks);
    for (const auto& threadblock : gpu["threadblocks"]) {
      for (const auto& channel : threadblock["channels"]) {
        ChannelType channelType = convertToChannelType(channel["channel_type"]);
        ChannelKey key = {BufferType::NONE, channelType};
        if (channel.contains("buff")) {
          key = {convertToBufferType(channel["buff"]), channelType};
        }
        for (int id : channel["channel_ids"]) {
          if (channelType == ChannelType::MEMORY) {
            this->threadblockMemoryChannelMap[rank][threadblock["id"]].emplace_back(channelMap[key][id]);
          } else if (channelType == ChannelType::PORT) {
            this->threadblockPortChannelMap[rank][threadblock["id"]].emplace_back(channelMap[key][id]);
          } else if (channelType == ChannelType::SWITCH) {
            this->threadblockNvlsChannelMap[rank][threadblock["id"]].emplace_back(channelMap[key][id]);
          }
        }
      }
    }
  }
}

void ExecutionPlan::Impl::setupRemoteBuffers(const json& gpus) {
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    this->parseRemoteBuffer(gpu, rank);
  }

  // setup threadblockBufferMap
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    int nthreadblocks = gpu["threadblocks"].size();
    this->threadblockMemoryChannelBufferMap[rank].resize(nthreadblocks);
    this->threadblockPortChannelBufferMap[rank].resize(nthreadblocks);
    for (const auto& threadblock : gpu["threadblocks"]) {
      if (!threadblock.contains("remote_buffer_refs")) {
        continue;
      }
      for (const auto& remoteBuffRef : threadblock["remote_buffer_refs"]) {
        ChannelType accessChanType = convertToChannelType(remoteBuffRef["access_channel_type"]);
        if (accessChanType == ChannelType::PORT) {
          for (const auto& bufferId : remoteBuffRef["remote_buffer_ids"]) {
            this->threadblockPortChannelBufferMap[rank][threadblock["id"]].push_back(
                this->bufferIndexMap_[rank][{bufferId, accessChanType}]);
          }
        } else if (accessChanType == ChannelType::MEMORY) {
          for (const auto& bufferId : remoteBuffRef["remote_buffer_ids"]) {
            this->threadblockMemoryChannelBufferMap[rank][threadblock["id"]].push_back(
                this->bufferIndexMap_[rank][{bufferId, accessChanType}]);
          }
        }
      }
    }
  }
}

void ExecutionPlan::Impl::setupSemaphores(const json& gpu) {
  if (!gpu.contains("semaphores")) {
    return;
  }
  for (const auto& sem : gpu["semaphores"]) {
    SemaphoreInfo info;
    info.initValue = sem["init_value"];
    this->semaphoreInfos.push_back(info);
  }
}

void ExecutionPlan::Impl::setupOperations(const json& gpu, size_t constSrcOffset, size_t constDstOffset) {
  // setup threadblocks and operations
  for (const auto& threadblock : gpu["threadblocks"]) {
    int threadBlockId = threadblock["id"];
    std::vector<Operation> ops;
    for (const auto& op : threadblock["ops"]) {
      Operation operation = {};
      this->setupOperation(op, operation, rank, threadBlockId, constSrcOffset, constDstOffset);
      ops.push_back(operation);
      if (operation.type == OperationType::PIPELINE) {
        for (const auto& innerOp : op["ops"]) {
          Operation pipelineOp = {};
          this->setupOperation(innerOp, pipelineOp, rank, threadBlockId, constSrcOffset, constDstOffset);
          ops.push_back(pipelineOp);
        }
      }
    }
    this->operations.push_back(ops);
  }
}

void ExecutionPlan::Impl::setupOperation(const nlohmann::json& op, Operation& operation, int rank, int threadBlockId,
                                         size_t constSrcOffset, size_t constDstOffset) {
  auto getConstOffset = [&](BufferType type) -> size_t {
    switch (type) {
      case BufferType::INPUT:
        return constSrcOffset;
      case BufferType::OUTPUT:
        return constDstOffset;
      case BufferType::SCRATCH:
        return 0;
      default:
        throw Error("Invalid buffer type", ErrorCode::ExecutorError);
    }
  };

  auto getRemoteBufferTypeWithId = [&](int bufferId, int rank, int threadBlockId,
                                       ChannelType channelType) -> BufferType {
    int id = -1;
    if (channelType == ChannelType::MEMORY) {
      id = this->threadblockMemoryChannelBufferMap[rank][threadBlockId][bufferId];
    } else if (channelType == ChannelType::PORT) {
      id = this->threadblockPortChannelBufferMap[rank][threadBlockId][bufferId];
    } else {
      throw Error("Invalid channel type", ErrorCode::ExecutorError);
    }
    return this->remoteBufferInfos[rank][id].bufferType;
  };

  operation.type = static_cast<mscclpp::OperationType>(getOpType(op["name"]));
  if (op.contains("channel_type")) {
    operation.channelType = convertToChannelType(op["channel_type"]);
  }
  if (op.contains("channel_ids")) {
    operation.nChannels = op["channel_ids"].size();
    for (uint32_t i = 0; i < op["channel_ids"].size(); i++) {
      operation.channelIndexes[i] = op["channel_ids"][i];
    }
  }
  if (op.contains("src_buff")) {
    operation.nInputs = op["src_buff"].size();
    for (int i = 0; i < operation.nInputs; i++) {
      auto& buff = op["src_buff"][i];
      size_t constOffset = 0;
      if (buff.contains("type")) {
        operation.inputBufferRefs[i].type = convertToBufferType(buff["type"]);
        constOffset = getConstOffset(operation.inputBufferRefs[i].type);
      }
      if (buff.contains("buff_id")) {
        operation.inputBufferRefs[i].id = buff["buff_id"];
        BufferType bufferType = getRemoteBufferTypeWithId(buff["buff_id"], rank, threadBlockId, operation.channelType);
        constOffset = getConstOffset(bufferType);
      }
      if (buff.contains("switch_channel_id")) {
        int switchChannelIdx = this->threadblockNvlsChannelMap[rank][threadBlockId][buff["switch_channel_id"]];
        BufferType bufferType = this->nvlsInfos[rank][switchChannelIdx].bufferType;
        constOffset = getConstOffset(bufferType);
        operation.nvlsInputBufferType = bufferType;
        operation.nvlsInputIndex = buff["switch_channel_id"];
      }
      // TODO: here we need to use another offset, if scratch and algo reusable, get another scrathc offset
      operation.inputOffsets[i] = this->getOffset(this->inputSize, this->outputSize, buff["index"]) + constOffset;
      operation.inputBufferSizes[i] =
          this->getBufferSize(this->inputSize, this->outputSize, buff["index"], buff["size"]);
    }
  }
  if (op.contains("dst_buff")) {
    operation.nOutputs = op["dst_buff"].size();
    for (int i = 0; i < operation.nOutputs; i++) {
      auto& buff = op["dst_buff"][i];
      size_t constOffset = 0;
      if (buff.contains("type")) {
        operation.outputBufferRefs[i].type = convertToBufferType(buff["type"]);
        constOffset = getConstOffset(operation.outputBufferRefs[i].type);
      }
      if (buff.contains("buff_id")) {
        operation.outputBufferRefs[i].id = buff["buff_id"];
        BufferType bufferType = getRemoteBufferTypeWithId(buff["buff_id"], rank, threadBlockId, operation.channelType);
        constOffset = getConstOffset(bufferType);
      }
      if (buff.contains("switch_channel_id")) {
        int switchChannelIdx = this->threadblockNvlsChannelMap[rank][threadBlockId][buff["switch_channel_id"]];
        BufferType bufferType = this->nvlsInfos[rank][switchChannelIdx].bufferType;
        constOffset = getConstOffset(bufferType);
        operation.nvlsOutputBufferType = bufferType;
        operation.nvlsOutputIndex = buff["switch_channel_id"];
      }
      operation.outputOffsets[i] = this->getOffset(this->inputSize, this->outputSize, buff["index"]) + constOffset;
      operation.outputBufferSizes[i] =
          this->getBufferSize(this->inputSize, this->outputSize, buff["index"], buff["size"]);
    }
  }
  if (op.contains("barrier_id")) {
    operation.deviceSyncerIndex = op["barrier_id"];
  }
  if (op.contains("nthread_blocks")) {
    operation.nThreadBlocks = op["nthread_blocks"];
  }
  if (op.contains("semaphore_ids")) {
    operation.nDeviceSemaphores = op["semaphore_ids"].size();
    for (uint32_t id = 0; id < operation.nDeviceSemaphores; id++) {
      operation.deviceSemaphoreIds[id] = op["semaphore_ids"][id];
    }
  }
  if (op.contains("iter_context")) {
    operation.unitSize = op["iter_context"]["unit_size"];
    operation.nOperations = op["ops"].size();
    int nChunks = op["iter_context"]["num_chunks"];
    size_t sizes = nChunks * getUpperBoundChunkSize(this->inputSize, this->outputSize);
    operation.nIterations = (sizes + (operation.unitSize - 1)) / operation.unitSize;
  }
}

std::pair<size_t, uint32_t> ExecutionPlan::Impl::getSizeAndChunks(size_t inputSize, size_t outputSize) const {
  std::pair<size_t, uint32_t> sizePerRank;
  if (this->inputChunks == 0 && this->outputChunks == 0) {
    throw mscclpp::Error("Output or Input chunks must be greater than 0", mscclpp::ErrorCode::ExecutorError);
  } else if (this->inputChunks != 0 && this->outputChunks != 0) {
    if (inputSize / this->inputChunks != outputSize / this->outputChunks)
      throw mscclpp::Error("Size per chunks inconsistent: inputSize " + std::to_string(inputSize) + " inputChunks " +
                               std::to_string(this->inputChunks) + " outputSize " + std::to_string(outputSize) +
                               " outputChunks " + std::to_string(this->outputChunks),
                           mscclpp::ErrorCode::ExecutorError);
    else
      sizePerRank = std::make_pair(inputSize, this->inputChunks);
  } else if (this->inputChunks != 0) {
    sizePerRank = std::make_pair(inputSize, this->inputChunks);
  } else if (this->outputChunks != 0) {
    sizePerRank = std::make_pair(outputSize, this->outputChunks);
  }
  return sizePerRank;
}

size_t ExecutionPlan::Impl::getOffset(size_t inputSize, size_t outputSize, uint32_t chunkIndex) const {
  if (inputSize % this->bufferAlignment != 0) {
    throw Error("inputSize must be a multiple of alignment", ErrorCode::ExecutorError);
  }

  auto rankSizeAndChunks = getSizeAndChunks(inputSize, outputSize);
  uint32_t nChunks = rankSizeAndChunks.second;
  uint32_t nelems = rankSizeAndChunks.first / (this->bufferAlignment * sizeof(uint8_t));

  uint32_t minNelems = nelems / nChunks;
  uint32_t remainder = nelems % nChunks;
  uint32_t offset = chunkIndex * minNelems + (chunkIndex % nelems < remainder ? chunkIndex % nelems : remainder);
  return static_cast<size_t>(offset) * this->bufferAlignment;
}

size_t ExecutionPlan::Impl::getBufferSize(size_t inputSize, size_t outputSize, uint32_t index, uint32_t nChunks) const {
  uint32_t beginOff = getOffset(inputSize, outputSize, index);
  uint32_t endOff = getOffset(inputSize, outputSize, index + nChunks);
  return endOff - beginOff;
}

size_t ExecutionPlan::Impl::getUpperBoundChunkSize(size_t inputSize, size_t outputSize) const {
  size_t nInputChunks = this->inputChunks;
  size_t nOutputChunks = this->outputChunks;
  if (nInputChunks != 0) {
    size_t nelems = inputSize / this->bufferAlignment;
    return (nelems + nInputChunks - 1) / nInputChunks * this->bufferAlignment;
  }
  if (nOutputChunks != 0) {
    size_t nelems = outputSize / this->bufferAlignment;
    return (nelems + nOutputChunks - 1) / nOutputChunks * this->bufferAlignment;
  }
  throw mscclpp::Error("Output or Input chunks must be greater than 0", mscclpp::ErrorCode::ExecutorError);
}

void ExecutionPlan::Impl::reset() {
  this->operations.clear();
  this->channelInfos.clear();
  this->nvlsInfos.clear();
  this->threadblockMemoryChannelMap.clear();
  this->threadblockPortChannelMap.clear();
  this->threadblockNvlsChannelMap.clear();
}

void ExecutionPlan::Impl::operationsReset() { this->operations.clear(); }

ExecutionPlan::ExecutionPlan(const std::string& planPath, int rank) : impl_(std::make_shared<Impl>(planPath, rank)) {}

std::string ExecutionPlan::name() const { return this->impl_->name; }

std::string ExecutionPlan::collective() const { return this->impl_->collective; }

size_t ExecutionPlan::minMessageSize() const { return this->impl_->minMessageSize; }

size_t ExecutionPlan::maxMessageSize() const { return this->impl_->maxMessageSize; }

bool ExecutionPlan::isInPlace() const { return this->impl_->isInPlace; }

}  // namespace mscclpp
