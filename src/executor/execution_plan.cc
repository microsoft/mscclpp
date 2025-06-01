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

ExecutionPlan::Impl::Impl(const std::string planPath) : planPath(planPath), isUsingPacket(false) {
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
  size_t chunkSize = this->getUpperBoundChunkSize(rank, sendBuffserSize, recvBufferSize);
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

size_t ExecutionPlan::Impl::getScratchBufferSize(int rank, size_t inputSize, size_t outputSize) const {
  size_t sizePerRank = 0;
  if (this->inputChunks.at(rank) != 0)
    sizePerRank = inputSize / this->inputChunks.at(rank);
  else if (this->outputChunks.at(rank) != 0)
    sizePerRank = outputSize / this->outputChunks.at(rank);
  else
    throw mscclpp::Error("Output or Input chunks must be greater than 0", mscclpp::ErrorCode::ExecutorError);

  if (this->isUsingPacket) {
    return sizePerRank * this->scratchChunks.at(rank) * 2 /* data + flag*/ * 2 /*double buffer*/;
  }
  return sizePerRank * this->scratchChunks.at(rank);
}

size_t ExecutionPlan::Impl::getMaxScratchBufferSize(int rank) const {
  if (this->maxMessageSize == std::numeric_limits<uint64_t>::max()) {
    return std::numeric_limits<size_t>::max();
  }
  size_t sizePerChunk = 0;
  if (this->inputChunks.at(rank) != 0)
    sizePerChunk = maxMessageSize / this->inputChunks.at(rank);
  else if (this->outputChunks.at(rank) != 0)
    sizePerChunk = maxMessageSize / this->outputChunks.at(rank);
  else
    throw mscclpp::Error("Output or Input chunks must be greater than 0", mscclpp::ErrorCode::ExecutorError);

  return this->getScratchBufferSize(rank, sizePerChunk * this->inputChunks.at(rank),
                                    sizePerChunk * this->outputChunks.at(rank));
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
  this->isInPlace = obj["inplace"];
  const auto& gpus = obj["gpus"];

  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    this->inputChunks[rank] = gpu["input_chunks"];
    this->outputChunks[rank] = gpu["output_chunks"];
    this->scratchChunks[rank] = gpu["scratch_chunks"];
  }
  this->setupChannels(gpus);
  this->setupRemoteBuffers(gpus);
  this->setupOperations(gpus, rank, contsSrcOffset, constDstOffset);
}

void ExecutionPlan::Impl::lightLoadExecutionPlan(int rank, size_t inputSize, size_t outputSize, size_t contsSrcOffset,
                                                 size_t constDstOffset) {
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

  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    this->inputChunks[rank] = gpu["input_chunks"];
    this->outputChunks[rank] = gpu["output_chunks"];
    this->scratchChunks[rank] = gpu["scratch_chunks"];
  }

  this->inputSize = inputSize;
  this->outputSize = outputSize;
  this->setupOperations(gpus, rank, contsSrcOffset, constDstOffset);
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

void ExecutionPlan::Impl::setupOperations(const json& gpus, int rank, size_t constSrcOffset, size_t constDstOffset) {
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

  // setup threadblocks and operations
  auto gpu = gpus[rank];
  if (gpu["id"] != rank) {
    throw Error("GPU ID does not match rank", ErrorCode::ExecutorError);
  }
  for (const auto& threadblock : gpu["threadblocks"]) {
    int threadBlockId = threadblock["id"];
    std::unordered_map<ChannelKey, std::vector<int>> channelIndexes;
    std::vector<Operation> ops;
    for (const auto& op : threadblock["ops"]) {
      Operation operation = {};
      std::vector<uint32_t> chunkIndexes;
      operation.type = static_cast<mscclpp::OperationType>(getOpType(op["name"]));
      if (op.contains("channel_type")) {
        operation.channelType = convertToChannelType(op["channel_type"]);
      }
      if (op.contains("channel_ids")) {
        operation.nChannels = op["channel_ids"].size();
        if (operation.channelType == mscclpp::ChannelType::SWITCH) {
          operation.nvlsInputIndex = op["channel_ids"][0];
        } else {
          for (uint32_t i = 0; i < op["channel_ids"].size(); i++) {
            operation.channelIndexes[i] = op["channel_ids"][i];
          }
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
            BufferType bufferType =
                getRemoteBufferTypeWithId(buff["buff_id"], rank, threadBlockId, operation.channelType);
            constOffset = getConstOffset(bufferType);
          }
          if (buff.contains("switch_channel_id")) {
            int switchChannelIdx = this->threadblockNvlsChannelMap[rank][threadBlockId][buff["switch_channel_id"]];
            constOffset = getConstOffset(this->nvlsInfos[rank][switchChannelIdx].bufferType);
          }
          operation.inputOffsets[i] =
              this->getOffset(rank, this->inputSize, this->outputSize, buff["index"]) + constOffset;
          operation.inputBufferSizes[i] =
              this->getBufferSize(rank, this->inputSize, this->outputSize, buff["index"], buff["size"]);
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
            BufferType bufferType =
                getRemoteBufferTypeWithId(buff["buff_id"], rank, threadBlockId, operation.channelType);
            constOffset = getConstOffset(bufferType);
          }
          if (buff.contains("switch_channel_id")) {
            int switchChannelIdx = this->threadblockNvlsChannelMap[rank][threadBlockId][buff["switch_channel_id"]];
            constOffset = getConstOffset(this->nvlsInfos[rank][switchChannelIdx].bufferType);
          }
          operation.outputOffsets[i] =
              this->getOffset(rank, this->inputSize, this->outputSize, buff["index"]) + constOffset;
          operation.outputBufferSizes[i] =
              this->getBufferSize(rank, this->inputSize, this->outputSize, buff["index"], buff["size"]);
        }
      }
      if (op.contains("barrier_id")) {
        operation.deviceSyncerIndex = op["barrier_id"];
      }
      if (op.contains("nthread_blocks")) {
        operation.nThreadBlocks = op["nthread_blocks"];
      }
      ops.push_back(operation);
    }
    this->operations.push_back(ops);
  }
}

std::pair<size_t, uint32_t> ExecutionPlan::Impl::getSizeAndChunksForRank(int rank, size_t inputSize,
                                                                         size_t outputSize) const {
  std::pair<size_t, uint32_t> sizePerRank;
  if (this->inputChunks.at(rank) == 0 && this->outputChunks.at(rank) == 0) {
    throw mscclpp::Error("Output or Input chunks must be greater than 0", mscclpp::ErrorCode::ExecutorError);
  } else if (this->inputChunks.at(rank) != 0 && this->outputChunks.at(rank) != 0) {
    if (inputSize / this->inputChunks.at(rank) != outputSize / this->outputChunks.at(rank))
      throw mscclpp::Error("Size per chunks inconsistent: inputSize " + std::to_string(inputSize) + " inputChunks " +
                               std::to_string(this->inputChunks.at(rank)) + " outputSize " +
                               std::to_string(outputSize) + " outputChunks " +
                               std::to_string(this->outputChunks.at(rank)),
                           mscclpp::ErrorCode::ExecutorError);
    else
      sizePerRank = std::make_pair(inputSize, this->inputChunks.at(rank));
  } else if (this->inputChunks.at(rank) != 0) {
    sizePerRank = std::make_pair(inputSize, this->inputChunks.at(rank));
  } else if (this->outputChunks.at(rank) != 0) {
    sizePerRank = std::make_pair(outputSize, this->outputChunks.at(rank));
  }
  return sizePerRank;
}

size_t ExecutionPlan::Impl::getOffset(int rank, size_t inputSize, size_t outputSize, uint32_t chunkIndex) const {
  if (inputSize % this->bufferAlignment != 0) {
    throw Error("inputSize must be a multiple of alignment", ErrorCode::ExecutorError);
  }

  auto rankSizeAndChunks = getSizeAndChunksForRank(rank, inputSize, outputSize);
  uint32_t nChunks = rankSizeAndChunks.second;
  uint32_t nelems = rankSizeAndChunks.first / (this->bufferAlignment * sizeof(uint8_t));

  uint32_t minNelems = nelems / nChunks;
  uint32_t remainder = nelems % nChunks;
  uint32_t offset = chunkIndex * minNelems + (chunkIndex % nelems < remainder ? chunkIndex % nelems : remainder);
  return static_cast<size_t>(offset) * this->bufferAlignment;
}

size_t ExecutionPlan::Impl::getBufferSize(int rank, size_t inputSize, size_t outputSize, uint32_t index,
                                          uint32_t nChunks) const {
  uint32_t beginOff = getOffset(rank, inputSize, outputSize, index);
  uint32_t endOff = getOffset(rank, inputSize, outputSize, index + nChunks);
  return endOff - beginOff;
}

size_t ExecutionPlan::Impl::getUpperBoundChunkSize(int rank, size_t inputSize, size_t outputSize) const {
  size_t nInputChunks = this->inputChunks.at(rank);
  size_t nOutputChunks = this->outputChunks.at(rank);
  size_t inputChunkSize = 0;
  size_t outputChunkSize = 0;
  if (nInputChunks != 0) {
    inputChunkSize = inputSize / nInputChunks;
  }
  if (nOutputChunks != 0) {
    outputChunkSize = outputSize / nOutputChunks;
  }
  return std::max(inputChunkSize, outputChunkSize);
}

void ExecutionPlan::Impl::reset() {
  this->operations.clear();
  this->channelInfos.clear();
  this->nvlsInfos.clear();
  this->threadblockMemoryChannelMap.clear();
  this->threadblockPortChannelMap.clear();
  this->threadblockNvlsChannelMap.clear();
  this->inputChunks.clear();
  this->outputChunks.clear();
  this->scratchChunks.clear();
}

void ExecutionPlan::Impl::operationsReset() { this->operations.clear(); }

ExecutionPlan::ExecutionPlan(const std::string& planPath) : impl_(std::make_shared<Impl>(planPath)) {}

std::string ExecutionPlan::name() const { return this->impl_->name; }

std::string ExecutionPlan::collective() const { return this->impl_->collective; }

size_t ExecutionPlan::minMessageSize() const { return this->impl_->minMessageSize; }

size_t ExecutionPlan::maxMessageSize() const { return this->impl_->maxMessageSize; }

bool ExecutionPlan::isInPlace() const { return this->impl_->isInPlace; }

}  // namespace mscclpp
