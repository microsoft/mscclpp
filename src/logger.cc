// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "logger.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>

namespace mscclpp {

static LogLevel stringToLogLevel(const std::string& levelStr) {
  // capitalize
  std::string upperStr = levelStr;
  std::transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::toupper);
  if (upperStr == "NONE") {
    return LogLevel::NONE;
  } else if (upperStr == "DEBUG") {
    return LogLevel::DEBUG;
  } else if (upperStr == "INFO") {
    return LogLevel::INFO;
  } else if (upperStr == "WARN") {
    return LogLevel::WARN;
  } else if (upperStr == "ERROR") {
    return LogLevel::ERROR;
  }
  return LogLevel::ERROR;  // Shouldn't reach here
}

static unsigned int stringToSubsysFlags(const std::string& subsysStr) {
  bool invert = false;
  std::string str = subsysStr;
  if (!str.empty() && str[0] == '^') {
    invert = true;
    str = str.substr(1);
  }
  std::string upperStr = str;
  std::transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::toupper);
  unsigned int flag = 0;
  size_t start = 0;
  size_t end = upperStr.find(',');
  if (end == std::string::npos && !upperStr.empty()) {
    end = upperStr.length();
  }
  while (end != std::string::npos) {
    std::string token = upperStr.substr(start, end - start);
    if (token == "ENV") {
      flag |= LogSubsysFlag::ENV;
    } else if (token == "NET") {
      flag |= LogSubsysFlag::NET;
    } else if (token == "CONN") {
      flag |= LogSubsysFlag::CONN;
    } else if (token == "EXEC") {
      flag |= LogSubsysFlag::EXEC;
    } else if (token == "NCCL") {
      flag |= LogSubsysFlag::NCCL;
    } else if (token == "ALL") {
      flag |= LogSubsysFlag::ALL;
    }
    start = end + 1;
    end = upperStr.find(',', start);
  }
  if (invert) {
    flag = ~flag;
  }
  return flag;
}

namespace detail {

// Return the full path of the detected project root directory, or empty string if not found.
static std::string guessFindProjectRoot(const std::string& filePath) {
  const std::string rootIndicators[] = {".git", "VERSION", "README.md"};

  // Start from the directory containing the file
  size_t lastSlashInitial = filePath.find_last_of('/');
  if (lastSlashInitial == std::string::npos) {
    return "";  // No directory structure
  }
  std::string currentDir = filePath.substr(0, lastSlashInitial);

  // Walk up the directory tree towards root
  while (!currentDir.empty() && currentDir != "/") {
    // Check indicators in currentDir
    bool foundRoot = false;
    for (const auto& indicator : rootIndicators) {
      std::filesystem::path potential = std::filesystem::path(currentDir) / indicator;
      std::error_code ec;  // non-throwing exists
      if (std::filesystem::exists(potential, ec)) {
        foundRoot = true;
        break;
      }
    }
    if (foundRoot) {
      return currentDir;  // Full path to root directory
    }
    size_t lastSlash = currentDir.find_last_of('/');
    if (lastSlash == std::string::npos) break;
    currentDir = currentDir.substr(0, lastSlash);
  }
  return "";  // Not found
}

// Remove the project root prefix (including trailing '/') from filePath if present.
static std::string removeProjectRootPrefix(const std::string& filePath, const std::string& projectRoot) {
  if (projectRoot.empty()) return filePath;   // Nothing to remove
  if (filePath.rfind(projectRoot, 0) == 0) {  // starts with projectRoot
    size_t cut = projectRoot.size();
    if (cut < filePath.size() && filePath[cut] == '/') {
      ++cut;  // Skip slash after root
    }
    return filePath.substr(cut);
  }
  // projectRoot not a prefix, return original
  return filePath;
}

static std::string globalProjectRoot;
static std::mutex globalProjectRootMutex;

std::string guessRemoveProjectPrefix(const std::string& filePath) {
  if (globalProjectRoot.empty()) {
    std::lock_guard<std::mutex> lock(globalProjectRootMutex);
    if (globalProjectRoot.empty()) {
      std::string candidate = guessFindProjectRoot(filePath);
      if (!candidate.empty()) {
        globalProjectRoot = std::move(candidate);
      }
    }
  }
  return removeProjectRootPrefix(filePath, globalProjectRoot);
}

std::string timestamp(const char* format) {
  // Thread-safe per-second UTC timestamp cache.
  // Uses mutex + shared_ptr (atomic<T> requires trivially copyable T in C++17).
  struct TimeCache {
    time_t second;
    std::string str;
  };
  static std::shared_ptr<const TimeCache> cachePtr;  // guarded by cacheMutex
  static std::mutex cacheMutex;

  auto now = std::chrono::system_clock::now();
  time_t currentTime = std::chrono::system_clock::to_time_t(now);

  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    if (cachePtr && cachePtr->second == currentTime) {
      return cachePtr->str;  // Safe stale/current value.
    }
  }

  // Build new formatted timestamp (UTC) for this second.
  std::tm tmBuf;
  if (::gmtime_r(&currentTime, &tmBuf) == nullptr) {
    return "";  // Conversion failure fallback.
  }
  std::stringstream ss;
  ss << std::put_time(&tmBuf, format);
  auto newCache = std::make_shared<TimeCache>(TimeCache{currentTime, ss.str()});

  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    cachePtr = std::move(newCache);
    return cachePtr->str;
  }
}

std::string subsysFlagToString(unsigned int flag) {
  switch (flag) {
    case LogSubsysFlag::ENV:
      return "ENV";
    case LogSubsysFlag::NET:
      return "NET";
    case LogSubsysFlag::CONN:
      return "CONN";
    case LogSubsysFlag::EXEC:
      return "EXEC";
    case LogSubsysFlag::NCCL:
      return "NCCL";
    case LogSubsysFlag::ALL:
      return "ALL";
    default:
      return "UNKNOWN";
  }
}

}  // namespace detail

static std::once_flag globalLoggerInitFlag;
static std::shared_ptr<Logger> globalLoggerPtr;

Logger::Logger(const std::string& header, const LogLevel level, const char delimiter)
    : header_(header), level_(level), delimiter_(delimiter) {
  subsysFlags_ = stringToSubsysFlags(env()->logSubsys);
  const std::string& path = env()->logFile;
  if (!path.empty()) {
    logFileStream_.open(path, std::ios::out | std::ios::app);
    if (!logFileStream_.is_open()) {
      // Fallback notice
      std::cerr << "MSCCLPP Logger: failed to open log file '" << path << "', using stdout." << std::endl;
    }
  }
}

Logger& logger(const std::string& header, const std::string& levelStr, char delimiter) {
  std::call_once(globalLoggerInitFlag, [&]() {
    LogLevel level = stringToLogLevel(levelStr);
    globalLoggerPtr = std::make_shared<Logger>(header, level, delimiter);
  });
  return *globalLoggerPtr;
}

}  // namespace mscclpp
