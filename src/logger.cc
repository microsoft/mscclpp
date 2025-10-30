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
  } else if (upperStr == "VERSION") {
    return LogLevel::VERSION;
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

std::string guessRemoveProjectPrefix(const std::string& filePath) {
  // Common project root indicators
  const std::string rootIndicators[] = {".git", "VERSION", "README.md"};

  // Start from the directory containing the file
  size_t lastSlashInitial = filePath.find_last_of('/');
  if (lastSlashInitial == std::string::npos) {
    return filePath;  // No directory separators
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
      // Return path relative to the detected root directory (exclude the root directory name)
      // currentDir = /path/to/repo  filePath = /path/to/repo/src/file.cpp -> want src/file.cpp
      if (filePath.size() > currentDir.size() + 1) {
        return filePath.substr(currentDir.size() + 1);  // skip trailing '/'
      }
      return filePath;  // Fallback / unexpected (file directly at root)
    }

    // Move up one directory
    size_t lastSlash = currentDir.find_last_of('/');
    if (lastSlash == std::string::npos) break;
    currentDir = currentDir.substr(0, lastSlash);
  }
  // No project root indicator found; return original path
  return filePath;
}

std::string timestamp(const char* format) {
  // Cache formatted UTC timestamp per second to reduce formatting overhead
  static std::atomic<time_t> cachedSecond{0};
  static std::string cachedString;
  static std::mutex cacheMutex;

  auto now = std::chrono::system_clock::now();
  time_t currentTime = std::chrono::system_clock::to_time_t(now);

  // Fast path: same second, return cached value
  time_t last = cachedSecond.load(std::memory_order_acquire);
  if (last == currentTime && !cachedString.empty()) {
    return cachedString;
  }

  // Compute new formatted time in UTC
  std::tm tmBuf;
  if (::gmtime_r(&currentTime, &tmBuf) == nullptr) {
    return "";  // Fallback on conversion failure
  }
  std::stringstream ss;
  ss << std::put_time(&tmBuf, format);
  std::string newString = ss.str();

  // Update cache
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    cachedString = std::move(newString);
    cachedSecond.store(currentTime, std::memory_order_release);
    return cachedString;
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
