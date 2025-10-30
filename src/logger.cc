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

static LogSubsysSet stringToLogSubsysSet(const std::string& subsysStr) {
  bool invert = false;
  std::string str = subsysStr;
  if (!str.empty() && str[0] == '^') {
    invert = true;
    str = str.substr(1);
  }
  std::string upperStr = str;
  std::transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::toupper);
  LogSubsysSet set;  // all bits start cleared
  size_t start = 0;
  size_t end = upperStr.find(',');
  if (end == std::string::npos && !upperStr.empty()) {
    end = upperStr.length();
  }
  while (end != std::string::npos) {
    std::string token = upperStr.substr(start, end - start);
    if (token == "ENV") {
      set.set(static_cast<size_t>(LogSubsys::ENV));
    } else if (token == "NET") {
      set.set(static_cast<size_t>(LogSubsys::NET));
    } else if (token == "CONN") {
      set.set(static_cast<size_t>(LogSubsys::CONN));
    } else if (token == "EXEC") {
      set.set(static_cast<size_t>(LogSubsys::EXEC));
    } else if (token == "NCCL") {
      set.set(static_cast<size_t>(LogSubsys::NCCL));
    } else if (token == "ALL") {
      set.set();  // all bits
    }
    start = end + 1;
    end = upperStr.find(',', start);
  }
  if (invert) {
    set.flip();
  }
  return set;
}

namespace detail {

// Return the full path of the detected project root directory, or empty string if not found.
static std::filesystem::path guessFindProjectRoot(const std::filesystem::path& filePath) {
  const std::string rootIndicators[] = {".git", "VERSION", "README.md"};

  // Start from the directory containing the file
  std::filesystem::path currentDir = filePath.parent_path();
  if (currentDir.empty()) return {};

  // Walk up the directory tree towards root
  while (!currentDir.empty() && currentDir != currentDir.root_path()) {
    bool foundRoot = false;
    for (const auto& indicator : rootIndicators) {
      std::filesystem::path potential = currentDir / indicator;
      std::error_code ec;  // non-throwing exists
      if (std::filesystem::exists(potential, ec)) {
        foundRoot = true;
        break;
      }
    }
    if (foundRoot) {
      return currentDir;  // Found root directory path
    }
    currentDir = currentDir.parent_path();
  }
  return {};  // Not found
}

// Remove the project root prefix (including trailing '/') from filePath if present.
static std::string removeProjectRootPrefix(const std::filesystem::path& filePath,
                                           const std::filesystem::path& projectRoot) {
  if (projectRoot.empty()) return filePath.generic_string();

  std::error_code ec;
  std::filesystem::path rel = std::filesystem::relative(filePath, projectRoot, ec);
  if (!ec && !rel.empty()) {
    std::string relStr = rel.generic_string();
    if (!(relStr.size() >= 2 && relStr[0] == '.' && relStr[1] == '.')) {
      return relStr;  // Within project root.
    }
  }

  // Fallback: not within project root.
  return filePath.generic_string();
}

static std::filesystem::path globalProjectRoot;
static std::mutex globalProjectRootMutex;

std::string guessRemoveProjectPrefix(const std::string& filePathStr) {
  std::filesystem::path filePath(filePathStr);
  if (globalProjectRoot.empty()) {
    std::lock_guard<std::mutex> lock(globalProjectRootMutex);
    if (globalProjectRoot.empty()) {
      auto candidate = guessFindProjectRoot(filePath);
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

std::string logSubsysToString(LogSubsys subsys) {
  switch (subsys) {
    case LogSubsys::ENV:
      return "ENV";
    case LogSubsys::NET:
      return "NET";
    case LogSubsys::CONN:
      return "CONN";
    case LogSubsys::EXEC:
      return "EXEC";
    case LogSubsys::NCCL:
      return "NCCL";
    case LogSubsys::COUNT:
      return "ALL";  // COUNT isn't a subsystem; treat only if misused.
    default:
      return "UNKNOWN";
  }
}

}  // namespace detail

static std::once_flag globalLoggerInitFlag;
static std::shared_ptr<Logger> globalLoggerPtr;

Logger::Logger(const std::string& header, const LogLevel level, const char delimiter)
    : header_(header), level_(level), delimiter_(delimiter) {
  subsysSet_ = stringToLogSubsysSet(env()->logSubsys);
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
