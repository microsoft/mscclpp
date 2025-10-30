// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
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
  return LogLevel::ERROR;  // Won't reach here
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
  const std::string root_indicators[] = {".git", "README.md", "README", "readme"};

  std::string currentDir = filePath;

  // Walk up the directory tree
  while (!currentDir.empty() && currentDir != "/") {
    // Get parent directory
    size_t lastSlash = currentDir.find_last_of('/');
    if (lastSlash == std::string::npos) break;

    currentDir = currentDir.substr(0, lastSlash);

    // Check if any root indicator exists in this directory
    for (const auto& indicator : root_indicators) {
      std::string potentialFile = currentDir + "/" + indicator;
      // You'd need to check if file exists (using filesystem or stat)
      // If found, return the filePath excluding the directory name
      size_t nameStart = currentDir.find_last_of('/');
      if (nameStart != std::string::npos) {
        return filePath.substr(nameStart + 1);
      }
    }
  }
  return filePath;  // No project root found
}

[[maybe_unused]] std::string removeProjectPrefix(const std::string& filePath, const std::string& projectPrefix = "/") {
  size_t pos = filePath.find(projectPrefix);
  if (pos != std::string::npos) {
    return filePath.substr(pos + projectPrefix.length());
  }
  return filePath;  // No prefix found, return original path
}

std::string timestamp(const char* format) {
  auto now = std::chrono::system_clock::now();
  auto inTime = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&inTime), format);
  return ss.str();
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

static std::unordered_map<std::string, std::unique_ptr<Logger>> allLoggers;

Logger::Logger(const std::string& header, const LogLevel level, const char delimeter)
    : header_(header), level_(level), delimeter_(delimeter) {
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

Logger& logger(const std::string& name, const std::string& header, const std::string& levelStr, char delimeter) {
  auto it = allLoggers.find(name);
  if (it != allLoggers.end()) {
    return *(it->second);
  }
  LogLevel level = stringToLogLevel(levelStr);
  allLoggers[name] = std::make_unique<Logger>(header, level, delimeter);
  return *(allLoggers[name]);
}

}  // namespace mscclpp
