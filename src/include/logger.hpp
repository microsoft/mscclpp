// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_LOGGER_HPP_
#define MSCCLPP_LOGGER_HPP_

#include <unistd.h>

#include <array>
#include <bitset>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mscclpp/env.hpp>
#include <mscclpp/errors.hpp>
#include <sstream>
#include <string>
#include <string_view>

namespace mscclpp {

typedef enum : unsigned int { NONE = 0, DEBUG, INFO, WARN, ERROR } LogLevel;
typedef enum : std::size_t { ENV = 0, GPU, NET, CONN, EXEC, NCCL, ALGO, COUNT } LogSubsys;

namespace detail {

constexpr std::string_view filenameFromPath(const char* path) {
  if (path == nullptr) return "";
  const char* last = path;
  for (const char* p = path; *p; ++p) {
    if (*p == '/' || *p == '\\') {
      last = p + 1;  // character after separator
    }
  }
  return std::string_view(last);
}

constexpr std::string_view logLevelToString(LogLevel level) {
  switch (level) {
    case LogLevel::NONE:
      return "NONE";
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARN:
      return "WARN";
    case LogLevel::ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
}

constexpr std::string_view logSubsysToString(LogSubsys subsys) {
  // NOTE: keep this in sync with LogSubsys enum and stringToLogSubsysSet function.
  switch (subsys) {
    case LogSubsys::ENV:
      return "ENV";
    case LogSubsys::GPU:
      return "GPU";
    case LogSubsys::NET:
      return "NET";
    case LogSubsys::CONN:
      return "CONN";
    case LogSubsys::EXEC:
      return "EXEC";
    case LogSubsys::NCCL:
      return "NCCL";
    case LogSubsys::ALGO:
      return "ALGO";
    case LogSubsys::COUNT:
      return "ALL";
    default:
      return "UNKNOWN";
  }
}

std::string timestamp(const char* format = "%Y-%m-%d %X");

}  // namespace detail

// Bitset holding enabled subsystems.
using LogSubsysSet = std::bitset<static_cast<std::size_t>(LogSubsys::COUNT)>;

class Logger {
 private:
  // Overload for string literals: const char(&)[N]
  template <std::size_t N>
  std::string toStringHelper(const char (&value)[N]) const {
    // N includes the terminating '\0'; std::string(value) stops at '\0'
    return std::string(value);
  }

  template <typename T>
  std::string toStringHelper(T&& value) const {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string_view>) {
      return std::string(value);
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
      return std::forward<T>(value);
    } else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
      return std::to_string(value);
    } else if constexpr (std::is_pointer_v<std::decay_t<T>>) {
      std::stringstream ss;
      ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(value);
      return ss.str();
    } else {
      std::stringstream ss;
      ss << value;
      return ss.str();
    }
  }

  std::string header_;
  LogLevel level_;
  char delimiter_;
  LogSubsysSet subsysSet_;
  std::ofstream logFileStream_;

 public:
  Logger(const std::string& header, const LogLevel level, const char delimiter);

  const std::string& header() const { return header_; }

  const LogLevel& level() const { return level_; }

  const char& delimiter() const { return delimiter_; }

  void setHeader(const std::string& header) { header_ = header; }

  void setLevel(LogLevel level) { level_ = level; }

  void setDelimiter(char delimiter) { delimiter_ = delimiter; }

  inline bool shouldLog(LogLevel level, LogSubsys subsys) const {
    return level >= level_ && subsysSet_.test(static_cast<std::size_t>(subsys));
  }

  template <bool NewLine, typename... Args>
  std::string message(Args&&... args) {
    if (sizeof...(args) == 0) {
      if constexpr (NewLine) {
        return header_ + "\n";
      }
      return header_;
    }

    // Convert all arguments to strings
    std::array<std::string, sizeof...(args)> argStrings = {toStringHelper(std::forward<Args>(args))...};

    std::stringstream ss;
    size_t argIndex = 0;

    // Replace "%@" placeholders by iterating through header_
    for (size_t i = 0; i < header_.size(); ++i) {
      if (i + 1 < header_.size() && header_[i] == '%' && header_[i + 1] == '@' && argIndex < argStrings.size()) {
        ss << argStrings[argIndex];
        ++argIndex;
        ++i;  // Skip the '@' character
      } else {
        ss << header_[i];
      }
    }

    // Append remaining arguments
    for (size_t i = argIndex; i < argStrings.size(); ++i) {
      if (delimiter_) {
        ss << delimiter_;
      }
      ss << argStrings[i];
    }

    if constexpr (NewLine) {
      ss << '\n';
    }
    return ss.str();
  }

  template <typename... Args>
  void log(Args&&... args) {
    auto msg = message<true>(std::forward<Args>(args)...);
    if (msg.empty()) return;
    if (logFileStream_.is_open()) {
      logFileStream_ << msg;
      logFileStream_.flush();
    } else {
      // Fallback to stdout if no file stream
      std::cout << msg;
    }
  }
};

Logger& logger(const std::string& header, const std::string& level, char delimiter);

}  // namespace mscclpp

// Helper to build log message arguments
#define LOGGER_BUILD_ARGS(level__, subsys__, ...)                                                              \
  ::mscclpp::detail::timestamp(), "MSCCLPP", ::getpid(), ::mscclpp::detail::logLevelToString(level__),         \
      ::mscclpp::detail::logSubsysToString(subsys__), ::mscclpp::detail::filenameFromPath(__FILE__), __LINE__, \
      __VA_ARGS__

#define LOGGER_LOG(level__, subsys__, ...)                                                      \
  do {                                                                                          \
    auto& logger__ = ::mscclpp::logger("%@ %@ %@ %@ %@ %@:%@ ", ::mscclpp::env()->logLevel, 0); \
    if (logger__.shouldLog(level__, subsys__)) {                                                \
      logger__.log(LOGGER_BUILD_ARGS(level__, subsys__, __VA_ARGS__));                          \
    }                                                                                           \
  } while (0)

#define LOG(level__, subsys__, ...) LOGGER_LOG(level__, subsys__, __VA_ARGS__)
#define DEBUG(subsys__, ...) LOGGER_LOG(::mscclpp::LogLevel::DEBUG, subsys__, __VA_ARGS__)
#define INFO(subsys__, ...) LOGGER_LOG(::mscclpp::LogLevel::INFO, subsys__, __VA_ARGS__)
#define WARN(subsys__, ...) LOGGER_LOG(::mscclpp::LogLevel::WARN, subsys__, __VA_ARGS__)
#define ERROR(subsys__, ...) LOGGER_LOG(::mscclpp::LogLevel::ERROR, subsys__, __VA_ARGS__)
#define THROW(subsys__, exception__, errorCode__, ...)                                                           \
  do {                                                                                                           \
    const auto errorCodeCopy__ = errorCode__;                                                                    \
    throw exception__(::mscclpp::logger("%@ %@ %@ %@ %@ %@:%@ ", ::mscclpp::env()->logLevel, 0)                  \
                          .message<false>(LOGGER_BUILD_ARGS(::mscclpp::LogLevel::ERROR, subsys__, __VA_ARGS__)), \
                      errorCodeCopy__);                                                                          \
  } while (0)

#endif  // MSCCLPP_LOGGER_HPP_
