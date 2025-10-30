// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_LOGGER_HPP_
#define MSCCLPP_LOGGER_HPP_

#include <bitset>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mscclpp/env.hpp>
#include <sstream>
#include <string>

namespace mscclpp {

typedef enum : unsigned int { NONE = 0, DEBUG, INFO, WARN, ERROR } LogLevel;
typedef enum : std::size_t { ENV = 0, NET, CONN, EXEC, NCCL, COUNT } LogSubsys;

namespace detail {
std::string guessRemoveProjectPrefix(const std::string& filePathStr);
std::string timestamp(const char* format = "%Y-%m-%d %X");
std::string logSubsysToString(LogSubsys subsys);
int pid();
}  // namespace detail

// Bitset holding enabled subsystems.
using LogSubsysSet = std::bitset<static_cast<std::size_t>(LogSubsys::COUNT)>;

class Logger {
 private:
  // Overload only for string literals: const char(&)[N]
  template <std::size_t N>
  std::string toStringHelper(const char (&value)[N]) const {
    // N includes the terminating '\0'; std::string(value) stops at '\0'
    return std::string(value);
  }

  template <typename T>
  std::string toStringHelper(T&& value) const {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
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

  template <bool NewLine, typename... Args>
  std::string message(LogLevel level, LogSubsys subsys, Args&&... args) {
    if (level < level_) return "";
    if (!subsysSet_.test(static_cast<std::size_t>(subsys))) return "";

    if (sizeof...(args) == 0) {
      if constexpr (NewLine) {
        return header_ + "\n";
      }
      return header_;
    }

    // Convert all arguments to strings
    std::array<std::string, sizeof...(args)> argStrings = {toStringHelper(std::forward<Args>(args))...};

    std::stringstream ss;
    std::string formattedHeader = header_;
    size_t argIndex = 0;
    size_t pos = 0;

    // Replace "%@" placeholders
    while ((pos = formattedHeader.find("%@", pos)) != std::string::npos && argIndex < argStrings.size()) {
      formattedHeader.replace(pos, 2, argStrings[argIndex]);
      pos += argStrings[argIndex].length();
      ++argIndex;
    }

    ss << formattedHeader;

    // Append remaining arguments
    for (size_t i = argIndex; i < argStrings.size(); ++i) {
      if (delimiter_) {
        ss << delimiter_;
      }
      ss << argStrings[i];
    }

    if constexpr (NewLine) {
      ss << std::endl;
    }
    return ss.str();
  }

  template <typename... Args>
  void log(LogLevel level, LogSubsys subsys, Args&&... args) {
    auto msg = message<true>(level, subsys, std::forward<Args>(args)...);
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

#define LOGGER_LOG(level__, subsys__, ...)                                                                          \
  do {                                                                                                              \
    ::mscclpp::logger("%@ %@ %@ %@ %@:%@ ", ::mscclpp::env()->logLevel, 0)                                          \
        .log(level__, subsys__, ::mscclpp::detail::timestamp(), "MSCCLPP", ::mscclpp::detail::pid(),                \
             ::mscclpp::detail::logSubsysToString(subsys__), ::mscclpp::detail::guessRemoveProjectPrefix(__FILE__), \
             __LINE__, __VA_ARGS__);                                                                                \
  } while (0)

#define LOG(level__, subsys__, ...) LOGGER_LOG(level__, subsys__, __VA_ARGS__)
#define DEBUG(subsys__, ...) LOGGER_LOG(::mscclpp::LogLevel::DEBUG, subsys__, __VA_ARGS__)
#define INFO(subsys__, ...) LOGGER_LOG(::mscclpp::LogLevel::INFO, subsys__, __VA_ARGS__)
#define WARN(subsys__, ...) LOGGER_LOG(::mscclpp::LogLevel::WARN, subsys__, __VA_ARGS__)
#define ERROR(subsys__, ...) LOGGER_LOG(::mscclpp::LogLevel::ERROR, subsys__, __VA_ARGS__)
#define THROW(subsys__, exception__, errorCode__, ...)                                                       \
  do {                                                                                                       \
    const auto errorCodeCopy__ = errorCode__;                                                                \
    throw exception__(                                                                                       \
        ::mscclpp::logger("%@ %@ %@ %@ %@:%@ ", ::mscclpp::env()->logLevel, 0)                               \
            .message<false>(::mscclpp::LogLevel::ERROR, subsys__, ::mscclpp::detail::timestamp(), "MSCCLPP", \
                            ::mscclpp::detail::pid(), ::mscclpp::detail::logSubsysToString(subsys__),        \
                            ::mscclpp::detail::guessRemoveProjectPrefix(__FILE__), __LINE__, __VA_ARGS__),   \
        errorCodeCopy__);                                                                                    \
  } while (0)

#endif  // MSCCLPP_LOGGER_HPP_
