// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_LOGGER_HPP_
#define MSCCLPP_LOGGER_HPP_

#include <iomanip>
#include <iostream>
#include <memory>
#include <mscclpp/env.hpp>
#include <sstream>
#include <string>

namespace mscclpp {

namespace detail {
std::string guessRemoveProjectPrefix(const std::string& filePath);
std::string timestamp(const char* format = "%Y-%m-%d %X");
std::string subsysFlagToString(unsigned int flag);
}  // namespace detail

typedef enum : unsigned int { NONE = 0, DEBUG, INFO, VERSION, WARN, ERROR } LogLevel;
typedef enum : unsigned int {
  ENV = 0x1,
  NET = 0x2,
  CONN = 0x4,
  EXEC = 0x8,
  NCCL = 0x10,
  ALL = 0xFFFFFFFF
} LogSubsysFlag;

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
  char delimeter_;
  unsigned int subsysFlags_;

 public:
  Logger(const std::string& header, const LogLevel level, const char delimeter);

  const std::string& header() const { return header_; }

  const LogLevel& level() const { return level_; }

  const char& delimeter() const { return delimeter_; }

  void setHeader(const std::string& header) { header_ = header; }

  void setLevel(LogLevel level) { level_ = level; }

  void setDelimeter(char delimeter) { delimeter_ = delimeter; }

  template <bool NewLine, typename... Args>
  std::string message(LogLevel level, LogSubsysFlag flag, Args&&... args) {
    if (level < level_) return "";
    if ((flag & subsysFlags_) == 0) return "";

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

    // Replace "%s" placeholders
    while ((pos = formattedHeader.find("%s", pos)) != std::string::npos && argIndex < argStrings.size()) {
      formattedHeader.replace(pos, 2, argStrings[argIndex]);
      pos += argStrings[argIndex].length();
      ++argIndex;
    }

    ss << formattedHeader;

    // Append remaining arguments
    for (size_t i = argIndex; i < argStrings.size(); ++i) {
      if (delimeter_) {
        ss << delimeter_;
      }
      ss << argStrings[i];
    }

    if constexpr (NewLine) {
      ss << std::endl;
    }
    return ss.str();
  }

  template <typename... Args>
  void log(LogLevel level, LogSubsysFlag flag, Args&&... args) {
    auto msg = message<true>(level, flag, std::forward<Args>(args)...);
    if (!msg.empty()) {
      std::cout << msg;
    }
  }
};

Logger& logger(const std::string& name, const std::string& header = "", const std::string& level = "ERROR",
               char delimeter = 0);

}  // namespace mscclpp

#define LOGGER_LOG(level__, flag__, ...)                                                                           \
  do {                                                                                                             \
    ::mscclpp::logger("MSCCLPP", "%s %s %s %s:%s ", ::mscclpp::env()->logLevel, 0)                                 \
        .log(level__, flag__, ::mscclpp::detail::timestamp(), "MSCCLPP",                                           \
             ::mscclpp::detail::subsysFlagToString(flag__), ::mscclpp::detail::guessRemoveProjectPrefix(__FILE__), \
             __LINE__, __VA_ARGS__);                                                                               \
    break;                                                                                                         \
  } while (0)

#define LOG(level__, flag__, ...) LOGGER_LOG(level__, flag__, __VA_ARGS__)
#define DEBUG(flag__, ...) LOGGER_LOG(::mscclpp::LogLevel::DEBUG, flag__, __VA_ARGS__)
#define INFO(flag__, ...) LOGGER_LOG(::mscclpp::LogLevel::INFO, flag__, __VA_ARGS__)
#define WARN(flag__, ...) LOGGER_LOG(::mscclpp::LogLevel::WARN, flag__, __VA_ARGS__)
#define ERROR(flag__, ...) LOGGER_LOG(::mscclpp::LogLevel::ERROR, flag__, __VA_ARGS__)
#define THROW(flag__, exception__, errorCode__, ...)                                                       \
  do {                                                                                                     \
    throw exception__(                                                                                     \
        ::mscclpp::logger("MSCCLPP", "%s %s %s %s:%s ", ::mscclpp::env()->logLevel, 0)                     \
            .message<false>(::mscclpp::LogLevel::ERROR, flag__, ::mscclpp::detail::timestamp(), "MSCCLPP", \
                            ::mscclpp::detail::subsysFlagToString(flag__),                                 \
                            ::mscclpp::detail::guessRemoveProjectPrefix(__FILE__), __LINE__, __VA_ARGS__), \
        errorCode__);                                                                                      \
  } while (0)
// #define THROW(flag__, exception__, errorCode__, ...) std::exit(EXIT_FAILURE);

#endif  // MSCCLPP_LOGGER_HPP_
