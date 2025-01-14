// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstdlib>
#include <type_traits>

// clang-format off
#include <mscclpp/env.hpp>
#include <mscclpp/errors.hpp>
// clang-format on

#include "debug.h"

template <typename T>
T readEnv(const std::string &envName, const T &defaultValue) {
  const char *envCstr = getenv(envName.c_str());
  if (envCstr == nullptr) return defaultValue;
  if constexpr (std::is_same_v<T, int>) {
    return atoi(envCstr);
  } else if constexpr (std::is_same_v<T, bool>) {
    return (std::string(envCstr) != "0");
  }
  return T(envCstr);
}

template <typename T>
void readAndSetEnv(const std::string &envName, T &env) {
  const char *envCstr = getenv(envName.c_str());
  if (envCstr == nullptr) return;
  if constexpr (std::is_same_v<T, int>) {
    env = atoi(envCstr);
  } else if constexpr (std::is_same_v<T, bool>) {
    env = (std::string(envCstr) != "0");
  } else {
    env = std::string(envCstr);
  }
}

template <typename T>
void logEnv(const std::string &envName, const T &env) {
  if (!getenv(envName.c_str())) return;
  INFO(MSCCLPP_ENV, "%s=%d", envName.c_str(), env);
}

template <>
void logEnv(const std::string &envName, const std::string &env) {
  if (!getenv(envName.c_str())) return;
  INFO(MSCCLPP_ENV, "%s=%s", envName.c_str(), env.c_str());
}

namespace mscclpp {

Env::Env()
    : debug(readEnv<std::string>("MSCCLPP_DEBUG", "")),
      debugSubsys(readEnv<std::string>("MSCCLPP_DEBUG_SUBSYS", "")),
      debugFile(readEnv<std::string>("MSCCLPP_DEBUG_FILE", "")),
      hcaDevices(readEnv<std::string>("MSCCLPP_HCA_DEVICES", "")),
      hostid(readEnv<std::string>("MSCCLPP_HOSTID", "")),
      socketFamily(readEnv<std::string>("MSCCLPP_SOCKET_FAMILY", "")),
      socketIfname(readEnv<std::string>("MSCCLPP_SOCKET_IFNAME", "")),
      commId(readEnv<std::string>("MSCCLPP_COMM_ID", "")),
      executionPlanDir(readEnv<std::string>("MSCCLPP_EXECUTION_PLAN_DIR", "")),
      npkitDumpDir(readEnv<std::string>("MSCCLPP_NPKIT_DUMP_DIR", "")),
      cudaIpcUseDefaultStream(readEnv<bool>("MSCCLPP_CUDAIPC_USE_DEFAULT_STREAM", false)) {}

std::shared_ptr<Env> env() {
  static std::shared_ptr<Env> globalEnv = std::shared_ptr<Env>(new Env());
  static bool logged = false;
  if (!logged) {
    logged = true;
    // cannot log inside the constructor because of circular dependency
    logEnv("MSCCLPP_DEBUG", globalEnv->debug);
    logEnv("MSCCLPP_DEBUG_SUBSYS", globalEnv->debugSubsys);
    logEnv("MSCCLPP_DEBUG_FILE", globalEnv->debugFile);
    logEnv("MSCCLPP_HCA_DEVICES", globalEnv->hcaDevices);
    logEnv("MSCCLPP_HOSTID", globalEnv->hostid);
    logEnv("MSCCLPP_SOCKET_FAMILY", globalEnv->socketFamily);
    logEnv("MSCCLPP_SOCKET_IFNAME", globalEnv->socketIfname);
    logEnv("MSCCLPP_COMM_ID", globalEnv->commId);
    logEnv("MSCCLPP_EXECUTION_PLAN_DIR", globalEnv->executionPlanDir);
    logEnv("MSCCLPP_NPKIT_DUMP_DIR", globalEnv->npkitDumpDir);
    logEnv("MSCCLPP_CUDAIPC_USE_DEFAULT_STREAM", globalEnv->cudaIpcUseDefaultStream);
  }
  return globalEnv;
}

}  // namespace mscclpp
