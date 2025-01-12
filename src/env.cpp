// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstdlib>
#include <memory>
#include <type_traits>

// clang-format off
#include <mscclpp/env.hpp>
#include <mscclpp/errors.hpp>
// clang-format on

#include "debug.h"

template <typename T>
void readEnv(const std::string &envName, T &env) {
  const char *envCstr = getenv(envName.c_str());
  if (envCstr == nullptr) return;
  if constexpr (std::is_same_v<T, int>) {
    env = atoi(envCstr);
  } else if constexpr (std::is_same_v<T, bool>) {
    env = (std::string(envCstr) != "0");
  } else {
    env = std::string(envCstr);
  }
  INFO(MSCCLPP_ENV, "%s=%s", envName.c_str(), envCstr);
}

namespace mscclpp {

Env::Env()
    : debug(),
      debugSubsys(),
      debugFile(),
      hcaDevices(),
      hostid(),
      socketFamily(),
      socketIfname(),
      commId(),
      executionPlanDir(),
      npkitDumpDir(),
      cudaIpcUseDefaultStream(false) {
  readEnv("MSCCLPP_DEBUG", debug);
  readEnv("MSCCLPP_DEBUG_SUBSYS", debugSubsys);
  readEnv("MSCCLPP_DEBUG_FILE", debugFile);
  readEnv("MSCCLPP_HCA_DEVICES", hcaDevices);
  readEnv("MSCCLPP_HOSTID", hostid);
  readEnv("MSCCLPP_SOCKET_FAMILY", socketFamily);
  readEnv("MSCCLPP_SOCKET_IFNAME", socketIfname);
  readEnv("MSCCLPP_COMM_ID", commId);
  readEnv("MSCCLPP_EXECUTION_PLAN_DIR", executionPlanDir);
  readEnv("MSCCLPP_NPKIT_DUMP_DIR", npkitDumpDir);
  readEnv("MSCCLPP_CUDAIPC_USE_DEFAULT_STREAM", cudaIpcUseDefaultStream);
}

const Env &env() {
  static std::unique_ptr<Env> globalEnv = std::make_unique<Env>();
  return *globalEnv;
}

}  // namespace mscclpp
