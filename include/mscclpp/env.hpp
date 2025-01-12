// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ENV_HPP_
#define MSCCLPP_ENV_HPP_

#include <string>

namespace mscclpp {

struct Env {
  Env();
  std::string debug;
  std::string debugSubsys;
  std::string debugFile;
  std::string hcaDevices;
  std::string hostid;
  std::string socketFamily;
  std::string socketIfname;
  std::string commId;
  std::string executionPlanDir;
  std::string npkitDumpDir;
  bool cudaIpcUseDefaultStream;
};

const Env &env();

}  // namespace mscclpp

#endif  // MSCCLPP_ENV_HPP_
