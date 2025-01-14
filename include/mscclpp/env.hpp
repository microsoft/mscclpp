// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ENV_HPP_
#define MSCCLPP_ENV_HPP_

#include <memory>
#include <string>

namespace mscclpp {

class Env;

std::shared_ptr<Env> env();

class Env {
 public:
  const std::string debug;
  const std::string debugSubsys;
  const std::string debugFile;
  const std::string hcaDevices;
  const std::string hostid;
  const std::string socketFamily;
  const std::string socketIfname;
  const std::string commId;
  const std::string executionPlanDir;
  const std::string npkitDumpDir;
  const bool cudaIpcUseDefaultStream;

 protected:
  Env();

  friend std::shared_ptr<Env> env();
};

}  // namespace mscclpp

#endif  // MSCCLPP_ENV_HPP_
