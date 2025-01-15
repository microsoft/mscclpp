// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ENV_HPP_
#define MSCCLPP_ENV_HPP_

#include <memory>
#include <string>

namespace mscclpp {

class Env;

/// Get the MSCCL++ environment.
/// @return A reference to the global environment object.
std::shared_ptr<Env> env();

/// The MSCCL++ environment. The constructor reads environment variables and sets the corresponding fields.
/// Use the @ref env() function to get the environment object.
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

 private:
  Env();

  friend std::shared_ptr<Env> env();
};

}  // namespace mscclpp

#endif  // MSCCLPP_ENV_HPP_
