// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONFIG_H_
#define MSCCLPP_CONFIG_H_

namespace mscclpp {

class Config {
 public:
  int bootstrapConnectionTimeout = 30;

  static Config* getInstance();
  int getBootstrapConnectionTimeoutConfig();
  void setBootstrapConnectionTimeoutConfig(int timeout);

 private:
  Config() = default;
  Config(const Config&) = delete;
  Config& operator=(const Config&) = delete;

  static Config instance_;
};

}  // namespace mscclpp

#endif  // end include guard
