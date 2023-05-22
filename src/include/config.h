#ifndef MSCCLPP_CONFIG_H_
#define MSCCLPP_CONFIG_H_

#include <time.h>

class mscclppConfig {
 public:
  time_t bootstrapConnectionTimeout = 30;

  static mscclppConfig* getInstance();
  time_t getBootstrapConnectionTimeoutConfig();
  void setBootstrapConnectionTimeoutConfig(time_t timeout);

 private:
  mscclppConfig() = default;
  mscclppConfig(const mscclppConfig&) = delete;
  mscclppConfig& operator=(const mscclppConfig&) = delete;

  static mscclppConfig _instance;
};

#endif  // end include guard
