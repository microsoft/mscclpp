#ifndef MSCCLPP_CONFIG_H_
#define MSCCLPP_CONFIG_H_

#include <time.h>

namespace mscclpp {
class Config {
 public:
  time_t bootstrapConnectionTimeout = 30;

  static Config* getInstance();
  time_t getBootstrapConnectionTimeoutConfig();
  void setBootstrapConnectionTimeoutConfig(time_t timeout);

 private:
  Config() = default;
  Config(const Config&) = delete;
  Config& operator=(const Config&) = delete;

  static Config instance_;
};
}  // namespace mscclpp

#endif  // end include guard
