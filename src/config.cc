#include "config.hpp"

namespace mscclpp {
Config Config::instance_;

Config* Config::getInstance() { return &instance_; }

time_t Config::getBootstrapConnectionTimeoutConfig() { return bootstrapConnectionTimeout; }

void Config::setBootstrapConnectionTimeoutConfig(time_t timeout) { bootstrapConnectionTimeout = timeout; }
}  // namespace mscclpp
