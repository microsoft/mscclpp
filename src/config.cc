#include "config.hpp"

namespace mscclpp {
Config Config::instance_;

Config* Config::getInstance() { return &instance_; }

int Config::getBootstrapConnectionTimeoutConfig() { return bootstrapConnectionTimeout; }

void Config::setBootstrapConnectionTimeoutConfig(int timeout) { bootstrapConnectionTimeout = timeout; }
}  // namespace mscclpp
