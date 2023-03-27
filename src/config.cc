#include "config.h"

mscclppConfig mscclppConfig::_instance;

mscclppConfig* mscclppConfig::getInstance()
{
    return &_instance;
}

time_t mscclppConfig::getBootstrapConnectionTimeoutConfig()
{
    return bootstrapConnectionTimeout;
}

void mscclppConfig::setBootstrapConnectionTimeoutConfig(time_t timeout)
{
    bootstrapConnectionTimeout = timeout;
}
