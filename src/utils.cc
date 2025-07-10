// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <unistd.h>

#include <chrono>
#include <mscclpp/errors.hpp>
#include <mscclpp/utils.hpp>
#include <string>

namespace mscclpp {

std::string getHostName(int maxlen, const char delim) {
  std::string hostname(maxlen + 1, '\0');
  if (gethostname(const_cast<char*>(hostname.data()), maxlen) != 0) {
    throw Error("gethostname failed", ErrorCode::SystemError);
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
  hostname[i] = '\0';
  return hostname.substr(0, i);
}

}  // namespace mscclpp
