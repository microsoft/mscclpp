// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/core.hpp>

struct ncclComm {
    std::shared_ptr<mscclpp::Communicator> comm;
};
