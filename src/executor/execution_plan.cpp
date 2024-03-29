// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_plan.hpp"

#include <nlohmann/json.hpp>

namespace mscclpp {
using json = nlohmann::json;
void ExecutionPlan::loadExecutionPlan(std::ifstream& file) {
    json obj = json::parse(file);
}
}  // namespace mscclpp
