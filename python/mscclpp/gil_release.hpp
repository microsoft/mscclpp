// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PYTHON_GIL_RELEASE_HPP_
#define MSCCLPP_PYTHON_GIL_RELEASE_HPP_

#include <functional>
#include <nanobind/nanobind.h>

template <typename ReturnType, typename ClassType, typename... Args>
std::function<ReturnType(ClassType*, Args...)> gil_release_wrapper(ReturnType (ClassType::*method)(Args...)) {
    return [method](ClassType* instance, Args... args) -> ReturnType {
        nanobind::gil_scoped_release release;
        return (instance->*method)(std::forward<Args>(args)...);
    };
}

#endif  // MSCCLPP_PYTHON_GIL_RELEASE_HPP_
