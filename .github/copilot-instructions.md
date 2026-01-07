# Project Contribution Guidelines

This document outlines a few useful informations for contributing to this project.

## C/C++ Headers Layout
This project has two C/C++ header directories: `include/mscclpp/` and `src/include/`. Headers in `include/mscclpp/` are public headers that define the public API of the project. Headers in `src/include/` are internal headers used only within the project.

When adding new headers, place them in the appropriate directory based on their intended usage (public API vs. internal use). To prevent confusion, do not have duplicate names for headers in these two directories.

Symbols declared in public headers must be properly documented using Doxygen-style comments, except forward declarations and private class members. In a few cases, we may need to add declarations in public headers that are not intended for public use. In such cases, declare them under `mscclpp::detail` namespace, where we do not necessarily document every symbol.

## License Header
A license header must be included at the top of each source code file in the project.

For Python source code:
```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
```

For C/C++/CUDA source code:
```cpp
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
```

## Formatting
If you have modified any code in the project, run `./tools/lint.sh` to automatically format the entire source code before finishing iterations. Note that this script formats only staged files.

## Building and Testing
The following commands are commonly used for building and testing the project. See `docs/quickstart.md` for more detailed instructions.

For building libraries and tests:
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..
```

For testing after successful build:
```bash
# To run all tests
mpirun -np 2 ./build/test/mp_unit_tests
# To run tests excluding IB-related ones (when IB is not available)
mpirun -np 2 ./build/test/mp_unit_tests --gtest_filter=-*Ib*
```

For building a Python package:
```bash
python3 -m pip install -e .
```

For building documentation (see dependencies in `docs/requirements.txt`):
```bash
cd docs
doxygen
make html
cd ..
```
