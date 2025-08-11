---
applyTo: 'python/mscclpp/language/*.py'
---

# Instructions for DSL API Documentation

## Overview
The MSCCL++ DSL (Domain Specific Language) provides a Python API for defining distributed GPU communication patterns. All API functions should have comprehensive Google-style docstrings.

## Documentation Requirements
- Add google-style docstrings to the DSL API functions in the `mscclpp.language` package.
- Ensure that each function's docstring includes:
  - A brief description of what the function does.
  - Parameters with their types and descriptions.
  - Return type and description.
  - Any exceptions raised by the function, if applicable.
  - Usage examples where appropriate.

## Implementation Steps
1. Open each Python file in the `python.mscclpp.language` folder, exclude `__init__.py` and internal folders.
2. For each function in the file, add a Google-style docstring that follows the documentation requirements outlined above.
3. Ensure that the docstrings are clear, concise, and accurately describe the function's behavior.
4. Review the docstrings for consistency in style and formatting.
