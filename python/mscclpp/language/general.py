# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mscclpp.language.internal.globals import get_program


def JSON():
    """Convert the current MSCCL++ program to JSON format.

    This function post-processes all operations in the current program and
    returns the program representation as a JSON string. This is typically
    called after defining a complete communication program to serialize it
    for execution.

    Returns:
        str: A JSON string representation of the current MSCCL++ program,
            including all ranks, operations, channels, and configuration.
    """
    return get_program().to_json()
