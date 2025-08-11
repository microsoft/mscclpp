# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

_current_program = None


def set_program(program):
    global _current_program
    _current_program = program


def get_program():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program
