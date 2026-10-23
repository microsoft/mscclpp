# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compatibility imports for GPU helpers now provided by :mod:`mscclpp.gpu`."""

from mscclpp import gpu as _gpu

__all__ = _gpu.__all__


def __getattr__(name):
    return getattr(_gpu, name)
