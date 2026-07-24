# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

__all__ = [
    "MscclppAllReduce1",
    "MscclppAllReduce2",
    "MscclppAllReduce3",
    "MscclppAllReduce4",
    "MscclppAllReduce5",
]


def __getattr__(name):
    if name in __all__:
        from . import mscclpp_op

        return getattr(mscclpp_op, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
