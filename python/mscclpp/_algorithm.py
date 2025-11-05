# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ._mscclpp import (
    Algorithm as _Algorithm,
)

class Algorithm():
    def __init__(self, handle: int):
        self._handle = handle

    @classmethod
    def create_from_handle(cls, handle: int) -> "Algorithm":
        return cls(handle)

    def launch(self):
        pass
