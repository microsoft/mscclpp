# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum


# Scratch buffer slice with manual indexing
class BufferSlice:
    def __init__(self, buf, name):
        self.name = name
        self.buf = buf
        self.offset = -1  # Offset into the global scratch buffer
        self.chunks = []

    # Returns the global index into the scratch buffer
    def get_global_index(self, index):
        assert self.offset > -1, "set_offset needs to be called first"
        return self.offset + index

    def get_buffer(self):
        return self.buf

    def instance_size(self):
        return len(self.chunks)

    def set_offset(self, offset):
        self.offset = offset

    def __getitem__(self, index):
        return self.chunks[index]

    def __setitem__(self, index, value):
        current_size = len(self.chunks)
        while index > current_size:
            self.chunks.append(None)
            current_size = len(self.chunks)
        if index == current_size:
            self.chunks.append(value)
        else:
            self.chunks[index] = value

    def __len__(self):
        return len(self.chunks)


class Buffer(Enum):
    input = "i"
    output = "o"
    scratch = "s"

    def __str__(self):
        return self.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value < other.value
