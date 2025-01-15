# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from dataclasses import dataclass


@dataclass
class Chunk:
    origin_rank: int  # Rank the chunk initially started at
    origin_index: int  # Index the chunk initially started at
    dst_rank: int = -1
    dst_index: int = -1

    def reduce(self, dst, chunk):
        if type(chunk) is ReduceChunk:
            return chunk.reduce(dst, self)
        elif type(chunk) is Chunk:
            chunks = [self, chunk]
            return ReduceChunk(dst, chunks)
        else:
            raise ValueError("Trying to reduce with chunk of None")

    def __hash__(self):
        return hash((self.origin_rank, self.origin_index))

    def __eq__(self, other):
        return (
            type(other) is Chunk and self.origin_rank == other.origin_rank and self.origin_index == other.origin_index
        )

    def __lt__(self, other):
        return self.origin_rank < other.origin_rank or (
            self.origin_rank == other.origin_rank and self.origin_index < other.origin_index
        )


@dataclass
class ReduceChunk:
    creation_rank: int  # Rank the Reduce Chunk is created. Necessary since the same ReduceChunk can be created on multiple ranks independently
    chunks: list  # List of chunks reduced

    def reduce(self, dst, chunk):
        if type(chunk) is ReduceChunk:
            chunks = self.chunks + chunk.chunks
        elif type(chunk) is Chunk:
            chunks = self.chunks + [chunk]
        else:
            raise ValueError("Trying to reduce with chunk of None")
        return ReduceChunk(self.creation_rank, chunks)

    def sort(self):
        self.chunks.sort()

    def __hash__(self):
        self.sort()
        return hash((self.creation_rank,) + tuple(self.chunks))

    # Two reduce chunks are equal if they contain the same list of
    # chunks being reduced
    def __eq__(self, other):
        self.sort()
        other.sort()
        return self.chunks == other.chunks
