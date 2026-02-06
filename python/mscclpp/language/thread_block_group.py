# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Dict, Set


class ThreadBlockGroup:
    """
    A group of thread blocks with unique identifiers.

    This class manages a collection of thread blocks, ensuring uniqueness
    and providing efficient lookup of thread block IDs.
    """

    def __init__(self, tb_list: List[int]):
        """
        Initialize a ThreadBlockGroup with a list of thread blocks.

        Args:
            tb_list: List of thread block objects
        """

        self.tb_list: Set[int] = set(tb_list)
        self._tb_id: Dict[int, int] = {}

        seen = set()
        for i, tb in enumerate(self.tb_list):
            if tb in seen:
                raise ValueError(f"Duplicate thread block found at index {i}: {tb}")
            seen.add(tb)
            self._tb_id[tb] = i

    def get_internal_id(self, tb: int) -> int:
        """
        Get the ID of a thread block in this group.

        Args:
            tb: The thread block object to look up

        Returns:
            The integer ID of the thread block (0-indexed)

        Raises:
            ValueError: If the thread block is not in this group
        """
        if tb not in self._tb_id:
            raise ValueError(f"Thread block {tb} not found in thread block group")
        return self._tb_id[tb]

    def numtb(self) -> int:
        """Return the number of thread blocks in the group."""
        return len(self.tb_list)

    def tbg_overlap(self, other):
        for tb in self.tb_list:
            if tb in other.tb_list:
                return True
        return False

    def tb_overlap(self, tb_id):
        return tb_id in self.tb_list

    def to_dict(self, tb):
        return {"tb_id": self.get_internal_id(tb), "tbg_size": self.numtb()}

    def start_offset(self, tb, size):
        tb_id = self.get_internal_id(tb)
        return (size / self.numtb()) * tb_id

    def end_offset(self, tb, size):
        tb_id = self.get_internal_id(tb)
        return (size / self.numtb()) * (tb_id + 1)
