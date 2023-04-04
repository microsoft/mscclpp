import os
import atexit
from typing import Any
import json
import pickle
import logging

logger = logging.getLogger(__file__)

from . import _py_mscclpp

__all__ = (
    "MscclppUniqueId",
    "MSCCLPP_UNIQUE_ID_BYTES",
)

_Comm = _py_mscclpp._Comm

MscclppUniqueId = _py_mscclpp.MscclppUniqueId
MSCCLPP_UNIQUE_ID_BYTES = _py_mscclpp.MSCCLPP_UNIQUE_ID_BYTES

def _setup_logging(level='INFO'):
    os.environ['MSCCLPP_DEBUG'] = level
    _py_mscclpp._bind_log_handler(logger.info)
    # needed to prevent a segfault at exit.
    atexit.register(_py_mscclpp._release_log_handler)

_setup_logging()

class Comm:
    _comm: _Comm

    @staticmethod
    def init_rank_from_address(
        address: str,
        rank: int,
        world_size: int,
    ):
        return Comm(
            _comm = _Comm.init_rank_from_address(
                address=address,
                rank=rank,
                world_size=world_size,
            ),
        )

    def __init__(self, *, _comm: _Comm):
        self._comm = _comm

    def close(self) -> None:
        self._comm.close()
        self._comm = None

    @property
    def rank(self) -> int:
        return self._comm.rank

    @property
    def world_size(self) -> int:
        return self._comm.world_size

    def bootstrap_all_gather_int(self, val: int) -> list[int]:
        return self._comm.bootstrap_all_gather_int(val)

    def all_gather_bytes(self, item: bytes) -> list[bytes]:
        return self._comm.all_gather_bytes(item)

    def all_gather_json(self, item: Any) -> list[Any]:
        return [
            json.loads(b.decode('utf-8'))
            for b in self.all_gather_bytes(bytes(json.dumps(item), 'utf-8'))
        ]

    def all_gather_pickle(self, item: Any) -> list[Any]:
        return [
            pickle.loads(b)
            for b in self.all_gather_bytes(pickle.dumps(item))
        ]

