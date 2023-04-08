import atexit
import json
import logging
import os
import pickle
import re
from typing import Any, Optional, final

logger = logging.getLogger(__file__)

from . import _py_mscclpp

__all__ = (
    "Comm",
    "MscclppUniqueId",
    "MSCCLPP_UNIQUE_ID_BYTES",
    "TransportType",
)

_Comm = _py_mscclpp._Comm
_P2PHandle = _py_mscclpp._P2PHandle
TransportType = _py_mscclpp.TransportType

MscclppUniqueId = _py_mscclpp.MscclppUniqueId
MSCCLPP_UNIQUE_ID_BYTES = _py_mscclpp.MSCCLPP_UNIQUE_ID_BYTES


def _mscclpp_log_cb(msg: str) -> None:
    """Log callback hook called from inside _py_mscclpp."""

    # Attempt to parse out the original log level:
    level = logging.INFO
    if match := re.search(r"MSCCLPP (\w+)", msg):
        level = logging._nameToLevel.get(match.group(1), logging.INFO)

    # actually log the event.
    logger.log(level, msg)


# The known log levels used by MSCCLPP.
# Set in os.environ['MSCCLPP_DEBUG'] and only parsed on first init.
MSCCLPP_LOG_LEVELS: set[str] = {
    "DEBUG",
    "INFO",
    "WARN",
    "ABORT",
    "TRACE",
}


def _setup_logging(level: str = "INFO"):
    """Setup log hooks for the C library."""
    level = level.upper()
    if level not in MSCCLPP_LOG_LEVELS:
        level = "INFO"
    os.environ["MSCCLPP_DEBUG"] = level

    _py_mscclpp._bind_log_handler(_mscclpp_log_cb)
    # needed to prevent a segfault at exit.
    atexit.register(_py_mscclpp._release_log_handler)


_setup_logging()


@final
class Comm:
    """Comm object; represents a mscclpp connection."""

    _comm: _Comm

    @staticmethod
    def init_rank_from_address(
            address: str,
            rank: int,
            world_size: int,
            *,
            port: Optional[int] = None,
    ):
        """Initialize a Comm from an address.

        :param address: the address as a string, with optional port.
        :param rank: this Comm's rank.
        :param world_size: the total world size.
        :param port: (optional) port, appended to address.
        :return: a newly initialized Comm.
        """
        if port is not None:
            address = f"{address}:{port}"
        return Comm(
            _comm=_Comm.init_rank_from_address(
                address=address,
                rank=rank,
                world_size=world_size,
            ),
        )

    def __init__(self, *, _comm: _Comm):
        """Construct a Comm object wrapping an internal _Comm handle."""
        self._comm = _comm

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Close the connection."""
        if self._comm:
            self._comm.close()
            self._comm = None

    @property
    def rank(self) -> int:
        """Return the rank of the Comm.

        Assumes the Comm is open.
        """
        return self._comm.rank

    @property
    def world_size(self) -> int:
        """Return the world_size of the Comm.

        Assumes the Comm is open.
        """
        return self._comm.world_size

    def bootstrap_all_gather_int(self, val: int) -> list[int]:
        """AllGather an int value through the bootstrap interface."""
        return self._comm.bootstrap_all_gather_int(val)

    def all_gather_bytes(self, item: bytes) -> list[bytes]:
        """AllGather bytes (of different sizes) through the bootstrap interface.

        :param item: the bytes object for this rank.
        :return: a list of bytes objects; the ret[rank] object will be a new copy.
        """
        return self._comm.all_gather_bytes(item)

    def all_gather_json(self, item: Any) -> list[Any]:
        """AllGather JSON objects through the bootstrap interface.

        :param item: the JSON object for this rank.
        :return: a list of JSON objects; the ret[rank] object will be a new copy.
        """
        return [
            json.loads(b.decode("utf-8"))
            for b in self.all_gather_bytes(bytes(json.dumps(item), "utf-8"))
        ]

    def all_gather_pickle(self, item: Any) -> list[Any]:
        """AllGather pickle-able objects through the bootstrap interface.

        :param item: the object for this rank.
        :return: a list of de-pickled objects. Note, the ret[rank] item will be a new copy.
        """
        return [pickle.loads(b) for b in self.all_gather_bytes(pickle.dumps(item))]

    def connect(
            self,
            remote_rank: int,
            tag: int,
            data_ptr,
            data_size: int,
            transport: int,
    ) -> None:
        self._comm.connect(
            remote_rank,
            tag,
            data_ptr,
            data_size,
            transport,
        )

    def connection_setup(self) -> None:
        self._comm.connection_setup()

    def launch_proxies(self) -> None:
        self._comm.launch_proxies()

    def stop_proxies(self) -> None:
        self._comm.stop_proxies()

    def register_buffer(
            self,
            data_ptr,
            data_size: int,
    ) -> list[_P2PHandle]:
        return [
            P2PHandle(self, h) for h in self._comm.register_buffer(
                data_ptr,
                data_size,
            )
        ]


class P2PHandle:
    _comm: Comm
    _handle: _P2PHandle

    def __init__(self,
                 comm: Comm,
                 handle: _P2PHandle):
        self._comm = comm
        self._handle = handle
