# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os as _os

from ._mscclpp import (
    Communicator,
    Connection,
    Fifo,
    Host2DeviceSemaphore,
    Host2HostSemaphore,
    numa,
    ProxyService,
    RegisteredMemory,
    SimpleProxyChannel,
    SmChannel,
    SmDevice2DeviceSemaphore,
    TcpBootstrap,
    Transport,
    TransportFlags,
)


def get_include():
    """Return the directory that contains the MSCCL++ headers."""
    return _os.path.join(_os.path.dirname(__file__), "include")


def get_lib():
    """Return the directory that contains the MSCCL++ headers."""
    return _os.path.join(_os.path.dirname(__file__), "lib")
