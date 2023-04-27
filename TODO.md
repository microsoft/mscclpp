# Core API extraction

- Add a test for host side Communicator/RegisteredMemory/Connection use.
- Implement a standalone "epoch" synchronization construct that can be used as a component in custom proxies. epoch.hpp/cc has the beginnings of this.
- Reimplement the "standard" proxy service + DeviceConnection on top of the new Communicator/RegisteredMemory/Connection core API. Remants of the old code is in channel.hpp, basic_proxy_handler.hpp/cc and host_connection.hpp/cc. Probably need a manager class to wrap all of this.
- Change the new IBConnection and Communicator to use the new C++ IbCtx and IbQp classes.
- Implement IbQp::~IbQp()