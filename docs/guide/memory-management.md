# Memory Management

The MSCCL++ stack handles most of the resource management automatically, so users don't need to explicitly store or destroy objects constructed by MSCCL++ APIs in most cases. For example:
* The `Context` object will be alive iff itself or any Connections created by it are alive.
* The `Connection` object will be alive iff itself or any SemaphoreStubs created from it are alive.
* The `SemaphoreStub` object will be alive iff itself or any Semaphores created from it are alive.
* The `Semaphore` object will be alive iff itself or any Channels created from it are alive.

However, there are still a few things put on **users' responsibility**:
* The objects that are serialized and sent to other processes (like Endpoints, SemaphoreStubs, and RegisteredMemory) should be kept alive until the remote endpoint has finished using them.
* The channel objects on the host should be kept alive until the GPU kernels are finished using the device handles.
