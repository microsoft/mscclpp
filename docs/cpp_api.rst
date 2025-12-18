C++ API Reference
=================

This reference organizes the MSCCL++ C++ API into two main categories: :ref:`host-side-interfaces` for CPU code and :ref:`device-side-interfaces` for GPU kernels. Components that are used in both host and device code are documented in the Device-Side Interfaces section.

.. _host-side-interfaces:

Host-Side Interfaces
--------------------

These are the interfaces used in CPU code to set up connections, manage memory, and coordinate operations.

Bootstrap and Process Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: mscclpp::Bootstrap
   :members:

.. doxygenclass:: mscclpp::TcpBootstrap
   :members:

.. doxygentypedef:: mscclpp::UniqueId

.. doxygenvariable:: mscclpp::UniqueIdBytes

Connection Setup and Memory Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: mscclpp::Connection
   :members:

.. doxygenclass:: mscclpp::Context
   :members:

.. doxygenclass:: mscclpp::Communicator
   :members:

.. doxygenstruct:: mscclpp::Device
   :members:

.. doxygenclass:: mscclpp::Endpoint
   :members:

.. doxygenstruct:: mscclpp::EndpointConfig
   :members:

.. doxygenclass:: mscclpp::NvlsConnection
   :members:

.. doxygenclass:: mscclpp::RegisteredMemory
   :members:

.. doxygenclass:: mscclpp::TransportFlags
   :members:

.. doxygenenum:: mscclpp::DeviceType

.. doxygenenum:: mscclpp::Transport

.. doxygenfunction:: mscclpp::connectNvlsCollective

Semaphores
~~~~~~~~~~

.. doxygenclass:: mscclpp::Host2DeviceSemaphore
   :members:

.. doxygenclass:: mscclpp::Host2HostSemaphore
   :members:

.. doxygenclass:: mscclpp::MemoryDevice2DeviceSemaphore
   :members:

.. doxygenclass:: mscclpp::Semaphore
   :members:

.. doxygenclass:: mscclpp::SemaphoreStub
   :members:

Channels
~~~~~~~~

.. doxygenstruct:: mscclpp::BaseMemoryChannel
   :members:

.. doxygenstruct:: mscclpp::BasePortChannel
   :members:

.. doxygenstruct:: mscclpp::MemoryChannel
   :members:

.. doxygenstruct:: mscclpp::PortChannel
   :members:

.. doxygenstruct:: mscclpp::SwitchChannel
   :members:

Proxy Service and FIFO Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: mscclpp::BaseProxyService
   :members:

.. doxygenclass:: mscclpp::Fifo
   :members:

.. doxygenclass:: mscclpp::Proxy
   :members:

.. doxygenclass:: mscclpp::ProxyService
   :members:

.. doxygentypedef:: mscclpp::ProxyHandler

.. doxygenenum:: mscclpp::ProxyHandlerResult

.. doxygenvariable:: mscclpp::DEFAULT_FIFO_SIZE

Utilities
~~~~~~~~~

.. doxygenstruct:: mscclpp::AvoidCudaGraphCaptureGuard
   :members:

.. doxygenstruct:: mscclpp::CudaStreamWithFlags
   :members:

.. doxygenclass:: mscclpp::GpuBuffer
   :members:

.. doxygenclass:: mscclpp::GpuStream
   :members:

.. doxygenclass:: mscclpp::GpuStreamPool
   :members:

.. doxygenfunction:: mscclpp::getDeviceNumaNode

.. doxygenfunction:: mscclpp::getHostName

.. doxygenfunction:: mscclpp::getIBDeviceCount

.. doxygenfunction:: mscclpp::getIBDeviceName

.. doxygenfunction:: mscclpp::getIBTransportByDeviceName

.. doxygenfunction:: mscclpp::gpuMemcpy

.. doxygenfunction:: mscclpp::gpuMemcpyAsync

.. doxygenfunction:: mscclpp::gpuStreamPool

.. doxygenfunction:: mscclpp::isCuMemMapAllocated

.. doxygenfunction:: mscclpp::isNvlsSupported

.. doxygenfunction:: mscclpp::numaBind

Executor Interface
~~~~~~~~~~~~~~~~~~

.. doxygenclass:: mscclpp::ExecutionPlan
   :members:

.. doxygenclass:: mscclpp::Executor
   :members:

.. doxygenenum:: mscclpp::DataType

.. doxygenenum:: mscclpp::PacketType

Environment and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: mscclpp::Env
   :members:

.. doxygenfunction:: mscclpp::env

Error Handling
~~~~~~~~~~~~~~

.. doxygenclass:: mscclpp::BaseError
   :members:

.. doxygenclass:: mscclpp::CudaError
   :members:

.. doxygenclass:: mscclpp::CuError
   :members:

.. doxygenclass:: mscclpp::Error
   :members:

.. doxygenclass:: mscclpp::IbError
   :members:

.. doxygenclass:: mscclpp::SysError
   :members:

.. doxygenenum:: mscclpp::ErrorCode

.. doxygenfunction:: mscclpp::errorToString

Version
~~~~~~~

.. doxygenfunction:: mscclpp::version

Macro Functions
~~~~~~~~~~~~~~~

.. doxygendefine:: MSCCLPP_CUDATHROW

.. doxygendefine:: MSCCLPP_CUTHROW


.. _device-side-interfaces:

Device-Side Interfaces
----------------------

These device-side handle structures provide GPU kernel interfaces for MSCCL++ communication primitives. They are designed to be used directly in CUDA/HIP device code.

Channel Device Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: mscclpp::BaseMemoryChannelDeviceHandle
   :members:

.. doxygenstruct:: mscclpp::BasePortChannelDeviceHandle
   :members:

.. doxygenunion:: mscclpp::LL16Packet

.. doxygenunion:: mscclpp::LL8Packet

.. doxygenstruct:: mscclpp::MemoryChannelDeviceHandle
   :members:

.. doxygenstruct:: mscclpp::PortChannelDeviceHandle
   :members:

.. doxygenstruct:: mscclpp::SwitchChannelDeviceHandle
   :members:

.. doxygentypedef:: mscclpp::LLPacket

.. doxygentypedef:: mscclpp::MemoryId

.. doxygentypedef:: mscclpp::SemaphoreId

Semaphore Device Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: mscclpp::Host2DeviceSemaphoreDeviceHandle
   :members:

.. doxygenstruct:: mscclpp::MemoryDevice2DeviceSemaphoreDeviceHandle
   :members:

FIFO Device Interfaces
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: mscclpp::FifoDeviceHandle
   :members:

.. doxygenunion:: mscclpp::ProxyTrigger

.. doxygenvariable:: mscclpp::TriggerBitsFifoReserved

.. doxygenvariable:: mscclpp::TriggerBitsMemoryId

.. doxygenvariable:: mscclpp::TriggerBitsOffset

.. doxygenvariable:: mscclpp::TriggerBitsSemaphoreId

.. doxygenvariable:: mscclpp::TriggerBitsSize

.. doxygenvariable:: mscclpp::TriggerBitsType

.. doxygentypedef:: mscclpp::TriggerType

.. doxygenvariable:: mscclpp::TriggerData

.. doxygenvariable:: mscclpp::TriggerFlag

.. doxygenvariable:: mscclpp::TriggerSync

Device Utilities
~~~~~~~~~~~~~~~~

.. doxygenstruct:: mscclpp::DeviceSemaphore
   :members:

.. doxygenstruct:: mscclpp::DeviceSyncer
   :members:

.. doxygenunion:: mscclpp::VectorType

.. doxygenstruct:: mscclpp::Words
   :members:

.. doxygenfunction:: mscclpp::copy

.. doxygenfunction:: mscclpp::copyFromPackets

.. doxygenfunction:: mscclpp::copyToPackets

Atomics
~~~~~~~

.. doxygenvariable:: mscclpp::memoryOrderAcqRel

.. doxygenvariable:: mscclpp::memoryOrderAcquire

.. doxygenvariable:: mscclpp::memoryOrderRelaxed

.. doxygenvariable:: mscclpp::memoryOrderRelease

.. doxygenvariable:: mscclpp::memoryOrderSeqCst

.. doxygenvariable:: mscclpp::scopeDevice

.. doxygenvariable:: mscclpp::scopeSystem

.. doxygenfunction:: mscclpp::atomicFetchAdd

.. doxygenfunction:: mscclpp::atomicLoad

.. doxygenfunction:: mscclpp::atomicStore

Vector Data Types
~~~~~~~~~~~~~~~~~

.. doxygentypedef:: mscclpp::bf16x2

.. doxygentypedef:: mscclpp::bf16x4

.. doxygentypedef:: mscclpp::bf16x8

.. doxygentypedef:: mscclpp::f16x2

.. doxygentypedef:: mscclpp::f16x4

.. doxygentypedef:: mscclpp::f16x8

.. doxygentypedef:: mscclpp::f32x1

.. doxygentypedef:: mscclpp::f32x2

.. doxygentypedef:: mscclpp::f32x4

.. doxygentypedef:: mscclpp::f64x1

.. doxygentypedef:: mscclpp::i32x1

.. doxygentypedef:: mscclpp::i32x2

.. doxygentypedef:: mscclpp::i32x4

.. doxygentypedef:: mscclpp::u32x1

.. doxygentypedef:: mscclpp::u32x2

.. doxygentypedef:: mscclpp::u32x4

Macro Functions
~~~~~~~~~~~~~~~

.. doxygendefine:: MSCCLPP_ASSERT_DEVICE

.. doxygendefine:: OR_POLL_MAYBE_JAILBREAK

.. doxygendefine:: POLL_MAYBE_JAILBREAK
