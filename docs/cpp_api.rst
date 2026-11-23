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

.. doxygenclass:: mscclpp::PortDevice2DeviceSemaphore
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

.. doxygenenum:: mscclpp::GpuBufferGranularity

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

GPU channel types use ``DeviceHandle`` names for non-owning state.
``BasePortChannelDeviceHandle<Backend>`` and ``PortChannelDeviceHandle<Backend>`` default to Host;
the Memory and Switch handle types remain concrete. GPU handle fields and operations, header filenames,
and Python export names and properties are unchanged. Python Port bindings select Host extraction.

Construct ``BasePortChannel`` or ``PortChannel`` with a Host semaphore and proxy, or with a Device
semaphore and registered memories. Extract the matching GPU handle explicitly:

.. code-block:: cpp

   using Backend = mscclpp::PortChannelBackend;
   mscclpp::PortChannel host = proxy.portChannel(semaphoreId, dstId, srcId);
   mscclpp::PortChannel::DeviceHandle<Backend::Host> handle = host.deviceHandle<Backend::Host>();

Keep the owner and application buffers alive during GPU use. Copies share semaphore/counter ownership
and retain their memory bindings. Default owners are unassigned. Extraction throws ``Error`` with
``ErrorCode::InvalidUsage`` for an unassigned owner, the wrong backend, a moved-from Port semaphore,
or unassigned or mismatched memory bindings. Device registration admission also checks that memory
is initialized and compatible with the connection. Host registry ID0 is valid.

Whole-owner assignment copies synchronization and memory bindings together, including across backends.
Base-only assignment does not update memory bindings. A same-backend base replacement still requires
compatible proxy/connection/regions; these checks do not establish general resource identity. Prefer
whole-owner assignment when rebinding. Host and Device GPU handles remain different types and cannot
be assigned to each other; GPU operations do not dispatch on the CPU owner's runtime state.

Port owner source migration
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Replace ``BasePortChannel<B>`` and ``PortChannel<B>`` with ``BasePortChannel`` and ``PortChannel``.
* Replace the fixed ``Channel::DeviceHandle`` type with ``Channel::DeviceHandle<B>`` for Port owners.
* Replace no-argument Port extraction with ``channel.deviceHandle<B>()``. There is no default backend.
* Port owners do not support ``mscclpp::DeviceHandle<Channel>`` or ``mscclpp::deviceHandle(channel)``.
  Generic callers must carry the backend explicitly at the Port boundary. The generic helper and
  associated-type protocol for other owners are unchanged.

The canonical GPU Port types remain templates. In type positions, use ``PortChannelDeviceHandle<>*``
rather than ``PortChannelDeviceHandle*``, and likewise for the base handle, or name
``Channel::DeviceHandle<B>``. ``SmChannelDeviceHandle`` remains a deprecated direct alias to
``MemoryChannelDeviceHandle``. Existing compatibility aliases are best-effort source conveniences,
not an ABI or layout guarantee. Rebuild consumers.

.. doxygenstruct:: mscclpp::BaseMemoryChannelDeviceHandle
   :members:

.. doxygenstruct:: mscclpp::BasePortChannelDeviceHandle< PortChannelBackend::Host >
   :members:

.. doxygenstruct:: mscclpp::BasePortChannelDeviceHandle< PortChannelBackend::Device >
   :members:

.. doxygenunion:: mscclpp::LL16Packet

.. doxygenunion:: mscclpp::LL8Packet

.. doxygenstruct:: mscclpp::MemoryChannelDeviceHandle
   :members:

.. doxygenstruct:: mscclpp::PortChannelDeviceHandle< PortChannelBackend::Host >
   :members:

.. doxygenstruct:: mscclpp::PortChannelDeviceHandle< PortChannelBackend::Device >
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

.. doxygenstruct:: mscclpp::PortDevice2DeviceSemaphoreDeviceHandle
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

.. doxygendefine:: MSCCLPP_BULK_AVAILABLE

.. doxygenstruct:: mscclpp::BulkBarrier
   :members:

.. doxygenenum:: mscclpp::BulkRedOp

.. doxygenfunction:: mscclpp::bulkLoad

.. doxygenfunction:: mscclpp::bulkStore

.. doxygenfunction:: mscclpp::bulkReduceStore

.. doxygenfunction:: mscclpp::bulkStoreCommit

.. doxygenfunction:: mscclpp::bulkStoreWait

.. doxygenfunction:: mscclpp::bulkStoreWaitSource

.. doxygenfunction:: mscclpp::bulkFence

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
