# RDMA providers

Internal RDMA providers isolate device resources and metadata from generic
context, endpoint, and registered-memory code. `Connection` is the I/O abstraction.

## Resources and ownership

| Internal interface / IB implementation | Responsibility |
| --- | --- |
| `RdmaProvider` / `IbProvider` | Own `IbCtx`; provide endpoint and memory factories; cached per transport by `Context::Impl`. |
| `RdmaEndpoint` / `IbEndpoint` | Own a local host or GPUNetIO QP, or imported metadata; serialize it; construct `IBConnection` or `GpuNetIoConnection`. |
| `RdmaMemory` / `IbMemory` | Own a local MR or imported metadata; serialize it. |

```{mermaid}
flowchart LR
    C[Context] --> B["IbProvider : RdmaProvider"]
    E[Endpoint] --> Q["IbEndpoint : RdmaEndpoint"]
    R[RegisteredMemory] --> M["IbMemory : RdmaMemory"]
    Q -->|retain when local| B
    M -->|retain when local| B
    B --> I["IbCtx / PD"]
    Q --> QP[IbQp]
    M --> MR[IbMr]
    Q -. factory .-> Conn[IBConnection]
```

Solid arrows denote ownership. Endpoints and registrations can exist without a
`Connection` and outlive their `Context`, so local wrappers retain their provider.
QP/MR destruction precedes provider release, keeping the device and PD alive.
Imported wrappers own metadata only and require no local NIC. Providers do not
own wrappers, avoiding ownership cycles. User backing buffers are not owned.

## Metadata and dispatch

Registered-memory records contain a transport tag, IPC metadata and an owning
`RdmaMemory` pointer, with no IB fields or header dependency. Generic serialization
delegates to the resource's virtual `serialize()` method; the IB implementation
uses `detail::serialize` on its concrete wire values. The raw-byte helper cannot
serialize a polymorphic resource or its owning pointer as metadata.

Per-record ownership preserves imported duplicate transports and their order.
IB imports preserve metadata bytes, including padding, and work in IB-disabled
builds. Factories bind transports to concrete resource types; IB code checks the
tag before a `static_cast` to `IbMemory`, without per-write RTTI.

RDMA classification and compatibility live with provider dispatch. Different IB
device indices are compatible; classification as RDMA does not establish
compatibility across provider families.

## Connection data path

The diagram shows host-mode IB. Device mode creates `GpuNetIoConnection` through
the same endpoint factory, but GPU PortChannel operations bypass the FIFO/proxy.
GPUNetIO accesses endpoint and MR wrappers using the same provider-owned `IbCtx`
and PD; its QP is released before the endpoint releases the provider.

```{mermaid}
flowchart LR
    G[GPU kernel] --> F["PortChannel / FIFO"]
    F --> P[ProxyService]
    P --> C[Connection]
    C --> I[IBConnection]
    I --> Q[IbQp]
```

Factories and endpoint type checks run during setup. QP transitions occur during
connection construction. IB operations access typed provider-owned MRs directly,
without an additional virtual provider I/O layer. `Connection` implementations
define posting, signaling/forwarding, CQ polling and flush behavior.

Provider resource interfaces cover endpoints, registrations and metadata import
independently of I/O. Embedding provider-specific fields in generic framework
classes duplicates ownership and metadata handling; abstracting QP operations
duplicates the `Connection` boundary.

Interfaces live in `src/core/include/`; concrete IB code is in `ib.hpp`/`ib.cc`.
`BaseConnection` shares validation helpers across connection implementations.
