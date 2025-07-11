// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CORE_HPP_
#define MSCCLPP_CORE_HPP_

#define MSCCLPP_MAJOR 0
#define MSCCLPP_MINOR 7
#define MSCCLPP_PATCH 0
#define MSCCLPP_VERSION (MSCCLPP_MAJOR * 10000 + MSCCLPP_MINOR * 100 + MSCCLPP_PATCH)

#include <array>
#include <bitset>
#include <future>
#include <memory>
#include <mscclpp/errors.hpp>
#include <string>
#include <vector>

namespace mscclpp {

#define MSCCLPP_UNIQUE_ID_BYTES 128

/// Unique ID for initializing the TcpBootstrap.
using UniqueId = std::array<uint8_t, MSCCLPP_UNIQUE_ID_BYTES>;

/// Return a version string.
/// @return The MSCCL++ version string in "major.minor.patch" format.
std::string version();

/// Base class for bootstraps.
class Bootstrap {
 public:
  /// Constructor.
  Bootstrap(){};

  /// Destructor.
  virtual ~Bootstrap() = default;

  /// Return the rank of the process.
  /// @return The rank of the process.
  virtual int getRank() const = 0;

  /// Return the total number of ranks.
  /// @return The total number of ranks.
  virtual int getNranks() const = 0;

  /// Return the total number of ranks per node.
  /// @return The total number of ranks per node.
  virtual int getNranksPerNode() const = 0;

  /// Send arbitrary data to another process.
  ///
  /// Data sent via `send(senderBuff, size, receiverRank, tag)` can be received via `recv(receiverBuff, size,
  /// senderRank, tag)`. Multiple calls to send() with the same @p peer and @p tag will be ordered by
  /// the order of calls, corresponding to the order of recv() calls on the receiving side. In cases where
  /// the execution order of multiple send()s or recv()s between two ranks is unknown, they should be differentiated
  /// by using different @p tag values to prevent unexpected behavior.
  ///
  /// @param data The data to send.
  /// @param size The size of the data to send.
  /// @param peer The rank of the process to send the data to.
  /// @param tag The tag to send the data with.
  virtual void send(void* data, int size, int peer, int tag) = 0;

  /// Receive data sent from another process by send().
  ///
  /// Data sent via `send(senderBuff, size, receiverRank, tag)` can be received via `recv(receiverBuff, size,
  /// senderRank, tag)`. Multiple calls to send() with the same @p peer and @p tag will be ordered by
  /// the order of calls, corresponding to the order of recv() calls on the receiving side. In cases where
  /// the execution order of multiple send()s or recv()s between two ranks is unknown, they should be differentiated
  /// by using different @p tag values to prevent unexpected behavior.
  ///
  /// @param data The buffer to write the received data to.
  /// @param size The size of the data to receive.
  /// @param peer The rank of the process to receive the data from.
  /// @param tag The tag to receive the data with.
  virtual void recv(void* data, int size, int peer, int tag) = 0;

  /// Gather data from all processes.
  ///
  /// When called by rank `r`, this sends data from `allData[r * size]` to `allData[(r + 1) * size - 1]` to all other
  /// ranks. The data sent by rank `r` is received into `allData[r * size]` of other ranks.
  ///
  /// @param allData The buffer to write the received data to.
  /// @param size The size of the data each rank sends.
  virtual void allGather(void* allData, int size) = 0;

  /// Synchronize all processes.
  virtual void barrier() = 0;

  /// A partial barrier that synchronizes a group of ranks.
  /// @param ranks The ranks to synchronize.
  void groupBarrier(const std::vector<int>& ranks);

  /// Wrapper of send() that sends a vector of characters.
  /// @param data The data to send.
  /// @param peer The rank of the process to send the data to.
  /// @param tag The tag to send the data with.
  void send(const std::vector<char>& data, int peer, int tag);

  /// Wrapper of recv() that receives a vector of characters.
  /// @param data The buffer to write the received data to.
  /// @param peer The rank of the process to receive the data from.
  /// @param tag The tag to receive the data with.
  ///
  /// @note The data vector will be resized to the size of the received data.
  void recv(std::vector<char>& data, int peer, int tag);
};

/// A native implementation of the bootstrap using TCP sockets.
class TcpBootstrap : public Bootstrap {
 public:
  /// Create a random unique ID.
  /// @return The created unique ID.
  static UniqueId createUniqueId();

  /// Constructor.
  /// @param rank The rank of the process.
  /// @param nRanks The total number of ranks.
  TcpBootstrap(int rank, int nRanks);

  /// Destructor.
  ~TcpBootstrap();

  /// Return the unique ID stored in the TcpBootstrap.
  /// @return The unique ID stored in the TcpBootstrap.
  UniqueId getUniqueId() const;

  /// Initialize the TcpBootstrap with a given unique ID. The unique ID can be generated by any method;
  /// it can be created by createUniqueId() or can be any arbitrary bit array provided by the user.
  /// @param uniqueId The unique ID to initialize the TcpBootstrap with.
  /// @param timeoutSec The connection timeout in seconds.
  void initialize(UniqueId uniqueId, int64_t timeoutSec = 30);

  /// Initialize the TcpBootstrap with a string formatted as "ip:port" or "interface:ip:port".
  /// @param ifIpPortTrio The string formatted as "ip:port" or "interface:ip:port".
  /// @param timeoutSec The connection timeout in seconds.
  void initialize(const std::string& ifIpPortTrio, int64_t timeoutSec = 30);

  /// Return the rank of the process.
  int getRank() const override;

  /// Return the total number of ranks.
  int getNranks() const override;

  /// Return the total number of ranks per node.
  int getNranksPerNode() const override;

  /// Send arbitrary data to another process.
  ///
  /// Data sent via `send(senderBuff, size, receiverRank, tag)` can be received via `recv(receiverBuff, size,
  /// senderRank, tag)`. Multiple calls to send() with the same @p peer and @p tag will be ordered by
  /// the order of calls, corresponding to the order of recv() calls on the receiving side. In cases where
  /// the execution order of multiple send()s or recv()s between two ranks is unknown, they should be differentiated
  /// by using different @p tag values to prevent unexpected behavior.
  ///
  /// @param data The data to send.
  /// @param size The size of the data to send.
  /// @param peer The rank of the process to send the data to.
  /// @param tag The tag to send the data with.
  void send(void* data, int size, int peer, int tag) override;

  /// Receive data sent from another process by send().
  ///
  /// Data sent via `send(senderBuff, size, receiverRank, tag)` can be received via `recv(receiverBuff, size,
  /// senderRank, tag)`. Multiple calls to send() with the same @p peer and @p tag will be ordered by
  /// the order of calls, corresponding to the order of recv() calls on the receiving side. In cases where
  /// the execution order of multiple send()s or recv()s between two ranks is unknown, they should be differentiated
  /// by using different @p tag values to prevent unexpected behavior.
  ///
  /// @param data The buffer to write the received data to.
  /// @param size The size of the data to receive.
  /// @param peer The rank of the process to receive the data from.
  /// @param tag The tag to receive the data with.
  void recv(void* data, int size, int peer, int tag) override;

  /// Gather data from all processes.
  ///
  /// When called by rank `r`, this sends data from `allData[r * size]` to `allData[(r + 1) * size - 1]` to all other
  /// ranks. The data sent by rank `r` is received into `allData[r * size]` of other ranks.
  ///
  /// @param allData The buffer to write the received data to.
  /// @param size The size of the data each rank sends.
  void allGather(void* allData, int size) override;

  /// Broadcast data from the root process to all processes using a ring-based algorithm.
  ///
  /// When called by the root rank, this sends the `size` bytes starting at memory location `data` to all other
  /// ranks. Non-root ranks receive these bytes into their own `data` buffer, overwriting its previous contents.
  /// The data propagates sequentially through a logical ring of processes until all ranks have received it.
  ///
  /// @param data Pointer to the send buffer (root) or receive buffer (non-root)
  /// @param size Number of bytes to broadcast
  /// @param root Rank initiating the broadcast
  void broadcast(void* data, int size, int root);

  /// Synchronize all processes.
  void barrier() override;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

/// Enumerates the available transport types.
enum class Transport {
  Unknown,        // Unknown transport type.
  CudaIpc,        // CUDA IPC transport type.
  Nvls,           // NVLS transport type.
  IB0,            // InfiniBand device 0 transport type.
  IB1,            // InfiniBand device 1 transport type.
  IB2,            // InfiniBand device 2 transport type.
  IB3,            // InfiniBand device 3 transport type.
  IB4,            // InfiniBand device 4 transport type.
  IB5,            // InfiniBand device 5 transport type.
  IB6,            // InfiniBand device 6 transport type.
  IB7,            // InfiniBand device 7 transport type.
  Ethernet,       // Ethernet transport type.
  NumTransports,  // The number of transports.
};

namespace detail {
const size_t TransportFlagsSize = 12;
static_assert(TransportFlagsSize == static_cast<size_t>(Transport::NumTransports),
              "TransportFlagsSize must match the number of transports");
/// Bitset for storing transport flags.
using TransportFlagsBase = std::bitset<TransportFlagsSize>;
}  // namespace detail

/// Stores transport flags.
class TransportFlags : private detail::TransportFlagsBase {
 public:
  /// Constructor.
  TransportFlags() = default;

  /// Constructor.
  /// @param transport The transport to set the flag for.
  TransportFlags(Transport transport);

  /// Check if a specific transport flag is set.
  /// @param transport The transport to check the flag for.
  /// @return True if the flag is set, false otherwise.
  bool has(Transport transport) const;

  /// Check if no transport flags are set.
  /// @return True if no flags are set, false otherwise.
  bool none() const;

  /// Check if any transport flags are set.
  /// @return True if any flags are set, false otherwise.
  bool any() const;

  /// Check if all transport flags are set.
  /// @return True if all flags are set, false otherwise.
  bool all() const;

  /// Get the number of transport flags that are set.
  /// @return The number of flags that are set.
  size_t count() const;

  /// Bitwise OR assignment operator for TransportFlags.
  /// @param other The other TransportFlags to perform the OR operation with.
  /// @return A reference to the modified TransportFlags.
  TransportFlags& operator|=(TransportFlags other);

  /// Bitwise OR operator for TransportFlags.
  /// @param other The other TransportFlags to perform the OR operation with.
  /// @return A new TransportFlags object with the result of the OR operation.
  TransportFlags operator|(TransportFlags other) const;

  /// Bitwise OR operator for TransportFlags and Transport.
  /// @param transport The Transport to perform the OR operation with.
  /// @return A new TransportFlags object with the result of the OR operation.
  TransportFlags operator|(Transport transport) const;

  /// Bitwise AND assignment operator for TransportFlags.
  /// @param other The other TransportFlags to perform the AND operation with.
  /// @return A reference to the modified TransportFlags.
  TransportFlags& operator&=(TransportFlags other);

  /// Bitwise AND operator for TransportFlags.
  /// @param other The other TransportFlags to perform the AND operation with.
  /// @return A new TransportFlags object with the result of the AND operation.
  TransportFlags operator&(TransportFlags other) const;

  /// Bitwise AND operator for TransportFlags and Transport.
  /// @param transport The Transport to perform the AND operation with.
  /// @return A new TransportFlags object with the result of the AND operation.
  TransportFlags operator&(Transport transport) const;

  /// Bitwise XOR assignment operator for TransportFlags.
  /// @param other The other TransportFlags to perform the XOR operation with.
  /// @return A reference to the modified TransportFlags.
  TransportFlags& operator^=(TransportFlags other);

  /// Bitwise XOR operator for TransportFlags.
  /// @param other The other TransportFlags to perform the XOR operation with.
  /// @return A new TransportFlags object with the result of the XOR operation.
  TransportFlags operator^(TransportFlags other) const;

  /// Bitwise XOR operator for TransportFlags and Transport.
  /// @param transport The Transport to perform the XOR operation with.
  /// @return A new TransportFlags object with the result of the XOR operation.
  TransportFlags operator^(Transport transport) const;

  /// Bitwise NOT operator for TransportFlags.
  /// @return A new TransportFlags object with the result of the NOT operation.
  TransportFlags operator~() const;

  /// Equality comparison operator for TransportFlags.
  /// @param other The other TransportFlags to compare with.
  /// @return True if the two TransportFlags objects are equal, false otherwise.
  bool operator==(TransportFlags other) const;

  /// Inequality comparison operator for TransportFlags.
  /// @param other The other TransportFlags to compare with.
  /// @return True if the two TransportFlags objects are not equal, false otherwise.
  bool operator!=(TransportFlags other) const;

  /// Convert the TransportFlags object to a bitset representation.
  /// @return A detail::TransportFlagsBase object representing the TransportFlags object.
  detail::TransportFlagsBase toBitset() const;

 private:
  /// Private constructor for TransportFlags that takes a bitset representation.
  /// @param bitset The bitset representation of the TransportFlags object.
  TransportFlags(detail::TransportFlagsBase bitset);
};

/// Bitwise OR operator for two Transport objects.
///
/// @param transport1 The first Transport to perform the OR operation with.
/// @param transport2 The second Transport to perform the OR operation with.
/// @return A new TransportFlags object with the result of the OR operation.
inline TransportFlags operator|(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) | transport2;
}

/// Bitwise AND operator for two Transport objects.
///
/// @param transport1 The first Transport to perform the AND operation with.
/// @param transport2 The second Transport to perform the AND operation with.
/// @return A new TransportFlags object with the result of the AND operation.
inline TransportFlags operator&(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) & transport2;
}

/// Bitwise XOR operator for two Transport objects.
///
/// @param transport1 The first Transport to perform the XOR operation with.
/// @param transport2 The second Transport to perform the XOR operation with.
/// @return A new TransportFlags object with the result of the XOR operation.
inline TransportFlags operator^(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) ^ transport2;
}

/// Available device types.
enum class DeviceType {
  Unknown,  // Unknown device type.
  CPU,      // CPU device type.
  GPU,      // GPU device type.
};

struct Device {
  /// Constructor.
  Device() = default;

  /// Constructor.
  /// @param type Device type.
  /// @param id Device ID. Default is -1 (no specific ID).
  Device(DeviceType type, int id = -1) : type(type), id(id) {}

  /// Device Type.
  DeviceType type;

  /// Device ID.
  int id;
};

class Context;
class Connection;

/// Block of memory that has been registered to a Context.
/// RegisteredMemory does not own the memory it points to, but it provides a way to transfer metadata about the memory
/// to other processes, hence allowing their access to the memory block.
class RegisteredMemory {
 public:
  /// Constructor.
  RegisteredMemory() = default;

  /// Destructor.
  ~RegisteredMemory();

  /// Get a pointer to the memory block.
  /// @return A pointer to the memory block.
  void* data() const;

  /// Get a pointer to the original memory block.
  /// @return A pointer to the original memory block.
  void* originalDataPtr() const;

  /// Get the size of the memory block.
  /// @return The size of the memory block.
  size_t size() const;

  /// Get the transport flags associated with the memory block.
  /// @return The transport flags associated with the memory block.
  TransportFlags transports() const;

  /// Serialize the RegisteredMemory object to a vector of characters.
  /// @return A vector of characters representing the serialized RegisteredMemory object.
  std::vector<char> serialize() const;

  /// Deserialize a RegisteredMemory object from a vector of characters.
  /// @param data A vector of characters representing a serialized RegisteredMemory object.
  /// @return A deserialized RegisteredMemory object.
  static RegisteredMemory deserialize(const std::vector<char>& data);

 private:
  struct Impl;
  RegisteredMemory(std::shared_ptr<Impl> pimpl);
  std::shared_ptr<Impl> pimpl_;

  friend class Context;
  friend class Connection;
  friend class SemaphoreStub;
};

/// One end of a connection.
class Endpoint {
 public:
  /// Constructor.
  Endpoint() = default;

  /// Get the transport used.
  /// @return The transport used.
  Transport transport() const;

  /// Get the device used.
  /// @return The device used.
  const Device& device() const;

  /// Get the maximum write queue size.
  /// @return The maximum number of write requests that can be queued.
  int maxWriteQueueSize() const;

  /// Serialize the Endpoint object to a vector of characters.
  /// @return A vector of characters representing the serialized Endpoint object.
  std::vector<char> serialize() const;

  /// Deserialize an Endpoint object from a vector of characters.
  /// @param data A vector of characters representing a serialized Endpoint object.
  /// @return A deserialized Endpoint object.
  static Endpoint deserialize(const std::vector<char>& data);

 private:
  struct Impl;
  Endpoint(std::shared_ptr<Impl> pimpl);
  std::shared_ptr<Impl> pimpl_;

  friend class Context;
  friend class Connection;
};

/// Connection between two processes.
class Connection {
 public:
  /// Constructor.
  /// @param localEndpoint The local endpoint of the connection.
  Connection(std::shared_ptr<Context> context, const Endpoint& localEndpoint)
      : context_(context), localEndpoint_(localEndpoint), maxWriteQueueSize_(localEndpoint.maxWriteQueueSize()) {}

  /// Destructor.
  virtual ~Connection() = default;

  /// Write data from a source RegisteredMemory to a destination RegisteredMemory.
  ///
  /// @param dst The destination RegisteredMemory.
  /// @param dstOffset The offset in bytes from the start of the destination RegisteredMemory.
  /// @param src The source RegisteredMemory.
  /// @param srcOffset The offset in bytes from the start of the source RegisteredMemory.
  /// @param size The number of bytes to write.
  virtual void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) = 0;

  /// Update an 8-byte value in a destination RegisteredMemory and synchronize the change with the remote process.
  ///
  /// @param dst The destination RegisteredMemory.
  /// @param dstOffset The offset in bytes from the start of the destination RegisteredMemory.
  /// @param src A pointer to the value to update.
  /// @param newValue The new value to write.
  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) = 0;

  /// Flush any pending writes to the remote process.
  /// @param timeoutUsec Timeout in microseconds. Default: -1 (no timeout)
  virtual void flush(int64_t timeoutUsec = -1) = 0;

  /// Get the transport used by the local process.
  /// @return The transport used by the local process.
  virtual Transport transport() const = 0;

  /// Get the transport used by the remote process.
  /// @return The transport used by the remote process.
  virtual Transport remoteTransport() const = 0;

  /// Get the context associated with this connection.
  /// @return A shared pointer to the context associated with this connection.
  std::shared_ptr<Context> context() const { return context_; }

  /// Get the device used by the local endpoint.
  /// @return The device used by the local endpoint.
  const Device& localDevice() const;

  /// Get the maximum write queue size.
  /// @return The maximum number of write requests that can be queued.
  int getMaxWriteQueueSize() const;

 protected:
  static std::shared_ptr<RegisteredMemory::Impl> getImpl(RegisteredMemory& memory);
  static std::shared_ptr<Endpoint::Impl> getImpl(Endpoint& memory);

  std::shared_ptr<Context> context_;
  Endpoint localEndpoint_;
  int maxWriteQueueSize_;
};

/// Used to configure an endpoint.
struct EndpointConfig {
  static const int DefaultMaxCqSize = 1024;
  static const int DefaultMaxCqPollNum = 1;
  static const int DefaultMaxSendWr = 8192;
  static const int DefaultMaxWrPerSend = 64;

  Transport transport;
  Device device;
  int ibMaxCqSize;
  int ibMaxCqPollNum;
  int ibMaxSendWr;
  int ibMaxWrPerSend;
  int maxWriteQueueSize;

  /// Constructor that takes a transport and sets the other fields to their default values.
  ///
  /// @param transport The transport to use.
  /// @param device The device to use.
  /// @param ibMaxCqSize The maximum completion queue size.
  /// @param ibMaxCqPollNum The maximum completion queue poll number.
  /// @param ibMaxSendWr The maximum send work requests.
  /// @param ibMaxWrPerSend The maximum work requests per send.
  /// @param maxWriteQueueSize The maximum write queue size.
  EndpointConfig(Transport transport = Transport::Unknown, Device device = DeviceType::GPU,
                 int ibMaxCqSize = DefaultMaxCqSize, int ibMaxCqPollNum = DefaultMaxCqPollNum,
                 int ibMaxSendWr = DefaultMaxSendWr, int ibMaxWrPerSend = DefaultMaxWrPerSend,
                 int maxWriteQueueSize = -1)
      : transport(transport),
        device(device),
        ibMaxCqSize(ibMaxCqSize),
        ibMaxCqPollNum(ibMaxCqPollNum),
        ibMaxSendWr(ibMaxSendWr),
        ibMaxWrPerSend(ibMaxWrPerSend),
        maxWriteQueueSize(maxWriteQueueSize) {}
};

/// Context for communication. This provides a low-level interface for forming connections in use-cases
/// where the process group abstraction offered by Communicator is not suitable, e.g., ephemeral client-server
/// connections. Correct use of this class requires external synchronization when finalizing connections with the
/// connect() method.
///
/// As an example, a client-server scenario where the server will write to the client might proceed as follows:
///   1. The client creates an endpoint with createEndpoint() and sends it to the server.
///   2. The server receives the client endpoint, creates its own endpoint with createEndpoint(), sends it to the
///      client, and creates a connection with connect().
///   3. The client receives the server endpoint, creates a connection with connect() and sends a
///      RegisteredMemory to the server.
///   4. The server receives the RegisteredMemory and writes to it using the previously created connection.
/// The client waiting to create a connection before sending the RegisteredMemory ensures that the server cannot
/// write to the RegisteredMemory before the connection is established.
///
/// While some transports may have more relaxed implementation behavior, this should not be relied upon.
class Context : public std::enable_shared_from_this<Context> {
 public:
  /// Create a new Context instance.
  static std::shared_ptr<Context> create() { return std::shared_ptr<Context>(new Context()); }

  /// Destructor.
  ~Context();

  /// Register a region of GPU memory for use in this context.
  ///
  /// @param ptr Base pointer to the memory.
  /// @param size Size of the memory region in bytes.
  /// @param transports Transport flags.
  /// @return A RegisteredMemory object representing the registered memory region.
  RegisteredMemory registerMemory(void* ptr, size_t size, TransportFlags transports);

  /// Create an endpoint for establishing connections.
  ///
  /// @param config The configuration for the endpoint.
  /// @return The newly created endpoint.
  Endpoint createEndpoint(EndpointConfig config);

  /// Establish a connection between two endpoints. While this method immediately returns a connection object, the
  /// connection is only safe to use after the corresponding connection on the remote endpoint has been established.
  /// This method must be called on both endpoints to establish a connection.
  ///
  /// @param localEndpoint The local endpoint.
  /// @param remoteEndpoint The remote endpoint.
  /// @return A shared pointer to the connection.
  std::shared_ptr<Connection> connect(Endpoint localEndpoint, Endpoint remoteEndpoint);

 private:
  Context();

  struct Impl;
  std::unique_ptr<Impl> pimpl_;

  friend class RegisteredMemory;
  friend class Endpoint;
};

/// SemaphoreStub object only used for constructing Semaphore, not for direct use by the user.
class SemaphoreStub {
 public:
  /// Constructor.
  /// @param connection A shared pointer to the connection associated with this semaphore.
  SemaphoreStub(std::shared_ptr<Connection> connection);

  /// Get the memory associated with this semaphore.
  /// @return A reference to the registered memory for this semaphore.
  const RegisteredMemory& memory() const;

  /// Serialize into a vector of characters.
  /// @return A vector of characters representing the serialized SemaphoreStub object.
  std::vector<char> serialize() const;

  /// Deserialize a SemaphoreStub object from a vector of characters.
  /// @param data A vector of characters representing a serialized SemaphoreStub object.
  /// @return A deserialized SemaphoreStub object.
  static SemaphoreStub deserialize(const std::vector<char>& data);

 protected:
  struct Impl;
  SemaphoreStub(std::shared_ptr<Impl> pimpl);
  std::shared_ptr<Impl> pimpl_;

  friend class Semaphore;
};

/// Semaphore used by channels for synchronization.
class Semaphore {
 public:
  /// Constructor.
  Semaphore() = default;

  /// Constructor.
  /// @param localStub SemaphoreStub allocated on the local process.
  /// @param remoteStub SemaphoreStub allocated on the remote process.
  Semaphore(const SemaphoreStub& localStub, const SemaphoreStub& remoteStub);

  /// Get the connection associated with this semaphore.
  /// @return A shared pointer to the connection.
  std::shared_ptr<Connection> connection() const;

  /// Get the local memory associated with this semaphore.
  /// @return A reference to the local registered memory.
  const RegisteredMemory& localMemory() const;

  /// Get the remote memory associated with this semaphore.
  /// @return A reference to the remote registered memory.
  const RegisteredMemory& remoteMemory() const;

 protected:
  struct Impl;
  std::shared_ptr<Impl> pimpl_;
};

template <typename T>
using NonblockingFuture [[deprecated("Use std::shared_future instead. This will be removed in a future release.")]] =
    std::shared_future<T>;

/// A class that sets up all registered memories and connections between processes.
///
/// A typical way to use this class:
///   1. Call connect() to declare connections between the calling process and other processes.
///   2. Call registerMemory() to register memory regions that will be used for communication.
///   3. Call sendMemory() or recvMemory() to send/receive registered memory regions to/from
///      other processes.
///   4. Call get() on futures returned by connect(). Use the returned connections to create flags.
///   5. Call buildSemaphore() to create a Semaphore out of the flags.
///   6. Call get() on all futures returned by buildSemaphore() and recvMemory().
///   7. All done; use semaphores and registered memories to build channels.
///
/// CAUTION: The order of method calls matters when the same remote rank and tags are used. That is, the i-th
/// "sending" method call (sendMemory(), connect(), and buildSemaphore()) on the local rank must be matched
/// by the i-th "receiving" method call (recvMemory(), connect(), and buildSemaphore()) on the remote rank.
///
/// Correct Example 1:
/// ```cpp
/// // Rank 0
/// communicator.sendMemory(memory1, 1, tag);
/// communicator.sendMemory(memory2, 1, tag);
/// auto connection = communicator.connect(Transport::CudaIpc, 1, tag);
/// connection.get(); // This will return the connection.
/// // Rank 1
/// auto mem1 = communicator.recvMemory(0, tag);
/// auto mem2 = communicator.recvMemory(0, tag);
/// auto connection = communicator.connect(Transport::CudaIpc, 0, tag);
/// mem2.get();       // This will return memory2.
/// connection.get(); // This will return the connection.
/// mem1.get();       // This will return memory1.
/// ```
///
/// Correct Example 2:
/// ```cpp
/// // Rank 0
/// communicator.sendMemory(memory0, 1, tag);
/// auto mem1 = communicator.recvMemory(1, tag);
/// auto connection = communicator.connect(Transport::CudaIpc, 1, tag);
/// connection.get(); // This will return the connection.
/// mem1.get();       // This will return memory1.
/// // Rank 1
/// auto mem0 = communicator.recvMemory(0, tag);
/// communicator.sendMemory(memory1, 0, tag);
/// auto connection = communicator.connect(Transport::CudaIpc, 0, tag);
/// mem0.get();       // This will return memory0.
/// connection.get(); // This will return the connection.
/// ```
///
/// Wrong Example:
/// ```cpp
/// // Rank 0
/// communicator.sendMemory(memory0, 1, tag);
/// auto mem1 = communicator.recvMemory(1, tag);
/// auto connection = communicator.connect(Transport::CudaIpc, 1, tag);
/// // Rank 1
/// auto mem0 = communicator.recvMemory(0, tag);
/// auto connection = communicator.connect(Transport::CudaIpc, 0, tag); // undefined behavior
/// communicator.sendMemory(memory1, 0, tag);
/// ```
/// In the wrong example, the connection information from rank 1 will be sent to the `mem1` object on rank 0,
/// where the object type is RegisteredMemory, not Connection.
///
class Communicator {
 public:
  /// Initializes the communicator with a given bootstrap implementation.
  ///
  /// @param bootstrap An implementation of the Bootstrap that the communicator will use.
  /// @param context An optional context to use for the communicator. If not provided, a new context will be created.
  Communicator(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context = nullptr);

  /// Destroy the communicator.
  ~Communicator();

  /// Returns the bootstrap held by this communicator.
  /// @return The bootstrap held by this communicator.
  std::shared_ptr<Bootstrap> bootstrap();

  /// Returns the context held by this communicator.
  /// @return The context held by this communicator.
  std::shared_ptr<Context> context();

  /// Register a region of GPU memory for use in this communicator's context.
  ///
  /// @param ptr Base pointer to the memory.
  /// @param size Size of the memory region in bytes.
  /// @param transports Transport flags.
  /// @return A RegisteredMemory object representing the registered memory region.
  RegisteredMemory registerMemory(void* ptr, size_t size, TransportFlags transports);

  /// Send information of a registered memory to the remote side.
  ///
  /// The send will be started upon calling this function, but this function returns immediately without
  /// waiting for the completion of the send. RegisteredMemory sent via `sendMemory(memory, remoteRank, tag)` can be
  /// received via `recvMemory(remoteRank, tag)`.
  ///
  /// Multiple calls to either sendMemory() or connect() with the same @p remoteRank and @p tag will be ordered by
  /// the order of calls, corresponding to the order of recvMemory() or connect() calls on the receiving side.
  /// In cases where the execution order is unknown between two ranks, they should be differentiated by using
  /// different @p tag values to prevent unexpected behavior.
  ///
  /// @param memory The registered memory buffer to send information about.
  /// @param remoteRank The rank of the remote process.
  /// @param tag The tag to use for identifying the send.
  ///
  void sendMemory(RegisteredMemory memory, int remoteRank, int tag = 0);

  [[deprecated("Use sendMemory() instead. This will be removed in a future release.")]] void sendMemoryOnSetup(
      RegisteredMemory memory, int remoteRank, int tag) {
    sendMemory(memory, remoteRank, tag);
  }

  /// Receive memory information from a corresponding sendMemory call on the remote side.
  ///
  /// This function returns a future immediately. The actual receive will be performed upon calling
  /// the first get() on the future. RegisteredMemory sent via `sendMemory(memory, remoteRank, tag)` can be
  /// received via `recvMemory(remoteRank, tag)`.
  ///
  /// Multiple calls to either sendMemory() or connect() with the same @p remoteRank and @p tag will be ordered by
  /// the order of calls, corresponding to the order of recvMemory() or connect() calls on the receiving side.
  /// In cases where the execution order is unknown between two ranks, they should be differentiated by using
  /// different @p tag values to prevent unexpected behavior.
  ///
  /// @note To guarantee the receiving order, calling get() on a future returned by recvMemory() or connect()
  /// may start receiving other RegisteredMemory or Connection objects of which futures were returned by
  /// an earlier call to recvMemory() or connect() with the same @p remoteRank and @p tag. For example, if
  /// we call recvMemory() or connect() five times with the same @p remoteRank and @p tag and then call get()
  /// on the last future, it will start receiving the five RegisteredMemory or Connection objects in order,
  /// back to back.
  ///
  /// @param remoteRank The rank of the remote process.
  /// @param tag The tag to use for identifying the receive.
  /// @return A future of registered memory.
  ///
  std::shared_future<RegisteredMemory> recvMemory(int remoteRank, int tag = 0);

  [[deprecated(
      "Use recvMemory() instead. This will be removed in a future release.")]] NonblockingFuture<RegisteredMemory>
  recvMemoryOnSetup(int remoteRank, int tag) {
    return recvMemory(remoteRank, tag);
  }

  /// Connect to a remote rank.
  ///
  /// This function will start (but not wait for) sending metadata about the local endpoint to the remote rank,
  /// and return a future connection without waiting for the remote rank to respond.
  /// The connection will be established when the remote rank responds with its own endpoint and the local rank calls
  /// the first get() on the future.
  /// Note that this function is two-way and a connection from rank `i` to remote rank `j` needs
  /// to have a counterpart from rank `j` to rank `i`. Note that with IB, buffers are registered at a page level and if
  /// a buffer is spread through multiple pages and does not fully utilize all of them, IB's QP has to register for all
  /// involved pages. This potentially has security risks if the connection's accesses are given to a malicious process.
  ///
  /// Multiple calls to either sendMemory() or connect() with the same @p remoteRank and @p tag will be ordered by
  /// the order of calls, corresponding to the order of recvMemory() or connect() calls on the receiving side.
  /// In cases where the execution order is unknown between two ranks, they should be differentiated by using
  /// different @p tag values to prevent unexpected behavior.
  ///
  /// @note To guarantee the receiving order, calling get() on a future returned by recvMemory() or connect()
  /// may start receiving other RegisteredMemory or Connection objects of which futures were returned by
  /// an earlier call to recvMemory() or connect() with the same @p remoteRank and @p tag. For example, if
  /// we call recvMemory() or connect() five times with the same @p remoteRank and @p tag and then call get()
  /// on the last future, it will start receiving the five RegisteredMemory or Connection objects in order,
  /// back to back.
  ///
  /// @param localConfig The configuration for the local endpoint.
  /// @param remoteRank The rank of the remote process.
  /// @param tag The tag to use for identifying the send and receive.
  /// @return A future of shared pointer to the connection.
  ///
  std::shared_future<std::shared_ptr<Connection>> connect(EndpointConfig localConfig, int remoteRank, int tag = 0);

  [[deprecated("Use connect(localConfig, remoteRank, tag) instead. This will be removed in a future release.")]] std::
      shared_future<std::shared_ptr<Connection>>
      connect(int remoteRank, int tag, EndpointConfig localConfig);

  [[deprecated("Use connect() instead. This will be removed in a future release.")]] NonblockingFuture<
      std::shared_ptr<Connection>>
  connectOnSetup(int remoteRank, int tag, EndpointConfig localConfig) {
    return connect(localConfig, remoteRank, tag);
  }

  /// Build a semaphore for cross-process synchronization.
  /// @param connection The connection associated with this semaphore.
  /// @param remoteRank The rank of the remote process.
  /// @param tag The tag to use for identifying the operation.
  /// @return A future of the built semaphore.
  std::shared_future<Semaphore> buildSemaphore(std::shared_ptr<Connection> connection, int remoteRank, int tag = 0);

  /// Get the remote rank a connection is connected to.
  ///
  /// @param connection The connection to get the remote rank for.
  /// @return The remote rank the connection is connected to.
  int remoteRankOf(const Connection& connection);

  /// Get the tag a connection was made with.
  ///
  /// @param connection The connection to get the tag for.
  /// @return The tag the connection was made with.
  int tagOf(const Connection& connection);

  [[deprecated("setup() is now no-op and no longer needed. This will be removed in a future release.")]] void setup() {}

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/// A constant TransportFlags object representing no transports.
extern const TransportFlags NoTransports;

/// A constant TransportFlags object representing all InfiniBand transports.
extern const TransportFlags AllIBTransports;

/// A constant TransportFlags object representing all transports.
extern const TransportFlags AllTransports;

/// A type which could be safely used on the device side.
template <class T>
using DeviceHandle = typename T::DeviceHandle;

/// Retrieve the deviceHandle instance from a host object.
template <typename T>
DeviceHandle<std::remove_reference_t<T>> deviceHandle(T&& t) {
  return t.deviceHandle();
}

/// Packet value type.
template <class T>
using PacketPayload = typename T::Payload;

}  // namespace mscclpp

namespace std {

std::string to_string(const mscclpp::Transport& transport);

std::string to_string(const mscclpp::Device& device);

/// Specialization of the std::hash template for mscclpp::TransportFlags.
template <>
struct hash<mscclpp::TransportFlags>;

}  // namespace std

#endif  // MSCCLPP_CORE_HPP_
