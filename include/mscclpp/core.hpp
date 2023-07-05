// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CORE_HPP_
#define MSCCLPP_CORE_HPP_

#define MSCCLPP_MAJOR 0
#define MSCCLPP_MINOR 1
#define MSCCLPP_PATCH 0
#define MSCCLPP_VERSION (MSCCLPP_MAJOR * 10000 + MSCCLPP_MINOR * 100 + MSCCLPP_PATCH)

#include <bitset>
#include <future>
#include <memory>
#include <mscclpp/errors.hpp>
#include <string>
#include <vector>

namespace mscclpp {

#define MSCCLPP_UNIQUE_ID_BYTES 128

/// Unique ID for a process. This is a MSCCLPP_UNIQUE_ID_BYTES byte array that uniquely identifies a process.
struct UniqueId {
  char internal[MSCCLPP_UNIQUE_ID_BYTES];
};

/// Base class for bootstrappers.
class BaseBootstrap {
 public:
  BaseBootstrap(){};
  virtual ~BaseBootstrap() = default;
  virtual int getRank() = 0;
  virtual int getNranks() = 0;
  virtual void send(void* data, int size, int peer, int tag) = 0;
  virtual void recv(void* data, int size, int peer, int tag) = 0;
  virtual void allGather(void* allData, int size) = 0;
  virtual void barrier() = 0;

  void send(const std::vector<char>& data, int peer, int tag);
  void recv(std::vector<char>& data, int peer, int tag);
};

/// A native implementation of the bootstrapper.
class Bootstrap : public BaseBootstrap {
 public:
  /// Construct a Bootstrap.
  /// @param rank The rank of the process.
  /// @param nRanks The total number of ranks.
  Bootstrap(int rank, int nRanks);

  /// Destroy the Bootstrap.
  ~Bootstrap();

  /// Create a random unique ID and store it in the Bootstrap.
  /// @return The created unique ID.
  UniqueId createUniqueId();

  /// Return the unique ID stored in the Bootstrap.
  /// @return The unique ID stored in the Bootstrap.
  UniqueId getUniqueId() const;

  /// Initialize the Bootstrap with a given unique ID.
  /// @param uniqueId The unique ID to initialize the Bootstrap with.
  void initialize(UniqueId uniqueId);

  /// Initialize the Bootstrap with a string formatted as "ip:port" or "interface:ip:port".
  /// @param ifIpPortTrio The string formatted as "ip:port" or "interface:ip:port".
  void initialize(std::string ifIpPortTrio);

  /// Return the rank of the process.
  int getRank() override;

  /// Return the total number of ranks.
  int getNranks() override;

  /// Send data to another process.
  ///
  /// Data sent via `send(senderBuff, size, receiverRank, tag)` can be received via `recv(receiverBuff, size,
  /// senderRank, tag)`.
  ///
  /// @param data The data to send.
  /// @param size The size of the data to send.
  /// @param peer The rank of the process to send the data to.
  /// @param tag The tag to send the data with.
  void send(void* data, int size, int peer, int tag) override;

  /// Receive data from another process.
  ///
  /// Data sent via `send(senderBuff, size, receiverRank, tag)` can be received via `recv(receiverBuff, size,
  /// senderRank, tag)`.
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

  /// Synchronize all processes.
  void barrier() override;

 private:
  /// Implementation class for Bootstrap.
  class Impl;
  /// Pointer to the implementation class for Bootstrap.
  std::unique_ptr<Impl> pimpl_;
};

/// Enumerates the available transport types.
enum class Transport {
  Unknown,       // Unknown transport type.
  CudaIpc,       // CUDA IPC transport type.
  IB0,           // InfiniBand device 0 transport type.
  IB1,           // InfiniBand device 1 transport type.
  IB2,           // InfiniBand device 2 transport type.
  IB3,           // InfiniBand device 3 transport type.
  IB4,           // InfiniBand device 4 transport type.
  IB5,           // InfiniBand device 5 transport type.
  IB6,           // InfiniBand device 6 transport type.
  IB7,           // InfiniBand device 7 transport type.
  NumTransports  // The number of transports.
};

namespace detail {
const size_t TransportFlagsSize = 10;
static_assert(TransportFlagsSize == static_cast<size_t>(Transport::NumTransports),
              "TransportFlagsSize must match the number of transports");
/// Bitset for storing transport flags.
using TransportFlagsBase = std::bitset<TransportFlagsSize>;
}  // namespace detail

/// Stores transport flags.
class TransportFlags : private detail::TransportFlagsBase {
 public:
  /// Default constructor for TransportFlags.
  TransportFlags() = default;

  /// Constructor for TransportFlags that takes a Transport enum value.
  ///
  /// @param transport The transport to set the flag for.
  TransportFlags(Transport transport);

  /// Check if a specific transport flag is set.
  ///
  /// @param transport The transport to check the flag for.
  /// @return True if the flag is set, false otherwise.
  bool has(Transport transport) const;

  /// Check if no transport flags are set.
  ///
  /// @return True if no flags are set, false otherwise.
  bool none() const;

  /// Check if any transport flags are set.
  ///
  /// @return True if any flags are set, false otherwise.
  bool any() const;

  /// Check if all transport flags are set.
  ///
  /// @return True if all flags are set, false otherwise.
  bool all() const;

  /// Get the number of transport flags that are set.
  ///
  /// @return The number of flags that are set.
  size_t count() const;

  /// Bitwise OR assignment operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the OR operation with.
  /// @return A reference to the modified TransportFlags.
  TransportFlags& operator|=(TransportFlags other);

  /// Bitwise OR operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the OR operation with.
  /// @return A new TransportFlags object with the result of the OR operation.
  TransportFlags operator|(TransportFlags other) const;

  /// Bitwise OR operator for TransportFlags and Transport.
  ///
  /// @param transport The Transport to perform the OR operation with.
  /// @return A new TransportFlags object with the result of the OR operation.
  TransportFlags operator|(Transport transport) const;

  /// Bitwise AND assignment operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the AND operation with.
  /// @return A reference to the modified TransportFlags.
  TransportFlags& operator&=(TransportFlags other);

  /// Bitwise AND operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the AND operation with.
  /// @return A new TransportFlags object with the result of the AND operation.
  TransportFlags operator&(TransportFlags other) const;

  /// Bitwise AND operator for TransportFlags and Transport.
  ///
  /// @param transport The Transport to perform the AND operation with.
  /// @return A new TransportFlags object with the result of the AND operation.
  TransportFlags operator&(Transport transport) const;

  /// Bitwise XOR assignment operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the XOR operation with.
  /// @return A reference to the modified TransportFlags.
  TransportFlags& operator^=(TransportFlags other);

  /// Bitwise XOR operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the XOR operation with.
  /// @return A new TransportFlags object with the result of the XOR operation.
  TransportFlags operator^(TransportFlags other) const;

  /// Bitwise XOR operator for TransportFlags and Transport.
  ///
  /// @param transport The Transport to perform the XOR operation with.
  /// @return A new TransportFlags object with the result of the XOR operation.
  TransportFlags operator^(Transport transport) const;

  /// Bitwise NOT operator for TransportFlags.
  ///
  /// @return A new TransportFlags object with the result of the NOT operation.
  TransportFlags operator~() const;

  /// Equality comparison operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to compare with.
  /// @return True if the two TransportFlags objects are equal, false otherwise.
  bool operator==(TransportFlags other) const;

  /// Inequality comparison operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to compare with.
  /// @return True if the two TransportFlags objects are not equal, false otherwise.
  bool operator!=(TransportFlags other) const;

  /// Convert the TransportFlags object to a bitset representation.
  ///
  /// @return A detail::TransportFlagsBase object representing the TransportFlags object.
  detail::TransportFlagsBase toBitset() const;

 private:
  /// Private constructor for TransportFlags that takes a bitset representation.
  ///
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

/// Get the number of available InfiniBand devices.
///
/// @return The number of available InfiniBand devices.
int getIBDeviceCount();

/// Get the name of the InfiniBand device associated with the specified transport.
///
/// @param ibTransport The InfiniBand transport to get the device name for.
/// @return The name of the InfiniBand device associated with the specified transport.
std::string getIBDeviceName(Transport ibTransport);

/// Get the InfiniBand transport associated with the specified device name.
///
/// @param ibDeviceName The name of the InfiniBand device to get the transport for.
/// @return The InfiniBand transport associated with the specified device name.
Transport getIBTransportByDeviceName(const std::string& ibDeviceName);

class Communicator;
class Connection;

/// Represents a block of memory that has been registered to a @ref Communicator.
class RegisteredMemory {
 protected:
  struct Impl;

 public:
  /// Default constructor.
  RegisteredMemory() = default;

  /// Constructor that takes a shared pointer to an implementation object.
  ///
  /// @param pimpl A shared pointer to an implementation object.
  RegisteredMemory(std::shared_ptr<Impl> pimpl);

  /// Destructor.
  ~RegisteredMemory();

  /// Get a pointer to the memory block.
  ///
  /// @return A pointer to the memory block.
  void* data();

  /// Get the size of the memory block.
  ///
  /// @return The size of the memory block.
  size_t size();

  /// Get the rank of the process that owns the memory block.
  ///
  /// @return The rank of the process that owns the memory block.
  int rank();

  /// Get the transport flags associated with the memory block.
  ///
  /// @return The transport flags associated with the memory block.
  TransportFlags transports();

  /// Serialize the RegisteredMemory object to a vector of characters.
  ///
  /// @return A vector of characters representing the serialized RegisteredMemory object.
  std::vector<char> serialize();

  /// Deserialize a RegisteredMemory object from a vector of characters.
  ///
  /// @param data A vector of characters representing a serialized RegisteredMemory object.
  /// @return A deserialized RegisteredMemory object.
  static RegisteredMemory deserialize(const std::vector<char>& data);

  friend class Connection;
  friend class IBConnection;
  friend class Communicator;

 private:
  // A shared_ptr is used since RegisteredMemory is functionally immutable, although internally some state is populated
  // lazily.
  std::shared_ptr<Impl> pimpl;
};

/// Represents a connection between two processes.
class Connection {
 public:
  /// Write data from a source @ref RegisteredMemory to a destination @ref RegisteredMemory.
  ///
  /// @param dst The destination @ref RegisteredMemory.
  /// @param dstOffset The offset in bytes from the start of the destination @ref RegisteredMemory.
  /// @param src The source @ref RegisteredMemory.
  /// @param srcOffset The offset in bytes from the start of the source @ref RegisteredMemory.
  /// @param size The number of bytes to write.
  virtual void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) = 0;

  /// Update a 8-byte value in a destination @ref RegisteredMemory and synchronize the change with the remote process.
  ///
  /// @param dst The destination @ref RegisteredMemory.
  /// @param dstOffset The offset in bytes from the start of the destination @ref RegisteredMemory.
  /// @param src A pointer to the value to update.
  /// @param newValue The new value to write.
  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) = 0;

  /// Flush any pending writes to the remote process.
  virtual void flush() = 0;

  /// Get the rank of the remote process.
  ///
  /// @return The rank of the remote process.
  virtual int remoteRank() = 0;

  /// Get the tag associated with the connection.
  ///
  /// @return The tag associated with the connection.
  virtual int tag() = 0;

  /// Get the transport used by the local process.
  ///
  /// @return The transport used by the local process.
  virtual Transport transport() = 0;

  /// Get the transport used by the remote process.
  ///
  /// @return The transport used by the remote process.
  virtual Transport remoteTransport() = 0;

 protected:
  /// Get the implementation object associated with a @ref RegisteredMemory object.
  ///
  /// @param memory The @ref RegisteredMemory object.
  /// @return A shared pointer to the implementation object.
  static std::shared_ptr<RegisteredMemory::Impl> getRegisteredMemoryImpl(RegisteredMemory& memory);
};

/// A base class for objects that can be set up during @ref Communicator::setup().
struct Setuppable {
  /// Called inside @ref Communicator::setup() before any call to @ref endSetup() of any @ref Setuppable object that is
  /// being set up within the same @ref Communicator::setup() call.
  ///
  /// @param bootstrap A shared pointer to the bootstrap implementation.
  virtual void beginSetup(std::shared_ptr<BaseBootstrap> bootstrap);

  /// Called inside @ref Communicator::setup() after all calls to @ref beginSetup() of all @ref Setuppable objects that
  /// are being set up within the same @ref Communicator::setup() call.
  ///
  /// @param bootstrap A shared pointer to the bootstrap implementation.
  virtual void endSetup(std::shared_ptr<BaseBootstrap> bootstrap);
};

/// A non-blocking future that can be used to check if a value is ready and retrieve it.
template <typename T>
class NonblockingFuture {
  std::shared_future<T> future;

 public:
  /// Default constructor.
  NonblockingFuture() = default;

  /// Constructor that takes a shared future and moves it into the NonblockingFuture.
  ///
  /// @param future The shared future to move.
  NonblockingFuture(std::shared_future<T>&& future) : future(std::move(future)) {}

  /// Copy constructor.
  ///
  /// @param other The @ref NonblockingFuture to copy.
  NonblockingFuture(const NonblockingFuture& other) = default;

  /// Check if the value is ready to be retrieved.
  ///
  /// @return True if the value is ready, false otherwise.
  bool ready() const { return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

  /// Get the value.
  ///
  /// @return The value.
  ///
  /// @throws Error if the value is not ready.
  T get() const {
    if (!ready()) throw Error("NonblockingFuture::get() called before ready", ErrorCode::InvalidUsage);
    return future.get();
  }
};

/// A class that sets up all registered memories and connections between processes.
///
/// A typical way to use this class:
///   1. Call @ref connectOnSetup() to declare connections between the calling process with other processes.
///   2. Call @ref registerMemory() to register memory regions that will be used for communication.
///   3. Call @ref sendMemoryOnSetup() or @ref recvMemoryOnSetup() to send/receive registered memory regions to/from
///      other processes.
///   4. Call @ref setup() to set up all registered memories and connections declared in the previous steps.
///   5. Call @ref NonblockingFuture<RegisteredMemory>::get() to get the registered memory regions received from other
///      processes.
///   6. All done; use connections and registered memories to build channels.
///
class Communicator {
 protected:
  struct Impl;

 public:
  /// Initializes the communicator with a given bootstrap implementation.
  ///
  /// @param bootstrap An implementation of the BaseBootstrap that the communicator will use.
  Communicator(std::shared_ptr<BaseBootstrap> bootstrap);

  /// Destroy the communicator.
  ~Communicator();

  /// Returns the bootstrapper held by this communicator.
  ///
  /// @return std::shared_ptr<BaseBootstrap> The bootstrapper held by this communicator.
  std::shared_ptr<BaseBootstrap> bootstrapper();

  /// Register a region of GPU memory for use in this communicator.
  ///
  /// @param ptr Base pointer to the memory.
  /// @param size Size of the memory region in bytes.
  /// @param transports Transport flags.
  /// @return RegisteredMemory A handle to the buffer.
  RegisteredMemory registerMemory(void* ptr, size_t size, TransportFlags transports);

  /// Send information of a registered memory to the remote side on setup.
  ///
  /// This function registers a send to a remote process that will happen by a following call of @ref setup(). The send
  /// will carry information about a registered memory on the local process.
  ///
  /// @param memory The registered memory buffer to send information about.
  /// @param remoteRank The rank of the remote process.
  /// @param tag The tag to use for identifying the send.
  void sendMemoryOnSetup(RegisteredMemory memory, int remoteRank, int tag);

  /// Receive memory on setup.
  ///
  /// This function registers a receive from a remote process that will happen by a following call of @ref setup(). The
  /// receive will carry information about a registered memory on the remote process.
  ///
  /// @param remoteRank The rank of the remote process.
  /// @param tag The tag to use for identifying the receive.
  /// @return NonblockingFuture<RegisteredMemory> A non-blocking future of registered memory.
  NonblockingFuture<RegisteredMemory> recvMemoryOnSetup(int remoteRank, int tag);

  /// Connect to a remote rank on setup.
  ///
  /// This function only prepares metadata for connection. The actual connection is made by a following call of
  /// @ref setup(). Note that this function is two-way and a connection from rank `i` to remote rank `j` needs
  /// to have a counterpart from rank `j` to rank `i`. Note that with IB, buffers are registered at a page level and if
  /// a buffer is spread through multiple pages and do not fully utilize all of them, IB's QP has to register for all
  /// involved pages. This potentially has security risks if the connection's accesses are given to a malicious process.
  ///
  /// @param remoteRank The rank of the remote process.
  /// @param tag The tag of the connection for identifying it.
  /// @param transport The type of transport to be used.
  /// @return std::shared_ptr<Connection> A shared pointer to the connection.
  std::shared_ptr<Connection> connectOnSetup(int remoteRank, int tag, Transport transport);

  /// Add a custom Setuppable object to a list of objects to be setup later, when @ref setup() is called.
  ///
  /// @param setuppable A shared pointer to the Setuppable object.
  void onSetup(std::shared_ptr<Setuppable> setuppable);

  /// Setup all objects that have registered for setup.
  ///
  /// This includes previous calls of @ref sendMemoryOnSetup(), @ref recvMemoryOnSetup(), @ref connectOnSetup(), and
  /// @ref onSetup(). It is allowed to call this function multiple times, where the n-th call will only setup objects
  /// that have been registered after the (n-1)-th call.
  void setup();

  friend class RegisteredMemory::Impl;
  friend class IBConnection;

 private:
  /// Unique pointer to the implementation of the Communicator class.
  std::unique_ptr<Impl> pimpl;
};

/// A constant TransportFlags object representing no transports.
extern const TransportFlags NoTransports;

/// A constant TransportFlags object representing all InfiniBand transports.
extern const TransportFlags AllIBTransports;

/// A constant TransportFlags object representing all transports.
extern const TransportFlags AllTransports;

}  // namespace mscclpp

namespace std {

/// Specialization of the std::hash template for mscclpp::TransportFlags.
template <>
struct hash<mscclpp::TransportFlags>;

}  // namespace std

#endif  // MSCCLPP_CORE_HPP_
