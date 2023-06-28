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

/// Unique ID for a process. This is a @ref MSCCLPP_UNIQUE_ID_BYTES byte array that uniquely identifies a process.
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

  /// Initialize the Bootstrap with a string formatted as "ip:port".
  /// @param ipPortPair The string formatted as "ip:port".
  void initialize(std::string ipPortPair);

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

/// A constant TransportFlags object representing no transports.
extern const TransportFlags NoTransports;

/// A constant TransportFlags object representing all InfiniBand transports.
extern const TransportFlags AllIBTransports;

/// A constant TransportFlags object representing all transports.
extern const TransportFlags AllTransports;

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

class RegisteredMemory {
 protected:
  struct Impl;

 public:
  RegisteredMemory() = default;
  RegisteredMemory(std::shared_ptr<Impl> pimpl);
  ~RegisteredMemory();

  void* data();
  size_t size();
  int rank();
  TransportFlags transports();

  std::vector<char> serialize();
  static RegisteredMemory deserialize(const std::vector<char>& data);

  friend class Connection;
  friend class IBConnection;
  friend class Communicator;

 private:
  // A shared_ptr is used since RegisteredMemory is functionally immutable, although internally some state is populated
  // lazily.
  std::shared_ptr<Impl> pimpl;
};

class Connection {
 public:
  virtual void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) = 0;

  // src must be a CPU memory
  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) = 0;

  virtual void flush() = 0;

  virtual int remoteRank() = 0;

  virtual int tag() = 0;

  virtual Transport transport() = 0;

  virtual Transport remoteTransport() = 0;

 protected:
  static std::shared_ptr<RegisteredMemory::Impl> getRegisteredMemoryImpl(RegisteredMemory&);
};

struct Setuppable {
  virtual void beginSetup(std::shared_ptr<BaseBootstrap>) {}
  virtual void endSetup(std::shared_ptr<BaseBootstrap>) {}
};

template <typename T>
class NonblockingFuture {
  std::shared_future<T> future;

 public:
  NonblockingFuture() = default;
  NonblockingFuture(std::shared_future<T>&& future) : future(std::move(future)) {}
  NonblockingFuture(const NonblockingFuture&) = default;

  bool ready() const { return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

  T get() const {
    if (!ready()) throw Error("NonblockingFuture::get() called before ready", ErrorCode::InvalidUsage);
    return future.get();
  }
};

class Communicator {
 public:
  /* Initialize the communicator.
   *
   * Inputs:
   *   bootstrap: an implementation of the of BaseBootstrap that the communicator will use
   */
  Communicator(std::shared_ptr<BaseBootstrap> bootstrap);

  ~Communicator();

  /* Return the bootstrapper held by this communicator. */
  std::shared_ptr<BaseBootstrap> bootstrapper();

  /* Register a region of GPU memory for use in this communicator.
   *
   * Inputs:
   *  data: base pointer to the memory
   *  size: size of the memory region in bytes
   *
   * Returns: a handle to the buffer
   */
  RegisteredMemory registerMemory(void* ptr, size_t size, TransportFlags transports);

  void sendMemoryOnSetup(RegisteredMemory memory, int remoteRank, int tag);

  NonblockingFuture<RegisteredMemory> recvMemoryOnSetup(int remoteRank, int tag);

  /* Connect to a remote rank. This function only prepares metadata for connection. The actual connection
   * is made by a following call of mscclppConnectionSetup(). Note that this function is two-way and a connection
   * from rank i to remote rank j needs to have a counterpart from rank j to rank i.
   * Note that with IB, buffers are registered at a page level and if a buffer is spread through multiple pages
   * and do not fully utilize all of them, IB's QP has to register for all involved pages. This potentially has
   * security risks if the devConn's accesses are given to a malicious process.
   *
   * Inputs:
   *   remoteRank:    the rank of the remote process
   *   tag:           the tag of the connection. tag is copied into the corresponding mscclppDevConn_t, which can be
   *                  used to identify the connection inside a GPU kernel.
   *   transportType: the type of transport to be used (mscclppTransportP2P or mscclppTransportIB)
   *   ibDev:         the name of the IB device to be used. Expects a null for mscclppTransportP2P.
   */
  std::shared_ptr<Connection> connectOnSetup(int remoteRank, int tag, Transport transport);

  /* Add a custom Setuppable object to a list of objects to be setup later, when setup() is called. */
  void onSetup(std::shared_ptr<Setuppable> setuppable);

  /* Setup all objects that have registered for setup. This includes any connections created by connect(). */
  void setup();

  struct Impl;

 private:
  std::unique_ptr<Impl> pimpl;
};
}  // namespace mscclpp

namespace std {
template <>
struct hash<mscclpp::TransportFlags> {
  size_t operator()(const mscclpp::TransportFlags& flags) const {
    return hash<mscclpp::detail::TransportFlagsBase>()(flags.toBitset());
  }
};
}  // namespace std

#endif  // MSCCLPP_CORE_HPP_
