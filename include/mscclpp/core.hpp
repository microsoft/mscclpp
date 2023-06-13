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
struct UniqueId {
  char internal[MSCCLPP_UNIQUE_ID_BYTES];
};

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

class Bootstrap : public BaseBootstrap {
 public:
  Bootstrap(int rank, int nRanks);
  ~Bootstrap();

  UniqueId createUniqueId();
  UniqueId getUniqueId() const;

  void initialize(UniqueId uniqueId);
  void initialize(std::string ipPortPair);
  int getRank() override;
  int getNranks() override;
  void send(void* data, int size, int peer, int tag) override;
  void recv(void* data, int size, int peer, int tag) override;
  void allGather(void* allData, int size) override;
  void barrier() override;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

/* Create a unique ID for communication. Only needs to be called by one process.
 * Use with mscclppCommInitRankFromId().
 * All processes need to provide the same ID to mscclppCommInitRankFromId().
 *
 * Outputs:
 *  uniqueId: the unique ID to be created
 */
std::unique_ptr<UniqueId> getUniqueId();

enum class Transport { Unknown, CudaIpc, IB0, IB1, IB2, IB3, IB4, IB5, IB6, IB7, NumTransports };

namespace detail {
const size_t TransportFlagsSize = 10;
static_assert(TransportFlagsSize == static_cast<size_t>(Transport::NumTransports),
              "TransportFlagsSize must match the number of transports");
using TransportFlagsBase = std::bitset<TransportFlagsSize>;
}  // namespace detail

class TransportFlags : private detail::TransportFlagsBase {
 public:
  TransportFlags() = default;
  TransportFlags(Transport transport) : detail::TransportFlagsBase(1 << static_cast<size_t>(transport)) {}

  bool has(Transport transport) const { return detail::TransportFlagsBase::test(static_cast<size_t>(transport)); }

  bool none() const { return detail::TransportFlagsBase::none(); }

  bool any() const { return detail::TransportFlagsBase::any(); }

  bool all() const { return detail::TransportFlagsBase::all(); }

  size_t count() const { return detail::TransportFlagsBase::count(); }

  TransportFlags& operator|=(TransportFlags other) {
    detail::TransportFlagsBase::operator|=(other);
    return *this;
  }

  TransportFlags operator|(TransportFlags other) const { return TransportFlags(*this) |= other; }

  TransportFlags operator|(Transport transport) const { return *this | TransportFlags(transport); }

  TransportFlags& operator&=(TransportFlags other) {
    detail::TransportFlagsBase::operator&=(other);
    return *this;
  }

  TransportFlags operator&(TransportFlags other) const { return TransportFlags(*this) &= other; }

  TransportFlags operator&(Transport transport) const { return *this & TransportFlags(transport); }

  TransportFlags& operator^=(TransportFlags other) {
    detail::TransportFlagsBase::operator^=(other);
    return *this;
  }

  TransportFlags operator^(TransportFlags other) const { return TransportFlags(*this) ^= other; }

  TransportFlags operator^(Transport transport) const { return *this ^ TransportFlags(transport); }

  TransportFlags operator~() const { return TransportFlags(*this).flip(); }

  bool operator==(TransportFlags other) const { return detail::TransportFlagsBase::operator==(other); }

  bool operator!=(TransportFlags other) const { return detail::TransportFlagsBase::operator!=(other); }

  detail::TransportFlagsBase toBitset() const { return *this; }

 private:
  TransportFlags(detail::TransportFlagsBase bitset) : detail::TransportFlagsBase(bitset) {}
};

inline TransportFlags operator|(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) | transport2;
}

inline TransportFlags operator&(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) & transport2;
}

inline TransportFlags operator^(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) ^ transport2;
}

const TransportFlags NoTransports = TransportFlags();
const TransportFlags AllIBTransports = Transport::IB0 | Transport::IB1 | Transport::IB2 | Transport::IB3 |
                                       Transport::IB4 | Transport::IB5 | Transport::IB6 | Transport::IB7;
const TransportFlags AllTransports = AllIBTransports | Transport::CudaIpc;

int getIBDeviceCount();
std::string getIBDeviceName(Transport ibTransport);
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

  T get() {
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
