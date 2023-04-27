#ifndef MSCCLPP_HPP_
#define MSCCLPP_HPP_

#define MSCCLPP_MAJOR 0
#define MSCCLPP_MINOR 1
#define MSCCLPP_PATCH 0
#define MSCCLPP_VERSION (MSCCLPP_MAJOR * 10000 + MSCCLPP_MINOR * 100 + MSCCLPP_PATCH)

#include <vector>
#include <memory>
#include <string>


namespace mscclpp {

#define MSCCLPP_UNIQUE_ID_BYTES 128
struct UniqueId {
  char internal[MSCCLPP_UNIQUE_ID_BYTES];
};

class BaseBootstrap
{
public:
  BaseBootstrap(){};
  virtual ~BaseBootstrap() = default;
  virtual int getRank() = 0;
  virtual int getNranks() = 0;
  virtual void send(void* data, int size, int peer, int tag) = 0;
  virtual void recv(void* data, int size, int peer, int tag) = 0;
  virtual void allGather(void* allData, int size) = 0;
  virtual void barrier() = 0;
};

class Bootstrap : public BaseBootstrap
{
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

using TransportFlags = uint32_t;
const TransportFlags TransportNone = 0b0;
const TransportFlags TransportCudaIpc = 0b1;
const TransportFlags TransportIB0 = 0b10;
const TransportFlags TransportIB1 = 0b100;
const TransportFlags TransportIB2 = 0b1000;
const TransportFlags TransportIB3 = 0b10000;
const TransportFlags TransportIB4 = 0b100000;
const TransportFlags TransportIB5 = 0b1000000;
const TransportFlags TransportIB6 = 0b10000000;
const TransportFlags TransportIB7 = 0b100000000;

const TransportFlags TransportAll = 0b111111111;
const TransportFlags TransportAllIB = 0b111111110;

int getIBDeviceCount();
std::string getIBDeviceName(TransportFlags ibTransport);
TransportFlags getIBTransportByDeviceName(const std::string& ibDeviceName);

class Communicator;
class Connection;

class RegisteredMemory {
  struct Impl;
  std::shared_ptr<Impl> pimpl;
public:

  RegisteredMemory(std::shared_ptr<Impl> pimpl);
  ~RegisteredMemory();

  void* data();
  size_t size();
  int rank();
  TransportFlags transports();

  std::vector<char> serialize();
  static RegisteredMemory deserialize(const std::vector<char>& data);

  friend class Connection;
  friend class Communicator;
};

class Connection {
public:
  virtual void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size) = 0;

  virtual void flush() = 0;

  virtual TransportFlags transport() = 0;

  virtual TransportFlags remoteTransport() = 0;

protected:
  static std::shared_ptr<RegisteredMemory::Impl> getRegisteredMemoryImpl(RegisteredMemory&);
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
  
  /* Ring-based AllGather through the bootstrap socket.
  *
  * Inputs:
  *   data: data array to be gathered where `[r*size, (r+1)*size)` is the data for rank `r`
  *   size: data size per rank
  */
  void bootstrapAllGather(void* data, int size);

  /* A no-op function that is used to synchronize all processes via a bootstrap allgather*/
  void bootstrapBarrier();

  /* Register a region of GPU memory for use in this communicator.
   *
   * Inputs:
   *  data: base pointer to the memory
   *  size: size of the memory region in bytes
   * 
   * Returns: a handle to the buffer
   */
  RegisteredMemory registerMemory(void* ptr, size_t size, TransportFlags transports);

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
  std::shared_ptr<Connection> connect(int remoteRank, int tag, TransportFlags transport);

  /* Establish all connections declared by connect(). This function must be called after all connect()
  * calls are made. This function ensures that all remote ranks are ready to communicate when it returns.
  */
  void connectionSetup();

  struct Impl;
private:
  std::unique_ptr<Impl> pimpl;
};

} // namespace mscclpp

#endif // MSCCLPP_H_
