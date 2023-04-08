/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include "core.h"

#include <stdlib.h>
#include <string>

// Get current Compute Capability
// int mscclppCudaCompCap() {
//   int cudaDev;
//   if (cudaGetDevice(&cudaDev) != cudaSuccess) return 0;
//   int ccMajor, ccMinor;
//   if (cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev) != cudaSuccess) return 0;
//   if (cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, cudaDev) != cudaSuccess) return 0;
//   return ccMajor*10+ccMinor;
// }

mscclppResult_t int64ToBusId(int64_t id, char* busId)
{
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12, (id & 0xff0) >> 4, (id & 0xf));
  return mscclppSuccess;
}

mscclppResult_t busIdToInt64(const char* busId, int64_t* id)
{
  char hexStr[17]; // Longest possible int64 hex string + null terminator.
  int hexOffset = 0;
  for (int i = 0; hexOffset < sizeof(hexStr) - 1; i++) {
    char c = busId[i];
    if (c == '.' || c == ':')
      continue;
    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else
      break;
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  return mscclppSuccess;
}

// Convert a logical cudaDev index to the NVML device minor number
mscclppResult_t getBusId(int cudaDev, std::string* busId)
{
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdChar[] = "00000000:00:00.0";
  CUDACHECK(cudaDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), cudaDev));
  // we need the hex in lower case format
  for (int i = 0; i < sizeof(busIdChar); i++) {
    busIdChar[i] = std::tolower(busIdChar[i]);
  }
  *busId = busIdChar;
  return mscclppSuccess;
}

mscclppResult_t getDeviceNumaNode(int cudaDev, int* numaNode)
{
  std::string busId;
  MSCCLPPCHECK(getBusId(cudaDev, &busId));

  std::string pci_str = "/sys/bus/pci/devices/" + busId + "/numa_node";
  FILE* file = fopen(pci_str.c_str(), "r");
  if (file == NULL) {
    WARN("Could not open %s to detect the NUMA node for device %d", pci_str.c_str(), cudaDev);
    return mscclppSystemError;
  }
  int ret = fscanf(file, "%d", numaNode);
  if (ret != 1) {
    WARN("Could not read NUMA node for device %d", cudaDev);
    return mscclppSystemError;
  }
  fclose(file);
  return mscclppSuccess;
}

mscclppResult_t getHostName(char* hostname, int maxlen, const char delim)
{
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return mscclppSystemError;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1))
    i++;
  hostname[i] = '\0';
  return mscclppSuccess;
}

uint64_t getHash(const char* string, int n)
{
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the MSCCLPP_HOSTID env var.
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
uint64_t getHostHash(void)
{
  char hostHash[1024];
  char* hostId;

  // Fall back is the full hostname if something fails
  (void)getHostName(hostHash, sizeof(hostHash), '\0');
  int offset = strlen(hostHash);

  if ((hostId = getenv("MSCCLPP_HOSTID")) != NULL) {
    INFO(MSCCLPP_ENV, "MSCCLPP_HOSTID set by environment to %s", hostId);
    strncpy(hostHash, hostId, sizeof(hostHash));
  } else {
    FILE* file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
      char* p;
      if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash + offset, p, sizeof(hostHash) - offset - 1);
        free(p);
      }
    }
    fclose(file);
  }

  // Make sure the string is terminated
  hostHash[sizeof(hostHash) - 1] = '\0';

  TRACE(MSCCLPP_INIT, "unique hostname '%s'", hostHash);

  return getHash(hostHash, strlen(hostHash));
}

/* Generate a hash of the unique identifying string for this process
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $$ $(readlink /proc/self/ns/pid)
 */
uint64_t getPidHash(void)
{
  char pname[1024];
  // Start off with our pid ($$)
  sprintf(pname, "%ld", (long)getpid());
  int plen = strlen(pname);
  int len = readlink("/proc/self/ns/pid", pname + plen, sizeof(pname) - 1 - plen);
  if (len < 0)
    len = 0;

  pname[plen + len] = '\0';
  TRACE(MSCCLPP_INIT, "unique PID '%s'", pname);

  return getHash(pname, strlen(pname));
}

int parseStringList(const char* string, struct netIf* ifList, int maxList)
{
  if (!string)
    return 0;

  const char* ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr + 1);
        ifNum++;
        ifC = 0;
      }
      while (c != ',' && c != '\0')
        c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++;
        ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

static bool matchIf(const char* string, const char* ref, bool matchExact)
{
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool matchPort(const int port1, const int port2)
{
  if (port1 == -1)
    return true;
  if (port2 == -1)
    return true;
  if (port1 == port2)
    return true;
  return false;
}

bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact)
{
  // Make an exception for the case where no user list is defined
  if (listSize == 0)
    return true;

  for (int i = 0; i < listSize; i++) {
    if (matchIf(string, ifList[i].prefix, matchExact) && matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

mscclppResult_t numaBind(int node)
{
  int totalNumNumaNodes = numa_num_configured_nodes();
  if (node < 0 || node >= totalNumNumaNodes) {
    WARN("Invalid NUMA node %d, must be between 0 and %d", node, totalNumNumaNodes);
    return mscclppInvalidUsage;
  }
  nodemask_t mask;
  nodemask_zero(&mask);
  nodemask_set_compat(&mask, node);
  numa_bind_compat(&mask);
  return mscclppSuccess;
}

mscclppResult_t getNumaState(mscclppNumaState* state)
{

  mscclppNumaState state_ = numa_get_run_node_mask();
  if (state_ == NULL) {
    WARN("Failed to get NUMA node mask of the running process");
    return mscclppSystemError;
  }
  *state = state_;
  return mscclppSuccess;
}

mscclppResult_t setNumaState(mscclppNumaState state)
{
  if (state == NULL) {
    WARN("Invalid NUMA state");
    return mscclppInvalidUsage;
  }
  numa_bind(state);
  return mscclppSuccess;
}

inline mscclppTime_t getClock()
{
  return std::chrono::steady_clock::now();
}

inline int64_t elapsedClock(mscclppTime_t start, mscclppTime_t end)
{
  return std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
}
