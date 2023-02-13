#ifndef MSCCLPP_SHMUTILS_H_
#define MSCCLPP_SHMUTILS_H_

#include "mscclpp.h"

mscclppResult_t mscclppShmutilsMapCreate(const char *name, size_t size, int *fd, void **map);
mscclppResult_t mscclppShmutilsMapOpen(const char *name, size_t size, int *fd, void **map);
mscclppResult_t mscclppShmutilsMapClose(const char *name, size_t size, int fd, void *map);
mscclppResult_t mscclppShmutilsMapDestroy(const char *name, size_t size, int fd, void *map);

#endif
