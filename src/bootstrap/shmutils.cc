#include "shmutils.h"
#include "debug.h"
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>

#define SHM_MODE 0666

// Open a shme file and create an mmap.
static mscclppResult_t shmutilsMapOpen(const char *name, size_t size, int *fd, void **map, int flag)
{
    int _fd = shm_open(name, flag, SHM_MODE);
    if (_fd == -1) {
        WARN("Failed to open shm file %s (flag: %d)", name, flag);
        return mscclppInternalError;
    }
    void *_map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
    if (_map == MAP_FAILED) {
        WARN("Failed to mmap shm file %s", name);
        goto fail;
    }
    if (flag & O_CREAT) {
        if (ftruncate(_fd, 0) == -1) {
            WARN("Failed to ftruncate shm file %s", name);
            goto fail;
        }
    }
    if (ftruncate(_fd, size) == -1) {
        WARN("Failed to ftruncate shm file %s", name);
        goto fail;
    }
    *fd = _fd;
    *map = _map;
    return mscclppSuccess;
fail:
    close(_fd);
    shm_unlink(name);
    return mscclppInternalError;
}

// Open or create a shm file.
mscclppResult_t mscclppShmutilsMapCreate(const char *name, size_t size, int *fd, void **map)
{
    return shmutilsMapOpen(name, size, fd, map, O_CREAT | O_RDWR);
}

// Open an existing shm file.
mscclppResult_t mscclppShmutilsMapOpen(const char *name, size_t size, int *fd, void **map)
{
    return shmutilsMapOpen(name, size, fd, map, O_RDWR);
}

// Close a shm mmap.
mscclppResult_t mscclppShmutilsMapClose(const char *name, size_t size, int fd, void *map)
{
    int err = 0;
    if (munmap(map, size) == -1) {
        WARN("Failed to munmap shm file %s", name);
        err = 1;
    }
    close(fd);
    return err ? mscclppInternalError : mscclppSuccess;
}

// Close a shm mmap and destroy a shm file.
mscclppResult_t mscclppShmutilsMapDestroy(const char *name, size_t size, int fd, void *map)
{
    int err = 0;
    if (munmap(map, size) == -1) {
        WARN("Failed to munmap shm file %s", name);
        err = 1;
    }
    close(fd);
    if (shm_unlink(name) == -1) {
        WARN("Failed to unlink shm file %s: errno %d", name, errno);
        err = 1;
    }
    return err ? mscclppInternalError : mscclppSuccess;
}
