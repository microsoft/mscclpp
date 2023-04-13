#ifndef MSCCLPP_PYTHON_H_
#define MSCCLPP_PYTHON_H_
#include <mscclpp.h>

struct _Comm {
  int _rank;
  int _world_size;
  mscclppComm_t _handle;
  bool _is_open;
  bool _proxies_running;
  // Close should be safe to call on a closed handle.
  void close();

  void check_open();
};

#endif  // MSCCLPP_PYTHON_H_