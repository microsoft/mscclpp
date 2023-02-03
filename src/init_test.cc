#include <cassert>
#include "debug.h"
#include "mscclpp.h"

int main()
{
  mscclppUniqueId uid;
  mscclppResult_t res = mscclppGetUniqueId(&uid);
  if (res != mscclppSuccess) {
    ABORT("mscclppGetUniqueId failed");
  }
  INFO("init_test succeed");
  return 0;
}
