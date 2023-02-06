#include "debug.h"
#include "mscclpp.h"

int main()
{
  mscclppUniqueId uid;
  mscclppResult_t res = mscclppGetUniqueId(&uid);
  if (res != mscclppSuccess) {
    printf("mscclppGetUniqueId failed\n");
    return -1;
  }
  printf("Succeeded!\n");
  return 0;
}
