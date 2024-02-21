# NCCL Interfaces of MSCCL++

Compile

```bash
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_APPS_NCCL=ON -DBUILD_PYTHON_BINDINGS=OFF ..
make -j
```

Run rccl-tests

```bash
mpirun -np 8 --bind-to numa --allow-run-as-root -x LD_PRELOAD="$MSCCLPP_BUILD/libmscclpp.so $MSCCLPP_BUILD/apps/nccl/libmscclpp_nccl.so" -x MSCCLPP_DEBUG=WARN -x MSCCLPP_DEBUG_SUBSYS=ALL -x NCCL_DEBUG=WARN ./build/all_reduce_perf -b 1K -e 256M -f 2 -d half -G 20 -w 10 -n 50
```
