set -e

echo "=================Run allgather_test_perf on 2 nodes========================="
mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
  --bind-to numa -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 /root/mscclpp/build/test/mscclpp-test/allgather_test_perf -b 1K -e 1G -f 2 -k 0

mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
  --bind-to numa -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 /root/mscclpp/build/test/mscclpp-test/allgather_test_perf -b 1K -e 1G -f 2 -k 2

echo "==================Run allreduce_test_perf on 2 nodes========================="
mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
  --bind-to numa -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 1K -e 1G -f 2 -k 0

mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
  --bind-to numa -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 1K -e 1G -f 2 -k 1

echo "==================Run alltoall_test_perf on 2 nodes========================="
mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
  --bind-to numa -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 /root/mscclpp/build/test/mscclpp-test/alltoall_test_perf -b 1K -e 1G -f 2 -k 0
