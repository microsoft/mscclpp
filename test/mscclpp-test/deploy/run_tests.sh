set -e

echo "=================Run allgather_test_perf on 2 nodes========================="
/usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
  -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 /root/mscclpp/build/test/mscclpp-test/allgather_test_perf -b 1K -e 1G -f 2 -k 0

# For kernel 2, the message size must can be devided by 3
/usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
  -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 /root/mscclpp/build/test/mscclpp-test/allgather_test_perf -b 3K -e 3G -f 2 -k 2

echo "==================Run alltoall_test_perf on 2 nodes========================="
/usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
  -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 /root/mscclpp/build/test/mscclpp-test/alltoall_test_perf -b 1K -e 1G -f 2 -k 0
