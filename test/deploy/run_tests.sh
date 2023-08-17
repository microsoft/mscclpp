set -e

function run_mscclpp_test()
{
  echo "=================Run allgather_test_perf on 2 nodes========================="
  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allgather_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  # For kernel 2, the message size must can be divided by 3
  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allgather_test_perf -b 3K -e 3G -f 2 -k 2 -o /root/mscclpp/output.jsonl

  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allgather_test_perf -b 1K -e 1G -f 2 -k 3 -o /root/mscclpp/output.jsonl

  echo "==================Run allreduce_test_perf on 2 nodes========================="
  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 1K -e 1G -f 2 -k 1 -o /root/mscclpp/output.jsonl

  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 1K -e 1M -f 2 -k 2 -o /root/mscclpp/output.jsonl

  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 3K -e 3G -f 2 -k 3 -o /root/mscclpp/output.jsonl

  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 3K -e 3G -f 2 -k 4 -o /root/mscclpp/output.jsonl

  echo "==================Run alltoall_test_perf on 2 nodes========================="
  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile /root/mscclpp/hostfile_mpi \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/alltoall_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  echo "========================Run performance check==============================="
  python3 /root/mscclpp/check_perf_result.py --perf-file /root/mscclpp/output.jsonl \
    --baseline-file /root/mscclpp/perf_ndmv4.jsonl
}

function run_mp_ut()
{
  echo "============Run multi-process unit tests on 2 nodes (np=2, npernode=1)========================="
  /usr/local/mpi/bin/mpirun -allow-run-as-root -tag-output -np 2 --bind-to numa \
  -hostfile /root/mscclpp/hostfile_mpi -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 1 /root/mscclpp/build/test/mp_unit_tests -ip_port mscclpp-it-000000:20003

  echo "============Run multi-process unit tests on 2 nodes (np=16, npernode=8)========================="
  /usr/local/mpi/bin/mpirun -allow-run-as-root -tag-output -np 16 --bind-to numa \
  -hostfile /root/mscclpp/hostfile_mpi -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 /root/mscclpp/build/test/mp_unit_tests -ip_port mscclpp-it-000000:20003
}

function run_pytests()
{
  echo "==================Run python tests================================"
  /usr/local/mpi/bin/mpirun -allow-run-as-root -tag-output -np 16 --bind-to numa \
  -hostfile /root/mscclpp/hostfile_mpi -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 pytest /root/mscclpp/python/test/test_mscclpp.py -x -v
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <mscclpp-test/mp-ut>"
    exit 1
fi
test_name=$1
case $test_name in
  mscclpp-test)
    echo "==================Run mscclpp-test on 2 nodes========================="
    run_mscclpp_test
    ;;
  mp-ut)
    echo "==================Run mp-ut on 2 nodes================================"
    run_mp_ut
    ;;
  pytests)
    echo "==================Run python tests================================"
    run_pytests
    ;;
  *)
    echo "Unknown test name: $test_name"
    exit 1
    ;;
esac
