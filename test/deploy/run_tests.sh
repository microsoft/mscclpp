set -e
HOSTFILE=/root/mscclpp/test/deploy/hostfile_mpi
HEAD_HOST=$(head -1 ${HOSTFILE})
# Resolve HEAD_HOST to an IP address on eth0 to ensure bootstrap uses the correct interface
HEAD_IP=$(ssh -o StrictHostKeyChecking=no -p 22345 -i /root/mscclpp/sshkey ${HEAD_HOST} "ip -4 addr show eth0 | grep -oP 'inet \K[0-9.]+' | head -1" 2>/dev/null)
if [ -z "${HEAD_IP}" ]; then
    HEAD_IP=${HEAD_HOST}
fi
MPI_ARGS="--allow-run-as-root --bind-to numa -hostfile ${HOSTFILE} -mca btl_tcp_if_include eth0"
MSCCLPP_ENV="-x MSCCLPP_DEBUG=WARN -x MSCCLPP_SOCKET_IFNAME=eth0 -x LD_LIBRARY_PATH=/root/mscclpp/build/lib:\$LD_LIBRARY_PATH"

# Select perf baseline based on GPU type
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)
if echo "${GPU_NAME}" | grep -qi "H100"; then
    PERF_BASELINE=/root/mscclpp/test/deploy/perf_ndmv5.jsonl
else
    PERF_BASELINE=/root/mscclpp/test/deploy/perf_ndmv4.jsonl
fi

function run_mscclpp_test()
{
  echo "=================Run allgather_test_perf on 2 nodes========================="
  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/allgather_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  # For kernel 2, the message size must can be divided by 3
  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/allgather_test_perf -b 3K -e 3G -f 2 -k 2 -o /root/mscclpp/output.jsonl

  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/allgather_test_perf -b 1K -e 1G -f 2 -k 3 -o /root/mscclpp/output.jsonl

  echo "==================Run allreduce_test_perf on 2 nodes========================="
  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/allreduce_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/allreduce_test_perf -b 1K -e 1G -f 2 -k 1 -o /root/mscclpp/output.jsonl

  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/allreduce_test_perf -b 1K -e 1M -f 2 -k 2 -o /root/mscclpp/output.jsonl

  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/allreduce_test_perf -b 3K -e 3G -f 2 -k 3 -o /root/mscclpp/output.jsonl

  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/allreduce_test_perf -b 3K -e 3G -f 2 -k 4 -o /root/mscclpp/output.jsonl

  echo "==================Run alltoall_test_perf on 2 nodes========================="
  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/alltoall_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  echo "========================Run performance check==============================="
  python3 /root/mscclpp/test/mscclpp-test/check_perf_result.py --perf-file /root/mscclpp/output.jsonl \
    --baseline-file ${PERF_BASELINE}
}

function run_mscclpp_default_algos()
{
  echo "=================Run mscclpp default algos on 2 nodes========================="
  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/allgather_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/allreduce_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  mpirun ${MPI_ARGS} -np 16 \
    ${MSCCLPP_ENV} \
    -npernode 8 /root/mscclpp/build/bin/mscclpp-test/alltoall_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl
}

function run_mp_ut()
{
  echo "============Run multi-process unit tests on 2 nodes (np=2, npernode=1)========================="
  mpirun ${MPI_ARGS} -tag-output -np 2 \
  ${MSCCLPP_ENV} \
  -npernode 1 /root/mscclpp/build/bin/mp_unit_tests -ip_port ${HEAD_IP}:20003

  echo "============Run multi-process unit tests on 2 nodes (np=16, npernode=8)========================="
  mpirun ${MPI_ARGS} -tag-output -np 16 \
  ${MSCCLPP_ENV} \
  -npernode 8 /root/mscclpp/build/bin/mp_unit_tests -ip_port ${HEAD_IP}:20003
}

function run_pytests()
{
  echo "==================Run python tests================================"
  mpirun ${MPI_ARGS} -tag-output -np 16 \
  ${MSCCLPP_ENV} \
  -x MSCCLPP_HOME=/root/mscclpp -npernode 8 bash /root/mscclpp/test/deploy/pytest.sh
}

function run_py_benchmark()
{
  echo "==================Run python benchmark================================"
  mpirun ${MPI_ARGS} -np 16 \
  ${MSCCLPP_ENV} \
  -mca pml ob1 -mca btl ^openib -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x NCCL_SOCKET_IFNAME=eth0 \
  -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml \
  -x NCCL_NET_PLUGIN=none -x NCCL_IB_DISABLE=0 -x NCCL_MIN_NCHANNELS=32 -x NCCL_DEBUG=WARN -x NCCL_P2P_DISABLE=0 -x NCCL_SHM_DISABLE=0 \
  -x MSCCLPP_HOME=/root/mscclpp -npernode 8 python3 /root/mscclpp/python/mscclpp_benchmark/allreduce_bench.py
}

function run_allreduce_test()
{
  PLAN_DIR=/root/mscclpp/dsl_plans
  ALGO=/root/mscclpp/python/mscclpp/default_algos/allreduce_multi_nodes.py

  echo "=============Generate DSL plans on each node=============================="
  mpirun ${MPI_ARGS} -np 2 -npernode 1 \
    ${MSCCLPP_ENV} -x MSCCLPP_HOME=/root/mscclpp \
    bash -c "mkdir -p ${PLAN_DIR} && \
      python3 ${ALGO} --name allreduce_2nodes_1K_64K --num_gpus 16 --gpus_per_node 8 \
        --tbg 1 --min_message_size 1024 --max_message_size 65536 \
        > ${PLAN_DIR}/allreduce_2nodes_1K_64K.json"

  echo "=============Run DSL allreduce_multi_nodes on 2 nodes====================="
  for SIZE in 1K 16K 64K; do
    mpirun ${MPI_ARGS} -np 16 \
      ${MSCCLPP_ENV} -x MSCCLPP_HOME=/root/mscclpp -npernode 8 \
      python3 /root/mscclpp/python/test/executor_test.py \
        --execution_plan_path ${PLAN_DIR}/allreduce_2nodes_1K_64K.json \
        --size ${SIZE} --dtype float16 --in_place
  done
}

function run_allgather_test()
{
  PLAN_DIR=/root/mscclpp/dsl_plans
  ALGO=/root/mscclpp/python/mscclpp/default_algos/allgather_multi_nodes.py

  echo "=============Generate DSL allgather plans on each node===================="
  mpirun ${MPI_ARGS} -np 2 -npernode 1 \
    ${MSCCLPP_ENV} -x MSCCLPP_HOME=/root/mscclpp \
    bash -c "mkdir -p ${PLAN_DIR} && \
      python3 ${ALGO} --name allgather_2nodes_1K_8M --num_gpus 16 --gpus_per_node 8 \
        --no-in_place --min_message_size 1024 --max_message_size 8388608 \
        > ${PLAN_DIR}/allgather_2nodes_1K_8M.json"

  echo "=============Run DSL allgather_multi_nodes on 2 nodes====================="
  for SIZE in 1K 64K 1M; do
    mpirun ${MPI_ARGS} -np 16 \
      ${MSCCLPP_ENV} -x MSCCLPP_HOME=/root/mscclpp -npernode 8 \
      python3 /root/mscclpp/python/test/executor_test.py \
        --execution_plan_path ${PLAN_DIR}/allgather_2nodes_1K_8M.json \
        --size ${SIZE} --dtype float16
  done
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <mscclpp-test/allreduce_test/allgather_test/mp-ut/run_pytests/run_py_benchmark>"
    exit 1
fi
test_name=$1
case $test_name in
  mscclpp-test)
    echo "==================Run mscclpp-test on 2 nodes========================="
    run_mscclpp_test
    ;;
  allreduce_test)
    echo "==============Run DSL allreduce on 2 nodes============================="
    run_allreduce_test
    ;;
  allgather_test)
    echo "==============Run DSL allgather on 2 nodes============================="
    run_allgather_test
    ;;
  mp-ut)
    echo "==================Run mp-ut on 2 nodes================================"
    run_mp_ut
    ;;
  pytests)
    echo "==================Run python tests===================================="
    run_pytests
    ;;
  py-benchmark)
    echo "==================Run python benchmark================================"
    run_py_benchmark
    ;;
  *)
    echo "Unknown test name: $test_name"
    exit 1
    ;;
esac
