module load mpi/hpcx #mpi/hpcx-mrc #mpi/hpcx-mrc-2.23.1

MPI_ARGS=""
MPI_ARGS+=" -x CUDA_VISIBLE_DEVICES=1 -mca coll_hcoll_enable 0 --mca coll ^ucc,hcoll --mca btl tcp,vader,self --mca pml ob1   --mca oob_tcp_if_include enP22p1s0f1 --mca btl_tcp_if_include enP22p1s0f1"
MPI_ARGS+=" -x MSCCLPP_IBV_SO=/opt/microsoft/mrc/Azure-Compute-AI-HPC-Perf-verbs-mrc/libibverbs.so -x UCX_NET_DEVICES=enP22p1s0f1 -x LD_LIBRARY_PATH=/opt/microsoft/mrc/Azure-Compute-AI-HPC-Perf-verbs-mrc/mrc-header-lib:$LD_LIBRARY_PATH"
MPI_ARGS+=" -x MSCCLPP_SOCKET_IFNAME=enP22p1s0f1 -x MSCCLPP_IBV_MODE=host-no-atomic  -x VMRC_LIBMRC_SO=/opt/mellanox/doca/lib/aarch64-linux-gnu/libnv_mrc.so"
MPI_ARGS+=" -x VMRC_LIBIBVERBS_SO=/lib/aarch64-linux-gnu/libibverbs.so.1 -x PATH=/home/azhpcuser/mahdieh/mscclpp-unittest/mscclpp/mscclpp/bin/:$PATH "
MPI_ARGS+=" -x MSCCLPP_LOG_LEVEL=ERROR -x MSCCLPP_DEBUG=ERROR  -x MSCCLPP_IB_GID_INDEX=3 -x MSCCLPP_HCA_DEVICES=mlx5_0"
MPI_ARGS+=" /home/azhpcuser/mahdieh/mscclpp-unittest/mscclpp/mscclpp/bin/python3   /home/azhpcuser/mahdieh/mscclpp-unittest/mscclpp/python/test/executor_test.py   -path /home/azhpcuser/mahdieh/mscclpp-unittest/mscclpp/sendrecv.json"


mpirun -np 2 --hostfile ./hosts --map-by ppr:1:node  $MPI_ARGS --size 1K
