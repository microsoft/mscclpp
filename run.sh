
module load mpi/hpcx #mpi/hpcx-mrc #mpi/hpcx-mrc-2.23.1

# Check if the number of arguments is exactly 2
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <hostfile> <nnodes> "
    exit 1
fi

HOSTFILE=$1
NNODES=$2

MPI_ARGS=""
MPI_ARGS+=" -mca coll_hcoll_enable 0 --mca coll ^ucc,hcoll --mca btl tcp,vader,self --mca pml ob1   --mca oob_tcp_if_include enP22p1s0f1 --mca btl_tcp_if_include enP22p1s0f1"
MPI_ARGS+=" -x MSCCLPP_IBV_SO=/opt/microsoft/mrc/Azure-Compute-AI-HPC-Perf-verbs-mrc/libibverbs.so -x UCX_NET_DEVICES=enP22p1s0f1 -x LD_LIBRARY_PATH=/opt/microsoft/mrc/Azure-Compute-AI-HPC-Perf-verbs-mrc/mrc-header-lib:$LD_LIBRARY_PATH"
MPI_ARGS+=" -x MSCCLPP_SOCKET_IFNAME=enP22p1s0f1 -x MSCCLPP_IBV_MODE=host-no-atomic  -x VMRC_LIBMRC_SO=/opt/mellanox/doca/lib/aarch64-linux-gnu/libnv_mrc.so"
MPI_ARGS+=" -x VMRC_LIBIBVERBS_SO=/lib/aarch64-linux-gnu/libibverbs.so.1 -x PATH=/home/azhpcuser/mahdieh/mscclpp/mscclpp2/bin/:$PATH "
MPI_ARGS+=" -x MSCCLPP_LOG_LEVEL=ERROR -x MSCCLPP_DEBUG=ERROR  -x MSCCLPP_IB_GID_INDEX=3 -x MSCCLPP_HCA_DEVICES=mlx5_1,mlx5_0,mlx5_3,mlx5_2"
MPI_ARGS+=" /home/azhpcuser/mahdieh/mscclpp/mscclpp/bin/python3   /home/azhpcuser/mahdieh/mscclpp/python/test/executor_test.py   -path /home/azhpcuser/mahdieh/mscclpp/test.json"


mpirun -np $((4*$NNODES)) --hostfile $HOSTFILE --map-by ppr:4:node  $MPI_ARGS --size 1G:1G:2 --n_iters 30 #--n_graph_iters 100

#mpirun -np 8  --hostfile /home/azhpcuser/binyli/hostfile   --map-by ppr:4:node   -mca coll_hcoll_enable 0 --mca btl tcp,vader,self --mca pml ob1   --mca oob_tcp_if_include enP22p1s0f1 --mca btl_tcp_if_include enP22p1s0f1 -x MSCCLPP_IBV_SO=/opt/microsoft/mrc/Azure-Compute-AI-HPC-Perf-verbs-mrc/libibverbs.so -x UCX_NET_DEVICES=enP22p1s0f1 -x LD_LIBRARY_PATH=/opt/microsoft/mrc/Azure-Compute-AI-HPC-Perf-verbs-mrc/mrc-header-lib:$LD_LIBRARY_PATH -x MSCCLPP_IBV_MODE=host-no-atomic  -x MSCCLPP_SOCKET_IFNAME=enP22p1s0f1   -x VMRC_LIBMRC_SO=/opt/mellanox/doca/lib/aarch64-linux-gnu/libnv_mrc.so   -x VMRC_LIBIBVERBS_SO=/lib/aarch64-linux-gnu/libibverbs.so.1     -x MSCCLPP_HCA_DEVICES=mlx5_1,mlx5_0,mlx5_3,mlx5_2   -x PATH=/home/azhpcuser/binyli/mscclpp/bin:$PATH  -x MSCCLPP_LOG_LEVEL=ERROR -x MSCCLPP_DEBUG=WARN -x MSCCLPP_IB_GID_INDEX=3 /home/azhpcuser/binyli/mscclpp/bin/python3   /home/azhpcuser/binyli/mscclpp/python/test/executor_test.py   -path /home/azhpcuser/binyli/mscclpp/test.json   --size 1G --n_iters 30
