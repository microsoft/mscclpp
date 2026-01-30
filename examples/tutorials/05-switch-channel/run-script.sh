module load mpi/hpcx

export LD_LIBRARY_PATH=/home/azhpcuser/mahdieh/mscclpp/build/lib/:$LD_LIBRARY_PATH

FE_NETDEV=enP22p1s0f1 # This is the MANA NIC netdev name.

MPI_ARGS=""
MPI_ARGS+=" -mca coll_hcoll_enable 0 --mca btl tcp,vader,self --mca pml ob1 --mca btl_tcp_if_include $FE_NETDEV"
MPI_ARGS+=" -x UCX_IB_GID_INDEX=3 -x UCX_NET_DEVICES=$FE_NETDEV"

mpirun -n 2 --map-by ppr:1:node --hostfile ./hostfile -x UCX_TLS=sm,self,tcp -x UCX_RC_MLX5_TM_ENABLE=0 -x LD_LIBRARY_PATH $MPI_ARGS ./bidir_switch_channel
