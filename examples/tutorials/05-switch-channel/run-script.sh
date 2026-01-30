mpirun -n 2 -x UCX_TLS=sm,self,tcp -x UCX_RC_MLX5_TM_ENABLE=0 ./bidir_switch_channel
