#include "mscclpp.hpp"

#include <memory>

#include <mpi.h>

int main()
{
    int rank, worldSize;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::shared_ptr<mscclpp::DefaultBootstrap> bootstrap(new mscclpp::DefaultBootstrap(rank, worldSize));
    // bootstrap->Initialize("costsim-dev-00000A:50000");
    mscclpp::UniqueId id;
    if (rank == 0)
        id = bootstrap->createUniqueId();
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    bootstrap->initialize(id);

    std::vector<int> tmp(worldSize, 0);
    tmp[rank] = rank+1;
    bootstrap->allGather(tmp.data(), sizeof(int));
    for (int i = 0; i < worldSize; i++){
        if (tmp[i] != i+1)
            printf("error AllGather: rank %d: tmp[%d] = %d\n", rank, i, tmp[i]); 
    }
    printf("rank %d: AllGather test passed!\n", rank);

    bootstrap->barrier();
    printf("rank %d: Barrier test passed!\n", rank);

    for (int i = 0; i < worldSize; i++){
        if (i == rank)
            continue;
        int msg1 = (rank + 1)*2;
        int msg2 = (rank + 1)*2+1;
        bootstrap->send(&msg1, sizeof(int), i, 0);
        bootstrap->send(&msg2, sizeof(int), i, 1);
    }

    for (int i = 0; i < worldSize; i++){
        if (i == rank)
            continue;
        int msg1 = 0;
        int msg2 = 0;
        // recv them in the opposite order to check correctness
        bootstrap->recv(&msg2, sizeof(int), i, 1);
        bootstrap->recv(&msg1, sizeof(int), i, 0);
        if (msg1 != (i+1)*2 || msg2 != (i+1)*2+1)
            printf("error Send/Recv: rank %d: msg1 = %d, msg2 = %d\n", rank, msg1, msg2);
    }
    printf("rank %d: Send/Recv test passed!\n", rank);

    MPI_Finalize();
    return 0;
}