#include "bootstrap.h"

#include <memory>

#include <mpi.h>

int main()
{
    int rank, worldSize;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::shared_ptr<MscclppBootstrap> bootstrap(new MscclppBootstrap(rank, worldSize));
    bootstrap->Initialize("costsim-dev-00000A:50000");
    // UniqueId id;
    // if (rank == 0)
    //     id = bootstrap->GetUniqueId();
    // MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    // bootstrap->Initialize(id);
    // need to call initialization first

    MPI_Finalize();
    return 0;
}