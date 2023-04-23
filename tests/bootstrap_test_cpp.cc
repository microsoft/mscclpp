#include "bootstrap.h"

#include <memory>

#include <mpi.h>

int main()
{
    int rank, worldSize;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::shared_ptr<Bootstrap> bootstrap(new MscclppBootstrap("", rank, worldSize));
    // need to call initialization first

    MPI_Finalize();
    return 0;
}