#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
using namespace std;

int main(int argc, char **argv)
{
    int rank, nproc;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    MPI_Finalize();
}