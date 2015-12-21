#ifndef __HELPERS_H__
#define __HELPERS_H__

#include "mpi.h"
#include "half.hpp"

// static MPI_Datatype mpi_type_float16;
static MPI_Op mpi_fp16sum;
static MPI_Op mpi_fp32sum;

void my_fp16_sum(void* invec, void* inoutvec, int *len, MPI_Datatype *datatype);
void my_fp32_sum(void* invec, void* inoutvec, int *len, MPI_Datatype *datatype);
void my_fp16_sum_avx(void* invec, void* inoutvec, int *len, MPI_Datatype *datatype);
void my_fp32_sum_avx(void* invec, void* inoutvec, int *len, MPI_Datatype *datatype);
half_float::half* vec_float_to_half(float* vec, int len);
float* vec_half_to_float(half_float::half* vec, int len);

#endif

