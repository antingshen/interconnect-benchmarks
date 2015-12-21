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
void my_nop_sum(void* invec, void* inoutvec, int *len, MPI_Datatype *datatype);
void my_x32char_sum(void* invec, void* inoutvec, int *len, MPI_Datatype *datatype);
void my_x32char_copy(void* invec, void* inoutvec, int *len, MPI_Datatype *datatype);
void* vec_float_to_half(float* vec, int len);
float* vec_half_to_float(void* vec, int len);
void* vec_float_to_char(float* vec, int len);
float* vec_char_to_float(void* vec, int len);
void* vec_float_to_x32char(float* vec, int len);
float* vec_x32char_to_float(void* vec, int len);

#endif

