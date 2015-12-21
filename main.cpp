#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include <unistd.h>
#include "half.hpp"
#include "helpers.h"
using namespace std;
using half_float::half;

#define USE_FLOAT NULL
#define CHECK_CORRECTNESS 0

struct config {
  MPI_Datatype *datatype;
  MPI_Op *op;
  int count;
  void* (*convert_to_datatype) (float *vec, int len);
  float* (*convert_from_datatype) (void *vec, int len);
  double elapsed;
};

void fill_array_deterministic(float* buf, int len){
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  for(int i=0; i<len; i++){
    buf[i] = i%100 + rank; //arbitrary and reproducable... and different per rank.
  }
}

void check_array_deterministic(float* buf, int len){
  if (CHECK_CORRECTNESS == 0) {
    return;
  }
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int sum_ranks = 0;
  for(int i=0; i<nproc; i++){
    sum_ranks += i;
  }

  for(int i=0; i<len; i++){
    float ground_truth = nproc*(i%100) + sum_ranks;
    if(ground_truth != buf[i]){
      printf("Calculation Error\n");
      printf(" rank=%d, buf[%d]=%f, GT=%f\n", rank, i, buf[i], ground_truth); 
      exit(0);
    } 
  }
}

double benchmark_allreduce(struct config *config) {
  float* float_array = (float*)malloc(config->count * sizeof(float));
  fill_array_deterministic(float_array, config->count);

  void* datatype_array;
  if (config->convert_to_datatype == USE_FLOAT) {
    datatype_array = float_array;
  } else {
    datatype_array = config->convert_to_datatype(float_array, config->count);
    free(float_array);
  }

  double start_time = MPI_Wtime();
  MPI_Allreduce(MPI_IN_PLACE,
      datatype_array,
      config->count,
      *config->datatype,
      *config->op,
      MPI_COMM_WORLD);
  double elapsed_time = MPI_Wtime() - start_time;

  if (config->convert_to_datatype == USE_FLOAT) {
    check_array_deterministic((float*) datatype_array, config->count);
  } else {
    float_array = config->convert_from_datatype(datatype_array, config->count);
    check_array_deterministic((float*) float_array, config->count);
    free(float_array);
  }
  free(datatype_array);

  return elapsed_time;
}


int main (int argc, char **argv)
{
    int rank, nproc;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    int count = atoi(argv[1]);

    MPI_Datatype mpi_float = MPI_FLOAT;
    MPI_Datatype mpi_char = MPI_SIGNED_CHAR;
    MPI_Datatype byte_x2;
    MPI_Datatype byte_x32;
    MPI_Type_contiguous(2, MPI_BYTE, &byte_x2);
    MPI_Type_contiguous(32, MPI_BYTE, &byte_x32);
    MPI_Type_commit(&byte_x2);
    MPI_Type_commit(&byte_x32);

    MPI_Op fp16_halfcpp;
    MPI_Op fp16_avx;
    MPI_Op fp32_sum;
    MPI_Op fp32_avx;
    MPI_Op nop;
    MPI_Op mpi_sum = MPI_SUM;
    MPI_Op x32char_sum;
    MPI_Op x32char_copy;
    MPI_Op_create(&my_fp16_sum, 1, &fp16_halfcpp);
    MPI_Op_create(&my_fp16_sum_avx, 1, &fp16_avx);
    MPI_Op_create(&my_fp32_sum, 1, &fp32_sum);
    MPI_Op_create(&my_fp32_sum_avx, 1, &fp32_avx);
    MPI_Op_create(&my_nop_sum, 1, &nop);
    MPI_Op_create(&my_x32char_sum, 1, &x32char_sum);
    MPI_Op_create(&my_x32char_copy, 1, &x32char_copy);

    //// FLOAT ////
    struct config conf_fp32_mpisum;
    conf_fp32_mpisum.datatype = &mpi_float;
    conf_fp32_mpisum.op = &mpi_sum;
    conf_fp32_mpisum.convert_to_datatype = USE_FLOAT;
    conf_fp32_mpisum.convert_from_datatype = USE_FLOAT;
    conf_fp32_mpisum.count = count;
    conf_fp32_mpisum.elapsed = 0;

    struct config conf_fp32_sum = conf_fp32_mpisum;
    conf_fp32_sum.op = &fp32_sum;

    struct config conf_fp32_avx = conf_fp32_sum;
    conf_fp32_avx.op = &fp32_avx;

    struct config conf_fp32_nop = conf_fp32_sum;
    conf_fp32_nop.op = &nop;

    //// HALF ////
    struct config conf_fp16_halfcpp;
    conf_fp16_halfcpp.datatype = &byte_x2;
    conf_fp16_halfcpp.op = &fp16_halfcpp;
    conf_fp16_halfcpp.convert_to_datatype = vec_float_to_half;
    conf_fp16_halfcpp.convert_from_datatype = vec_half_to_float;
    conf_fp16_halfcpp.count = count;
    conf_fp16_halfcpp.elapsed = 0;

    struct config conf_fp16_avx = conf_fp16_halfcpp;
    conf_fp16_avx.op = &fp16_avx;

    struct config conf_fp16_nop = conf_fp16_halfcpp;
    conf_fp16_nop.op = &nop;

    //// CHAR //// 
    struct config conf_char_mpisum;
    conf_char_mpisum.datatype = &mpi_char;
    conf_char_mpisum.op = &mpi_sum;
    conf_char_mpisum.convert_to_datatype = vec_float_to_char;
    conf_char_mpisum.convert_from_datatype = vec_char_to_float;
    conf_char_mpisum.count = count;
    conf_char_mpisum.elapsed = 0;

    //// PACKED_HALF ////
    struct config conf_x32char_sum;
    conf_x32char_sum.datatype = &byte_x32;
    conf_x32char_sum.op = &x32char_sum;
    conf_x32char_sum.convert_to_datatype = vec_float_to_x32char;
    conf_x32char_sum.convert_from_datatype = vec_x32char_to_float;
    conf_x32char_sum.count = count;
    conf_x32char_sum.elapsed = 0;

    struct config conf_x32char_nop = conf_x32char_sum;
    conf_x32char_nop.op = &nop;

    struct config conf_x32char_copy = conf_x32char_sum;
    conf_x32char_copy.op = &x32char_copy;

    struct config configs[] = {
      conf_fp32_mpisum,
      conf_fp32_sum,
      conf_fp32_avx,
      conf_fp32_nop,
      conf_fp16_halfcpp,
      conf_fp16_avx,
      conf_fp16_nop,
      conf_char_mpisum,
      conf_x32char_sum,
      conf_x32char_nop,
      conf_x32char_copy
    };
    int num_configs = 11;

    int nRuns = 100;
    for (int i=0; i<nRuns; i++) {
      if (rank == 0) {
        printf("--- %d ---\n", i);
      }
      for (int c=0; c<num_configs; c++) {
        // printf("%d\n", configs[c].count);
        configs[c].elapsed += benchmark_allreduce(&configs[c]);
        //verified: all ranks get roughly the same elapsedTime.
        if(rank == 0 && i >= 1 && i % 2 == 0)
            printf("%f M/s \n", (double)count*(i+1)/configs[c].elapsed/1e6);
      }
    }

    MPI_Type_free(&byte_x2);
    MPI_Op_free(&fp16_halfcpp);
    MPI_Op_free(&fp16_avx);

    MPI_Finalize();

    return 0;
}
