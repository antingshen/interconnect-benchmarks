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

struct config {
  MPI_Datatype *datatype;
  MPI_Op *op;
  int count;
  void* (*convert_to_datatype) (float *vec, int len);
  float* (*convert_from_datatype) (void *vec, int len);
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

// double benchmark_allreduce_with_correctness_check(int weight_count){

//   //verified that the time spent in Malloc is trivial compared to Allreduce.
//   float* weight_diff = (float*)malloc(weight_count * sizeof(float)); //sum all weight diffs here

//   double elapsedTime = 0;

//   fill_array_deterministic(weight_diff, weight_count);

//   double start = MPI_Wtime(); //in seconds

//   MPI_Allreduce(MPI_IN_PLACE, //weight_diff_local, //send
//       weight_diff, //recv
//       weight_count, //count
//       MPI_FLOAT, 
//       mpi_fp32sum, //op
//       MPI_COMM_WORLD);

//   elapsedTime += ( MPI_Wtime() - start );
//   elapsedTime = elapsedTime;
//   check_array_deterministic(weight_diff, weight_count); 

//   free(weight_diff);
//   return elapsedTime;
// }

// double benchmark_allreduce_with_correctness_check_halfPrecision(int weight_count){
//   float* weight_diff = (float*)malloc(weight_count * sizeof(float)); //sum all weight diffs here

//   double elapsedTime = 0;

//   fill_array_deterministic(weight_diff, weight_count);
//   half* weight_diff_half = vec_float_to_half(weight_diff, weight_count); 

//   double start = MPI_Wtime(); //in seconds

//   MPI_Allreduce(MPI_IN_PLACE, //weight_diff_local, //send
//       weight_diff_half, //recv
//       weight_count, //count
//       mpi_type_float16, 
//       mpi_fp16sum, //op
//       MPI_COMM_WORLD);

//   elapsedTime += ( MPI_Wtime() - start );
//   elapsedTime = elapsedTime;

//   free(weight_diff);
//   weight_diff = vec_half_to_float(weight_diff_half, weight_count);
//   // check_array_deterministic(weight_diff, weight_count); 

//   free(weight_diff);
//   return elapsedTime;
// }

//@param argv = [(optional) size of data to transfer]
int main (int argc, char **argv)
{
    int rank, nproc;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    int count = atoi(argv[1]);

    struct config config;
    MPI_Datatype byte_x2;
    MPI_Type_contiguous(2, MPI_BYTE, &byte_x2);
    MPI_Type_commit(&byte_x2);

    MPI_Op fp16_halfcpp;
    MPI_Op_create(&my_fp16_sum, 1, &fp16_halfcpp);

    config.datatype = &byte_x2;
    config.op = &fp16_halfcpp;
    config.convert_to_datatype = vec_float_to_half;
    config.convert_from_datatype = vec_half_to_float;

    config.count = count;

    int nRuns = 100;
    double total = 0;
    for (int i=0; i<nRuns; i++) {
        double elapsedTime = benchmark_allreduce(&config);
        total += elapsedTime;
        //verified: all ranks get roughly the same elapsedTime.
        if(rank == 0 && i >= 10 && i % 5 == 0)
            printf("%f M/s \n", (double)count*(i+1)/total/1e6);
    }

    MPI_Type_free(&config.datatype);
    MPI_Op_free(&config.op);

    MPI_Finalize();

    return 0;
}
