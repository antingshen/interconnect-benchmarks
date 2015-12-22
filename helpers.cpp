
#include "helpers.h"
#include "immintrin.h"
#include "xmmintrin.h"
#include "emmintrin.h"
using namespace std;
using half_float::half;
using half_float::half_cast;
//thx: http://stackoverflow.com/questions/29638251/16-bit-float-mpi-reduce

// define custom reduce operation
void my_fp16_sum(void* invec, void* inoutvec, int *len,
              MPI_Datatype *datatype) {
    half* in = (half*)invec;
    half* inout = (half*)inoutvec;
    for (int i = 0; i < *len; i++) {
        *inout = *in + *inout; 
        in++;
        inout++;
    }
}

void my_fp32_sum(void* invec, void* inoutvec, int *len,
                  MPI_Datatype *datatype) {
  float *in = (float*)invec;
  float *inout = (float*)inoutvec;
  for (int i = 0; i < *len; i++) {
      *inout = *in + *inout; 
      in++;
      inout++;
  }
}

void my_fp16_sum_avx(void* invec, void* inoutvec, int *len,
                     MPI_Datatype *datatype) {
  __m128i *in = (__m128i*)invec;
  __m128i *inout = (__m128i*)inoutvec;
  __m256 ps1, ps2;
  __m128i ph_in, ph_inout;
  int i;
  for (i = 0; i < *len/8; i++) {
    ph_in = _mm_loadu_si128(in);
    ph_inout = _mm_loadu_si128(inout);
    ps1 = _mm256_cvtph_ps(ph_in);
    ps2 = _mm256_cvtph_ps(ph_inout);
    ph_inout = _mm256_cvtps_ph(_mm256_add_ps(ps1, ps2),  (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    _mm_storeu_si128(inout, ph_inout);
    in++;
    inout++;
  } 
  half *in_half = (half*)in;
  half *inout_half = (half*)inout;
  for (i = *len/8*8; i < *len; i++) {
    *inout_half = *in_half + *inout_half;
    in_half++;
    inout_half++;
  }
}

void my_fp32_sum_avx(void* invec, void* inoutvec, int *len,
                     MPI_Datatype *datatype) {
  float *in = (float*)invec;
  float *inout = (float*)inoutvec;
  __m256 packed_in, packed_inout;
  int i;
  for (i = 0; i < *len/8; i++) {
    packed_in = _mm256_loadu_ps(in);
    packed_inout = _mm256_loadu_ps(inout);
    packed_inout = _mm256_add_ps(packed_in, packed_inout);
    _mm256_storeu_ps(inout, packed_inout);
    in += 8;
    inout += 8;
  } 
  for (i = *len/8*8; i < *len; i++) {
    *inout = *in + *inout;
    in++;
    inout++;
  }
}

void my_nop_sum(void* invec, void* inoutvec, int *len,
                     MPI_Datatype *datatype) {
  return;
}

void my_fp32_charsum(void* invec, void* inoutvec, int *len,
                     MPI_Datatype *datatype) {
  char *in = (char*)invec;
  char *inout = (char*)inoutvec;
  for (int i = 0; i < *len*4; i++) {
      *inout = *in + *inout; 
      in++;
      inout++;
  }
}

void my_x32char_sum(void* invec, void* inoutvec, int *len,
                     MPI_Datatype *datatype) {
  char *in = (char*)invec;
  char *inout = (char*)inoutvec;
  for (int i = 0; i < *len*32; i++) {
      *inout = *in + *inout; 
      in++;
      inout++;
  }
}

void my_x32char_copy(void* invec, void* inoutvec, int *len,
                     MPI_Datatype *datatype) {
  memcpy(inoutvec, invec, *len*32);
  // char *in = (char*)invec;
  // char *inout = (char*)inoutvec;
  // for (int i = 0; i < *len*32; i++) {
  //     *inout = *in; 
  //     in++;
  //     inout++;
  // }
}


void* vec_float_to_half(float* vec, int len) {
  half* out = (half*)malloc(len * sizeof(half));
  for(int i=0; i<len; i++){
    out[i] = half_cast<half, std::round_to_nearest>(vec[i]);
  }
  return (void*)out;
}

float* vec_half_to_float(void* v, int len) {
  half* vec = (half*) v;
  float* out = (float*)malloc(len * sizeof(float));
  for(int i=0; i<len; i++){
    out[i] = half_cast<int, std::round_to_nearest>(vec[i]);
  }
  return out;
}

void* vec_float_to_char(float* vec, int len) {
  char* out = (char*)malloc(len * sizeof(char));
  for (int i=0; i<len; i++) {
    out[i] = (char)(vec[i]);
  }
  return out;
}

float* vec_char_to_float(void* v, int len) {
  char* vec = (char*) v;
  float* out = (float*)malloc(len * sizeof(float));
  for(int i=0; i<len; i++){
    out[i] = (float)(vec[i]);
  }
  return out;
}

void* vec_float_to_x32char(float* vec, int len) {
  char* out = (char*)malloc(len * 32 * sizeof(char));
  for (int i=0; i<len; i++) {
    for (int j=0; j<32; j++) {
      out[i*8+j] = (char)(vec[i]);
    }
  }
  return out;
}

float* vec_x32char_to_float(void* v, int len) {
  char* vec = (char*) v;
  float* out = (float*)malloc(len * sizeof(float));
  for(int i=0; i<len; i++){
    out[i] = 0;
    for (int j=0; j<32; j++) {
      out[i] += (float)(vec[i*8+j]);
    }
  }
  return out;
}

