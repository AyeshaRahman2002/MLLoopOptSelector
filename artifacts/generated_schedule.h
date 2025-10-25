#pragma once
#ifdef __cplusplus
#  include <cstring>
#else
#  include <string.h>
#endif
static inline const char* schedule_for(const char* kernel, int N){
  int B = (N+63)/64*64; if(B>1024) B=1024;
  if (!strcmp(kernel,"conv1d") && B==128) return "vec_pragma";
  if (!strcmp(kernel,"conv1d") && B==192) return "vec_pragma";
  if (!strcmp(kernel,"conv1d") && B==256) return "vec_pragma";
  if (!strcmp(kernel,"conv1d") && B==320) return "vec_pragma";
  if (!strcmp(kernel,"conv1d") && B==384) return "vec_pragma";
  if (!strcmp(kernel,"conv1d") && B==448) return "vec_pragma";
  if (!strcmp(kernel,"conv1d") && B==512) return "vec_pragma";
  if (!strcmp(kernel,"matmul") && B==128) return "omp";
  if (!strcmp(kernel,"matmul") && B==192) return "omp";
  if (!strcmp(kernel,"matmul") && B==256) return "tile_unroll_omp";
  if (!strcmp(kernel,"matmul") && B==320) return "tile_unroll_omp";
  if (!strcmp(kernel,"matmul") && B==384) return "tile_unroll_omp";
  if (!strcmp(kernel,"matmul") && B==448) return "tile_unroll_omp";
  if (!strcmp(kernel,"matmul") && B==512) return "tile_unroll_omp";
  if (!strcmp(kernel,"stencil2d") && B==128) return "baseline";
  if (!strcmp(kernel,"stencil2d") && B==192) return "baseline";
  if (!strcmp(kernel,"stencil2d") && B==256) return "baseline";
  if (!strcmp(kernel,"stencil2d") && B==320) return "baseline";
  if (!strcmp(kernel,"stencil2d") && B==384) return "omp";
  if (!strcmp(kernel,"stencil2d") && B==448) return "omp";
  if (!strcmp(kernel,"stencil2d") && B==512) return "omp";
  return "baseline";
}