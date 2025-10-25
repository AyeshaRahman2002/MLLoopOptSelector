// MLLoopOptSelector/csrc/common.h
#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

// wall time in seconds
static inline double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

// deterministic filler
static inline void fill(float* a, size_t n){
    for(size_t i=0;i<n;i++){
        a[i] = (float)((i*1315423911u) % 1000) / 1000.0f;
    }
}

// simple checksum to avoid DCE and to sanity-check outputs
static inline float checksum(const float* a, size_t n){
    double s=0.0;
    for(size_t i=0;i<n;i++) s += a[i];
    return (float)s;
}

// median of 3
static inline double median3(double a, double b, double c){
    double x = a>b ? a : b;
    double y = a>b ? b : a;
    double z = c;
    if (z > x) return x;
    if (z < y) return y;
    return z;
}

// aligned allocation helper (portable on macOS)
static inline void* aalloc(size_t alignment, size_t nbytes){
    void* p=NULL;
    if (posix_memalign(&p, alignment, nbytes)!=0) return NULL;
    return p;
}

#ifndef TILE_I
#define TILE_I 32
#endif
#ifndef TILE_J
#define TILE_J 32
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

// pragma toggles
#ifdef PRAGMA_VEC
#define VEC_PRAGMA _Pragma("clang loop vectorize(enable) interleave(enable)")
#else
#define VEC_PRAGMA
#endif

#ifdef PRAGMA_UNROLL
#define UNROLL_PRAGMA _Pragma("clang loop unroll_count(4)")
#else
#define UNROLL_PRAGMA
#endif

#ifdef OMP_PARALLEL
#include <omp.h>
#define OMP_FOR _Pragma("omp parallel for schedule(static)")
#else
#define OMP_FOR
#endif