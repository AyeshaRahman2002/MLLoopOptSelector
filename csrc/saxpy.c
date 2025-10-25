// MLLoopOptSelector/csrc/saxpy.c
#include "common.h"
// y = a*x + y (single-precision)
int main(int argc, char** argv){
    if(argc < 2){ fprintf(stderr, "usage: %s N\n", argv[0]); return 1; }
    int N = atoi(argv[1]);
    float a = 2.0f;
    float* x = (float*) aalloc(64, sizeof(float)*N);
    float* y = (float*) aalloc(64, sizeof(float)*N);
    fill(x, (size_t)N); fill(y, (size_t)N);

    double t1=0,t2=0,t3=0;
    for(int rep=0; rep<3; ++rep){
        double t0 = wall_time();
#ifdef TILE_TRANSFORM
        OMP_FOR
        for(int ii=0; ii<N; ii+=TILE_I){
            int im = (ii+TILE_I<N)? ii+TILE_I : N;
            VEC_PRAGMA
            UNROLL_PRAGMA
            for(int i=ii;i<im;++i) y[i] = a*x[i] + y[i];
        }
#else
        OMP_FOR
        for(int i=0;i<N;++i){ VEC_PRAGMA UNROLL_PRAGMA y[i] = a*x[i] + y[i]; }
#endif
        double t = wall_time() - t0;
        if(rep==0)t1=t; else if(rep==1)t2=t; else t3=t;
    }
    double med = median3(t1,t2,t3);
    float chk = checksum(y, (size_t)N);
    printf("{\"time_sec\": %.6f, \"checksum\": %.3f}\n", med, chk);
    free(x); free(y); return 0;
}
