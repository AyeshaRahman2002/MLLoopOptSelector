// MLLoopOptSelector/csrc/conv1d.c
#include "common.h"

// y[i] = sum_{k=0..K-1} x[i+k] * w[k], valid conv, output length N-K+1
int main(int argc, char** argv){
    if(argc < 2){
        fprintf(stderr, "usage: %s N [K]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int K = (argc >= 3) ? atoi(argv[2]) : (N/16 > 2 ? N/16 : 3);
    int O = N - K + 1;
    if (O <= 0){
        fprintf(stderr, "N must be >= K\n");
        return 2;
    }

    float* x = (float*) aalloc(64, sizeof(float)*N);
    float* w = (float*) aalloc(64, sizeof(float)*K);
    float* y = (float*) aalloc(64, sizeof(float)*O);
    fill(x, (size_t)N);
    fill(w, (size_t)K);
    for(int i=0;i<O;i++) y[i]=0.0f;

    double t1=0,t2=0,t3=0;
    for(int rep=0; rep<3; ++rep){
        for(int i=0;i<O;i++) y[i]=0.0f;
        double t0 = wall_time();

#ifdef TILE_TRANSFORM
        OMP_FOR
        for(int ii=0; ii<O; ii+=TILE_I){
            int i_max = ii + TILE_I < O ? ii + TILE_I : O;
            for(int i=ii; i<i_max; ++i){
                float acc = 0.0f;
                VEC_PRAGMA
                UNROLL_PRAGMA
                for(int k=0; k<K; ++k){
                    acc += x[i+k] * w[k];
                }
                y[i] = acc;
            }
        }
#else
        OMP_FOR
        for(int i=0;i<O;i++){
            float acc = 0.0f;
            VEC_PRAGMA
            UNROLL_PRAGMA
            for(int k=0; k<K; ++k){
                acc += x[i+k] * w[k];
            }
            y[i] = acc;
        }
#endif
        double t = wall_time() - t0;
        if(rep==0) t1=t; else if(rep==1) t2=t; else t3=t;
    }

    double med = median3(t1,t2,t3);
    float chk = checksum(y, (size_t)O);
    printf("{\"time_sec\": %.6f, \"checksum\": %.3f}\n", med, chk);

    free(x); free(w); free(y);
    return 0;
}
