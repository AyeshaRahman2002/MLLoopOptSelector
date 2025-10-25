// MLLoopOptSelector/csrc/stencil2d.c
#include "common.h"

// 5-point stencil on NxM grid
int main(int argc, char** argv){
    if(argc < 3){
        fprintf(stderr, "usage: %s N M\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    float* A = (float*) aalloc(64, sizeof(float)*N*M);
    float* B = (float*) aalloc(64, sizeof(float)*N*M);
    fill(A, (size_t)N*M);
    for(int i=0;i<N*M;i++) B[i]=0.0f;

    double t1=0,t2=0,t3=0;
    for(int rep=0; rep<3; ++rep){
        double t0 = wall_time();
#ifdef TILE_TRANSFORM
        OMP_FOR
        for(int ii=1; ii<N-1; ii+=TILE_I){
            int i_max = ii + TILE_I < N-1 ? ii + TILE_I : N-1;
            for(int jj=1; jj<M-1; jj+=TILE_J){
                int j_max = jj + TILE_J < M-1 ? jj + TILE_J : M-1;
                for(int i=ii; i<i_max; ++i){
                    VEC_PRAGMA
                    UNROLL_PRAGMA
                    for(int j=jj; j<j_max; ++j){
                        B[i*M + j] = 0.25f*A[i*M + j] + 0.125f*(A[(i-1)*M + j] + A[(i+1)*M + j] + A[i*M + (j-1)] + A[i*M + (j+1)]);
                    }
                }
            }
        }
#else
        OMP_FOR
        for(int i=1; i<N-1; ++i){
            VEC_PRAGMA
            UNROLL_PRAGMA
            for(int j=1; j<M-1; ++j){
                B[i*M + j] = 0.25f*A[i*M + j] + 0.125f*(A[(i-1)*M + j] + A[(i+1)*M + j] + A[i*M + (j-1)] + A[i*M + (j+1)]);
            }
        }
#endif
        double t = wall_time() - t0;
        if(rep==0) t1=t; else if(rep==1) t2=t; else t3=t;
    }

    double med = median3(t1,t2,t3);
    float chk = checksum(B, (size_t)N*M);
    printf("{\"time_sec\": %.6f, \"checksum\": %.3f}\n", med, chk);

    free(A); free(B);
    return 0;
}
