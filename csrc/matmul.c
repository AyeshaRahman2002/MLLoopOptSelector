// MLLoopOptSelector/csrc/matmul.c
#include "common.h"

// C = A (N x K) * B (K x M)
int main(int argc, char** argv){
    if(argc < 4){
        fprintf(stderr, "usage: %s N M K\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int K = atoi(argv[3]);

    float* A = (float*) aalloc(64, sizeof(float)*N*K);
    float* B = (float*) aalloc(64, sizeof(float)*K*M);
    float* C = (float*) aalloc(64, sizeof(float)*N*M);
    fill(A, (size_t)N*K);
    fill(B, (size_t)K*M);
    for(int i=0;i<N*M;i++) C[i]=0.0f;

    double t1=0,t2=0,t3=0;
    for(int rep=0; rep<3; ++rep){
        for(int i=0;i<N*M;i++) C[i]=0.0f;
        double t0 = wall_time();

#ifdef TILE_TRANSFORM
        OMP_FOR
        for(int ii=0; ii<N; ii+=TILE_I){
            for(int jj=0; jj<M; jj+=TILE_J){
                for(int kk=0; kk<K; kk+=TILE_K){
                    int i_max = ii + TILE_I < N ? ii + TILE_I : N;
                    int j_max = jj + TILE_J < M ? jj + TILE_J : M;
                    int k_max = kk + TILE_K < K ? kk + TILE_K : K;
                    for(int i=ii; i<i_max; ++i){
                        for(int j=jj; j<j_max; ++j){
                            VEC_PRAGMA
                            UNROLL_PRAGMA
                            for(int k=kk; k<k_max; ++k){
                                C[i*M + j] += A[i*K + k] * B[k*M + j];
                            }
                        }
                    }
                }
            }
        }
#else
        OMP_FOR
        for(int i=0;i<N;i++){
            for(int j=0;j<M;j++){
                float sum = 0.0f;
                VEC_PRAGMA
                UNROLL_PRAGMA
                for(int k=0;k<K;k++){
                    sum += A[i*K + k] * B[k*M + j];
                }
                C[i*M + j] = sum;
            }
        }
#endif
        double t = wall_time() - t0;
        if(rep==0) t1=t; else if(rep==1) t2=t; else t3=t;
    }

    double med = median3(t1,t2,t3);
    float chk = checksum(C, (size_t)N*M);
    printf("{\"time_sec\": %.6f, \"checksum\": %.3f}\n", med, chk);

    free(A); free(B); free(C);
    return 0;
}
