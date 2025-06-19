#include <immintrin.h>



template <typename T,  int blocky = 4, int blockx = 4>
inline void cpu_gemm_v2_1(const T* A, const T* B, T* C,
                          int M, int N, int K) {
    static_assert(blockx % 4 == 0, "blockx must be multiple of 4 for AVX");
    constexpr int vecWidth = 4; // AVX2 256-bit = 4 Ts
    for (int by = 0; by < M; by += blocky) {
        for (int bx = 0; bx < N; bx += blockx) {

            __m256d acc[blocky][blockx / vecWidth] = {};
            for(int k=0;k<K;k++){
                for(int iy=0;iy<blocky;iy++){   
                    __m256d fa = _mm256_broadcast_sd(&A[(by+iy)*K+k]); 
                    for(int ix=0;ix<blockx;ix+=vecWidth){
                        __m256d fb = _mm256_loadu_pd(&B[k*N + bx + ix]);
                        acc[iy][ix/vecWidth] = _mm256_fmadd_pd(fa, fb, acc[iy][ix/vecWidth]);
                    }
                }
            }
            // store
            for(int iy=0;iy<blocky;iy++){
                for(int ix=0;ix<blockx;ix+=vecWidth){
                    _mm256_storeu_pd(&C[(by+iy)*N + bx + ix], acc[iy][ix/vecWidth]);
                }
            }
        }
    }
}