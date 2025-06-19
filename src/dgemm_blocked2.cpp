#include <immintrin.h> // AVX2
#define BLOCK 4

template <typename T, int blocky = 64, int blockx = 64>
inline void cpu_gemm_v2(const T *A, const T *B, T *C,
                        int M, int N, int K)
{
    constexpr int vecWidth = 4; 

    for (int by = 0; by < M; by += blocky)
    {
        for (int bx = 0; bx < N; bx += blockx * BLOCK)
        {
            __m256d tc[blocky][blockx] = {}; //actalliy the size of c tile is blocky * blockx * vecWidth

            for (int k = 0; k < K; ++k)
            {
                for (int y = 0; y < blocky; ++y)
                {
                    __m256d a4 = _mm256_broadcast_sd(&A[(by + y) * K + k]);  //deal with 1 line 
                    for (int x = 0; x < blockx; ++x)
                    {
                        // __m256d b4 = _mm256_loadu_pd(&B[((bx + x * BLOCK) * N + k * vecWidth) / vecWidth]);
                        // tc[y][x] = _mm256_fmadd_pd(a4, Bfm[((bx+x*BLOCK)*K + k*vecWidth)/vecWidth], tc[y][x]);
                        __m256d b4 = _mm256_loadu_pd(&B[(k * N  + (x +bx) * BLOCK * vecWidth)/vecWidth]);
                        tc[y][x] = _mm256_fmadd_pd(a4, b4, tc[y][x]);
                    }
                }
            }

            // Store back tc[y][x]
            for (int y = 0; y < blocky; ++y)
            {
                for (int x = 0; x < blockx; ++x)
                {
                    _mm256_storeu_pd(&C[((by + y) * N + bx + x * BLOCK) / vecWidth], tc[y][x]);
                }
            }
        }
    }
    // free(Bf); // 释放预处理的 B 缓存
}

