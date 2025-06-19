#include <immintrin.h>

template <typename T>
inline void micro_kernel(const T *A, const T *B, T *C,
                         int N, int K)
{ // make 4x4 to C
    __m256d c_reg_0 = {}, c_reg_1 = {}, c_reg_2 = {}, c_reg_3 = {};

    for (int k = 0; k < K; k++)
    {                                                   // accumulate
        __m256d b_reg = _mm256_loadu_pd(&B[k * N]); // 假设B按行主序存储4个double
        __m256d a_reg_0 = _mm256_broadcast_sd(&A[0 * K + k]);
        __m256d a_reg_1 = _mm256_broadcast_sd(&A[1 * K + k]);
        __m256d a_reg_2 = _mm256_broadcast_sd(&A[2 * K + k]);
        __m256d a_reg_3 = _mm256_broadcast_sd(&A[3 * K + k]);
    
        c_reg_0 = _mm256_fmadd_pd(a_reg_0, b_reg, c_reg_0);
        c_reg_1 = _mm256_fmadd_pd(a_reg_1, b_reg, c_reg_1);
        c_reg_2 = _mm256_fmadd_pd(a_reg_2, b_reg, c_reg_2);
        c_reg_3 = _mm256_fmadd_pd(a_reg_3, b_reg, c_reg_3);
    }

    // store the results back to C
    _mm256_storeu_pd(&C[0], c_reg_0);
    _mm256_storeu_pd(&C[N], c_reg_1);
    _mm256_storeu_pd(&C[N * 2], c_reg_2);
    _mm256_storeu_pd(&C[N * 3], c_reg_3);
}

template <typename T, int blocky = 4, int blockx = 4>
inline void cpu_gemm_v2_2(const T *A, const T *B, T *C,
                          int M, int N, int K)
{
    static_assert(blockx % 4 == 0, "blockx must be multiple of 4 for AVX");
    constexpr int vecWidth = 4; // AVX2 256-bit = 4 Ts
    for (int by = 0; by < M; by += blocky)
    {
        for (int bx = 0; bx < N; bx += blockx)
        {
            micro_kernel(&A[by * K], &B[bx], &C[by * N + bx],  N, K);
        }
    }
}